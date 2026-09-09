#!/usr/bin/env python3
"""Remove only stale cross-view matches from a copied COLMAP database."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Mapping

from prepare_lightglue_workspace import (
    DEFAULT_MANIFEST,
    DEFAULT_SOURCE_WORKSPACE,
    DEFAULT_WORKSPACE,
    WorkspacePreparationError,
    load_manifest,
    resolve_project_path,
)


MAX_IMAGE_ID = 2**31 - 1
MATCH_TABLES = ("matches", "two_view_geometries")
DEFAULT_REPORT_NAME = "side_top_cleanup_report.json"


class DatabaseCleanupError(RuntimeError):
    """A user-correctable copied-database cleanup error."""


@dataclass(frozen=True)
class GroupCounts:
    records: int = 0
    rows: int = 0


@dataclass(frozen=True)
class DatabaseSnapshot:
    images: int
    keypoint_images: int
    keypoint_rows: int
    descriptor_images: int
    descriptor_rows: int
    match_groups: Mapping[str, Mapping[str, GroupCounts]]


@dataclass(frozen=True)
class CleanupResult:
    database: Path
    target_group: str
    dry_run: bool
    before: DatabaseSnapshot
    after: DatabaseSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transactionally remove only side-to-top match records from the copied "
            "LightGlue COLMAP database."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Prepared isolated LightGlue workspace.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Combined multiview dataset manifest.",
    )
    parser.add_argument("--left-view", default="side")
    parser.add_argument("--right-view", default="top45")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report exactly what would be removed without writing to the database.",
    )
    return parser.parse_args()


def pair_id_from_image_ids(image_id1: int, image_id2: int) -> int:
    if image_id1 == image_id2:
        raise DatabaseCleanupError("A COLMAP match pair cannot repeat an image ID.")
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    if image_id1 <= 0 or image_id2 >= MAX_IMAGE_ID:
        raise DatabaseCleanupError(
            f"COLMAP image IDs must be between 1 and {MAX_IMAGE_ID - 1}."
        )
    return image_id1 * MAX_IMAGE_ID + image_id2


def image_ids_from_pair_id(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    if image_id1 <= 0 or image_id2 <= 0 or image_id1 >= image_id2:
        raise DatabaseCleanupError(f"Invalid COLMAP pair_id: {pair_id}")
    return image_id1, image_id2


def group_label(view1: str, view2: str) -> str:
    return "<->".join(sorted((view1, view2)))


def sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro&immutable=1"


def manifest_views(path: Path) -> dict[str, str]:
    sequences = load_manifest(path)
    return {
        frame.filename: frame.view
        for sequence in sequences.values()
        for frame in sequence
    }


def validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required = {
        "images",
        "keypoints",
        "descriptors",
        "matches",
        "two_view_geometries",
    }
    missing = required - tables
    if missing:
        raise DatabaseCleanupError(
            "COLMAP database is missing table(s): " + ", ".join(sorted(missing))
        )


def snapshot_database(
    connection: sqlite3.Connection, filename_to_view: Mapping[str, str]
) -> DatabaseSnapshot:
    validate_schema(connection)
    integrity = connection.execute("PRAGMA quick_check").fetchone()
    if integrity != ("ok",):
        raise DatabaseCleanupError(f"SQLite quick_check failed: {integrity}")

    image_names = {
        image_id: name
        for image_id, name in connection.execute("SELECT image_id, name FROM images")
    }
    database_filenames = set(image_names.values())
    manifest_filenames = set(filename_to_view)
    if database_filenames != manifest_filenames:
        missing = manifest_filenames - database_filenames
        extra = database_filenames - manifest_filenames
        raise DatabaseCleanupError(
            "Manifest/database image mismatch: "
            f"{len(missing)} missing and {len(extra)} unexpected image(s)."
        )

    match_groups: dict[str, dict[str, GroupCounts]] = {}
    for table in MATCH_TABLES:
        mutable_counts: dict[str, list[int]] = {}
        for pair_id, rows in connection.execute(
            f"SELECT pair_id, rows FROM {table}"
        ):
            image_id1, image_id2 = image_ids_from_pair_id(pair_id)
            try:
                filename1 = image_names[image_id1]
                filename2 = image_names[image_id2]
            except KeyError as error:
                raise DatabaseCleanupError(
                    f"{table} pair {pair_id} refers to a missing image ID."
                ) from error
            label = group_label(
                filename_to_view[filename1], filename_to_view[filename2]
            )
            counts = mutable_counts.setdefault(label, [0, 0])
            counts[0] += 1
            counts[1] += rows
        match_groups[table] = {
            label: GroupCounts(records=counts[0], rows=counts[1])
            for label, counts in sorted(mutable_counts.items())
        }

    keypoint_images, keypoint_rows = connection.execute(
        "SELECT COUNT(1), COALESCE(SUM(rows), 0) FROM keypoints"
    ).fetchone()
    descriptor_images, descriptor_rows = connection.execute(
        "SELECT COUNT(1), COALESCE(SUM(rows), 0) FROM descriptors"
    ).fetchone()
    return DatabaseSnapshot(
        images=len(image_names),
        keypoint_images=keypoint_images,
        keypoint_rows=keypoint_rows,
        descriptor_images=descriptor_images,
        descriptor_rows=descriptor_rows,
        match_groups=match_groups,
    )


def target_pair_ids(
    connection: sqlite3.Connection,
    filename_to_view: Mapping[str, str],
    target_group: str,
) -> dict[str, tuple[int, ...]]:
    image_names = dict(connection.execute("SELECT image_id, name FROM images"))
    result: dict[str, tuple[int, ...]] = {}
    for table in MATCH_TABLES:
        pair_ids = []
        for (pair_id,) in connection.execute(f"SELECT pair_id FROM {table}"):
            image_id1, image_id2 = image_ids_from_pair_id(pair_id)
            label = group_label(
                filename_to_view[image_names[image_id1]],
                filename_to_view[image_names[image_id2]],
            )
            if label == target_group:
                pair_ids.append(pair_id)
        result[table] = tuple(pair_ids)
    return result


def validate_transition(
    before: DatabaseSnapshot,
    after: DatabaseSnapshot,
    target_group: str,
) -> None:
    feature_fields = (
        "images",
        "keypoint_images",
        "keypoint_rows",
        "descriptor_images",
        "descriptor_rows",
    )
    for field in feature_fields:
        if getattr(before, field) != getattr(after, field):
            raise DatabaseCleanupError(
                f"Cleanup unexpectedly changed database field '{field}'."
            )
    for table in MATCH_TABLES:
        before_groups = dict(before.match_groups[table])
        after_groups = dict(after.match_groups[table])
        before_groups.pop(target_group, None)
        after_target = after_groups.pop(target_group, None)
        if after_target not in (None, GroupCounts()):
            raise DatabaseCleanupError(
                f"Cleanup did not remove all {target_group} records from {table}."
            )
        if before_groups != after_groups:
            raise DatabaseCleanupError(
                f"Cleanup unexpectedly changed preserved groups in {table}."
            )


def read_only_snapshot(
    database: Path, filename_to_view: Mapping[str, str]
) -> DatabaseSnapshot:
    try:
        connection = sqlite3.connect(sqlite_read_only_uri(database), uri=True)
        try:
            return snapshot_database(connection, filename_to_view)
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise DatabaseCleanupError(
            f"Could not inspect COLMAP database {database}: {error}"
        ) from error


def remove_empty_sqlite_sidecars(database: Path) -> None:
    wal_path = Path(f"{database}-wal")
    shm_path = Path(f"{database}-shm")
    if wal_path.exists() and wal_path.stat().st_size != 0:
        raise DatabaseCleanupError(
            f"SQLite WAL was not fully checkpointed: {wal_path}"
        )
    for path in (wal_path, shm_path):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise DatabaseCleanupError(
                f"Could not remove generated SQLite sidecar {path}: {error}"
            ) from error


def execute_cleanup(
    database: Path,
    filename_to_view: Mapping[str, str],
    target_group: str,
) -> tuple[DatabaseSnapshot, DatabaseSnapshot]:
    try:
        connection = sqlite3.connect(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            before = snapshot_database(connection, filename_to_view)
            pair_ids_by_table = target_pair_ids(
                connection, filename_to_view, target_group
            )
            if not any(pair_ids_by_table.values()):
                raise DatabaseCleanupError(
                    f"No {target_group} records remain; cleanup was not applied."
                )
            for table, pair_ids in pair_ids_by_table.items():
                connection.executemany(
                    f"DELETE FROM {table} WHERE pair_id = ?",
                    ((pair_id,) for pair_id in pair_ids),
                )
            after = snapshot_database(connection, filename_to_view)
            validate_transition(before, after, target_group)
            connection.commit()
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if journal_mode.lower() == "wal":
                busy, _, _ = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if busy:
                    raise DatabaseCleanupError(
                        "SQLite WAL checkpoint remained busy after cleanup."
                    )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise DatabaseCleanupError(
            f"Could not clean COLMAP database {database}: {error}"
        ) from error

    remove_empty_sqlite_sidecars(database)
    committed = read_only_snapshot(database, filename_to_view)
    if committed != after:
        raise DatabaseCleanupError(
            "Committed database does not match the verified cleanup transaction."
        )
    return before, committed


def snapshot_to_json(snapshot: DatabaseSnapshot) -> dict[str, object]:
    return {
        "images": snapshot.images,
        "keypoint_images": snapshot.keypoint_images,
        "keypoint_rows": snapshot.keypoint_rows,
        "descriptor_images": snapshot.descriptor_images,
        "descriptor_rows": snapshot.descriptor_rows,
        "match_groups": {
            table: {
                label: asdict(counts) for label, counts in groups.items()
            }
            for table, groups in snapshot.match_groups.items()
        },
    }


def write_report(path: Path, result: CleanupResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(result.database),
        "target_group": result.target_group,
        "before": snapshot_to_json(result.before),
        "after": snapshot_to_json(result.after),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DatabaseCleanupError(
            f"Could not write cleanup report: {error}"
        ) from error


def cleanup_database(
    *,
    database: Path,
    manifest: Path,
    left_view: str,
    right_view: str,
    dry_run: bool,
    report_path: Path | None = None,
) -> CleanupResult:
    if not database.is_file():
        raise DatabaseCleanupError(f"Copied COLMAP database was not found: {database}")
    if left_view == right_view:
        raise DatabaseCleanupError("Cleanup views must be different.")
    filename_to_view = manifest_views(manifest)
    known_views = set(filename_to_view.values())
    missing_views = {left_view, right_view} - known_views
    if missing_views:
        raise DatabaseCleanupError(
            "Manifest is missing cleanup view(s): " + ", ".join(sorted(missing_views))
        )
    target_group = group_label(left_view, right_view)

    if dry_run:
        before = read_only_snapshot(database, filename_to_view)
        after = before
    else:
        before, after = execute_cleanup(
            database, filename_to_view, target_group
        )
    result = CleanupResult(database, target_group, dry_run, before, after)
    if report_path is not None and not dry_run:
        write_report(report_path, result)
    return result


def print_result(result: CleanupResult) -> None:
    action = "Would remove" if result.dry_run else "Removed"
    print(f"Database: {result.database}")
    print(f"Target group: {result.target_group}")
    for table in MATCH_TABLES:
        counts = result.before.match_groups[table].get(
            result.target_group, GroupCounts()
        )
        print(
            f"{action} from {table}: {counts.records} records, "
            f"{counts.rows} correspondence rows"
        )
    print(
        "Preserved features: "
        f"{result.after.images} images, "
        f"{result.after.keypoint_rows} keypoints, "
        f"{result.after.descriptor_rows} descriptors"
    )
    if not result.dry_run:
        for table in MATCH_TABLES:
            remaining = result.after.match_groups[table].get(
                result.target_group, GroupCounts()
            )
            print(
                f"Remaining {result.target_group} records in {table}: "
                f"{remaining.records}"
            )
        print("All non-target match groups were preserved exactly.")


def main() -> int:
    args = parse_args()
    try:
        workspace = resolve_project_path(args.workspace, "Workspace")
        manifest = resolve_project_path(args.manifest, "Manifest")
        database = workspace / "database.db"
        original_database = (DEFAULT_SOURCE_WORKSPACE / "database.db").resolve()
        if database.resolve() == original_database:
            raise DatabaseCleanupError(
                "Refusing to clean the original multiview COLMAP database."
            )
        if workspace.resolve() == DEFAULT_SOURCE_WORKSPACE.resolve():
            raise DatabaseCleanupError(
                "Workspace must not be the original multiview COLMAP workspace."
            )
        report_path = workspace / "logs" / DEFAULT_REPORT_NAME
        result = cleanup_database(
            database=database,
            manifest=manifest,
            left_view=args.left_view.strip(),
            right_view=args.right_view.strip(),
            dry_run=args.dry_run,
            report_path=report_path,
        )
        print_result(result)
        if not args.dry_run:
            print(f"Audit report: {report_path}")
        return 0
    except (DatabaseCleanupError, WorkspacePreparationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
