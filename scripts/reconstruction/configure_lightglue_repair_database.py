"""Assign one COLMAP camera and rig to each repaired capture sequence.

This utility only changes the isolated LightGlue repair database. It preserves
all images, keypoints, descriptors, matches, and two-view geometries while
splitting the original shared camera into independent side, top, and underside
calibrations that match the repaired pose model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import struct
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPAIR_ROOT = (
    REPO_ROOT
    / "data/processed/pot1-unglazed_multiview/"
    "colmap_sparse_masked_multiview_lightglue_repaired"
)
DEFAULT_DATABASE = DEFAULT_REPAIR_ROOT / "database.db"
DEFAULT_SIDE_CAMERA = DEFAULT_REPAIR_ROOT / "source_text/side/cameras.txt"
DEFAULT_TOP_CAMERA = DEFAULT_REPAIR_ROOT / "source_text/top/cameras.txt"
DEFAULT_UNDERSIDE_CAMERA = DEFAULT_REPAIR_ROOT / "source_text/combined/cameras.txt"

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}
EXPECTED_GROUP_COUNTS = {"side": 273, "top45": 233, "underside": 222}


class DatabaseConfigurationError(RuntimeError):
    """Raised when the copied database is not safe to reconfigure."""


@dataclass(frozen=True)
class CameraDefinition:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def params_blob(self) -> bytes:
        return struct.pack(f"<{len(self.params)}d", *self.params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the copied LightGlue database into three cameras/rigs."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--side-camera", type=Path, default=DEFAULT_SIDE_CAMERA)
    parser.add_argument("--top-camera", type=Path, default=DEFAULT_TOP_CAMERA)
    parser.add_argument(
        "--underside-camera", type=Path, default=DEFAULT_UNDERSIDE_CAMERA
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_camera(path: Path, camera_id: int) -> CameraDefinition:
    if not path.is_file():
        raise DatabaseConfigurationError(f"Missing camera file: {path}")
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(rows) != 1 or len(rows[0]) < 5:
        raise DatabaseConfigurationError(
            f"Expected exactly one valid camera in {path}."
        )
    fields = rows[0]
    model_name = fields[1]
    if model_name not in CAMERA_MODEL_IDS:
        raise DatabaseConfigurationError(
            f"Unsupported camera model {model_name!r} in {path}."
        )
    return CameraDefinition(
        camera_id=camera_id,
        model_id=CAMERA_MODEL_IDS[model_name],
        width=int(fields[2]),
        height=int(fields[3]),
        params=tuple(float(value) for value in fields[4:]),
    )


def image_group(name: str) -> str:
    basename = Path(name).name.lower()
    if basename.startswith("side_"):
        return "side"
    if basename.startswith("top45_"):
        return "top45"
    if basename.startswith("underside_"):
        return "underside"
    raise DatabaseConfigurationError(f"Unknown image group for {name!r}.")


def validate_database(connection: sqlite3.Connection) -> dict[str, int]:
    required = {"cameras", "images", "rigs", "frames", "frame_data"}
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = required - tables
    if missing:
        raise DatabaseConfigurationError(
            f"Database is missing tables: {', '.join(sorted(missing))}."
        )
    counts = {group: 0 for group in EXPECTED_GROUP_COUNTS}
    image_rows = list(connection.execute("SELECT image_id, name FROM images"))
    for _, name in image_rows:
        counts[image_group(name)] += 1
    if counts != EXPECTED_GROUP_COUNTS:
        raise DatabaseConfigurationError(
            f"Unexpected image groups {counts}; expected {EXPECTED_GROUP_COUNTS}."
        )
    frame_count = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    frame_data_count = connection.execute(
        "SELECT COUNT(*) FROM frame_data WHERE sensor_type = 0"
    ).fetchone()[0]
    if frame_count != len(image_rows) or frame_data_count != len(image_rows):
        raise DatabaseConfigurationError(
            "Expected exactly one camera frame and frame-data row per image."
        )
    unknown_frame_data = connection.execute(
        """
        SELECT COUNT(*)
        FROM frame_data AS fd
        LEFT JOIN images AS i ON i.image_id = fd.data_id
        WHERE fd.sensor_type = 0 AND i.image_id IS NULL
        """
    ).fetchone()[0]
    if unknown_frame_data:
        raise DatabaseConfigurationError(
            f"Found {unknown_frame_data} frame-data rows without an image."
        )
    return counts


def configure_database(
    database: Path,
    cameras: tuple[CameraDefinition, CameraDefinition, CameraDefinition],
    *,
    dry_run: bool,
) -> None:
    database = database.resolve()
    if not database.is_file():
        raise DatabaseConfigurationError(f"Missing copied database: {database}")
    connection = sqlite3.connect(str(database), timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        counts = validate_database(connection)
        before = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        print(f"Database: {database}")
        print(f"Images: {sum(counts.values())} ({counts})")
        print(f"Existing cameras: {before}")
        for group, camera in zip(("side", "top45", "underside"), cameras):
            print(
                f"camera {camera.camera_id} <- {group}: model={camera.model_id}, "
                f"size={camera.width}x{camera.height}, params={camera.params}"
            )
        if dry_run:
            print("Dry run only: database was not modified.")
            return

        connection.execute("BEGIN IMMEDIATE")
        for camera in cameras:
            connection.execute(
                """
                INSERT INTO cameras(
                    camera_id, model, width, height, params, prior_focal_length
                ) VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(camera_id) DO UPDATE SET
                    model = excluded.model,
                    width = excluded.width,
                    height = excluded.height,
                    params = excluded.params,
                    prior_focal_length = excluded.prior_focal_length
                """,
                (
                    camera.camera_id,
                    camera.model_id,
                    camera.width,
                    camera.height,
                    camera.params_blob,
                ),
            )
        connection.execute(
            """
            UPDATE images
            SET camera_id = CASE
                WHEN lower(name) LIKE 'side\_%' ESCAPE '\\' THEN 1
                WHEN lower(name) LIKE 'top45\_%' ESCAPE '\\' THEN 2
                WHEN lower(name) LIKE 'underside\_%' ESCAPE '\\' THEN 3
                ELSE camera_id
            END
            """
        )
        for rig_id in (1, 2, 3):
            connection.execute(
                """
                INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type)
                VALUES (?, ?, 0)
                ON CONFLICT(rig_id) DO UPDATE SET
                    ref_sensor_id = excluded.ref_sensor_id,
                    ref_sensor_type = excluded.ref_sensor_type
                """,
                (rig_id, rig_id),
            )
        connection.execute(
            """
            UPDATE frame_data
            SET sensor_id = (
                SELECT images.camera_id
                FROM images
                WHERE images.image_id = frame_data.data_id
            )
            WHERE sensor_type = 0
            """
        )
        connection.execute(
            """
            UPDATE frames
            SET rig_id = (
                SELECT frame_data.sensor_id
                FROM frame_data
                WHERE frame_data.frame_id = frames.frame_id
                  AND frame_data.sensor_type = 0
            )
            """
        )
        connection.commit()

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise DatabaseConfigurationError(
                f"SQLite integrity check failed after update: {integrity}"
            )
        validate_database(connection)
        assigned = dict(
            connection.execute(
                "SELECT camera_id, COUNT(*) FROM images GROUP BY camera_id"
            )
        )
        if assigned != {1: 273, 2: 233, 3: 222}:
            raise DatabaseConfigurationError(
                f"Unexpected final camera assignments: {assigned}."
            )
        print(f"Configured cameras: {len(cameras)}")
        print(f"Final image assignments: {assigned}")
        print("Database integrity: ok")
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    try:
        cameras = (
            read_camera(args.side_camera.resolve(), 1),
            read_camera(args.top_camera.resolve(), 2),
            read_camera(args.underside_camera.resolve(), 3),
        )
        configure_database(args.database, cameras, dry_run=args.dry_run)
    except (OSError, ValueError, sqlite3.Error, DatabaseConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
