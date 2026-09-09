"""Conservative, auditable cleanup of the untextured classical mesh.

No smoothing, hole filling, pose changes, or inferred base plane. Uses NumPy
only for geometry and the project's existing PyYAML configuration loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from run_colmap_dense import PROJECT_ROOT, PipelineError, load_yaml, resolve_project_path

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/mesh_cleanup_pot1_unglazed_every6.yml'
PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'int': '<i4', 'uint': '<u4'}


def read_ply(path: Path, require_faces: bool = True):
    """Read the COLMAP binary triangle PLY layout, checking actual payload."""
    with path.open('rb') as stream:
        lines = []
        while True:
            line = stream.readline(4096)
            if not line or sum(map(len, lines)) > 65536:
                raise PipelineError(f'Invalid PLY header: {path}')
            lines.append(line)
            if line.strip() == b'end_header':
                break
        header = b''.join(lines).decode('ascii').splitlines()
        if header[0] != 'ply' or 'format binary_little_endian 1.0' not in header:
            raise PipelineError(f'Expected binary little-endian COLMAP PLY: {path}')
        counts, properties, element = {}, {}, None
        for line in header:
            fields = line.split()
            if fields[:1] == ['element']:
                element = fields[1]
                counts[element] = int(fields[2])
                properties[element] = []
            elif fields[:1] == ['property']:
                properties[element].append(fields[1:])
        dtype = np.dtype([(p[1], PLY_TYPES[p[0]]) for p in properties['vertex']])
        vertices = np.fromfile(stream, dtype=dtype, count=counts['vertex'])
        if len(vertices) != counts['vertex']:
            raise PipelineError('Truncated vertex data')
        xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
        if not require_faces:
            return xyz
        expected = [[['list', 'uchar', 'int', name]]
                    for name in ('vertex_indices', 'vertex_index')]
        if properties.get('face') not in expected:
            raise PipelineError('Expected untextured triangle mesh; texture the cleaned mesh later.')
        records = np.fromfile(stream, dtype=[('n', 'u1'), ('indices', '<i4', (3,))],
                              count=counts['face'])
        if len(records) != counts['face'] or stream.read(1) or not (records['n'] == 3).all():
            raise PipelineError('Truncated, nontriangular, or unexpected mesh payload')
        faces = records['indices'].copy()
        if not len(faces) or faces.min() < 0 or faces.max() >= len(xyz):
            raise PipelineError('Empty mesh or invalid vertex indices')
        return xyz, faces


def supported_vertices(vertices, cloud, cell):
    """Conservative voxel support, not an exact point-to-surface distance."""
    origin = np.minimum(vertices.min(0), cloud.min(0)) - 2 * cell
    maximum = np.maximum(vertices.max(0), cloud.max(0))
    dimensions = np.ceil((maximum - origin) / cell).astype(np.int64) + 4
    if np.prod(dimensions.astype(object)) >= 2**62:
        raise PipelineError('Support grid is too large')
    weights = np.array([dimensions[1] * dimensions[2], dimensions[2], 1])
    occupied = np.unique(np.floor((cloud - origin) / cell).astype(np.int64) @ weights)
    queries = np.floor((vertices - origin) / cell).astype(np.int64) @ weights
    supported = np.zeros(len(vertices), dtype=bool)
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                keys = queries + np.array([x, y, z]) @ weights
                indices = np.searchsorted(occupied, keys)
                supported |= occupied[np.minimum(indices, len(occupied) - 1)] == keys
    return supported


def component_mask(faces, vertex_count, minimum):
    parents = np.arange(vertex_count)

    def root(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v

    for a, b, c in faces:
        ra = root(a)
        parents[root(b)] = ra
        parents[root(c)] = ra
    labels = np.array([root(a) for a in faces[:, 0]])
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse] >= minimum, sorted(counts.tolist(), reverse=True)


def clean(vertices, faces, cloud, settings):
    fraction = float(settings['support_voxel_fraction'])
    minimum = int(settings['minimum_component_faces'])
    limit = float(settings['maximum_removed_fraction'])
    if not (0 < fraction <= 0.02 and minimum >= 1 and 0 < limit < 1):
        raise PipelineError('Invalid cleanup settings')
    if not len(cloud) or not np.isfinite(cloud).all():
        raise PipelineError('Support point cloud must be nonempty and finite')
    diagonal = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    if diagonal <= 0:
        raise PipelineError('Support cloud has zero extent')
    finite = np.isfinite(vertices).all(1)
    keep = finite[faces].all(1)
    reasons = {'nonfinite_faces': int((~keep).sum())}
    safe = np.where(np.isfinite(vertices), vertices, 0).astype(np.float64)
    tri = safe[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = (area2 <= diagonal**2 * 1e-14)
    reasons['degenerate_faces'] = int((keep & degenerate).sum())
    keep &= ~degenerate
    ids = np.flatnonzero(keep)
    _, first = np.unique(np.sort(faces[ids], axis=1), axis=0, return_index=True)
    unique = np.zeros(len(faces), dtype=bool)
    unique[ids[first]] = True
    reasons['duplicate_faces'] = int((keep & ~unique).sum())
    keep &= unique
    support = np.zeros(len(vertices), dtype=bool)
    support[finite] = supported_vertices(vertices[finite], cloud, diagonal * fraction)
    # Only discard a face if all three vertices lack support; border faces remain.
    unsupported = ~support[faces].any(1)
    reasons['unsupported_faces'] = int((keep & unsupported).sum())
    keep &= ~unsupported
    ids = np.flatnonzero(keep)
    if not len(ids):
        raise PipelineError('Cleanup would remove the whole surface')
    component_keep, component_counts = component_mask(faces[ids], len(vertices), minimum)
    reasons['small_component_faces'] = int((~component_keep).sum())
    keep[ids[~component_keep]] = False
    removed_fraction = float((~keep).mean())
    report = dict(input_vertices=len(vertices), input_faces=len(faces),
                  kept_faces=int(keep.sum()), removed_faces=int((~keep).sum()),
                  removed_fraction=removed_fraction, removal_reasons=reasons,
                  support_voxel_width=diagonal * fraction,
                  support_cloud_diagonal=diagonal,
                  component_count_before_filter=len(component_counts),
                  largest_component_face_counts=component_counts[:10],
                  ready_to_write=bool(keep.any() and removed_fraction <= limit))
    return keep, report


def write_mesh(path, vertices, faces):
    used, inverse = np.unique(faces, return_inverse=True)
    compact = vertices[used].astype('<f4')
    records = np.empty(len(faces), dtype=[('n', 'u1'), ('indices', '<i4', (3,))])
    records['n'] = 3
    records['indices'] = inverse.reshape(-1, 3)
    with path.open('xb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\n'
                      f'element vertex {len(compact)}\nproperty float x\nproperty float y\nproperty float z\n'
                      f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        compact.tofile(stream)
        records.tofile(stream)
    return len(compact)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = resolve_project_path(str(args.config), 'config')
    config = load_yaml(config_path)
    mesh, cloud_path, output = [resolve_project_path(config[k], k) for k in
                               ('input_mesh', 'support_cloud', 'output_directory')]
    for source in (mesh, cloud_path):
        if source.is_relative_to(output):
            raise PipelineError('Output directory must not contain input files')
    if output.exists():
        raise PipelineError(f'Output already exists; select a new candidate directory: {output}')
    start = time.perf_counter()
    vertices, faces = read_ply(mesh)
    cloud = read_ply(cloud_path, require_faces=False)
    keep, report = clean(vertices, faces, cloud, config['cleanup'])
    report.update(settings=dict(config['cleanup']), input_mesh=str(mesh),
                  support_cloud=str(cloud_path), input_sha256=sha256(mesh),
                  support_sha256=sha256(cloud_path), numpy_version=np.__version__,
                  elapsed_seconds=time.perf_counter() - start,
                  policy='Remove unsupported faces and tiny components; no smoothing or hole filling.')
    print(json.dumps(report, indent=2))
    if not report['ready_to_write']:
        raise PipelineError('Removal exceeds the configured limit; inspect the dry-run before tuning.')
    if args.dry_run:
        print('Dry run complete. No cleanup files written.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cleanup-', dir=output.parent))
    name = 'meshed-poisson-cleaned.ply'
    report['output_vertices'] = write_mesh(staging / name, vertices, faces[keep])
    if (~keep).any():
        write_mesh(staging / 'removed-geometry.ply', vertices, faces[~keep])
    np.save(staging / 'kept-source-face-indices.npy', np.flatnonzero(keep))
    reloaded, reloaded_faces = read_ply(staging / name)
    if not np.isfinite(reloaded).all() or len(reloaded_faces) != int(keep.sum()):
        raise PipelineError(f'Written output failed validation; inspect {staging}')
    report['output_sha256'] = sha256(staging / name)
    (staging / 'cleanup_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    staging.rename(output)
    print(f'Cleanup candidate saved: {output / name}')
    print('Inspect this candidate and removed-geometry.ply before generating new textures.')


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f'ERROR: {error}')
