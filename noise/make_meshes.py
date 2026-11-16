"""
Target surfaces of models A, B and C: one mesh per label, marching cubes at 0.5 on the binary mask,
vertices in world mm, normals pointing outwards, no smoothing.

    {data_root}/{dataset}/labels/{labelset}/{case}.nii.gz  ->  {data_root}/{dataset}/meshes/{labelset}/{case}.ply

    python make_meshes.py --data_root /path/to/data --dataset mmwhs              # every label set
    python make_meshes.py --data_root /path/to/data --dataset mmwhs --labelsets ref apex_v2_p50
"""

import argparse
import glob
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure


def label_to_world_mesh(mask, affine):
    padded = np.pad(mask.astype(np.float32), 1)  # closes surfaces touching the crop border
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = verts - 1.0
    verts_world = (affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    # outward normals, like the templates; the oriented normal loss of A and C depends on this
    v0, v1, v2 = (verts_world[faces[:, k]] for k in range(3))
    if np.einsum('ij,ij->', np.cross(v0, v1), v2) < 0:
        faces = faces[:, [0, 2, 1]]
    mesh = pv.PolyData(verts_world, np.c_[np.full(len(faces), 3), faces].ravel())
    # keep the largest surface piece: tiny enclosed pockets and slivers at 1-voxel-thin spots
    # otherwise become extra surfaces (the voxel label is not changed)
    pieces = mesh.connectivity('all').split_bodies()
    if len(pieces) <= 1:
        return mesh
    areas = np.array([p.area for p in pieces])
    return pieces[int(areas.argmax())].extract_surface().clean()


def write_mesh(paths):
    label_path, mesh_path = paths
    nii = nib.load(label_path)
    label_to_world_mesh(np.asanyarray(nii.dataobj) > 0, nii.affine).save(mesh_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', default='mmwhs')
    parser.add_argument('--labelsets', nargs='*', help='default: every folder in labels/')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    root = f'{args.data_root}/{args.dataset}'
    labelsets = args.labelsets or sorted(os.listdir(f'{root}/labels'))
    jobs = []
    for ls in labelsets:
        os.makedirs(f'{root}/meshes/{ls}', exist_ok=True)
        for f in sorted(glob.glob(f'{root}/labels/{ls}/*.nii.gz')):
            case = os.path.basename(f)[:-len('.nii.gz')]
            jobs.append((f, f'{root}/meshes/{ls}/{case}.ply'))
    with Pool(args.workers) as pool:
        pool.map(write_mesh, jobs)
    print(f'{len(jobs)} meshes written for {len(labelsets)} label sets')


if __name__ == '__main__':
    main()
