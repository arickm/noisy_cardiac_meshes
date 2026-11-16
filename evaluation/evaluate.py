"""
Accuracy of predicted LV myocardium surfaces against the manual labels at their native resolution.

Every prediction becomes a closed triangle surface in world mm:
  *.vtk            tet meshes of models A and B (outer surface)
  *.ply / *.obj    surface meshes (model C)
  *.nii.gz         label maps (nnU-Net), marching cubes at 0.5, no smoothing (--smooth: windowed-sinc
                   smoothing, the 'nnU-Net smoothed' variant)
and is compared with the surface of the manual label (marching cubes at 0.5, native grid):
  assd, hd95       100k area-weighted points on each surface, point-to-surface distances, both directions
  dice             prediction surface voxelized onto the native label grid

    python evaluate.py --pred_dir runs/mmwhs_A_ref/test_meshes --gt_dir /path/to/manual_labels \
        --gt_value 205 --out runs/mmwhs_A_ref/test_metrics.csv

gt_dir holds {case}.nii.gz (case ids as in splits.json), the original manual label maps. --gt_value:
myocardium value in those maps (MMWHS 205, MITEA 1).
"""

import argparse
import csv
import glob
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from skimage import measure
from vtk.util import numpy_support

N_POINTS = 100_000
SMOOTH_ITER, SMOOTH_PASS_BAND = 20, 0.1


def mask_to_world_surface(mask, affine):
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask.astype(np.float32), 1), 0.5)
    world = (affine @ np.c_[verts - 1.0, np.ones(len(verts))].T).T[:, :3]
    return pv.PolyData(world, np.c_[np.full(len(faces), 3), faces].ravel()).clean()


def sample_surface(surface, n_points, seed):
    """Area-weighted uniform sample of n_points on a triangle surface."""
    rng = np.random.default_rng(seed)
    tri = surface.faces.reshape(-1, 4)[:, 1:]
    p = surface.points[tri]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    idx = rng.choice(len(tri), n_points, p=area / area.sum())
    u, v = rng.random(n_points), rng.random(n_points)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = p[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def point_to_surface(points, surface):
    return np.abs(pv.PolyData(points).compute_implicit_distance(surface)['implicit_distance'])


def stencil_voxelize(surface, affine, shape):
    """Voxels whose centres lie inside the closed surface, on a grid with a diagonal (RAS) affine."""
    spacing, origin = np.diag(affine)[:3], affine[:3, 3]
    assert np.allclose(affine[:3, :3], np.diag(spacing)) and (spacing > 0).all(), 'needs a canonical grid'
    img = vtk.vtkImageData()
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    img.SetDimensions(*shape)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(surface)
    pol2stenc.SetOutputOrigin(*origin)
    pol2stenc.SetOutputSpacing(*spacing)
    pol2stenc.SetOutputWholeExtent(img.GetExtent())
    pol2stenc.Update()
    stenc = vtk.vtkImageStencil()
    stenc.SetInputData(img)
    stenc.SetStencilConnection(pol2stenc.GetOutputPort())
    stenc.ReverseStencilOff()
    stenc.SetBackgroundValue(0)
    stenc.Update()
    arr = numpy_support.vtk_to_numpy(stenc.GetOutput().GetPointData().GetScalars())
    return arr.reshape(shape[::-1]).transpose(2, 1, 0).astype(bool)


def prediction_surface(path, smooth):
    if path.endswith('.nii.gz'):
        nii = nib.load(path)
        surface = mask_to_world_surface(np.asanyarray(nii.dataobj) > 0, nii.affine)
        return surface.smooth_taubin(n_iter=SMOOTH_ITER, pass_band=SMOOTH_PASS_BAND) if smooth else surface
    return pv.read(path).extract_surface().triangulate().clean()


def evaluate_case(job):
    case, pred_path, gt_path, gt_value, smooth = job
    gt_nii = nib.as_closest_canonical(nib.load(gt_path))
    gt = np.asanyarray(gt_nii.dataobj) == gt_value
    gt_surface = mask_to_world_surface(gt, gt_nii.affine)
    surface = prediction_surface(pred_path, smooth).triangulate().clean()
    d = np.concatenate([point_to_surface(sample_surface(surface, N_POINTS, seed=2), gt_surface),
                        point_to_surface(sample_surface(gt_surface, N_POINTS, seed=1), surface)])
    mask = stencil_voxelize(surface, gt_nii.affine, gt.shape)
    return dict(case=case, dice=2 * (mask & gt).sum() / (mask.sum() + gt.sum()), assd=d.mean(),
                hd95=np.percentile(d, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, help='{case}.vtk / .ply / .obj / .nii.gz')
    parser.add_argument('--gt_dir', required=True, help='{case}.nii.gz manual labels, native resolution')
    parser.add_argument('--gt_value', type=int, default=1, help='myocardium value in the labels (MMWHS: 205)')
    parser.add_argument('--smooth', action='store_true', help='smooth marching-cubes surfaces of label maps')
    parser.add_argument('--out', required=True, help='CSV, one row per case')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    jobs = []
    for path in sorted(glob.glob(f'{args.pred_dir}/*')):
        case = os.path.basename(path).split('.')[0]
        if os.path.exists(f'{args.gt_dir}/{case}.nii.gz'):
            jobs.append((case, path, f'{args.gt_dir}/{case}.nii.gz', args.gt_value, args.smooth))
    with Pool(args.workers) as pool:
        rows = pool.map(evaluate_case, jobs)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'dice', 'assd', 'hd95'])
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} cases: ' + ', '.join(f'{k} {np.mean([r[k] for r in rows]):.3f}' for k in ['dice', 'assd', 'hd95']))


if __name__ == '__main__':
    main()
