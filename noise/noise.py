"""
Noisy label sets of the paper: three matched pairs of label errors, each with a biased and an unbiased
(random) member, at prevalences of 25 / 50 / 75 / 100 % of the training cases.

    pair        biased                               unbiased
    thickness   thickdil_d2: dilate 2 mm             thickrand_d2: dilate or erode 2 mm (50/50 per case)
    tissue      apex_v2: hole at the apex,           holerand_v2: same hole at a random myocardium voxel
                removing 2 % of the myocardium
    warp        warpbias: the label is compressed    warp_L2: random smooth B-spline warp, strength set
                along the LV long axis towards the   per case so that the label Dice is 0.75
                apex (a different valve-plane cut);
                one shift for all cases, set so that
                the mean label Dice is 0.75

Input: the reference labels on the 1 mm crop grid, train + val cases only (never the test cases):
    {data_root}/{dataset}/labels/ref/{case}.nii.gz     myocardium (one piece)
    {data_root}/{dataset}/lv_cavity/{case}.nii.gz      LV blood pool (only for the long axis)
    {data_root}/{dataset}/splits.json                  {"train": [...], "val": [...], "test": [...]}
Output: labels/ref/{case}.nii.gz of the train + val cases is rewritten as the largest connected piece of
the label (stray voxels removed; this is the 'ref' label set of the paper), and
{data_root}/{dataset}/labels/{labelset}/{case}.nii.gz for the 24 noisy label sets, with a labelset.json
per set (corrupted cases, per-case source and Dice / ASSD / HD95 vs ref).

Prevalence: one shared random order of the train + val cases; prevalence p corrupts the first
round(p * N) cases of that order, for every noise type. All random draws are seeded per case.

    python noise.py --data_root /path/to/data --dataset mmwhs
"""

import argparse
import hashlib
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

PREVALENCES = [25, 50, 75, 100]
THICK_D = 2  # mm
HOLE_V = 2  # % of the myocardium volume
HOLE_TOL = 0.25  # percentage points
WARP_DICE = 0.75
DICE_TOL = 0.01
WARP_CP_SPACING = 32  # mm
WARPBIAS_GRID = np.arange(0.0, 30.01, 1.0)  # mm, shifts evaluated per case to find the shared one
MAX_DETACHED_PERCENT = 1.0  # erosion: specks up to this share of the myocardium are dropped

# label set prefix: (type, pair, biased, variant per case: (case, thickrand sign) -> variant name)
PAIRS = {
    'thickdil_d2': ('thickdil', 'thickness', True, lambda c, s: 'dil'),
    'thickrand_d2': ('thickrand', 'thickness', False, lambda c, s: 'dil' if s > 0 else 'ero'),
    'apex_v2': ('apex', 'tissue', True, lambda c, s: 'apex'),
    'holerand_v2': ('holerand', 'tissue', False, lambda c, s: 'holerand'),
    'warpbias': ('warpbias', 'warp', True, lambda c, s: 'warpbias'),
    'warp_L2': ('warp', 'warp', False, lambda c, s: 'warp'),
}


# ---------------------------------------------------------------------------- helpers

def case_seed(dataset, noise_type, case_id):
    digest = hashlib.sha256(f'{dataset}|{noise_type}|{case_id}'.encode()).hexdigest()
    return int(digest[:8], 16)


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask)


def label_stats(mask, ref):
    """Dice, ASSD and HD95 (boundary voxel distances, mm on the 1 mm grid) and volume vs ref."""
    if (mask == ref).all():
        return dict(dice=1.0, assd=0.0, hd95=0.0, volume_ml=float(mask.sum() / 1000))
    ba, bb = boundary(mask), boundary(ref)
    d = np.concatenate([ndimage.distance_transform_edt(~bb)[ba], ndimage.distance_transform_edt(~ba)[bb]])
    return dict(dice=dice(mask, ref), assd=float(d.mean()), hd95=float(np.percentile(d, 95)),
                volume_ml=float(mask.sum() / 1000))


def bisect(fn, target, lo, hi, tol, max_iter=40):
    """x in [lo, hi] with fn(x) ~ target, fn decreasing in x. Expands hi if needed."""
    while fn(hi) > target:
        hi *= 2
        if hi > 1e4:
            raise RuntimeError('could not bracket target')
    x = hi
    for _ in range(max_iter):
        x = 0.5 * (lo + hi)
        v = fn(x)
        if abs(v - target) <= tol / 2:
            return x
        if v > target:
            lo = x
        else:
            hi = x
    return x


# ---------------------------------------------------------------------------- noise types

def random_bspline_field(shape, seed, spacing=WARP_CP_SPACING):
    """Smooth zero-mean displacement field: i.i.d. N(0,1) control points, cubic B-spline interpolation."""
    rng = np.random.default_rng(seed)
    n_cp = [int(np.ceil(s / spacing)) + 3 for s in shape]
    cp = rng.standard_normal((3, *n_cp))
    coords = np.meshgrid(*[np.arange(s) / spacing + 1.0 for s in shape], indexing='ij')
    return np.stack([ndimage.map_coordinates(cp[d], coords, order=3, mode='nearest') for d in range(3)])


def warp_label(mask, field, strength):
    """Warped label, main piece only (a strong warp can pinch off small pieces)."""
    grid = np.meshgrid(*[np.arange(s) for s in mask.shape], indexing='ij')
    coords = [g + strength * f for g, f in zip(grid, field)]
    return largest_component(
        ndimage.map_coordinates(mask.astype(np.float32), coords, order=1, mode='constant') >= 0.5)


def dilate(mask, d):
    return mask | (ndimage.distance_transform_edt(~mask) <= d)


def erode_capped(mask, d, step=0.1):
    """Erosion by d mm, returned as one piece. Specks of up to MAX_DETACHED_PERCENT of the myocardium
    are dropped; if more would come off (the wall really breaks), d is reduced in steps of 0.1 mm."""
    dist_in = ndimage.distance_transform_edt(mask)
    d_used = d
    while d_used > 0:
        eroded = dist_in > d_used
        main = largest_component(eroded)
        detached = 100.0 * (eroded.sum() - main.sum()) / mask.sum()
        if main.any() and detached <= MAX_DETACHED_PERCENT:
            return main, d_used
        d_used = round(d_used - step, 3)
    return mask.copy(), 0.0


def long_axis(myo, lv):
    """Unit long-axis direction (pointing to the apex), centroid and apex point (voxel coords).

    Axis: first principal component of the filled LV (myocardium + blood pool). Apex end: the end
    where the axis leaves the filled LV through myocardium (closed apical cap); at the base it leaves
    through the blood pool. Apex point: the outermost myocardium voxel on the axis at that end.
    """
    solid = myo | lv
    pts = np.argwhere(solid).astype(float)
    centroid = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    axis = vt[0]

    def last_inside(direction):
        last, last_is_myo = centroid, False
        for t in np.arange(0, 150, 0.5):
            idx = np.round(centroid + t * direction).astype(int)
            if (idx < 0).any() or (idx >= np.array(myo.shape)).any():
                break
            if solid[tuple(idx)]:
                last, last_is_myo = centroid + t * direction, bool(myo[tuple(idx)])
        return last, last_is_myo

    end_pos, pos_myo = last_inside(axis)
    end_neg, neg_myo = last_inside(-axis)
    if neg_myo and not pos_myo:
        axis, apex = -axis, end_neg
    elif pos_myo and not neg_myo:
        apex = end_pos
    else:  # ambiguous: the apex is the end farther from the blood-pool centroid
        lv_c = np.argwhere(lv).mean(0)
        if np.linalg.norm(end_neg - lv_c) > np.linalg.norm(end_pos - lv_c):
            axis, apex = -axis, end_neg
        else:
            apex = end_pos
    return axis, centroid, apex


def punch_hole(myo, centre, radius):
    """Remove myocardium within radius of centre; drop pieces cut off from the main body."""
    grid = np.ogrid[tuple(slice(0, s) for s in myo.shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grid, centre))
    return largest_component(myo & (dist2 > radius ** 2))


def hole_with_volume(myo, centre, v_percent):
    """Hole whose radius removes v_percent of the myocardium volume (incl. cut-off pieces)."""
    total = myo.sum()
    removed = lambda r: 100.0 * (1 - punch_hole(myo, centre, r).sum() / total)
    r = bisect(lambda r: -removed(r), -v_percent, 0.0, 10.0, tol=HOLE_TOL)
    return punch_hole(myo, centre, r)


def base_shift_label(myo, axis, apex, shift):
    """Label compressed along the long axis towards the apex: the basal rim (height L) moves to
    L - shift, heights in between scale linearly, the apex stays. Main piece only."""
    if shift <= 0:
        return myo.copy()
    pts = np.argwhere(myo).astype(float)
    L = ((pts - apex) @ -axis).max()
    k = 1 - shift / L
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in myo.shape], indexing='ij'), -1).astype(np.float32)
    h = (grid - apex) @ -axis
    src = grid + (np.clip(h, 0, None) * (1 / k - 1))[..., None] * -axis
    out = ndimage.map_coordinates(myo.astype(np.float32), src.reshape(-1, 3).T, order=1, mode='constant')
    return largest_component(out.reshape(myo.shape) >= 0.5)


# ---------------------------------------------------------------------------- per case

def load_case(root, case):
    nii = nib.load(f'{root}/labels/ref/{case}.nii.gz')
    myo = largest_component(np.asanyarray(nii.dataobj) > 0)
    lv = np.asanyarray(nib.load(f'{root}/lv_cavity/{case}.nii.gz').dataobj) > 0
    return myo, lv, nii.affine


def case_variants(args):
    """All corrupted versions of one case, except warpbias (needs the shift shared by all cases)."""
    dataset, root, case = args
    myo, lv, _ = load_case(root, case)
    field = random_bspline_field(myo.shape, case_seed(dataset, 'warp', case))
    s = bisect(lambda s: dice(warp_label(myo, field, s), myo), WARP_DICE, 0.0, 4.0, DICE_TOL)
    out = dict(warp=warp_label(myo, field, s), dil=dilate(myo, THICK_D), ero=erode_capped(myo, THICK_D)[0])

    axis, _, apex = long_axis(myo, lv)
    out['apex'] = hole_with_volume(myo, apex, HOLE_V)
    rng = np.random.default_rng(case_seed(dataset, 'holerand', case))
    myo_vox = np.argwhere(myo)
    out['holerand'] = hole_with_volume(myo, myo_vox[rng.integers(len(myo_vox))].astype(float), HOLE_V)
    return case, out


def warpbias_curve(args):
    """Label Dice of the base-plane shift over WARPBIAS_GRID for one case."""
    root, case = args
    myo, lv, _ = load_case(root, case)
    axis, _, apex = long_axis(myo, lv)
    return [dice(base_shift_label(myo, axis, apex, s), myo) for s in WARPBIAS_GRID]


def warpbias_variant(args):
    root, case, shift = args
    myo, lv, _ = load_case(root, case)
    axis, _, apex = long_axis(myo, lv)
    return case, base_shift_label(myo, axis, apex, shift)


# ---------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', default='mmwhs', help='folder name below data_root, also part of the seeds')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    root = f'{args.data_root}/{args.dataset}'
    split = json.load(open(f'{root}/splits.json'))
    cases = sorted(split['train'] + split['val'])
    refs, affines = {}, {}
    for c in cases:  # the reference label set is the largest piece of each label (rewritten in place)
        refs[c], _, affines[c] = load_case(root, c)
        nib.save(nib.Nifti1Image(refs[c].astype(np.uint8), affines[c]), f'{root}/labels/ref/{c}.nii.gz')

    with Pool(args.workers) as pool:
        variants = dict(pool.map(case_variants, [(args.dataset, root, c) for c in cases]))
        # warpbias: the one shift at which the mean label Dice over all cases reaches WARP_DICE
        mean_curve = np.mean(pool.map(warpbias_curve, [(root, c) for c in cases]), axis=0)
        shift = float(np.interp(-WARP_DICE, -mean_curve, WARPBIAS_GRID))
        for c, mask in pool.map(warpbias_variant, [(root, c, shift) for c in cases]):
            variants[c]['warpbias'] = mask
    print(f'warpbias: shared base-plane shift {shift:.2f} mm')

    rng = np.random.default_rng(case_seed(args.dataset, 'prevalence_order', 'all'))
    order = [cases[i] for i in rng.permutation(len(cases))]
    thick_sign = {c: (1 if np.random.default_rng(case_seed(args.dataset, 'thickrand', c)).random() < 0.5 else -1)
                  for c in cases}

    for prefix, (noise_type, pair, biased, variant_of) in PAIRS.items():
        for p in PREVALENCES:
            name = f'{prefix}_p{p}'
            corrupted = set(order[:int(round(p / 100 * len(order)))])
            os.makedirs(f'{root}/labels/{name}', exist_ok=True)
            per_case = {}
            for c in cases:
                source = variant_of(c, thick_sign[c]) if c in corrupted else 'ref'
                mask = variants[c][source] if c in corrupted else refs[c]
                nib.save(nib.Nifti1Image(mask.astype(np.uint8), affines[c]), f'{root}/labels/{name}/{c}.nii.gz')
                per_case[c] = dict(source=source, **label_stats(mask, refs[c]))
            manifest = dict(labelset=name, type=noise_type, pair=pair, biased=biased, prevalence=p,
                            corrupted_cases=sorted(corrupted), cases=per_case)
            if noise_type == 'warpbias':
                manifest['shift_mm'] = shift
            json.dump(manifest, open(f'{root}/labels/{name}/labelset.json', 'w'), indent=1)
            d = [per_case[c]['dice'] for c in corrupted]
            print(f'{name:18s} {len(corrupted):3d} corrupted cases, label Dice {np.mean(d):.3f}')


if __name__ == '__main__':
    main()
