# Noisy cardiac meshes

Code of the ISBI 2027 paper

> **TITLE** (TODO)<br>
> AUTHORS (TODO)<br>
> IEEE International Symposium on Biomedical Imaging (ISBI), 2027

Do template-deformation networks resist label noise better than voxel segmentation? We train three mesh
models and nnU-Net on left-ventricle (LV) myocardium labels with controlled errors. Each error comes as a
**biased** version (the same mistake in every corrupted case) and an **unbiased** version (random mistakes
of the same size). We then measure each model against manual labels.

| | Model | Template | Loss |
|---|---|---|---|
| **A** | 3D residual U-Net → B-spline velocity field → template deformation | LV surface | Vox2Cortex mesh losses |
| **B** | same network and deformation as A | LV tetrahedral mesh | Chamfer + ARAP + edge length |
| **C** | Vox2Cortex (graph-convolutional mesh deformation) | LV surface | Vox2Cortex mesh losses |
| nnU-Net | nnU-Net v2, 3d_fullres | – | – |

Datasets: MMWHS CT and MITEA 3D echocardiography.

## Repository

```
noise/noise.py            noisy label sets (the label errors of the paper)
noise/make_meshes.py      labels -> target surfaces (marching cubes)
models/ab/                models A and B: train.py, predict.py
models/c/                 model C: modified Vox2Cortex (see models/c/README.md)
nnunet/                   nnU-Net: 750-epoch trainer and the commands we ran
evaluation/evaluate.py    Dice, ASSD, HD95 against the manual labels
configs/A.json, B.json    every hyperparameter of models A and B
templates/                LV templates for MMWHS and MITEA (see below)
```

## Installation

Two environments: one for the noise, meshes and evaluation (`requirements.txt`), and one for models A and B
(`models/ab/requirements.txt`: PyTorch 2.4, Kaolin 0.16, MONAI 1.4). Model C needs its own environment
(PyTorch 1.10 + PyTorch3D, see `models/c/README.md`). nnU-Net: `pip install nnunetv2`.

```bash
pip install -r requirements.txt
```

## Data

- **MMWHS** (CT): https://zmiclab.github.io/zxh/0/mmwhs/. Manual labels exist only for the 20 cases of the
  original *training* set, so we swap the roles. The 40 original *test* cases are our training data
  (32 train / 8 val), and their reference labels are pseudo labels from
  [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (`heartchambers_highres`). The 20
  original *training* cases with manual labels are our test set.
- **MITEA** (3D echo): https://www.cardiacatlas.org/mitea/ (TODO: confirm link). We use the end-diastolic
  volumes of both scans of each subject and fold 1 of the MITEA subject split: 192 train / 22 val / 54 test
  volumes. The reference labels are the MITEA labels.

The code expects every image already cropped. For each case:

1. Reorient to RAS.
2. Resample to 1 mm isotropic, into a 128 × 128 × 128 box centred on the centroid of the reference LV
   myocardium. Use linear interpolation. Pad CT with −1024 HU and echo with 0. Keep CT in HU and echo as stored
   (0–255).
3. Resample the labels the same way: linear interpolation of the binary mask, then threshold at 0.5.

All models, the noise and nnU-Net read this layout:

```
{data_root}/{dataset}/                  dataset = mmwhs | mitea
    splits.json                         {"train": [...], "val": [...], "test": [...]}  (case ids)
    images/{case}.nii.gz                128^3 crop (all cases)
    labels/ref/{case}.nii.gz            myocardium mask on the crop grid: reference labels (train, val)
                                        and manual labels (test)
    lv_cavity/{case}.nii.gz             LV blood pool of the reference label (train, val; noise.py only)
    labels/{labelset}/                  written by noise/noise.py
    meshes/{labelset}/{case}.ply        written by noise/make_meshes.py
```

Evaluation compares against the manual labels **at their native resolution**. Put those label maps, as
distributed with the datasets, into a separate folder as `{case}.nii.gz`.

## 1. Noisy label sets

```bash
cd noise
python noise.py --data_root /path/to/data --dataset mmwhs
python make_meshes.py --data_root /path/to/data --dataset mmwhs
```

`noise.py` corrupts the reference labels of the train and val cases. It first reduces each reference label
to its largest connected piece, in place. The three noise pairs, each biased vs unbiased and matched in
size:

| Pair | Biased | Unbiased |
|---|---|---|
| thickness | `thickdil_d2`: dilate by 2 mm | `thickrand_d2`: dilate or erode by 2 mm (50/50 per case) |
| tissue | `apex_v2`: spherical hole at the apex, 2 % of the myocardium | `holerand_v2`: the same hole at a random myocardium voxel |
| warp | `warpbias`: label compressed along the LV long axis towards the apex (a different valve-plane cut); one shift for all cases, mean label Dice 0.75 | `warp_L2`: random smooth B-spline warp, label Dice 0.75 per case |

Each type comes at prevalences of 25, 50, 75 and 100 % (`_p25` … `_p100`): the share of training cases that
are corrupted. All types use one shared random case order, so `_p25` corrupts the same cases for every type.
Every random draw is seeded per case, so the label sets are the same on every machine. Each label set has a
`labelset.json` with the corrupted cases and the per-case Dice / ASSD / HD95 against `ref`.

`make_meshes.py` turns every label into the target surface of models A, B and C. It uses marching cubes at
0.5, world mm, outward normals and no smoothing, and keeps the largest piece.

## 2. Training and prediction

**Models A and B**:

```bash
cd models/ab
python train.py --config ../../configs/B.json --data_root /path/to/data --dataset mmwhs \
    --labelset apex_v2_p50 --out_dir runs/mmwhs_B_apex_v2_p50
python predict.py --run_dir runs/mmwhs_B_apex_v2_p50 --data_root /path/to/data --split test
```

`configs/A.json` and `configs/B.json` hold every hyperparameter. Label sets differ only in `--labelset`.
Settings for both models: 750 epochs, seed 3, AdamW (learning rate 1e-3, weight decay 1e-4), learning rate
halved on a plateau of the validation loss (computed every 25 epochs), last checkpoint, no rigid
pre-alignment. Augmentation: random affine transforms and intensity changes. The template is centred in the
crop. The network output is integrated in 5 Euler steps on a B-spline grid with 3-voxel spacing.

- A: Vox2Cortex mesh losses on the template's outer surface (Chamfer 4, normals 0.01, Laplacian 1, normal
  consistency 0.001, edge 5). A uses the same template file as B but only its outer surface.
- B: Chamfer + ARAP 0.01 + edge length 1.

Loss weights were chosen on the MMWHS validation cases (against the TotalSegmentator labels, 500 epochs)
and used unchanged for every label set and for MITEA.

A run on MMWHS takes about 3.5 GB of GPU memory. Weights & Biases logging is optional (`--wandb_project`).

**Model C**: see `models/c/README.md`. **nnU-Net**: see `nnunet/README.md`.

## 3. Evaluation

```bash
cd evaluation
python evaluate.py --pred_dir ../models/ab/runs/mmwhs_B_apex_v2_p50/test_meshes \
    --gt_dir /path/to/manual_labels/mmwhs --gt_value 205 --out mmwhs_B_apex_v2_p50.csv
```

The script works for every model: tet meshes (A, B), surfaces (C) and nnU-Net label maps (meshed with
marching cubes; `--smooth` for the smoothed variant). It computes ASSD and HD95 from 100,000 points on each
surface, in both directions, and Dice after voxelizing the predicted surface onto the native label grid.
The myocardium value is 205 in MMWHS and 1 in MITEA.

## Templates

`templates/` holds two LV templates per dataset (MMWHS, MITEA):

- `{DATASET}_template_*`: the templates used in the paper, each the LV myocardium mesh of one subject. They are
  the default of every script.
- `{DATASET}_avg_template_*`: an average template of the training labels. It is a bit smoother and its
  tetrahedra are better shaped. To use it, pass `--template templates/{DATASET}_avg_template_LV_volumetric.vtk`
  to models A and B, or link its surface for model C (`models/c/README.md`).

Each template is a tetrahedral mesh (`*_LV_volumetric.vtk`), its outer surface (`*_LV_surface.ply`, used by
model C), and the indices of the surface vertices in the tet mesh (`*_LV_volumetric_LV_surface_vertices.npy`,
used by model B). Coordinates are in world mm.

## Acknowledgements and code we built on

- **Vox2Cortex**: F. Bongratz, A.-M. Rickmann, S. Pölsterl, C. Wachinger, "Vox2Cortex: Fast Explicit
  Reconstruction of Cortical Surfaces from 3D MRI Scans with Geometric Deep Neural Networks", CVPR 2022.
  Code: https://github.com/ai-med/Vox2Cortex (GPL-3.0). Model C is a modified version of it
  (`models/c/`). The residual U-Net of models A and B (`models/ab/model.py`) is partly taken from its
  `vox2organ/models/u_net.py`. The model A losses (`models/ab/losses.py`) re-implement its mesh losses.
- **Deep Cardiac Volumetric Mesh** by Daniel H. Pak: https://github.com/danpak94/Deep-Cardiac-Volumetric-Mesh.
  Models A and B follow its template-deformation approach. The B-spline transformation
  (`models/ab/bspline.py`) comes from its `airlab_transforms.py`. Please also cite
  - D. H. Pak et al., "Patient-specific Heart Geometry Modeling for Solid Biomechanics using Deep Learning",
    IEEE Transactions on Medical Imaging, 2023.
  - D. H. Pak, M. Liu, T. Kim, L. Liang, R. McKay, W. Sun, J. S. Duncan, "Distortion Energy for Deep
    Learning-Based Volumetric Finite Element Mesh Generation for Aortic Valves", MICCAI 2021.
- **AirLab**: https://github.com/airlab-unibas/airlab (Apache-2.0), origin of the B-spline transformation.
- **nnU-Net**: F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, K. H. Maier-Hein, "nnU-Net: a
  self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021.
  Code: https://github.com/MIC-DKFZ/nnUNet.
- **TotalSegmentator**: J. Wasserthal et al., "TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images", Radiology: Artificial Intelligence 2023. We used it for the MMWHS pseudo labels.
- **MONAI** (https://github.com/Project-MONAI/MONAI) and **Kaolin** (https://github.com/NVIDIAGameWorks/kaolin)
  for the augmentation and the point-cloud distances of models A and B.

### Third-party code in this repository

| File | Origin | Licence |
|---|---|---|
| `models/ab/bspline.py` | AirLab, via Deep Cardiac Volumetric Mesh (modified) | Apache-2.0 |
| `models/ab/model.py` (residual U-Net) | Vox2Cortex `vox2organ/models/u_net.py` (modified) | GPL-3.0 |
| `models/c/vox2organ/` | Vox2Cortex (modified, see `models/c/README.md`) | GPL-3.0 |

## Citation

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027}
}
```

## Licence

GPL-3.0 (see `LICENSE`). Third-party files keep their licence headers.
