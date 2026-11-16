# Model C: Vox2Cortex

This folder is a **modified version of Vox2Cortex** (https://github.com/ai-med/Vox2Cortex, GPL-3.0), taken from
commit `ef10ca2` (May 21, 2025) of its `main` branch. Vox2Cortex was written by Fabian Bongratz et al.
(Bongratz, Rickmann, Pölsterl, Wachinger, CVPR 2022). Changes made in 2026 for this paper:

- New dataset class `vox2organ/data/mmwhs.py` (`MMWHSDataset`, `MITEADataset`). It holds one closed LV
  myocardium surface, reads the data layout of the main README (with the same `splits.json` and label sets
  as models A and B), and applies the same intensity window as A and B.
- Case-specific template placement (`template_trans_affine` in `utils/template.py`, used by `utils/train.py`
  and `utils/test.py`). The crops differ by a translation, so the template's bounding-box centre is placed
  at the patch centre, as in A and B. Templates: `heart_template/{mmwhs,mitea}/LV_myo.ply`, links to the surface
  templates in the top-level `templates/` folder:
  `mkdir -p heart_template/mmwhs && ln -s ../../../../templates/MMWHS_template_LV_surface.ply heart_template/mmwhs/LV_myo.ply`
  (likewise for `mitea`; link `{DATASET}_avg_template_LV_surface.ply` instead for the average template).
- Parameter groups `Vox2Cortex-MMWHS` and `Vox2Cortex-MITEA` (`params/groups.py`), with the default Vox2Cortex
  loss weights of the white-matter surface: Chamfer 4, cosine 0.01, Laplacian 0.1, normal consistency 0.001,
  edge 5. Weights & Biases is off in these groups.
- `main.py`: options `--labelset`, `--raw_data_dir` and `--params_file`. The last model is saved as `final.pt`.
- `isbi2027_predict.py`: test meshes in world mm, one `.ply` per case.
- Removed: the cortex and abdomen datasets, templates and parameter groups, V2C-Flow / UNetFlow, the
  pretrained models, preprocessing scripts and docker files.

Modified files carry a short notice at the top.

## Environment

Python 3.9, PyTorch 1.10.0 + CUDA 11.3, and the PyTorch3D fork https://github.com/fabibo3/pytorch3d at commit
`9d05fbbd4fb01ec3d7a8a9972713c9daa25467dd`:

```bash
conda create -n vox2organ --file env/conda_explicit.txt
conda activate vox2organ
pip install --no-deps -r env/pip_requirements.txt
git clone https://github.com/fabibo3/pytorch3d && cd pytorch3d && git checkout 9d05fbbd4fb01ec3d7a8a9972713c9daa25467dd
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=<your GPU arch, e.g. 8.6> pip install -e .   # CUDA 11.3 needs GCC <= 10
```

CUDA 11.3 has no kernels for Hopper or Blackwell GPUs. We trained on A40s.

## Train and predict

Settings of the paper: 750 epochs, seed 3, last model. Validation uses the run's own label set.

```bash
cd vox2organ
python main.py --train --group Vox2Cortex-MMWHS --raw_data_dir /path/to/data/mmwhs --labelset apex_v2_p50 \
    --n_epochs 750 --master_seed 3 -n mmwhs_C_apex_v2_p50 --experiment_base_dir ../experiments
python isbi2027_predict.py --exp_dir ../experiments/mmwhs_C_apex_v2_p50 --split test \
    --out_dir ../experiments/mmwhs_C_apex_v2_p50/test_meshes
```

For MITEA use `--group Vox2Cortex-MITEA` and the `mitea` folder. `configs/mmwhs_C_params.json` is the full
parameter set of our MMWHS `ref` run, as Vox2Cortex saved it (`params.json`), with machine-specific paths and
logging settings removed.
