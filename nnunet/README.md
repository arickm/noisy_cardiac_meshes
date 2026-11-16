# nnU-Net baseline

nnU-Net v2, `3d_fullres`, one fold that is our train/val split, 750 epochs, no postprocessing. One nnU-Net
dataset per label set: `Dataset{ID}_{DATASET}_{labelset}`. We used IDs 600, 601, ... for MMWHS and 700, 701, ...
for MITEA.

**One non-default step: the plans of every label set are copied from `ref`.** nnU-Net's CT normalization
uses intensity statistics of the foreground voxels, which here are the (noisy) labels. Planning each noisy
label set on its own would give each run a different intensity normalization. So we plan and
preprocess `ref` once and copy its plans to every other label set
(`nnUNetv2_move_plans_between_datasets`) before preprocessing. That way every run gets the same input,
as models A, B and C do.

## Setup

```bash
pip install nnunetv2
export nnUNet_raw=/path/to/nnUNet_raw nnUNet_preprocessed=/path/to/nnUNet_preprocessed nnUNet_results=/path/to/nnUNet_results

# install the 750-epoch trainer into the nnunetv2 package (nnU-Net only looks for trainers there)
NNDIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp nnUNetTrainer_750epochs.py $NNDIR/training/nnUNetTrainer/variants/training_length/
```

## 1. Raw dataset per label set

Images and labels are symlinks into the data folder (layout: see the main README). The channel name
picks nnU-Net's normalization: `CT` for MMWHS, `echo` (z-score per image) for MITEA.

```bash
DATA=/path/to/data/mmwhs; DATASET=MMWHS; CHANNEL=CT

make_raw() {  # make_raw <labelset> <id>
    local name=Dataset$2_${DATASET}_$1
    mkdir -p $nnUNet_raw/$name/{imagesTr,labelsTr,imagesTs}
    for c in $(python -c "import json; s=json.load(open('$DATA/splits.json')); print(*s['train'], *s['val'])"); do
        ln -sf $DATA/images/$c.nii.gz $nnUNet_raw/$name/imagesTr/${c}_0000.nii.gz
        ln -sf $DATA/labels/$1/$c.nii.gz $nnUNet_raw/$name/labelsTr/$c.nii.gz
    done
    for c in $(python -c "import json; print(*json.load(open('$DATA/splits.json'))['test'])"); do
        ln -sf $DATA/images/$c.nii.gz $nnUNet_raw/$name/imagesTs/${c}_0000.nii.gz
    done
    python -c "import json; s=json.load(open('$DATA/splits.json')); json.dump(dict(channel_names={'0': '$CHANNEL'},
        labels={'background': 0, 'myocardium': 1}, numTraining=len(s['train']) + len(s['val']),
        file_ending='.nii.gz'), open('$nnUNet_raw/$name/dataset.json', 'w'), indent=2)"
}
# our split as nnU-Net fold 0
python -c "import json; s=json.load(open('$DATA/splits.json')); json.dump([dict(train=s['train'], val=s['val'])],
    open('splits_final.json', 'w'), indent=2)"

make_raw ref 600
make_raw apex_v2_p50 601   # ... one ID per label set
```

## 2. Plan on `ref`, copy the plans, preprocess

```bash
# ref: plan and preprocess once
nnUNetv2_plan_and_preprocess -d 600 -c 3d_fullres --verify_dataset_integrity
cp splits_final.json $nnUNet_preprocessed/Dataset600_MMWHS_ref/

# every other label set (here 601): ref plans, then preprocess
nnUNetv2_extract_fingerprint -d 601 --verify_dataset_integrity
nnUNetv2_move_plans_between_datasets -s 600 -t 601 -sp nnUNetPlans -tp nnUNetPlans
cp $nnUNet_raw/Dataset601_MMWHS_apex_v2_p50/dataset.json $nnUNet_preprocessed/Dataset601_MMWHS_apex_v2_p50/
nnUNetv2_preprocess -d 601 -plans_name nnUNetPlans -c 3d_fullres
cp splits_final.json $nnUNet_preprocessed/Dataset601_MMWHS_apex_v2_p50/
```

## 3. Train fold 0 and predict the test cases

```bash
nnUNetv2_train 601 3d_fullres 0 -tr nnUNetTrainer_750epochs
OUT=$nnUNet_results/Dataset601_MMWHS_apex_v2_p50/nnUNetTrainer_750epochs__nnUNetPlans__3d_fullres/fold_0
nnUNetv2_predict -i $nnUNet_raw/Dataset601_MMWHS_apex_v2_p50/imagesTs -o $OUT/test_predictions \
    -d 601 -c 3d_fullres -f 0 -tr nnUNetTrainer_750epochs -chk checkpoint_final.pth
```

No postprocessing (we do not run `nnUNetv2_find_best_configuration` or `nnUNetv2_apply_postprocessing`). The
predictions are label maps on the crop grid. `evaluation/evaluate.py` turns them into surfaces with marching
cubes; add `--smooth` for the smoothed variant.
