#!/usr/bin/env python3

""" Predict LV myocardium meshes (world mm, one .ply per case) with a trained
model C (ISBI 2027; added to Vox2Cortex in 2026). Default: the last model
(final.pt), as in the paper.

Example:
    python isbi2027_predict.py --exp_dir ../experiments/mmwhs_C_ref \
        --split test --out_dir ../experiments/mmwhs_C_ref/test_meshes
"""

import os
import json
from argparse import ArgumentParser

import torch
import trimesh
from pytorch3d.structures.utils import transform_mesh_affine

from data.dataset_split_handler import dataset_split_handler
from data.supported_datasets import dataset_paths
from models.model_handler import ModelHandler
from utils.utils import load_checkpoint, update_dict
from utils.template import MeshTemplate, TEMPLATE_SPECS, template_trans_affine
from utils.model_names import FINAL_MODEL_NAME
from utils.graph_conv import *


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--split', default='validation',
                        choices=('train', 'validation', 'test'))
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model', default=FINAL_MODEL_NAME,
                        help="Checkpoint file name in exp_dir.")
    parser.add_argument('--raw_data_dir', default=None,
                        help="Overrides the data dir stored in params.json,"
                        " e.g. on another machine.")
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    with open(os.path.join(args.exp_dir, 'params.json'), 'r') as f:
        hps = json.load(f)
    if args.raw_data_dir is not None:
        hps['RAW_DATA_DIR'] = args.raw_data_dir
    hps['MODEL_CONFIG']['GC'] = eval(hps['MODEL_CONFIG']['GC'])
    # Only the images are needed for prediction: voxel labels are not read
    # (the test label volumes are not part of the export)
    hps['SEG_GROUND_TRUTH'] = 'voxelized_meshes'
    hps['SANITY_CHECK_DATA'] = False
    hps_lower = {k.lower(): v for k, v in hps.items()}
    model_config = {k.lower(): v for k, v in hps['MODEL_CONFIG'].items()}

    datasets = dataset_split_handler[hps['DATASET']](
        save_dir=args.out_dir if os.path.isdir(args.out_dir) else
        os.makedirs(args.out_dir) or args.out_dir,
        load_only=(args.split,),
        check_dir=args.out_dir,
        **hps_lower
    )
    dataset = dict(zip(('train', 'validation', 'test'), datasets))[args.split]

    template = MeshTemplate(
        mesh_label_names=list(dataset.mesh_label_names.keys()),
        trans_affine=template_trans_affine(dataset, hps['MESH_TEMPLATE_ID']),
        **TEMPLATE_SPECS[hps['MESH_TEMPLATE_ID']]
    )

    model = ModelHandler[hps['ARCHITECTURE']].value(
        ndims=hps['NDIMS'],
        n_v_classes=hps['N_V_CLASSES'],
        n_m_classes=hps['N_M_CLASSES'],
        patch_size=hps['PATCH_SIZE'],
        **model_config
    ).float()
    model, _, _, epoch = load_checkpoint(
        model, os.path.join(args.exp_dir, args.model), 'cpu'
    )
    model.to(args.device).eval()
    # load_checkpoint returns the epoch to start from
    print(f"Loaded {args.model} (epoch {epoch - 1}), {len(dataset)} {args.split} cases")

    for i in range(len(dataset)):
        case = dataset.get_ID_from_index(i)
        data = dataset.get_data_element(i)
        norm2world = torch.inverse(
            torch.tensor(data['trans_affine_label']).float()
        ).to(args.device)
        with torch.no_grad():
            img = data['img'].float().to(args.device)
            img = img.expand(model.batch_size, *img.shape)
            pred = model(
                img,
                template.create_template_batch_size(
                    model.batch_size, device=args.device
                )
            )
        mesh = model.__class__.pred_to_final_mesh_pred(pred)
        # Also flips the faces if the transformation flips the winding
        v, f = transform_mesh_affine(
            mesh.verts_list()[0], mesh.faces_list()[0], norm2world
        )
        trimesh.Trimesh(
            v.cpu().numpy(), f.cpu().numpy(), process=False
        ).export(os.path.join(args.out_dir, f"{case}.ply"))
        print(f"  {case}")


if __name__ == '__main__':
    main()
