"""
Predict meshes with a trained model A or B: the deformed tet template per case, in world mm
({out_dir}/{case}.vtk). Model A only uses the template's outer surface; evaluate.py extracts it.

    python predict.py --run_dir runs/mmwhs_A_ref --data_root /path/to/data --split test
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import LVDataset, apply_affine, load_template
from train import MeshModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='output folder of train.py')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out_dir', default=None, help='default: {run_dir}/{split}_meshes')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    cfg = json.load(open(f'{args.run_dir}/config.json'))
    out_dir = args.out_dir or f'{args.run_dir}/{args.split}_meshes'
    os.makedirs(out_dir, exist_ok=True)
    template = load_template(cfg['template'])
    dataset = LVDataset(f'{args.data_root}/{cfg["dataset"]}', args.split, cfg['labelset'], template,
                        cfg['patch_size'], cfg['intensity_window'])
    model = MeshModel(cfg, template, args.device)
    checkpoint = torch.load(f'{args.run_dir}/checkpoint.pth', map_location=args.device)
    model.net.load_state_dict(checkpoint['model'])
    model.net.eval()
    print(f'model {cfg["model"]}, epoch {checkpoint["epoch"]}, {len(dataset)} {args.split} cases')

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1):
            points = model(batch)[0].cpu().numpy()
            mesh = template['grid'].copy()
            mesh.points = apply_affine(points, batch['affine'][0].numpy()).astype(np.float64)
            mesh.save(f'{out_dir}/{batch["case"][0]}.vtk', binary=True)
    print(f'meshes in {out_dir}')


if __name__ == '__main__':
    main()
