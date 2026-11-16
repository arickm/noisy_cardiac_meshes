"""
Train model A or B on one label set; validation loss on the val cases (same label set) every
validate_every epochs, which also steps the learning-rate scheduler and saves the checkpoint.

    python train.py --config ../../configs/A.json --data_root /path/to/data --dataset mmwhs \
        --labelset ref --out_dir runs/mmwhs_A_ref
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Augmentation, LVDataset, load_template
from losses import SurfaceTopology, model_a_loss, model_b_loss, outward_faces
from model import ResidualUNet, TemplateDeformation

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'templates')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def template_path(args):
    return args.template or os.path.join(TEMPLATES, f'{args.dataset.upper()}_template_LV_volumetric.vtk')


class MeshModel:
    """Network + deformation + loss of model A or B for one template."""

    def __init__(self, cfg, template, device):
        self.cfg, self.device = cfg, device
        net = cfg['network']
        self.net = ResidualUNet(1, net['encoder_channels'], net['decoder_channels'], net['dropout']).to(device)
        d = cfg['deformation']
        self.deform = TemplateDeformation(cfg['patch_size'], d['cp_spacing'], d['bspline_order'],
                                          d['integration_steps'], device)
        self.tmpl = dict(cells=torch.from_numpy(template['cells']).long().to(device),
                         edges=torch.from_numpy(template['edges']).long().to(device),
                         surface_vertices=torch.from_numpy(template['surface_vertices']).long().to(device))
        faces = torch.from_numpy(template['surface_faces']).to(device)
        points = torch.from_numpy(template['points']).float()[None].to(device)
        self.topo = SurfaceTopology(outward_faces(points, faces), len(template['points']))

    def __call__(self, batch):
        return self.deform(self.net(batch['image'].float().to(self.device)), batch['template'].to(self.device))

    def loss(self, deformed, batch, epoch, batch_idx):
        loss = self.cfg['loss']
        template = batch['template'].to(self.device)
        gt_vertices = batch['gt_vertices'].to(self.device)
        if self.cfg['model'] == 'A':
            return model_a_loss(deformed, template, gt_vertices, batch['gt_faces'][0].to(self.device), self.topo,
                                loss['weights'], loss['n_sample_points'], self.cfg['patch_size'])
        return model_b_loss(deformed, template, gt_vertices, self.tmpl, loss['weights'], loss['n_sample_points'],
                            epoch, batch_idx)


def run_epoch(model, loader, epoch, optimizer=None, grad_clip=None):
    """One pass over loader (training if an optimizer is given); mean loss and terms per case."""
    model.net.train(optimizer is not None)
    total, terms_sum, n = 0.0, {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch_idx, batch in enumerate(loader):
            loss, terms = model.loss(model(batch), batch, epoch, batch_idx)
            if not torch.isfinite(loss):
                print(f'epoch {epoch + 1} batch {batch_idx}: non-finite loss, step skipped')
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=grad_clip)
                if not torch.isfinite(grad_norm):
                    print(f'epoch {epoch + 1} batch {batch_idx}: non-finite gradient, step skipped')
                    continue
                optimizer.step()
            total += loss.item()
            for k, v in terms.items():
                terms_sum[k] = terms_sum.get(k, 0.0) + v.item()
            n += len(batch['case'])
    return total / n, {k: v / n for k, v in terms_sum.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='configs/A.json or configs/B.json')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', default='mmwhs')
    parser.add_argument('--labelset', default='ref')
    parser.add_argument('--template', default=None, help='default: templates/{DATASET}_template_LV_volumetric.vtk')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--epochs', type=int, default=None, help='overrides the config (e.g. for a quick test)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb_project', default=None, help='log to Weights & Biases (off by default)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(cfg, dataset=args.dataset, labelset=args.labelset, template=template_path(args)),
              open(f'{args.out_dir}/config.json', 'w'), indent=2)
    set_seed(cfg['seed'])
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=os.path.basename(os.path.normpath(args.out_dir)), config=cfg)

    root = f'{args.data_root}/{args.dataset}'
    template = load_template(template_path(args))
    common = dict(root=root, labelset=args.labelset, template=template, patch_size=cfg['patch_size'],
                  window=cfg['intensity_window'])
    train_set = LVDataset(split='train', augmentation=Augmentation(cfg['augmentation']), **common)
    val_set = LVDataset(split='val', **common)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    print(f'{len(train_set)} training and {len(val_set)} validation cases, label set {args.labelset}')

    model = MeshModel(cfg, template, args.device)
    opt = cfg['optimizer']
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **cfg['lr_scheduler'])

    for epoch in range(cfg['epochs']):
        train_loss, train_terms = run_epoch(model, train_loader, epoch, optimizer, cfg['grad_clip_norm'])
        print(f'epoch {epoch + 1}: train loss {train_loss:.5f} '
              + ' '.join(f'{k} {v:.5f}' for k, v in train_terms.items()), flush=True)
        log = {'epoch': epoch + 1, 'train/loss': train_loss, 'lr': optimizer.param_groups[0]['lr'],
               **{f'train/{k}': v for k, v in train_terms.items()}}
        if (epoch + 1) % cfg['validate_every'] == 0 or epoch + 1 == cfg['epochs']:
            val_loss, val_terms = run_epoch(model, val_loader, epoch)
            print(f'epoch {epoch + 1}: val loss {val_loss:.5f}', flush=True)
            log.update({'val/loss': val_loss, **{f'val/{k}': v for k, v in val_terms.items()}})
            scheduler.step(val_loss)
            torch.save(dict(epoch=epoch + 1, model=model.net.state_dict(), optimizer=optimizer.state_dict()),
                       f'{args.out_dir}/checkpoint.pth')
        if args.wandb_project:
            wandb.log(log, step=(epoch + 1) * len(train_loader))


if __name__ == '__main__':
    main()
