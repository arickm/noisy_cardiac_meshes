"""
Data of models A and B: image crop, template and target mesh per case, in the voxel coordinates of the
crop. The template is centred in the patch (bounding-box centre at patch_size / 2); the target mesh
keeps its position.

    {data_root}/{dataset}/images/{case}.nii.gz           128^3 crop (1 mm, RAS)
    {data_root}/{dataset}/meshes/{labelset}/{case}.ply   target surface, world mm
    {data_root}/{dataset}/splits.json
"""

import json
import os
import random

import nibabel as nib
import numpy as np
import pyvista as pv
import torch
from monai.transforms import RandAdjustContrast, RandAffine, RandGaussianNoise, RandGaussianSmooth
from torch.utils.data import Dataset


def world_to_voxel(points, affine):
    return (np.linalg.inv(affine) @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T[:, :3]


def apply_affine(points, affine):
    return (affine @ np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T).T[:, :3]


def load_template(path):
    """Tet template (world mm): points, tets, unique tet edges, surface vertex indices (from
    {name}_LV_surface_vertices.npy next to it) and the outer surface triangles (indices into points)."""
    grid = pv.read(path)
    cells = grid.cells_dict[pv.CellType.TETRA]
    edges = np.vstack([cells[:, [0, 1]], cells[:, [0, 2]], cells[:, [0, 3]],
                       cells[:, [1, 2]], cells[:, [1, 3]], cells[:, [2, 3]]])
    edges.sort(axis=1)
    surface = grid.extract_surface(pass_pointid=True).triangulate()
    surface_faces = np.asarray(surface['vtkOriginalPointIds'])[surface.faces.reshape(-1, 4)[:, 1:]]
    return dict(grid=grid, points=grid.points.copy(), cells=cells, edges=np.unique(edges, axis=0),
                surface_vertices=np.load(path.replace('.vtk', '_LV_surface_vertices.npy')),
                surface_faces=surface_faces.astype(np.int64))


class Augmentation:
    """Random affine transform of image, template and target (p = 0.5), then intensity changes of the
    image (p = 0.5: contrast p = 0.5, Gaussian noise p = 0.3, Gaussian smoothing p = 0.3; MONAI)."""

    def __init__(self, cfg):
        self.rand_affine = RandAffine(prob=cfg['affine_prob'],
                                      rotate_range=[d * np.pi / 180 for d in cfg['rotate_range_degree']],
                                      shear_range=cfg['shear_range'], translate_range=cfg['translate_range'],
                                      scale_range=cfg['scale_range'])
        self.intensity_prob = cfg['intensity_prob']
        self.contrast = RandAdjustContrast(prob=1.0, gamma=tuple(cfg['contrast_range']))
        self.noise = RandGaussianNoise(prob=1.0, mean=0.0, std=cfg['noise_std'])
        sigma = tuple(cfg['gaussian_smooth_sigma'])
        self.smooth = RandGaussianSmooth(prob=1.0, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)

    def __call__(self, image, template, gt_vertices):
        augmented = self.rand_affine(image)
        if not torch.equal(augmented, image):
            # MONAI's affine pulls image values in voxel coordinates centred on the image centre; the
            # forward transform of the mesh points is its inverse, shifted to corner coordinates
            A = self.rand_affine.rand_affine_grid.get_transformation_matrix().cpu().numpy()
            center = (np.array(image.shape[1:]) - 1) / 2.0
            T_plus, T_minus = np.eye(4), np.eye(4)
            T_plus[:3, 3], T_minus[:3, 3] = center, -center
            M = T_plus @ np.linalg.inv(A) @ T_minus
            template = torch.from_numpy(apply_affine(template.numpy(), M)).float()
            gt_vertices = torch.from_numpy(apply_affine(gt_vertices.numpy(), M)).float()
        image = augmented

        if random.random() < self.intensity_prob:
            if random.random() < 0.5:
                image = self.contrast(image)
            if random.random() < 0.3:
                image = self.noise(image)
            if random.random() < 0.3:
                image = self.smooth(image)
        return image, template, gt_vertices


class LVDataset(Dataset):
    """One item per case: image (1, D, H, W) scaled to [0, 1] after the intensity window, template points
    and target vertices / faces in patch voxel coordinates, the image affine (voxel -> world)."""

    def __init__(self, root, split, labelset, template, patch_size, window, augmentation=None):
        self.cases = json.load(open(f'{root}/splits.json'))[split]
        self.template = template
        self.patch_size = np.array(patch_size)
        self.window = window
        self.augmentation = augmentation
        self.images, self.affines, self.meshes = [], [], []
        for c in self.cases:
            img = nib.load(f'{root}/images/{c}.nii.gz')
            assert tuple(img.shape) == tuple(patch_size), f'{c}: image must be {patch_size}'
            self.images.append(img.get_fdata().astype(np.float32))
            self.affines.append(img.affine)
            mesh_path = f'{root}/meshes/{labelset}/{c}.ply'
            if os.path.exists(mesh_path):  # not needed for prediction
                mesh = pv.read(mesh_path)
                self.meshes.append((world_to_voxel(mesh.points, img.affine), mesh.faces.reshape(-1, 4)[:, 1:]))
            else:
                self.meshes.append((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        affine = self.affines[idx]
        template = world_to_voxel(self.template['points'], affine)
        template = template + (self.patch_size / 2.0 - (template.max(axis=0) + template.min(axis=0)) / 2.0)
        image = torch.from_numpy(self.images[idx][None].copy()).float()
        template = torch.from_numpy(template).float()
        gt_vertices = torch.from_numpy(self.meshes[idx][0]).float()
        if self.augmentation is not None:
            image, template, gt_vertices = self.augmentation(image, template, gt_vertices)
        image = image.as_tensor() if hasattr(image, 'as_tensor') else image  # MONAI MetaTensor -> tensor
        image = image.float().clamp(*self.window)
        image = (image - image.min()) / (image.max() - image.min())
        return dict(case=self.cases[idx], image=image, template=template, gt_vertices=gt_vertices,
                    gt_faces=torch.from_numpy(self.meshes[idx][1]).long(), affine=torch.from_numpy(affine))
