""" MMWHS CT dataset (LV myocardium, one closed surface) for the ISBI 2027
pseudo-label experiments (model C). Added to Vox2Cortex in 2026.

Layout (RAW_DATA_DIR = {data_root}/mmwhs, see the repository README):
    images/{case}.nii.gz              128^3 crop, 1 mm, RAS
    labels/{labelset}/{case}.nii.gz   myocardium labels (test cases: in ref)
    meshes/{labelset}/{case}.ply      target surfaces, world mm
    splits.json                       {"train": [...], "val": [...], "test": [...]}
Training and validation use the same label set; the test targets are always
the reference ('ref').

In contrast to the other datasets, the image affines differ between cases (by
a translation), so the world -> normalized transformation is stored per case
and the template gets its own transformation (see template_trans_affine).
"""

import json
import os
from enum import IntEnum

import numpy as np
import nibabel as nib
import torch
import trimesh
from pytorch3d.structures import Meshes
from tqdm import tqdm

from data.image_and_mesh_dataset import ImageAndMeshDataset
from utils.modes import DataModes
from utils.utils import normalize_min_max
from utils.coordinate_transform import normalize_vertices
from logger import logger

log = logger.get_std_logger(__name__)


class MMWHSLabels(IntEnum):
    LV_myo = 1


class MMWHSDataset(ImageAndMeshDataset):
    """ MMWHS CT dataset with the LV myocardium as a single surface.

    :param labelset: The label set of the training and validation targets,
    e.g. 'ref'.
    :param kwargs: Parameters for ImageAndMeshDataset
    """

    # Same intensity window as models A/B (WindowLevelClip(700, 250))
    HU_MIN = -100.0
    HU_MAX = 600.0
    PAD_VALUE = -1024.0
    # Test targets are always the manual GT
    TEST_LABELSET = "ref"

    LabelMap = MMWHSLabels

    @classmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        if structure_type != "LV_myo":
            raise NotImplementedError()
        return {"LV_myo": ("LV_myo",)}, {"LV_myo": "LV_myo"}

    @classmethod
    def split(cls, raw_data_dir, save_dir, fixed_split="splits.json", **kwargs):
        """ Fixed split from splits.json in the dataset folder. """
        s = json.load(open(os.path.join(raw_data_dir, fixed_split)))
        return super(MMWHSDataset, cls).split(
            raw_data_dir, save_dir, fixed_split={
                "train": s["train"], "validation": s["val"], "test": s["test"]
            }, **kwargs
        )

    def __init__(
        self,
        structure_type="LV_myo",
        labelset="ref",
        **kwargs
    ):
        (self.voxel_label_names,
         self.mesh_label_names) = self._get_seg_and_mesh_label_names(
             structure_type
         )
        mode = kwargs['mode']
        if mode in (DataModes.TRAIN, DataModes.VALIDATION):
            self.labelset = labelset
        else:
            self.labelset = self.TEST_LABELSET
        self.img_norm = self.window_and_min_max
        # Vertices of reference meshes are not used as points directly
        self.registered_gt_meshes = False
        log.info(f"{self.__class__.__name__} {mode.name}: label set '{self.labelset}'")

        super().__init__(
            image_file_name="image",
            mesh_file_names=list(self.mesh_label_names.values()),
            seg_file_name="label",
            **kwargs
        )

    def window_and_min_max(self, img):
        return normalize_min_max(img.clamp(self.HU_MIN, self.HU_MAX))

    def _image_path(self, case):
        return os.path.join(self._raw_data_dir, "images", f"{case}.nii.gz")

    def _label_path(self, case):
        return os.path.join(
            self._raw_data_dir, "labels", self.labelset, f"{case}.nii.gz"
        )

    def _mesh_path(self, case):
        return os.path.join(
            self._raw_data_dir, "meshes", self.labelset, f"{case}.ply"
        )

    def image_affine(self, index: int):
        return nib.load(self._image_path(self.ids[index])).affine

    def _load_data3D_and_transform(self, filename: str, is_label: bool):
        """ Load images ('image') or voxel labels ('label'). """
        get_path = self._label_path if is_label else self._image_path
        data, transformations = [], []
        for case in tqdm(self.ids, position=0, leave=True, desc=f"Loading {filename}s..."):
            img = nib.load(get_path(case))
            img_data = img.get_fdata()
            if self._orig_img_size is None:
                self._orig_img_size = img_data.shape
            else:
                assert np.array_equal(
                    np.array(img_data.shape), np.array(self._orig_img_size)
                ), "All images should be of equal size"
            img_data, trans_affine = self._get_single_patch(img_data, is_label)
            data.append(img_data)
            transformations.append(trans_affine)

        return data, transformations

    def _load_dataMesh_raw(self, meshnames):
        """ Load meshes (world mm) and map them to the voxel space of the
        respective image. """
        data = []
        assert len(self.trans_affine) == 0, "Should be empty."
        for case in tqdm(self.ids, position=0, leave=True, desc="Loading meshes..."):
            world2vox_affine = np.linalg.inv(
                nib.load(self._image_path(case)).affine
            )
            self.trans_affine.append(world2vox_affine)
            mesh = trimesh.load_mesh(self._mesh_path(case), process=False)
            mesh.apply_transform(world2vox_affine)
            n_v = mesh.vertices.shape[0]
            self.n_max_vertices = max(n_v, self.n_max_vertices or 0)
            self.n_min_vertices = min(n_v, self.n_min_vertices or n_v)
            data.append(Meshes(
                [torch.from_numpy(mesh.vertices).float()],
                [torch.from_numpy(mesh.faces).long()]
            ))

        return data

    def template_trans_affine(self, template_path):
        """ Transformation of the template from world mm to normalized image
        coordinates: the bounding-box centre is moved to the patch centre
        (voxel patch_size/2, as in models A/B); orientation and scale are
        those of the image grid (all crops share the linear part of the
        affine). """
        ref_affine = self.image_affine(0)
        verts = trimesh.load_mesh(template_path, process=False).vertices
        world2vox_lin = np.linalg.inv(ref_affine[:3, :3])
        v_vox = verts @ world2vox_lin.T
        bbox_centre = (v_vox.max(0) + v_vox.min(0)) / 2.0
        centring = np.eye(4)
        centring[:3, :3] = world2vox_lin
        centring[:3, 3] = np.array(self.patch_size) / 2.0 - bbox_centre
        _, norm_affine = normalize_vertices(
            v_vox, self.patch_size, return_affine=True
        )
        return norm_affine @ centring


class MITEADataset(MMWHSDataset):
    """ MITEA 3D echo dataset (ED frames of scan 1 and 2), LV myocardium.

    Same data layout and label sets as MMWHS. Images are 0-255 with 0 padding
    and are min-max scaled per image; models A/B apply the CT window, which
    does nothing to these values. The test targets are the MITEA labels.
    """

    PAD_VALUE = 0.0

    def window_and_min_max(self, img):
        return normalize_min_max(img)
