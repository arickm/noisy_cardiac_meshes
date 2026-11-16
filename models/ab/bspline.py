# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# B-spline kernel transformation from AirLab (https://github.com/airlab-unibas/airlab,
# airlab/transformation/pairwise.py and airlab/utils/kernelFunction.py), as adapted in Daniel H. Pak's
# Deep-Cardiac-Volumetric-Mesh (https://github.com/danpak94/Deep-Cardiac-Volumetric-Mesh,
# airlab_transforms.py). Modified (2026): 3D only, non-diffeomorphic only, merged into one class.

import numpy as np
import torch
import torch.nn.functional as F


def bspline_kernel_3d(sigma, order):
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1
    for _ in range(1, order + 1):
        kernel = F.conv3d(kernel, kernel_ones, padding=padding.tolist()) / (sigma[0] * sigma[1] * sigma[2])
    return kernel[0, 0, ...]


class BsplineTransformation(torch.nn.Module):
    """Dense displacement field of shape image_size from B-spline control points with spacing sigma."""

    def __init__(self, image_size, sigma, order, dtype=torch.float32, device='cpu'):
        super().__init__()
        image_size = np.array(image_size)
        stride = np.array(sigma)
        kernel = bspline_kernel_3d(sigma, order).to(dtype=dtype)
        self._padding = ((np.array(kernel.size()) - 1) / 2).astype(int).tolist()
        self._kernel = kernel[None, None].expand(3, -1, -1, -1, -1).to(device=device)

        cp_grid_shape = np.ceil(np.divide(image_size, stride)).astype(int)
        # new image size after convolution
        inner_image_size = np.multiply(stride, cp_grid_shape) - (stride - 1)
        # add one control point at each side
        cp_grid_shape = cp_grid_shape + 2
        # center image between control points
        image_size_diff = inner_image_size - image_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        self._crop_start = (image_size_diff_floor + np.remainder(image_size_diff, 2)
                            * np.sign(image_size_diff)).astype(int)
        self._crop_end = image_size_diff_floor.astype(int)
        self._stride = stride.astype(int).tolist()
        self.cp_grid_shape = [1, 3] + cp_grid_shape.tolist()

    def forward(self, trans_parameters):
        displacement = F.conv_transpose3d(trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=3)
        s, c0, c1 = self._stride, self._crop_start, self._crop_end
        return displacement[:, :,
                            s[0] + c0[0]:-s[0] - c1[0],
                            s[1] + c0[1]:-s[1] - c1[1],
                            s[2] + c0[2]:-s[2] - c1[2]]
