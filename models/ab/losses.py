"""
Losses of models A and B. All coordinates are patch voxel coordinates.

Model B (tetrahedral template): Chamfer distance between the template's surface vertices and the
target mesh vertices, plus as-rigid-as-possible (ARAP) energy of the tets and edge-length preservation.

Model A (surface template): our re-implementation of the default Vox2Cortex mesh loss (Bongratz et al.,
CVPR 2022; https://github.com/ai-med/Vox2Cortex, vox2organ/utils/losses.py), following the PyTorch3D
definitions it uses, without PyTorch3D:
  - Chamfer (squared distances, mean per direction, both directions summed) and oriented cosine
    distance between the normals of matched points
  - uniform Laplacian of the deformation field on the mesh
  - normal consistency between adjacent faces
  - squared edge length with target 0
Vox2Cortex computes them in coordinates normalized to [-1, 1] over the input patch, so the published
weights only mean the same thing there: points are normalized first.
"""

import numpy as np
import torch
import kaolin.metrics.pointcloud


# ---------------------------------------------------------------------------- model B

def deformation_gradients(X_ref, X_def):
    """Deformation gradient per tet. X_ref, X_def: (B, n_tets, 4, 3) -> (B, n_tets, 3, 3)."""
    E = torch.stack([X_ref[:, :, k, :] - X_ref[:, :, 0, :] for k in (1, 2, 3)], dim=-1)
    e = torch.stack([X_def[:, :, k, :] - X_def[:, :, 0, :] for k in (1, 2, 3)], dim=-1)
    if (torch.abs(torch.linalg.det(E)) < 1e-10).any() or (torch.abs(torch.linalg.det(e)) < 1e-10).any():
        E_inv = torch.linalg.pinv(E)  # degenerate tets
    else:
        E_inv = torch.linalg.inv(E)
    return torch.matmul(e, E_inv)


def arap_energy(F_batch):
    """Mean squared Frobenius norm of F - R, R the closest rotation (polar decomposition, no gradient)."""
    F_flat = torch.clamp(F_batch.reshape(-1, 3, 3), min=-100.0, max=100.0)
    U, _, Vh = torch.linalg.svd(F_flat.detach(), full_matrices=False)
    R = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0) @ torch.nan_to_num(Vh, nan=0.0, posinf=0.0, neginf=0.0)
    diff = torch.nan_to_num(F_flat - R, nan=0.0, posinf=0.0, neginf=0.0)
    return diff.square().sum(dim=(-2, -1)).mean()


def edge_length_loss(points_ref, points_def, edges):
    """Mean squared relative change of the edge lengths."""
    l_ref = (points_ref[:, edges[:, 0]] - points_ref[:, edges[:, 1]]).norm(dim=-1).clamp_min(1e-8)
    l_def = (points_def[:, edges[:, 0]] - points_def[:, edges[:, 1]]).norm(dim=-1)
    return ((l_def - l_ref) / l_ref).square().mean()


def model_b_loss(deformed, template, gt_vertices, tmpl, weights, n_points, epoch, batch_idx):
    """Chamfer on up to n_points random surface / target vertices (the draw depends on the epoch and
    batch index only), + ARAP + edge. Returns (total, dict of unweighted terms)."""
    b = deformed.shape[0]
    generator = torch.Generator(device=deformed.device).manual_seed(42 + epoch * 1000 + batch_idx)
    pred = deformed[:, tmpl['surface_vertices']]
    pred = pred[:, torch.randperm(pred.shape[1], device=deformed.device, generator=generator)[:n_points]]
    gt = gt_vertices[:, torch.randperm(gt_vertices.shape[1], device=deformed.device, generator=generator)[:n_points]]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred, gt, w1=0.5, w2=0.5, squared=False).mean()

    cells = tmpl['cells'].expand(b, -1, -1)
    batch = torch.arange(b, device=deformed.device)[:, None, None]
    F_def = deformation_gradients(template[batch, cells], deformed[batch, cells])
    terms = dict(chamfer=chamfer, edge=edge_length_loss(template, deformed, tmpl['edges']), arap=arap_energy(F_def))
    total = weights['chamfer'] * chamfer + (weights['edge'] * terms['edge'] + weights['arap'] * terms['arap'])
    return total, terms


# ---------------------------------------------------------------------------- model A (Vox2Cortex losses)

def normalize_points(points, patch_size):
    """Voxel coordinates in a patch of patch_size -> [-1, 1]."""
    half = (torch.as_tensor(patch_size, dtype=points.dtype, device=points.device) - 1) / 2
    return (points - half) / half


def face_normals(verts, faces):
    """Unit face normals and face areas. verts (B, V, 3), faces (F, 3)."""
    v0, v1, v2 = verts[:, faces[:, 0]], verts[:, faces[:, 1]], verts[:, faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True)
    return cross / norm.clamp_min(1e-12), 0.5 * norm.squeeze(-1)


def outward_faces(verts, faces):
    """Flip the winding if the closed surface has negative signed volume (normals pointing in)."""
    v = verts[0].detach()
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return faces if (torch.cross(v0, v1, dim=-1) * v2).sum() / 6 > 0 else faces[:, [0, 2, 1]]


def sample_points_with_normals(verts, faces, n_points):
    """Area-weighted uniform samples on the surface (differentiable w.r.t. verts)."""
    normals, areas = face_normals(verts, faces)
    b = verts.shape[0]
    face_idx = torch.multinomial(areas.detach(), n_points, replacement=True)
    u = torch.rand(b, n_points, 1, device=verts.device)
    v = torch.rand(b, n_points, 1, device=verts.device)
    flip = (u + v) > 1
    u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    f = faces[face_idx]
    batch = torch.arange(b, device=verts.device)[:, None]
    p0, p1, p2 = verts[batch, f[..., 0]], verts[batch, f[..., 1]], verts[batch, f[..., 2]]
    return p0 + u * (p1 - p0) + v * (p2 - p0), normals[batch, face_idx]


class SurfaceTopology:
    """Edges, vertex degrees and adjacent face pairs of the template's outer surface."""

    def __init__(self, faces, n_verts):
        faces_np = faces.cpu().numpy()
        edges = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
        face_of_edge = np.tile(np.arange(len(faces_np)), 3)
        uniq, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        # the two faces sharing each edge (closed manifold: exactly two)
        pairs = face_of_edge[np.argsort(inverse.ravel(), kind='stable')].reshape(-1, 2)
        device = faces.device
        self.faces = faces
        self.edges = torch.from_numpy(uniq).long().to(device)
        self.face_pairs = torch.from_numpy(pairs).long().to(device)
        # vertices on the surface (the tet template also has interior vertices)
        self.surface_verts = torch.from_numpy(np.unique(faces_np)).long().to(device)
        self.degree = torch.from_numpy(np.bincount(uniq.ravel(), minlength=n_verts).astype(np.float32)).to(device)

    def uniform_laplacian_norm(self, features):
        """Mean over surface vertices of || f_i - mean_{j in N(i)} f_j || (PyTorch3D 'uniform')."""
        i, j = self.edges[:, 0], self.edges[:, 1]
        nbr_sum = torch.zeros_like(features)
        nbr_sum.index_add_(1, i, features[:, j])
        nbr_sum.index_add_(1, j, features[:, i])
        lap = nbr_sum / self.degree.clamp_min(1)[None, :, None] - features
        return lap[:, self.surface_verts].norm(dim=-1).mean()

    def normal_consistency(self, verts):
        normals, _ = face_normals(verts, self.faces)
        n1, n2 = normals[:, self.face_pairs[:, 0]], normals[:, self.face_pairs[:, 1]]
        return (1 - (n1 * n2).sum(-1)).mean()

    def edge_length_sq(self, verts):
        e = verts[:, self.edges[:, 0]] - verts[:, self.edges[:, 1]]
        return e.square().sum(-1).mean()


def model_a_loss(deformed, template, gt_vertices, gt_faces, topo, weights, n_points, patch_size):
    """Weighted Vox2Cortex mesh loss on the template's outer surface. Returns (total, dict of terms)."""
    pred = normalize_points(deformed, patch_size)
    ref = normalize_points(template, patch_size)
    gt = normalize_points(gt_vertices, patch_size)

    pred_pts, pred_nrm = sample_points_with_normals(pred, topo.faces, n_points)
    with torch.no_grad():
        gt_pts, gt_nrm = sample_points_with_normals(gt, outward_faces(gt_vertices, gt_faces), n_points)
    d_pg, idx_pg = kaolin.metrics.pointcloud.sided_distance(pred_pts, gt_pts)
    d_gp, idx_gp = kaolin.metrics.pointcloud.sided_distance(gt_pts, pred_pts)
    batch = torch.arange(pred_pts.shape[0], device=pred_pts.device)[:, None]
    cos_pg = (pred_nrm * gt_nrm[batch, idx_pg]).sum(-1)
    cos_gp = (gt_nrm * pred_nrm[batch, idx_gp]).sum(-1)
    terms = {
        'chamfer': (d_pg.mean(dim=1) + d_gp.mean(dim=1)).mean(),
        'cosine': ((1 - cos_pg).mean(dim=1) + (1 - cos_gp).mean(dim=1)).mean(),
        'laplacian': topo.uniform_laplacian_norm(pred - ref),
        'normal_consistency': topo.normal_consistency(pred),
        'edge': topo.edge_length_sq(pred),
    }
    return sum(weights[k] * v for k, v in terms.items()), terms
