# Templates

Two LV templates per dataset (`MMWHS`, `MITEA`), in world mm:

- `{DATASET}_template_*`: used in the paper; the LV myocardium mesh of one subject. Default of every script.
- `{DATASET}_avg_template_*`: average template of the training labels; a bit smoother, with better-shaped
  tetrahedra.

Each consists of:

- `*_LV_volumetric.vtk`: tetrahedral LV myocardium mesh (models A and B)
- `*_LV_volumetric_LV_surface_vertices.npy`: indices of its outer-surface vertices (model B)
- `*_LV_surface.ply`: its closed outer surface with outward normals (model C)
