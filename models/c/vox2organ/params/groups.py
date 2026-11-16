""" Experiment-specific parameters. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

# Modified (2026, ISBI 2027 LV myocardium experiments): groups of the MMWHS
# and MITEA runs (model C); the other experiment groups are removed

from params.default import hyper_ps_default
from utils.utils import update_dict
from utils.losses import *
from utils.graph_conv import (
    GraphConvNorm,
    LinearLayer,
)
from utils.eval_metrics import *

# This dict contains groups of parameters that kind of belong together in order
# to conduct certain experiments
hyper_ps_groups = {

    ### Vox2Cortex ###
    'Vox2Cortex': {
        'BASE_GROUP': None,
        'BATCH_SIZE': 1,
    },


    ### Vox2Cortex on the MMWHS LV myocardium (ISBI 2027, model C) ###
    # Loss weights are the Vox2Cortex weights of the white surface
    'Vox2Cortex-MMWHS': {
        'BASE_GROUP': 'Vox2Cortex',
        'DATASET': 'MMWHS',
        'STRUCTURE_TYPE': 'LV_myo',
        'MESH_TEMPLATE_ID': 'mmwhs-lv',
        'PATCH_SIZE': [128, 128, 128],
        'SELECT_PATCH_SIZE': [128, 128, 128],
        'N_M_CLASSES': 1,
        'N_V_CLASSES': 2,
        'REDUCED_GT': False,
        'SEG_GROUND_TRUTH': 'voxel_seg',
        'MODEL_CONFIG': {
            'N_VERTEX_CLASSES': 1,
        },
        'MESH_LOSS_FUNC_WEIGHTS': [
            [4.0],  # Chamfer
            [0.01],  # Cosine
            [0.1],  # Laplace
            [0.001],  # NormalConsistency
            [5.0]  # Edge
        ],
        'USE_WANDB': False,
    },


    ### Vox2Cortex on the MITEA LV myocardium (ISBI 2027, model C): MMWHS
    # settings, MITEA data and template
    'Vox2Cortex-MITEA': {
        'BASE_GROUP': 'Vox2Cortex-MMWHS',
        'DATASET': 'MITEA',
        'MESH_TEMPLATE_ID': 'mitea-lv',
    },
}


def assemble_group_params(group_name: str):
    """ Combine group params for a certain group name and potential base
    groups.
    """
    group_params = hyper_ps_groups[group_name]
    if group_params['BASE_GROUP'] is not None:
        base_params = assemble_group_params(group_params['BASE_GROUP'])
    else:
        base_params = hyper_ps_default

    return update_dict(base_params, group_params)
