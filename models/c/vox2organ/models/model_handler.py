
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

# Modified (2026, ISBI 2027 LV myocardium experiments): Vox2Cortex only

from utils.utils import ExtendedEnum

from models.vox2cortex import Vox2Cortex

class ModelHandler(ExtendedEnum):
    vox2cortex = Vox2Cortex
