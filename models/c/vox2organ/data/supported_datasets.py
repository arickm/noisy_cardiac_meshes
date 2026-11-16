
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

# Modified (2026, ISBI 2027 LV myocardium experiments): only the MMWHS (CT)
# and MITEA (3D echo) LV myocardium datasets. RAW_DATA_DIR is the dataset
# folder of the data layout in the repository README and is set with
# --raw_data_dir.

from enum import IntEnum


class SupportedDatasets(IntEnum):
    """ List supported datasets """
    MMWHS = 30
    MITEA = 31

class MMWHSDatasets(IntEnum):
    """ Cardiac CT """
    MMWHS = SupportedDatasets.MMWHS.value

class MITEADatasets(IntEnum):
    """ 3D echo """
    MITEA = SupportedDatasets.MITEA.value


dataset_paths = {
    SupportedDatasets.MMWHS.name: {'RAW_DATA_DIR': None, 'FIXED_SPLIT': "splits.json"},
    SupportedDatasets.MITEA.name: {'RAW_DATA_DIR': None, 'FIXED_SPLIT': "splits.json"},
}
