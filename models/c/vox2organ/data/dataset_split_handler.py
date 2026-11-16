""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

# Modified (2026, ISBI 2027 LV myocardium experiments): only the MMWHS and
# MITEA datasets

from data.mmwhs import MMWHSDataset, MITEADataset
from data.supported_datasets import MMWHSDatasets, MITEADatasets

# Mapping supported datasets to split functions
dataset_split_handler = {
    **{x.name: MMWHSDataset.split for x in MMWHSDatasets},
    **{x.name: MITEADataset.split for x in MITEADatasets},
}
