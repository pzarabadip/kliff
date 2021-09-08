import os
from pathlib import Path
from typing import Callable, List, Optional

import torch
from kliff.dataset import Configuration
from kliff.dataset.configuration import SUPPORTED_FORMAT
from kliff.descriptors.descriptor import load_fingerprints
from kliff.utils import to_path
from loguru import logger
from torch.utils.data import Dataset as Torch_Dataset


class Dataset:
    """
    A dataset of multiple configurations (:class:`~kliff.dataset.Configuration`).

    Args:
        path: Path of a file storing a configuration or filename to a directory containing
            multiple files. If given a directory, all the files in this directory and its
            subdirectories with the extension corresponding to the specified file_format
            will be read.
        file_format: Format of the file that stores the configuration, e.g. `xyz`.
    """

    def __init__(self, path: Optional[Path] = None, file_format="xyz"):
        self.file_format = file_format

        if path is not None:
            self.configs = self._read(path, file_format)

        else:
            self.configs = []

    def add_configs(self, path: Path):
        """
        Read configurations from filename and added them to the existing set of
        configurations.

        This is a convenience function to read configurations from different directory
        on disk.

        Args:
            path: Path the directory (or filename) storing the configurations.
        """

        configs = self._read(path, self.file_format)
        self.configs.extend(configs)

    def get_configs(self) -> List[Configuration]:
        """
        Get the configurations.
        """
        return self.configs

    def get_num_configs(self) -> int:
        """
        Return the number of configurations in the dataset.
        """
        return len(self.configs)

    @staticmethod
    def _read(path: Path, file_format: str = "xyz"):
        """
        Read atomic configurations from path.
        """
        try:
            extension = SUPPORTED_FORMAT[file_format]
        except KeyError:
            raise DatasetError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

        path = to_path(path)

        if path.is_dir():
            parent = path
            all_files = []
            for root, dirs, files in os.walk(parent):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(to_path(root).joinpath(f))
            all_files = sorted(all_files)
        else:
            parent = path.parent
            all_files = [path]

        configs = [Configuration.from_file(f, file_format) for f in all_files]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )

        logger.info(f"{len(configs)} configurations read from {path}")

        return configs


class FingerprintsDataset(Torch_Dataset):
    """
    Atomic environment fingerprints dataset used by torch models.

    Args:
        filename: to the fingerprints file.
        transform: transform to be applied on a sample.
    """

    def __init__(self, filename: Path, transform: Optional[Callable] = None):
        self.fp = load_fingerprints(filename)
        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


def fingerprints_collate_fn(batch):
    """
    Convert a batch of samples into tensor.

    Unlike the default collate_fn(), which stack samples in the batch (requiring each
    sample having the same dimension), this function does not do the stack.

    Args:
        batch: A batch of samples.

    Returns:
        A list of tensor.
    """
    tensor_batch = []
    for i, sample in enumerate(batch):
        tensor_sample = {}
        for key, value in sample.items():
            if type(value).__module__ == "numpy":
                value = torch.from_numpy(value)
            tensor_sample[key] = value
        tensor_batch.append(tensor_sample)

    return tensor_batch


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
