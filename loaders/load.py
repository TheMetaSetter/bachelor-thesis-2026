# TODO: Write summarization of this file
# TODO: Add code to load more datasets

from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import numpy as np

from dataset import DataEntity, Dataset

MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9',
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9',
            'machine-3-10', 'machine-3-11']
smap_data_set_number = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
msl_data_set_number = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']

# URLs to data
SMD_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"

from pathlib import Path
import requests
from tqdm import tqdm
import zipfile


def download_file(
    filename: str,
    directory: str,
    source_url: str,
    decompress: bool = False
) -> None:
    """
    Download a data file from a given URL into the specified directory,
    with optional decompression support.

    Parameters
    ----------
    filename : str
        The name of the file to save locally.
    directory : str
        The directory path where the file should be stored.
    source_url : str
        The URL from which to download the file.
    decompress : bool, optional
        Whether to automatically extract the file after download
        (supports .zip or other archive formats if patool is available).
    """

    # Ensure the directory exists
    directory = Path(directory) if isinstance(directory, str) else directory
    print(f"Directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    # Construct full file path
    filepath = directory / filename

    # Stream the download for memory efficiency
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(source_url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KiB

    # Display progress bar while downloading
    with tqdm(total=total_size, unit="iB", unit_scale=True) as progress:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(block_size):
                progress.update(len(chunk))
                f.write(chunk)
                f.flush()

    # Verify file size after download
    downloaded_size = filepath.stat().st_size
    print(f"Downloaded: {downloaded_size / 1024:.2f} KiB")

    # Optionally decompress the downloaded file
    if decompress:
        if filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(directory)
                print(f"Extracted ZIP archive to: {directory}")
        else:
            from patoolib import extract_archive
            extract_archive(str(filepath), outdir=directory)
            print(f"Extracted archive to: {directory}")

from typing import Union, List, Callable


def load_data(
    dataset: str,
    group: str,
    entities: Union[str, List[str]],
    downsampling: float = None,
    min_length: float = None,
    root_dir: str = "./data",
    normalize: bool = True,
    verbose: bool = True,
    validation: bool = False,
):
    """
    Load a time-series anomaly detection dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load. Must be one of VALID_DATASETS.
    group : str
        Data split to load, e.g., 'train' or 'test'.
    entities : Union[str, List[str]]
        Entity or list of entities/channels/machines to load.
    downsampling : float, optional
        Factor by which to downsample the data.
    min_length : float, optional
        Minimum sequence length (used for 'anomaly_archive' datasets).
    root_dir : str, default='./data'
        Directory where datasets are stored.
    normalize : bool, default=True
        Whether to normalize the target series Y.
    verbose : bool, default=True
        Whether to print dataset loading information.
    validation : bool, default=False
        Whether to load the validation split (if available).

    Returns
    -------
    Dataset
        The loaded dataset object corresponding to the specified parameters.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.
    """

    # Define a dispatch table mapping dataset names to their loader functions
    loaders: dict[str, Callable] = {
        "smd": load_smd,
        "msl": load_msl,
        "smap": load_smap,
        "anomaly_archive": load_anomaly_archive,
        "iops": load_iops,
    }

    # Validate dataset name
    if dataset not in loaders:
        raise ValueError(
            f"Dataset must be one of {list(loaders.keys())}, but '{dataset}' was passed!"
        )

    # Handle dataset-specific entity name normalization
    if dataset == "msl" and entities == "msl":
        entities = msl_data_set_number
    elif dataset == "smap" and entities == "smap":
        entities = smap_data_set_number

    # Common arguments for all loaders
    common_args = dict(
        group=group,
        downsampling=downsampling,
        root_dir=root_dir,
        normalize=normalize,
        verbose=verbose,
        validation=validation,
    )

    # Handle dataset-specific keyword differences
    if dataset == "smd":
        return loaders["smd"](machines=entities, **common_args)
    elif dataset == "msl":
        return loaders["msl"](channels=entities, **common_args)
    elif dataset == "smap":
        return loaders["smap"](channels=entities, **common_args)
    elif dataset == "anomaly_archive":
        return loaders["anomaly_archive"](
            datasets=entities, min_length=min_length, **common_args
        )
    elif dataset == "iops":
        return loaders["iops"](filename=entities, **common_args)

def load_smd(
    group: str,
    machines: list[str] | str | None = None,
    downsampling: int | None = None,
    root_dir: str = "./data",
    normalize: bool = True,
    verbose: bool = True,
    validation: bool = False,
):
    """
    Load the Server Machine Dataset (SMD) for time-series anomaly detection.

    Parameters
    ----------
    group : str
        Which data split to load: 'train' or 'test'.
    machines : list[str] or str, optional
        Machine IDs to load. If None, all default machines (MACHINES) are used.
    downsampling : int, optional
        Factor by which to downsample the data using max pooling.
    root_dir : str, default='./data'
        Base directory where the dataset is stored or will be downloaded to.
    normalize : bool, default=True
        Unused argument kept for compatibility; SMD is pre-normalized.
    verbose : bool, default=True
        Whether to print progress and dataset info.
    validation : bool, default=False
        Whether to create a 90/10 train-validation split when loading train data.

    Returns
    -------
    Dataset or (Dataset, Dataset)
        Returns a Dataset object for train/test, or a tuple (train, val)
        when `validation=True` and `group='train'`.
    """

    # SMD is already normalized; `normalize` is kept for compatibility
    if machines is None:
        machines = MACHINES
    elif isinstance(machines, str):
        machines = [machines]

    root_dir = Path(root_dir) / "ServerMachineDataset"

    # ------------------------------------------------------------------
    # 1. Download missing files
    # ------------------------------------------------------------------
    for machine in machines:
        train_path = root_dir / "train" / f"{machine}.txt"

        if not train_path.exists():
            print(f"Downloading SMD files for machine: {machine}")

            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "train",
                source_url=f"{SMD_URL}/train/{machine}.txt",
            )
            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "test",
                source_url=f"{SMD_URL}/test/{machine}.txt",
            )
            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "test_label",
                source_url=f"{SMD_URL}/test_label/{machine}.txt",
            )

    # ------------------------------------------------------------------
    # 2. Load TRAIN split
    # ------------------------------------------------------------------
    if group == "train":
        entities, entities_val = [], []

        for machine in machines:
            name, name_val = "smd-train", "smd-val"
            train_file = root_dir / "train" / f"{machine}.txt"
            Y = np.loadtxt(train_file, delimiter=",").T

            # Downsampling via max pooling
            if downsampling is not None:
                num_features, num_timesteps = Y.shape
                right_padding = downsampling - (num_timesteps % downsampling)
                Y = np.pad(Y, ((0, 0), (right_padding, 0)))
                Y = Y.reshape(
                    num_features, Y.shape[-1] // downsampling, downsampling
                ).max(axis=2)

            # Optional 90/10 split for validation
            if validation:
                split_idx = int(Y.shape[1] * 0.9)
                train_entity = DataEntity(Y=Y[:, :split_idx], name=machine, verbose=verbose)
                val_entity = DataEntity(Y=Y[:, split_idx:], name=machine, verbose=verbose)
                entities.append(train_entity)
                entities_val.append(val_entity)
            else:
                entity = DataEntity(Y=Y, name=machine, verbose=verbose)
                entities.append(entity)

        if validation:
            smd_train = Dataset(entities=entities, name=name, verbose=verbose)
            smd_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return smd_train, smd_val

        return Dataset(entities=entities, name=name, verbose=verbose)

    # ------------------------------------------------------------------
    # 3. Load TEST split
    # ------------------------------------------------------------------
    elif group == "test":
        entities = []

        for machine in machines:
            name = "smd-test"
            test_file = root_dir / "test" / f"{machine}.txt"
            label_file = root_dir / "test_label" / f"{machine}.txt"

            Y = np.loadtxt(test_file, delimiter=",").T
            labels = np.loadtxt(label_file, delimiter=",")

            # Downsampling via max pooling
            if downsampling is not None:
                num_features, num_timesteps = Y.shape
                right_padding = downsampling - (num_timesteps % downsampling)
                Y = np.pad(Y, ((0, 0), (right_padding, 0)))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(
                    num_features, Y.shape[-1] // downsampling, downsampling
                ).max(axis=2)
                labels = labels.reshape(
                    labels.shape[0] // downsampling, downsampling
                ).max(axis=1)

            labels = labels[None, :]  # make shape (1, T)
            entity = DataEntity(Y=Y, name=machine, labels=labels, verbose=verbose)
            entities.append(entity)

        return Dataset(entities=entities, name=name, verbose=verbose)

def load_msl():
    pass

def load_smap():
    pass

def load_anomaly_archive():
    pass

def load_iops():
    pass