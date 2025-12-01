# TODO: Write summarization of this file
# TODO: Add code to load more datasets

# Import standard library modules for file path handling and type hints
from pathlib import Path
from typing import Union, List, Callable
import shutil
import os
import sys

# Import numerical computing and data handling libraries
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import zipfile

# Import data scaler from sklearn
from sklearn.preprocessing import MinMaxScaler

# Get parent directory of this file
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

# Import custom dataset classes for data representation
from dataset import DataEntity, Dataset

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-1-4",
    "machine-1-5",
    "machine-1-6",
    "machine-1-7",
    "machine-1-8",
    "machine-2-1",
    "machine-2-2",
    "machine-2-3",
    "machine-2-4",
    "machine-2-5",
    "machine-2-6",
    "machine-2-7",
    "machine-2-8",
    "machine-2-9",
    "machine-3-1",
    "machine-3-2",
    "machine-3-3",
    "machine-3-4",
    "machine-3-5",
    "machine-3-6",
    "machine-3-7",
    "machine-3-8",
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]
smap_data_set_number = [
    "A-1",
    "A-2",
    "A-3",
    "A-4",
    "A-7",
    "B-1",
    "D-1",
    "D-11",
    "D-13",
    "D-2",
    "D-3",
    "D-4",
    "D-5",
    "D-6",
    "D-7",
    "D-8",
    "D-9",
    "E-1",
    "E-10",
    "E-11",
    "E-12",
    "E-13",
    "E-2",
    "E-3",
    "E-4",
    "E-5",
    "E-6",
    "E-7",
    "E-8",
    "E-9",
    "F-1",
    "F-2",
    "F-3",
    "G-1",
    "G-2",
    "G-3",
    "G-4",
    "G-6",
    "G-7",
    "P-1",
    "P-2",
    "P-2",
    "P-3",
    "P-4",
    "P-7",
    "R-1",
    "S-1",
    "T-1",
    "T-2",
    "T-3",
]

msl_data_set_number = [
    "C-1",
    "D-14",
    "D-15",
    "D-16",
    "F-4",
    "F-5",
    "F-7",
    "F-8",
    "M-1",
    "M-2",
    "M-3",
    "M-4",
    "M-5",
    "M-6",
    "M-7",
    "P-10",
    "P-11",
    "P-14",
    "P-15",
    "T-12",
    "T-13",
    "T-4",
    "T-5",
]

# URLs to data
SMD_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
ANOMALY_ARCHIVE_URI = r"https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"


def download_file(
    filename: str, directory: Path, source_url: str, decompress: bool = False
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

            extract_archive(str(filepath), outdir=directory)  # type: ignore
            print(f"Extracted archive to: {directory}")


def load_data(
    dataset: str,
    group: str,
    entities: Union[str, List[str]],
    downsampling: float = None,  # type: ignore
    min_length: float = None,  # type: ignore
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
        DataEntity or list of entities/channels/machines to load.
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

    root_dir = Path(root_dir) / "ServerMachineDataset"  # type: ignore

    # ------------------------------------------------------------------
    # 1. Download missing files
    # ------------------------------------------------------------------
    for machine in machines:
        train_path = root_dir / "train" / f"{machine}.txt"  # type: ignore

        if not train_path.exists():
            print(f"Downloading SMD files for machine: {machine}")

            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "train",  # type: ignore
                source_url=f"{SMD_URL}/train/{machine}.txt",
            )
            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "test",  # type: ignore
                source_url=f"{SMD_URL}/test/{machine}.txt",
            )
            download_file(
                filename=f"{machine}.txt",
                directory=root_dir / "test_label",  # type: ignore
                source_url=f"{SMD_URL}/test_label/{machine}.txt",
            )

    # ------------------------------------------------------------------
    # 2. Load TRAIN split
    # ------------------------------------------------------------------
    if group == "train":
        entities, entities_val = [], []

        for machine in machines:
            name, name_val = "smd-train", "smd-val"
            train_file = root_dir / "train" / f"{machine}.txt"  # type: ignore
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
                train_entity = DataEntity(
                    Y=Y[:, :split_idx], name=machine, verbose=verbose
                )
                val_entity = DataEntity(
                    Y=Y[:, split_idx:], name=machine, verbose=verbose
                )
                entities.append(train_entity)
                entities_val.append(val_entity)
            else:
                entity = DataEntity(Y=Y, name=machine, verbose=verbose)
                entities.append(entity)

        if validation:
            smd_train = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
            smd_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)  # type: ignore
            return smd_train, smd_val

        return Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore

    # ------------------------------------------------------------------
    # 3. Load TEST split
    # ------------------------------------------------------------------
    elif group == "test":
        entities = []

        for machine in machines:
            name = "smd-test"
            test_file = root_dir / "test" / f"{machine}.txt"  # type: ignore
            label_file = root_dir / "test_label" / f"{machine}.txt"  # type: ignore

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

        return Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore


def _load_nasa(
    group: str,
    spacecraft: str,
    channels: str = None,  # type: ignore
    downsampling: int = None,  # type: ignore
    root_dir="./data",
    normalize=True,
    verbose=True,
    validation=False,
):
    """
    Read data streams from every channels of a particular spacecraft (SMAP or MSL).
    1 channel emits 1 data stream.
    """

    # Read metadata file
    root_dir = f"{root_dir}/NASA"
    meta_data = pd.read_csv(f"{root_dir}/labeled_anomalies.csv")

    # `spacecraft` is the particular spacecraft the user want to load data
    # List all channels of this spacecraft
    # Treat 1 stream from 1 channel as 1 data entity.
    CHANNEL_IDS = list(
        meta_data.loc[meta_data["spacecraft"] == spacecraft]["chan_id"].values
    )
    if verbose:
        print(f"Number of Entities: {len(CHANNEL_IDS)}")

    # Debug
    # Print out channels of the particular spacecraft (SMAP or MSL) that user want to load
    print("channels", channels)

    # Print out full channels of this particular spacecraft
    print("CHANNELS", sorted(CHANNEL_IDS))

    # If user didn't specify which channel of this spacecraft to load, we will load all by defautl.
    if channels is None:
        channels = CHANNEL_IDS

    # If user want to load a particular channel of this spacecraft
    if isinstance(channels, str):
        channels = [channels]  # type: ignore

    # If the user want to load training group for this spacecraft.
    # A training group for a particular spacecraft contains data streams that belong to that spacecraft.
    if group == "train":
        # List of data entities used for training
        entities: List[DataEntity] = []
        # List of data entities used for validation
        entities_val: List[DataEntity] = []

        # Loop over each channel (of this particular spacecraft) the user want to load
        for channel_id in channels:
            ## 1. Initialize normalizer
            # If the user want to normalize data, we will init the normalizer first.
            # Though we always normalize data by default
            if normalize:
                with open(f"{root_dir}/train/{channel_id}.npy", "rb") as f:
                    Y = np.load(f)  # Transpose dataset
                scaler = MinMaxScaler()
                scaler.fit(Y)

            ## 2. Setup name for the returned datasets
            name = f"{spacecraft}-train"  # Name of the training dataset
            name_val = f"{spacecraft}-val"  # Name of the testing dataset

            ## 3. Perform normalization
            # Read data stream from the current channel
            with open(f"{root_dir}/train/{channel_id}.npy", "rb") as f:
                Y = np.load(f).T  # Transpose dataset

            if normalize:
                Y = scaler.transform(Y.T).T  # type: ignore

            ## 4. Downsampling to make sure the stream is not too lengthy.
            if downsampling is not None:
                n_features, n_timesteps = Y.shape

                right_padding = downsampling - n_timesteps % downsampling
                Y = np.pad(Y, ((0, 0), (right_padding, 0)))

                Y = Y.reshape(
                    n_features, Y.shape[-1] // downsampling, downsampling
                ).max(axis=2)

            ## 5. Load validation set
            # If the user want to load validation set for this spacecraft.
            # Though we will NOT load validation set by default.
            if validation:
                train_length = int(Y.shape[1] * 0.9)
                entity = DataEntity(
                    Y=Y[:, :train_length], name=channel_id, verbose=verbose
                )
                entities.append(entity)
                entity_val = DataEntity(
                    Y=Y[:, train_length:], name=channel_id, verbose=verbose
                )
                entities_val.append(entity_val)
            else:
                # print('Y', Y.shape)
                # entity = DataEntity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, verbose=verbose)
                entity = DataEntity(Y=Y, name=channel_id, verbose=verbose)
                entities.append(entity)

        # If the user want validation set, we will return both training set and validation set.
        if validation:
            data = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
            data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)  # type: ignore
            return data, data_val
        # We will return only training set by default.
        else:
            data = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
            return data
    # If user want to load the testing group of this spacecraft.
    elif group == "test":
        # List to data entities we will return as testing group of this spacecraft
        entities: List[DataEntity] = []

        # Loop over each channel of this spacecraft
        for channel_id in channels:
            ## 1. Init normalizer
            if normalize:
                with open(f"{root_dir}/train/{channel_id}.npy", "rb") as f:
                    Y = np.load(f)  # Transpose dataset
                scaler = MinMaxScaler()
                scaler.fit(Y)

            ## 2. Normalize
            name = f"{spacecraft}-test"
            with open(f"{root_dir}/test/{channel_id}.npy", "rb") as f:
                Y = np.load(f).T  # Transpose dataset

            if normalize:
                Y = scaler.transform(Y.T).T  # type: ignore

            ## 3. Prepare labels for these testing entities
            # Init labels for these data entities
            labels = np.zeros(Y.shape[1])  # 1D NumPy array of length n_timesteps
            # Read an anomalous sequence for each entity from metadata
            anomalous_sequences = eval(
                meta_data.loc[meta_data["chan_id"] == channel_id][
                    "anomaly_sequences"
                ].values[0]
            )
            if verbose:
                print("Anomalous sequences:", anomalous_sequences)

            # Loop over each anomalous sequence
            # All time-steps within anomalous sequence will be marked 1.
            for interval in anomalous_sequences:
                labels[interval[0] : interval[1]] = 1

            ## 4. Downsampling to make sure testing entities are not too lengthy
            if downsampling is not None:
                n_features, n_timesteps = Y.shape
                right_padding = downsampling - n_timesteps % downsampling

                Y = np.pad(Y, ((0, 0), (right_padding, 0)))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(
                    n_features, Y.shape[-1] // downsampling, downsampling
                ).max(axis=2)
                labels = labels.reshape(
                    labels.shape[0] // downsampling, downsampling
                ).max(axis=1)

            # Expand the 1D NumPy array into 2D NumPy array (a matrix) with 1 row
            labels = labels[None, :]  # Shape: (1, n_timesteps)
            # entity = DataEntity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, labels=labels, verbose=verbose)
            entity = DataEntity(Y=Y, name=channel_id, labels=labels, verbose=verbose)
            entities.append(entity)

        # Package all the (training or testing) entities, labels into one dataset instance of this spacecraft
        data = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
        return data


def load_msl(
    group,
    channels=None,
    downsampling=None,
    root_dir="./data",
    normalize=True,
    verbose=True,
    validation=False,
):
    return _load_nasa(
        group=group,
        spacecraft="MSL",
        channels=channels,  # type: ignore
        downsampling=downsampling,  # type: ignore
        root_dir=root_dir,
        normalize=normalize,
        verbose=verbose,
        validation=validation,
    )


def load_smap(
    group,
    channels: str = None,  # type: ignore
    downsampling: int = None,  # type: ignore
    root_dir="./data",
    normalize=True,
    verbose=True,
    validation=False,
):
    return _load_nasa(
        group=group,
        spacecraft="SMAP",
        channels=channels,
        downsampling=downsampling,
        root_dir=root_dir,
        normalize=normalize,
        verbose=verbose,
        validation=validation,
    )


def download_anomaly_archive(root_dir: str = "./data"):
    """Convenience function to download the Timeseries Anomaly Archive datasets"""

    # Download the data
    download_file(
        filename="AnomalyArchive",
        directory=Path(root_dir),
        source_url=ANOMALY_ARCHIVE_URI,
        decompress=True,
    )

    # Reorganising the data
    shutil.move(
        src=f"{root_dir}/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData",
        dst=root_dir,
    )
    os.remove(os.path.join(root_dir, "AnomalyArchive"))
    shutil.rmtree(os.path.join(root_dir, "AnomalyDatasets_2021"))
    shutil.move(
        src=f"{root_dir}/UCR_Anomaly_FullData", dst=f"{root_dir}/AnomalyArchive"
    )


def load_anomaly_archive(
    group: str,
    datasets=None,
    downsampling: int = None,  # type: ignore
    min_length=None,
    root_dir="./data",
    normalize=True,
    verbose=True,
    validation=False,
):
    # If the raw dataset doesn't exist, download it
    if not os.path.exists(f"{root_dir}/AnomalyArchive/"):
        download_anomaly_archive(root_dir=root_dir)

    # List of data entities as sources of this dataset
    ANOMALY_ARCHIVE_ENTITIES = [
        "_".join(entity.split("_")[:4])
        for entity in os.listdir(os.path.join(root_dir, "AnomalyArchive"))
    ]
    ANOMALY_ARCHIVE_ENTITIES = sorted(ANOMALY_ARCHIVE_ENTITIES)

    # If user doesn't specify entities, we load all by default.
    # Data streams of a particular entity will be packaged into a single Dataset instance.
    if datasets is None:
        datasets = ANOMALY_ARCHIVE_ENTITIES
    if verbose:
        print(f"Number of datasets: {len(datasets)}")

    entities: List[DataEntity] = []  # List of training entities
    entities_val: List[DataEntity] = []  # List of validation entities

    # Normalize datasets to always be a list for consistent checking
    datasets_list = datasets if isinstance(datasets, list) else [datasets]

    # Loop over each file inside AnomalyArchive directory
    for file in os.listdir(os.path.join(root_dir, "AnomalyArchive")):
        downsampling_entity = downsampling  # True or False

        # Infer name of dataset from its path
        file_parts = file.split("_")
        dataset_name = "_".join(file_parts[:4])

        # Check if the file matches any of the requested datasets
        if (
            dataset_name in datasets_list
            or file_parts[0] in datasets_list
            or (len(file_parts) > 2 and file_parts[2] in datasets_list)
        ):
            with open(os.path.join(root_dir, "AnomalyArchive", file)) as f:
                # Read the primary time-series which we will use to train or test
                Y = f.readlines()

                # Parse the raw time-series data from the file into a NumPy array
                # Single-line format: convert space-separated values into a 1D sequence
                if len(Y) == 1:
                    Y = Y[0].strip()
                    # Convert string values to floats and reshape to (1, n_timesteps)
                    # for univariate time-series representation
                    Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape(
                        (1, -1)
                    )
                # Multi-line format: each line contains one observation/measurement
                elif len(Y) > 1:
                    # Convert each line to a float value and reshape to (1, n_timesteps)
                    # representing a single univariate time-series
                    Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

            fields = file.split("_")
            meta_data: dict = {
                "name": "_".join(fields[:4]),
                "train_end": int(fields[4]),
                "anomaly_start_in_test": int(fields[5]) - int(fields[4]),
                "anomaly_end_in_test": int(fields[6][:-4]) - int(fields[4]),
            }
            if verbose:
                print(f"DataEntity meta-data: {meta_data}")

            if normalize:
                # Current shape of Y is (1, n_timesteps)
                Y_train = Y[0, 0 : meta_data["train_end"]].reshape((-1, 1))  # type: ignore
                scaler = MinMaxScaler()
                scaler.fit(Y_train)
                Y = scaler.transform(Y.T).T  # type: ignore

            n_timesteps = Y.shape[-1]  # type: ignore
            len_train = meta_data["train_end"]
            len_test = n_timesteps - len_train

            # No downsampling if n_timesteps < min_length
            if (downsampling_entity is not None) and (min_length is not None):
                if (len_train // downsampling_entity < min_length) or (
                    len_test // downsampling_entity < min_length
                ):
                    downsampling_entity = None

            if group == "train":
                name = f"{meta_data['name']}-train"
                name_val = f"{meta_data['name']}-val"
                Y = Y[0, 0 : meta_data["train_end"]].reshape((1, -1))  # type: ignore

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_timesteps = Y.shape

                    right_padding = (
                        downsampling_entity - n_timesteps % downsampling_entity
                    )
                    Y = np.pad(Y, ((0, 0), (right_padding, 0)))

                    Y = Y.reshape(
                        n_features,
                        Y.shape[-1] // downsampling_entity,
                        downsampling_entity,
                    ).max(axis=2)

                if validation:
                    train_length = int(Y.shape[1] * 0.9)
                    entity = DataEntity(
                        Y=Y.reshape((1, -1))[:, :train_length],
                        name=meta_data["name"],
                        verbose=verbose,
                    )
                    entities.append(entity)
                    entity_val = DataEntity(
                        Y=Y.reshape((1, -1))[:, train_length:],
                        name=meta_data["name"],
                        verbose=verbose,
                    )
                    entities_val.append(entity_val)
                else:
                    entity = DataEntity(
                        Y=Y.reshape((1, -1)), name=meta_data["name"], verbose=verbose
                    )
                    entities.append(entity)

            elif group == "test":
                name = f"{meta_data['name']}-test"
                Y = Y[0, meta_data["train_end"] + 1 :].reshape((1, -1))  # type: ignore

                # Label the data
                labels = np.zeros(Y.shape[1])
                labels[
                    meta_data["anomaly_start_in_test"] : meta_data[
                        "anomaly_end_in_test"
                    ]
                ] = 1

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_timesteps = Y.shape
                    right_padding = (
                        downsampling_entity - n_timesteps % downsampling_entity
                    )

                    Y = np.pad(Y, ((0, 0), (right_padding, 0)))
                    labels = np.pad(labels, (right_padding, 0))

                    Y = Y.reshape(
                        n_features,
                        Y.shape[-1] // downsampling_entity,
                        downsampling_entity,
                    ).max(axis=2)
                    labels = labels.reshape(
                        labels.shape[0] // downsampling_entity, downsampling_entity
                    ).max(axis=1)

                labels = labels[None, :]
                entity = DataEntity(
                    Y=Y.reshape((1, -1)),
                    name=meta_data["name"],
                    labels=labels,
                    verbose=verbose,
                )
                entities.append(entity)

    if validation:
        data = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
        data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)  # type: ignore
        return data, data_val
    else:
        data = Dataset(entities=entities, name=name, verbose=verbose)  # type: ignore
        return data


def load_iops():
    pass
