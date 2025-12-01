import numpy as np

from typing import Optional, List


class DataEntity(object):
    """
    This class represents a single data stream from a particular data source.
    Data source can be a particular machine in a factory, or a particular patient in a hospital, or a particular sensor in an IoT system, etc.
    We use the word "data entity" to refer to a single data stream like this.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        # Load input arguments into attributes of the entity
        ## Load primary time-series data - main data (simply the data we will find anomalies inside)
        self.Y = Y
        self.num_features, self.num_timesteps = Y.shape

        ## Load exogenous covariates - external features that might influence the main data
        if X is None:  # If the user didn't pass any exogenous covariates
            self.X = None
            self.num_exogenous = 0
        else:  # If the user did pass exogenous covariates
            num_exogenous, num_exo_timesteps = X.shape

            # The number of exogenous time-steps (X) must be the same as main time-steps (Y)
            assert num_exo_timesteps == self.num_timesteps, (
                "The number of timesteps in the main data (Y) and optional/exogenous data (X) must be the same."
            )

            self.X = X
            self.num_exogenous = num_exogenous

        ## Load labels
        if (
            labels is None
        ):  # If the user didn't pass any anomaly labels for the main data
            # Treat all the main data as "normal" -- which means label `0`
            self.labels = np.zeros((1, self.num_timesteps))
        else:
            # The number of labels must be the same as the number of time-steps
            # A.k.a mapping 1 label - 1 time-step
            assert labels.shape[-1] == self.num_timesteps, (
                "The number of timesteps in the main data (Y) and the labels array must be the same."
            )
            self.labels = labels

        ## Load name of the particular data source
        if name is None:  # If the user didn't pass any name
            self.name = "N/A"  # TODO: Random name generator
        else:
            self.name = name

        ## Load anomaly mask
        if mask is None:  # If the user didn't pass any anomaly mask
            self.mask = np.ones((self.num_features, self.num_timesteps))
        else:
            # Shape of anomaly mask and shape of the main time-series must be the same
            assert mask.shape == Y.shape, (
                "The main time-series data (Y) and the anomaly mask must have the same shape."
            )
            self.mask = mask

        # Handling `verbose` - print or not print out information of this entity
        self.verbose = verbose
        if self.verbose:
            print(42 * "-")
            print("Entity: ", self.name)
            print("num_features: ", self.num_features)
            print("num_exogenous: ", self.num_exogenous)
            print("num_timesteps: ", self.num_timesteps)
            print("Anomaly %: ", np.mean(self.labels))
            print("Mask %: ", np.mean(self.mask))
            print(42 * "-")


class Dataset(object):
    """
    This class represents a set of data entities having the same type such as machines or patients or sensors.
    """

    def __init__(self, entities: List[DataEntity], name: str, verbose: bool = False):
        self.entities = entities  # The number of distinct entities in the dataset (e.g, machine-1-1, machine-2-1, ...)
        self.name = name  # Name of the dataset
        self.verbose = verbose  # Print out info of the dataset or not

        # Load important information of the entities
        self.num_features = entities[0].num_features
        self.num_exogenous = entities[0].num_exogenous
        self.num_entities = len(entities)

        self.total_timesteps = 0
        self.num_anomalies = 0
        for entity in entities:
            self.total_timesteps += entity.num_timesteps
            self.num_anomalies += np.sum(entity.labels)

        # Check if all entities have the same number of features
        assert all(
            [(entity.num_features == self.num_features) for entity in self.entities]
        ), "All entities must have the same number of features."

        # Check if all entities have the same number of exogenous covariates
        assert all(
            [(entity.num_exogenous == self.num_exogenous) for entity in self.entities]
        ), "All entities must have the same number of exogenous covariates/variables."

        # Print out information of the dataset
        if self.verbose:
            print(42 * "-")
            print(self)
            print(42 * "-")

    def get_entity(self, entity_name: str):
        """
        Returns a specific entity in the dataset as user's request
        """

        for entity in self.entities:
            if entity.name == entity_name:
                return entity
            else:
                raise ValueError("Entity not found!")

    def __len__(self):
        # Treat the number of entities in a dataset as the length of that dataset
        return self.num_entities

    def __iter__(self):
        """
        This method will allow user to write snippet like the one below
        to loop over data entities in a dataset.
        ```
        dataset = Dataset(...)
        for entity in dataset:
            # Do something here
        ```
        """
        for entity in self.entities:
            yield entity

    def __str__(self):
        """
        Return a human-readable string representation of the Dataset object.

        This method summarizes the key attributes of the dataset—such as its name,
        number of entities, features, exogenous variables, total time steps, and
        anomaly percentage—in a structured, indented format. It replaces any values
        whose keys end with `_token` by placeholder tags (e.g., `<API_TOKEN>`) to
        prevent sensitive information from being displayed.

        Returns
        -------
        str
            A formatted string representation of the dataset, suitable for printing
            or logging. Example output:

            Dataset(
                name=MachineHealth,
                num_entities=8,
                num_features=5,
                num_exogenous=2,
                total_timesteps=12000,
                anomaly_percentage=0.47,
            )
        """

        info_dict = {
            "name": self.name,
            "num_entities": self.num_entities,
            "num_features": self.num_features,
            "num_exogenous": self.num_exogenous,
            "total_timesteps": self.total_timesteps,
            "anomaly_percentage": 100 * (self.num_anomalies / self.total_timesteps),
        }
        cleaned_info = {
            key: (f"<{key.upper()}>" if key.endswith("_token") else value)
            for key, value in info_dict.items()
        }
        attrs_as_str = [f"\t{key}={value},\n" for key, value in cleaned_info.items()]

        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
