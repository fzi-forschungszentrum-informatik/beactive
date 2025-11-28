"""Extractor module to load and preprocess data from various studies."""
import logging

import numpy as np

from adapters.extract1 import load_dataset_1
from Dataset import Dataset
from IdCreator import IdCreator

logging.basicConfig(level=logging.INFO)

class Extractor:
    """Class to handle data extraction and preprocessing from multiple studies.
    Initialize the Extractor with a dict of paths to the root folder of the dataset and an affiliated id.
    """

    def __init__(self, paths):
        self.paths = paths
        self.ID_creator = IdCreator()

    def load_studies(self, use_raw_data=True, use_lagged_mood=False,
                     window=15, locations=['hip'], criteria={}):
        """Load and preprocess data from specified studies and locations."""

        data_list = []
        mood_list = []
        id_list = []

        for location in locations:
            # Load all study IDs if adapter is available
            for path, datasetID in self.paths.items():

                data = None
                mood = None

                if datasetID == 1:
                    logging.info("Dataset 1")
                    data, mood,ids = load_dataset_1(path,
                                                 use_raw_data=use_raw_data,
                                                 use_lagged_mood=use_lagged_mood,
                                                 window=window,location=location,
                                                 id_creator=self.ID_creator,
                                                 criteria=criteria)
                    data_list.append(data)
                    mood_list.append(mood)
                    id_list = id_list + ids
                else:
                    raise ValueError("No extractor available for dataset %s", datasetID)

                if data is not None and mood is not None:

                    try:
                        X = np.append(X,data,axis=0)
                        y = np.append(y,mood,axis=0)
                    except Exception:
                        X = data
                        y = mood

                    if X.shape[0] != y.shape[0]:
                        logging.error("Data and labels dimensions dont match, check the adapters!")
        return X, y, id_list

    def create_dataset(self, use_raw_data=False, use_lagged_mood=False,
                       window=15, locations=['hip'], criteria={}):
        """Create a Dataset object from the loaded studies."""

        data, mood, id_list = self.load_studies(use_raw_data=use_raw_data,
                                                use_lagged_mood=use_lagged_mood,
                                                window=window,
                                                locations=locations,
                                                criteria=criteria)

        mood = mood.astype(int)

        inpute_shape = data[0].shape
        output_shape = mood[0].shape
        logging.debug("Input Shape: %s", data.shape)
        logging.debug("Output Shape: %s", mood.shape)

        dataset = Dataset(data,mood,inpute_shape,output_shape, id_list)

        return dataset
    