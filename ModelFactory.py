"""Factory class to create and train models based on the provided dataset."""
import logging

from Model import Model

logging.basicConfig(level=logging.INFO)

class ModelFactory:
    """Factory class to create and train models based on the provided dataset."""

    def __init__(self,dataset):
        self.dataset = dataset
        self.available_model_types = ["lstm", "gru","xgb", "svm"]

        self.input_shape, self.output_shape = self.dataset.get_io()

        self.model = None

    def create_model(self, model_type, train=True, epochs=20, split=0.8) -> Model:
        """Create and optionally train a model of the specified type. If given model type is not supported raises ValueError."""

        if model_type not in self.available_model_types:
            raise ValueError("Model type %s is currently not supported or misspelled.", model_type)

        model = Model(model_type,self.input_shape,self.output_shape)

        if train is True:
            model.train(self.dataset,epochs,split)
        else:
            logging.warning("Creating an untrained model, please manually train your model later")

        return model

    def get_available_model_types(self):
        """Return a list of available model types."""
        return self.available_model_types
