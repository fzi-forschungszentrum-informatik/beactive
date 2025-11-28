"""Example script to demonstrate the usage of the framework."""

from Extractor import Extractor
from ModelFactory import ModelFactory
from utils import write_overall_metrics_to_csv, write_per_id_metrics_to_csv

datasets = {
    "./sample_data": 1,
    }

TIME_WINDOW = 5
WEARING_LOCATIONS = ['hip']
MODEL_TYPE = 'gru'
TRAIN_TEST_SPLIT = 0.7
NUMBER_OF_EPOCHS = 1
USE_RAW_DATA = True

# Create an extractor for the given dataset/id dictionary
my_extractor = Extractor(datasets)

# Create a datset using the extractor
my_dataset = my_extractor.create_dataset(use_raw_data=USE_RAW_DATA,
                                         use_lagged_mood=False,
                                         window=TIME_WINDOW,
                                         locations=WEARING_LOCATIONS)

# Create a model factory based on the dataset
my_factory = ModelFactory(my_dataset)

# We can now create multiple models featuring our dataset because the dataset is tied to the ModelFactory
my_model = my_factory.create_model(MODEL_TYPE, split=TRAIN_TEST_SPLIT, epochs=NUMBER_OF_EPOCHS)

#my_model_2 = my_factory.create_model(params)

# Validate the model with the given dataset
individual_results, overall_results = my_model.validate(my_dataset)

# Write per-ID results to CSV
write_per_id_metrics_to_csv(individual_results, f'{str(TIME_WINDOW)}_{str(MODEL_TYPE)}_{str(TRAIN_TEST_SPLIT)}_{str(NUMBER_OF_EPOCHS)}_{str(WEARING_LOCATIONS)}_per_person_results.csv')
# Write overall results to CSV
write_overall_metrics_to_csv(overall_results, f'{str(TIME_WINDOW)}_{str(MODEL_TYPE)}_{str(TRAIN_TEST_SPLIT)}_{str(NUMBER_OF_EPOCHS)}_{str(WEARING_LOCATIONS)}_overall_results.csv')