import os
import logging
import warnings

import pandas as pd
import numpy as np
import unisens
import tsfel

from utils import * 
from adapters.extract_metadata import meets_criteria

MOOD_COLS = ['Participant', 'Trigger_time', 'Form', MOOD_ITEM_EA1_ID, MOOD_ITEM_V1_ID, MOODE_ITEM_C1_ID, MOOD_ITEM_EA2_ID, MOOD_ITEM_V2_ID, MOODE_ITEM_C2_ID]
HIP_COLS = ['Time abs [hh:mm:ss]', 'ActivityClass []', 'BodyPosition []', 'MovementAcceleration [g]']


def load_mood_data(filepath: str) -> pd.DataFrame:
    '''
    Loads the mood data from an excel file

            Parameters:
                    filepath (string): The folderpath of the excel file

            Returns:
                    mood (pandas.DataFrame): Dataframe containing the mood values
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pd.read_excel(filepath, usecols=MOOD_COLS)
    mood = data[data['Form'] == 'EMA_daily_prior_task'].drop(columns=['Form'])
    mood.dropna(inplace=True)
    return mood

def load_preprocessed_data(filepath: str, mood_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Loads the data calculated by the movisens DataAnalyzer from an excel file

            Parameters:
                    filepath (string): The path of the excel file
                    mood_df (pandas.DataFrame): The EMA results corresponding to the data

            Returns:
                    data_processed (pandas.DataFrame): Dataframe containing the feature data
    '''
    # Load the Excel file into a DataFrame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pd.read_excel(filepath)

    # Ensure required columns are present
    for column in ['ActivityClass []', 'BodyPosition []']:
        if column not in data.columns:
            data[column] = 0

    # Filter the necessary columns
    raw_data = data[HIP_COLS]

    # Get the participant info
    participant = mood_df['Participant'].iloc[0]

    # Container to hold the rows for hip data
    rows_to_concat = []

    # Iterate over the mood_df DataFrame
    for _, row in mood_df.iterrows():
        time_end = pd.to_datetime(row['Trigger_time'])
        time_start = time_end - pd.Timedelta(minutes=45)

        # Filter the raw_data based on time window
        mask = (pd.to_datetime(raw_data['Time abs [hh:mm:ss]']) >= time_start) & \
               (pd.to_datetime(raw_data['Time abs [hh:mm:ss]']) <= time_end)
        temp_data = raw_data[mask].copy()  # Create a copy to avoid SettingWithCopyWarning

        # Check for data validity and if valid, add to our list
        if temp_data.shape[0] == 45 and not temp_data.isnull().values.any():
            temp_data.insert(0, 'Participant', participant)
            temp_data = temp_data.rename(columns={
                'Time abs [hh:mm:ss]': 'Time',
                'ActivityClass []': 'Activity_Class',
                'BodyPosition []': 'Body_Position',
                'MovementAcceleration [g]': 'Movement_Intensity'
            })
            rows_to_concat.append(temp_data)
        else:
            mood_df.drop(row.name, inplace=True)

    # Convert the list of DataFrames into a single DataFrame
    processed_data = pd.concat(rows_to_concat, ignore_index=True)

    return processed_data

def extract_rawdata_features(unisens_path, data_temp):
    '''
    Calculates featrues based on the available data

            Parameters:
                    unisens_path (string): The path to the unisens-file
                    data_temp (pandas.DataFrame): The extracted data from the sensor 

            Returns:
                    data_temp (pandas.DataFrame): The calculated features
                    invalid_time_indices (List): List of invalid time indices
    '''
    u = unisens.Unisens(unisens_path)

    # Extracting the data
    temp_data = np.asarray(u['temp.bin'].get_data())[0]
    press_data = np.asarray(u['press.bin'].get_data())[0]

    accelerate_data_x = np.asarray(u['acc.bin'].get_data())[0]
    accelerate_data_y = np.asarray(u['acc.bin'].get_data())[1]
    accelerate_data_z = np.asarray(u['acc.bin'].get_data())[2]

    angularrate_data_x = np.asarray(u['angularrate.bin'].get_data())[0]
    angularrate_data_y = np.asarray(u['angularrate.bin'].get_data())[1]
    angularrate_data_z = np.asarray(u['angularrate.bin'].get_data())[2]

    # Get the starting timestamp
    start_timestamp = pd.Timestamp(u.timestampStart)

    sample_rates = {
        "temp": 1,
        "press": 8,
        "acc": 64,
        "ar": 64
    }

    # Helper for getting indices 
    def get_index(start_timestamp, sample_rate, target_timestamp):
        time_delta = (target_timestamp - start_timestamp).total_seconds()
        index = int(time_delta * sample_rate)
        return index

    all_features = []
    invalid_time_indices = []

    all_feature_set = tsfel.get_features_by_domain()
    desired_features = {
        "statistical": ["Mean", "Standard deviation", "Max", "Min", "Entropy", "Skewness", "Kurtosis", "Absolute energy"],
        "spectral": ["Max power spectrum", "Fundamental frequency"],
        "temporal": ["Neighbourhood peaks", "Zero crossing rate", "Autocorrelation"]
    }
    feature_set = {}
    for domain, features in desired_features.items():
        feature_set[domain] = {feature: all_feature_set[domain][feature] for feature in features}

    for idx, time_str in enumerate(data_temp['Time']):
        features_dict = {}

        target_timestamp = pd.Timestamp(time_str)

        index_temp = get_index(start_timestamp, sample_rates["temp"], target_timestamp)
        if index_temp >= len(temp_data):  # Check if index is out of range
            invalid_time_indices.append(idx)
            continue
        features_dict["Temp"] = temp_data[index_temp]

        index_press = get_index(start_timestamp, sample_rates["press"], target_timestamp)
        if index_press >= len(press_data):  # Check if index is out of range
            invalid_time_indices.append(idx)
            continue
        features_dict["Press"] = np.mean(press_data[index_press - 8:index_press])

        # Extract acc.bin data
        index_acc = get_index(start_timestamp, sample_rates["acc"], target_timestamp)
        if index_acc >= len(accelerate_data_x):  # Check if index is out of range
            invalid_time_indices.append(idx)
            continue
        acc_window_data = {
            "x": accelerate_data_x[index_acc - 64*60:index_acc],
            "y": accelerate_data_y[index_acc - 64*60:index_acc],
            "z": accelerate_data_z[index_acc - 64*60:index_acc]
        }
        acc_window_data["m"] = np.sqrt(np.square(acc_window_data["x"]) +
                                       np.square(acc_window_data["y"]) +
                                       np.square(acc_window_data["z"]))

        # Extract ar.bin data
        index_ar = get_index(start_timestamp, sample_rates["ar"], target_timestamp)
        if index_ar >= len(angularrate_data_x):  # Check if index is out of range
            invalid_time_indices.append(idx)
            continue
        ar_window_data = {
            "x": angularrate_data_x[index_ar - 64*60:index_ar],
            "y": angularrate_data_y[index_ar - 64*60:index_ar],
            "z": angularrate_data_z[index_ar - 64*60:index_ar]
        }
        ar_window_data["m"] = np.sqrt(np.square(ar_window_data["x"]) +
                                      np.square(ar_window_data["y"]) +
                                      np.square(ar_window_data["z"]))

        # Extracting features
        for axis in ["x", "y", "z", "m"]:
            if axis in acc_window_data:
                acc_features = tsfel.time_series_features_extractor(feature_set, acc_window_data[axis],
                                                                    fs=sample_rates["acc"], verbose=0).iloc[0]
                for feature_name, value in acc_features.items():
                    features_dict[f"Acc_{axis.upper()}_{feature_name}"] = value
            if axis in ar_window_data:
                ar_features = tsfel.time_series_features_extractor(feature_set, ar_window_data[axis],
                                                                   fs=sample_rates["ar"], verbose=0).iloc[0]
                for feature_name, value in ar_features.items():
                    features_dict[f"Ar_{axis.upper()}_{feature_name}"] = value

        all_features.append(features_dict)

    features_df = pd.DataFrame(all_features)
    data_temp = pd.concat([data_temp, features_df], axis=1)

    return data_temp, invalid_time_indices


#================================= main function =================================

def load_dataset_1(dataset_path,use_raw_data=True,use_lagged_mood=False,window=15,location='hip',id_creator=None,criteria={}):
    '''
    Extracts data samples from dataset 1

            Parameters:
                    dataset_path (string): Folderpath to the root directory of the dataset
                    use_raw_data: Flag for switching between using raw data or calculated features
                    use_lagged_mood (bool): Flag for enabling the last mood scores to be included in the sample
                    window (int): Desired time window for each sample in minutes 
                    location (string): Desired position of the sensor
                    id_creator (IdCreator): IdCreator object for mapping individual samples to their corresponding subjects 
                    criteria (dict): List of criteria that need to be fulfilled by a subject to be included in the dataset

            Returns:
                    data (numpy.ndarray): The extracted samples from the entire dataset 
                    mood_labels (numpy.ndarray): The mood scores corresponding to the extracted samples
                    id_list (list): The subject ID corresponding to the origin of each extracted sample
    '''
    participants_count = 0
    samples_count = 0
    error_count = 0
    id_list = []

    # Loop participant folders
    for foldername in os.listdir(dataset_path):

        try:
            # Assert foldername to be a participant folder
            if foldername.startswith('P'):

                # Create paths for acc-data and mood data
                mood_path = os.path.join(dataset_path, foldername, 'mood')
                path = os.path.join(dataset_path,foldername)

                # Check criteria for the folder
                if not meets_criteria(path, criteria):
                    logging.info(f'Skipped processing {foldername} due to not meeting the provided critera')
                    continue


                #----- raw data extraction -----
                if use_raw_data == True:
                    
                    # Load acc-data and labels and preprocess mood scores
                    acc_data = process_acc_sensor_raw(path,location)
                    mood_temp = process_mood_folder(mood_path)
                    mood_temp = preprocess_mood(mood_temp)

                    # Create datasamples based on given parameters
                    acc_temp,mood_temp = create_raw_samples(acc_data,mood_temp,lagged_mood=use_lagged_mood,window_size=window)

                    # Check for invalid values
                    if np.isnan(mood_temp.astype(dtype=float)).any() or np.isnan(acc_temp.astype(dtype=float)).any():
                        logging.error(f"Encountered NaN in {foldername}")
                        error_count += 1
                        continue

                    # Append samples to data and mood scores
                    try:
                        mood_labels = np.append(mood_labels,mood_temp,axis=0)
                    except:
                        mood_labels = mood_temp 
                    try:
                        data = np.append(data,acc_temp,axis=0)
                    except:
                        data = acc_temp

                    # Get number of extracted samples
                    number_of_samples = acc_temp.shape[0]

                    # Assign subject id to extracted samples, keep in seperate list
                    id_temp = [id_creator.get_id()] * number_of_samples
                    id_list = id_list + id_temp

                    # Do statistics
                    logging.info(f'Finished processing {foldername}, extracted {number_of_samples} samples of length {acc_temp.shape[1]}')
                    if number_of_samples > 0:
                        participants_count += 1
                        samples_count += number_of_samples


                #----- feature extraction -----
                else:

                    # get all the relevant paths
                    mood_paths = [str(signal) for signal in Path(path).rglob('*.xlsx') if 'mood' in str(signal)]
                    da_results_paths = [signal for signal in Path(path).rglob('*.xlsx') if 'Result' in str(signal) and location in str(signal)]
                    unisens_paths = set([signal for signal in Path(path).rglob('acc.bin') if location in str(signal) and 'Result' not in str(signal)] )

                    mood_path = list(mood_paths)[0]
                    da_result_path = list(da_results_paths)[0]
                    unisens_path = list(unisens_paths)[0].parent

                    # extract all the relevant data
                    mood_temp = load_mood_data(mood_path)
                    location_features = load_preprocessed_data(da_result_path,mood_temp)
                    features_temp, invalid_time_indices = extract_rawdata_features(str(unisens_path), location_features)

                    # check for invalid time indices
                    if invalid_time_indices:                
                        features_temp.drop(invalid_time_indices, inplace=True)
                        mood_temp.drop(mood_temp.index[invalid_time_indices // 45].unique(), inplace=True)

                    features_temp, mood_temp = create_feature_samples(features_temp,mood_temp,window_size=window)


                    # concatenate data from each participant
                    try:
                        mood_labels = np.append(mood_labels,mood_temp,axis=0)
                    except:
                        mood_labels = mood_temp 
                    try:
                        data = np.append(data,features_temp,axis=0)
                    except:
                        data = features_temp

                    number_of_samples = features_temp.shape[0]

                    logging.info(f'Finished processing {foldername}, extracted {number_of_samples} samples of length {features_temp.shape[1]}')
                    if number_of_samples > 0:
                        participants_count += 1
                        samples_count += number_of_samples


        # Exception handling 
        except Exception as e:
            logging.error(f"An error occurred while processing the folder: {foldername}")
            logging.error(f"Error message: {str(e)}")
            error_count += 1

        try:
            data
        except Exception as e:
            logging.error(f"Error in folder {foldername}: No acceleration data present")
            error_count += 1

    logging.info(f"Finished processing {dataset_path}, extracted {samples_count} samples from {participants_count} participants, skipped {error_count} participants")

    return data, mood_labels, id_list