"""Utility functions for processing sensor and mood data."""
import logging
import os
import csv
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import unisens
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MOOD_ITEM_EA_ID = 'Mood_EA'
MOOD_ITEM_V_ID = 'Mood_V'
MOOD_ITEM_C_ID = 'Mood_C'

MOOD_ITEM_EA1_ID = 'Mood1_EA1'
MOOD_ITEM_V1_ID = 'Mood2_V1'
MOODE_ITEM_C1_ID = 'Mood3_C1'
MOOD_ITEM_EA2_ID = 'Mood4_EA2'
MOOD_ITEM_V2_ID = 'Mood5_V2'
MOODE_ITEM_C2_ID = 'Mood6_C2'

def process_acc_sensor_raw(participant_path, location):
    """ Process raw accelerometer data from unisens folder structure. """
    participant_path = Path(participant_path)
    locations = set([signal for signal in Path(participant_path).rglob('acc.bin') if location in str(signal)])
    location_path = str(list(locations)[-1].parent)
    u = unisens.Unisens(location_path)
    signal = u['acc.bin']
    try:
        acc_data = np.asarray(signal.get_data()).transpose()
    except:
        logging.error(f'Error occured while getting raw data from folder {participant_path} at location {location}')   

    return acc_data


def recode_mood_items(mood_data):
    mood_data[MOOD_ITEM_EA2_ID] = 100 - mood_data[MOOD_ITEM_EA2_ID]
    mood_data[MOOD_ITEM_V1_ID] = 100 - mood_data[MOOD_ITEM_V1_ID]
    mood_data[MOODE_ITEM_C2_ID] = 100 - mood_data[MOODE_ITEM_C2_ID]

    return mood_data

def combine_mood_items(mood_data):
    mood_data[MOOD_ITEM_EA_ID] = (mood_data[MOOD_ITEM_EA1_ID] + mood_data[MOOD_ITEM_EA2_ID]) / 2
    mood_data[MOOD_ITEM_V_ID] = (mood_data[MOOD_ITEM_V1_ID] + mood_data[MOOD_ITEM_V2_ID]) / 2
    mood_data[MOOD_ITEM_C_ID] = (mood_data[MOODE_ITEM_C1_ID] + mood_data[MOODE_ITEM_C2_ID]) / 2

    return mood_data


def process_mood_folder(mood_folder):
    """ Process mood folder to extract mood data. """
    mood_files = [f for f in os.listdir(mood_folder) if f.endswith(".xlsx")]
    if len(mood_files) < 1:
        logging.error(f"Did not find a xlsx-file in folder {mood_folder}")

    if platform.platform(terse=True) == 'Windows-10':
        file_path = os.path.join(mood_folder, mood_files[-1]) # use this for win10
    else:
        file_path = os.path.join(mood_folder, mood_files[0]) # use this for win11
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mood_data = pd.read_excel(file_path)

    mood_data = mood_data[['Participant', 'Trigger_date', 'Trigger_time', MOOD_ITEM_EA1_ID, MOOD_ITEM_V1_ID, MOODE_ITEM_C1_ID, MOOD_ITEM_EA2_ID, MOOD_ITEM_V2_ID, MOODE_ITEM_C2_ID]]

    mood_data = mood_data.dropna(subset=[MOOD_ITEM_EA1_ID])

    mood_data = recode_mood_items(mood_data)

    return mood_data


def preprocess_mood(mood_data):
    """ Preprocess mood data for studies. """
    mood_data = combine_mood_items(mood_data)

    mood_data.reset_index(drop=True, inplace=True)

    mood_data['Timestamp'] = pd.to_datetime(mood_data['Trigger_date'], format='%Y-%m-%d  %H:%M:%S')
    mood_data['MinTimestamp'] = mood_data.groupby('Participant')['Timestamp'].transform('min')
    mood_data['RelativeTime'] = (mood_data['Timestamp'] - mood_data['MinTimestamp']).dt.total_seconds()

    mood_data['AccIndex'] = mood_data['RelativeTime'].apply(lambda x: x*64)

    return mood_data


def create_feature_samples(features,mood_data,lagged_mood=False,window_size=20):
    """ Create feature samples for model training. """
    # Recoding mood scores
    mood_data[MOOD_ITEM_EA_ID] = ((100 - mood_data[MOOD_ITEM_EA2_ID]) + mood_data[MOOD_ITEM_EA1_ID]) / 2
    mood_data[MOOD_ITEM_V_ID] = ((100 - mood_data[MOOD_ITEM_V2_ID]) + mood_data[MOOD_ITEM_V1_ID]) / 2
    mood_data[MOOD_ITEM_C_ID] = ((100 - mood_data[MOODE_ITEM_C2_ID]) + mood_data[MOODE_ITEM_C1_ID]) / 2

    mood_data = recode_mood_items(mood_data)
    mood_data = combine_mood_items(mood_data)

    # Classification for Activity Intensity
    features['Movement_Intensity'] = features['Movement_Intensity'] * 1000
    features.loc[features['Temp'] > 50, 'Temp'] = 30.0

    if lagged_mood == True:
        features['lag_EA'] = np.nan
        features['lag_V'] = np.nan
        features['lag_C'] = np.nan

    group_data = mood_data.groupby('Participant')

    for name, group in group_data:
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            
            if i == 0:
                # For the first value
                prev_mood_EA = group[MOOD_ITEM_EA_ID].head(min(5, len(group))).mean()
                prev_mood_V = group[MOOD_ITEM_V_ID].head(min(5, len(group))).mean()
                prev_mood_C = group[MOOD_ITEM_C_ID].head(min(5, len(group))).mean()
            else:
                prev_mood_EA = group[MOOD_ITEM_EA_ID].iloc[i-1]
                prev_mood_V = group[MOOD_ITEM_V_ID].iloc[i-1]
                prev_mood_C = group[MOOD_ITEM_C_ID].iloc[i-1]
            
            interpolated_EA = np.round(np.linspace(prev_mood_EA, row[MOOD_ITEM_EA_ID], 30), 1)
            interpolated_V = np.round(np.linspace(prev_mood_V, row[MOOD_ITEM_V_ID], 30), 1)
            interpolated_C = np.round(np.linspace(prev_mood_C, row[MOOD_ITEM_C_ID], 30), 1)

            if lagged_mood == True:
                idx = features[features['Participant'] == name].index[i*30:(i+1)*30]
                features.loc[idx, 'lag_EA'] = interpolated_EA
                features.loc[idx, 'lag_V'] = interpolated_V
                features.loc[idx, 'lag_C'] = interpolated_C


    # Preprocessing
    num_samples = len(mood_data)
    num_features = features.shape[1] - 4 
    X = np.zeros((num_samples, window_size, num_features))
    y = mood_data[[MOOD_ITEM_EA_ID, MOOD_ITEM_V_ID, MOOD_ITEM_C_ID]].values

    for i in range(num_samples-1):
        start = i*window_size
        end = (i+1)*window_size
        sample = features.shape
        last_in = (num_samples) * window_size
        X[i] = features.iloc[i*window_size:(i+1)*window_size, 4:].values

    # Features Reduction
    pca = PCA(n_components=30) 

    # Reshape the data to 2D, fit PCA and then transform
    X_2d = X.reshape(-1, num_features)
    X_pca_2d = pca.fit_transform(X_2d)

    # Reshape the transformed data back to 3D
    X_pca = X_pca_2d.reshape(num_samples, window_size, 30)

    # Normalization
    scaler = StandardScaler()
    X_pca_2d_scaled = scaler.fit_transform(X_pca_2d)
    X_pca_scaled = X_pca_2d_scaled.reshape(num_samples, window_size, 30)

    return X_pca_scaled, y


def create_raw_samples(acc_data, mood, lagged_mood=False, window_size=20):
    """ Create raw samples for model training. """
    sequence_length = 64*60*window_size

    labels = np.zeros([1,3])
    labels_with_last = np.zeros([1,6])
    if lagged_mood == True:
        data = np.zeros([1,sequence_length,6])
    else:
        data = np.zeros([1,sequence_length,3])
    labels_with_last_step = np.zeros((1,6))

    # Recoding mood scores
    mood[MOOD_ITEM_EA_ID] = ((100 - mood[MOOD_ITEM_EA2_ID]) + mood[MOOD_ITEM_EA1_ID]) / 2
    mood[MOOD_ITEM_V_ID] = ((100 - mood[MOOD_ITEM_V2_ID]) + mood[MOOD_ITEM_V1_ID]) / 2
    mood[MOOD_ITEM_C_ID] = ((100 - mood[MOODE_ITEM_C2_ID]) + mood[MOODE_ITEM_C1_ID]) / 2

    for idx, timestamp in enumerate(mood['AccIndex']):

            if timestamp > 0:
                timestamp = int(timestamp)

                mood_data = mood.iloc[idx][[MOOD_ITEM_EA_ID, MOOD_ITEM_V_ID, MOOD_ITEM_C_ID]].to_numpy()
                mood_data_last = mood.iloc[idx-1][[MOOD_ITEM_EA_ID, MOOD_ITEM_V_ID, MOOD_ITEM_C_ID]].to_numpy()
                acc_step = acc_data[timestamp-sequence_length:timestamp]

                try:
                    mood_data = mood_data.reshape((1,3))
                    mood_data_last = mood_data_last.reshape((1,3))
                    acc_step = acc_step.reshape((1,sequence_length,3))

                    if lagged_mood == True:
                        mood_data_last = np.full([1,sequence_length,3],mood_data_last, dtype=float)
                        acc_step = np.append(acc_step,mood_data_last,axis=2)

                except ValueError:
                    continue

                if np.isnan(labels.astype(dtype=float)).any() or np.isnan(data.astype(dtype=float)).any():
                    break

                labels = np.append(labels, mood_data, axis=0)
                data = np.append(data, acc_step,axis=0)

    labels = labels.astype(dtype=float)

    return data[1:], labels[1:]

def write_per_id_metrics_to_csv(individual_results, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'RMSE', 'MAE', 'RMSE_EA', 'RMSE_V', 
                            'RMSE_C', 'MAE_EA', 'MAE_V', 'MAE_C', 'count'])
        for ir in individual_results:
            writer.writerow([ir[0], ir[1], ir[2], ir[3], ir[4], ir[5], ir[6], ir[7], ir[8], ir[9]])

def write_overall_metrics_to_csv(overall_results, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RMSE', 'MAE', 'RMSE_EA', 'RMSE_V', 'RMSE_C', 'MAE_EA', 'MAE_V', 'MAE_C'])
        writer.writerow([overall_results[0], overall_results[1], overall_results[2][0], overall_results[2][1], overall_results[2][2], overall_results[3][0], overall_results[3][1], overall_results[3][2]])