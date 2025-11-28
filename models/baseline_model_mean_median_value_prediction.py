import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

MOOD_COLS=['EA', 'V', 'C']

def find_all_mood_files(dataset_path):
    mood_files = []
    for foldername in os.listdir(dataset_path):
        if foldername.startswith('P'):
            mood_folder = os.path.join(dataset_path, foldername, 'mood')
            mood_files.extend([os.path.join(dataset_path, foldername, mood_folder, f) for f in os.listdir(mood_folder) if f.endswith(".xlsx")])
    return mood_files

def load_participant_mood_data(filepath):
    data = pd.read_excel(filepath, usecols=MOOD_COLS)
    return data

def calc_mean_model_metrics(dataset):

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    ea_train = np.array(train_set[:,0])
    ea_test = np.array(test_set[:,0])
    mean_ea_train = np.mean(ea_train)
    ea_pred = np.full(ea_test.size, mean_ea_train)

    v_train = np.array(train_set[:,1])
    v_test = np.array(test_set[:,1])
    mean_v_train = np.mean(v_train)
    v_pred = np.full(v_test.size, mean_v_train)

    c_train = np.array(train_set[:,2])
    c_test = np.array(test_set[:,2])
    mean_c_train = np.mean(c_train)
    c_pred = np.full(c_test.size, mean_c_train)

    return mean_ea_train.item(), mean_absolute_error(ea_test, ea_pred), root_mean_squared_error(ea_test, ea_pred), \
        mean_v_train.item(), mean_absolute_error(v_test, v_pred), root_mean_squared_error(v_test, v_pred), \
        mean_c_train.item(), mean_absolute_error(c_test, c_pred), root_mean_squared_error(c_test, c_pred)

def calc_median_model_metrics(dataset):

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    ea_train = np.array(train_set[:,0])
    ea_test = np.array(test_set[:,0])
    median_ea_train = np.median(ea_train)
    ea_pred = np.full(ea_test.size, median_ea_train)

    v_train = np.array(train_set[:,1])
    v_test = np.array(test_set[:,1])
    median_v_train = np.median(v_train)
    v_pred = np.full(v_test.size, median_v_train)

    c_train = np.array(train_set[:,2])
    c_test = np.array(test_set[:,2])
    median_c_train = np.median(c_train)
    c_pred = np.full(c_test.size, median_c_train)

    return median_ea_train.item(), mean_absolute_error(ea_test, ea_pred), root_mean_squared_error(ea_test, ea_pred), \
        median_v_train.item(), mean_absolute_error(v_test, v_pred), root_mean_squared_error(v_test, v_pred), \
        median_c_train.item(), mean_absolute_error(c_test, c_pred), root_mean_squared_error(c_test, c_pred)

if __name__ == '__main__':
    path = os.path.abspath('sample_data')

    mood_files = find_all_mood_files(path)
    for mood_file in mood_files:
        extracted_mood_data = load_participant_mood_data(mood_file)
        try:
            mood_data = np.append(mood_data, extracted_mood_data, axis=0)
        except:
            mood_data = extracted_mood_data
    
    mean_model_metrics = calc_mean_model_metrics(mood_data)

    median_model_metrics = calc_median_model_metrics(mood_data)


    print(f'Number of samples per dimension: {mood_data.shape[0]}')
    print()
    print('Mean prediction model performance:')
    print(f'EA | MAE: {round(mean_model_metrics[1], 2)} | RMSE: {round(mean_model_metrics[2], 2)}')
    print(f'V  | MAE: {round(mean_model_metrics[4], 2)} | RMSE: {round(mean_model_metrics[5], 2)}')
    print(f'C  | MAE: {round(mean_model_metrics[7], 2)} | RMSE: {round(mean_model_metrics[8], 2)}')

    print()
    print('Median prediction model performance:')
    print(f'EA | MAE: {round(median_model_metrics[1], 2)} | RMSE: {round(median_model_metrics[2], 2)}')
    print(f'V  | MAE: {round(median_model_metrics[4], 2)} | RMSE: {round(median_model_metrics[5], 2)}')
    print(f'C  | MAE: {round(median_model_metrics[7], 2)} | RMSE: {round(median_model_metrics[8], 2)}')