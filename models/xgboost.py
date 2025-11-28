"""XGBoost model architecture"""
import csv
import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split


class XGBmodel:
    """XGBoost architecture for raw acc -> ema"""
    def __init__(self) -> None:
        self.xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,
                                  subsample=0.8, colsample_bytree = 0.7, colsample_bylevel = 0.7,
                                  gamma=0.1, reg_alpha = 0.1, reg_lambda = 1, min_child_weight = 3, 
                                  tree_method = 'hist',early_stopping_rounds=5, eval_metric=["rmse", "mae"])
    

    def train(self, dataset,epochs,split=0.8):
        """Train the XGBoost model"""
        X,y = dataset.get_all()
        X = np.reshape(X,(X.shape[0],-1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-split), random_state=42)

        self.xgb_model.fit(X_train, y_train,eval_set=[(X_test,y_test)],verbose=True) # , eval_metric=["rmse", "mae"], eval_set=[(X_test, y_test)], verbose=50, early_stopping_rounds=epochs)

        # Validate
        logging.info(f"Finished training, validating...")
        y_pred = self.xgb_model.predict(X_test)
        RMSE_all = root_mean_squared_error(y_pred, y_test)
        MAE_all = mean_absolute_error(y_pred, y_test)
        rmse_individual = np.zeros(3)
        mae_individual = np.zeros(3)

        for i in range(3):
            rmse_individual[i] = root_mean_squared_error(y_pred[:, i], y_test[:, i])
            mae_individual[i] = mean_absolute_error(y_pred[:, i], y_test[:, i])

        logging.info("Test RMSE %.4f, test MAE %.4f" % (RMSE_all, MAE_all))
        logging.info("Individual values: RMSE(EA) %.4f, RMSE(V) %.4f, RMSE(C) %.4f, MAE(EA) %.4f, MAE(V) %.4f, MAE(C) %.4f" %
                     (rmse_individual[0], rmse_individual[1], rmse_individual[2], mae_individual[0], mae_individual[1], mae_individual[2]))


    def validate(self, validation_dataset, silent=False):
        """Validate the XGBoost model"""
        X, y = validation_dataset.get_all()
        ids = np.array(validation_dataset.id_list)  # convert to np.array for easy indexing
        X = np.reshape(X, (X.shape[0], -1))
        
        # Initialize results dictionary grouped by ID
        id_results = {}
        for id_val in np.unique(ids):
            id_results[id_val] = {
                'rmse_EA': 0.0,
                'rmse_V': 0.0,
                'rmse_C': 0.0,
                'mae_EA': 0.0,
                'mae_V': 0.0,
                'mae_C': 0.0,
                'count': 0
            }
        
        y_pred = self.xgb_model.predict(X)
        RMSE_all = root_mean_squared_error(y_pred, y)
        MAE_all = mean_absolute_error(y_pred, y)
        rmse_individual = np.zeros(3)
        mae_individual = np.zeros(3)

        # Loop through each unique ID and accumulate squared and absolute errors
        for id_val in np.unique(ids):
            id_idx = np.where(ids == id_val)[0]
            count = len(id_idx)

            # Squared errors for RMSE (accumulate sum of squared errors)
            se_EA = np.square(y_pred[id_idx, 0] - y[id_idx, 0])
            se_V  = np.square(y_pred[id_idx, 1] - y[id_idx, 1])
            se_C  = np.square(y_pred[id_idx, 2] - y[id_idx, 2])

            # Absolute errors for MAE (accumulate sums)
            ae_EA = np.abs(y_pred[id_idx, 0] - y[id_idx, 0])
            ae_V  = np.abs(y_pred[id_idx, 1] - y[id_idx, 1])
            ae_C  = np.abs(y_pred[id_idx, 2] - y[id_idx, 2])

            id_results[id_val]['rmse_EA'] += np.sum(se_EA)
            id_results[id_val]['rmse_V']  += np.sum(se_V)
            id_results[id_val]['rmse_C']  += np.sum(se_C)
            id_results[id_val]['mae_EA']  += np.sum(ae_EA)
            id_results[id_val]['mae_V']   += np.sum(ae_V)
            id_results[id_val]['mae_C']   += np.sum(ae_C)
            id_results[id_val]['count']  += count

        # Calculate per-output overall RMSE and MAE
        for i in range(3):
            rmse_individual[i] = root_mean_squared_error(y_pred[:, i], y[:, i])
            mae_individual[i] = mean_absolute_error(y_pred[:, i], y[:, i])

        overall_results = [RMSE_all, MAE_all, rmse_individual, mae_individual]

        individual_results = []

        # Write per-ID results to CSV, computing final RMSE and MAE averages
        for id_val, result in id_results.items():
            count = result['count']
            if count == 0:
                # Avoid division by zero
                continue
            rmse = np.sqrt((result['rmse_EA'] + result['rmse_V'] + result['rmse_C']) / (3 * count))
            rmse_EA = np.sqrt(result['rmse_EA'] / count)
            rmse_V  = np.sqrt(result['rmse_V']  / count)
            rmse_C  = np.sqrt(result['rmse_C']  / count)
            mae = (result['mae_EA'] + result['mae_V'] + result['mae_C']) / (3 * count)
            mae_EA  = result['mae_EA'] / count
            mae_V   = result['mae_V']  / count
            mae_C   = result['mae_C']  / count
            individual_results.append([id_val, rmse, mae, rmse_EA, rmse_V, rmse_C, mae_EA, mae_V, mae_C, count])
        
        if not silent:
            logging.info("Test RMSE %.4f, test MAE %.4f" % (RMSE_all, MAE_all))
            logging.info("Individual values: RMSE(EA) %.4f, RMSE(V) %.4f, RMSE(C) %.4f, MAE(EA) %.4f, MAE(V) %.4f, MAE(C) %.4f" %
                        (rmse_individual[0], rmse_individual[1], rmse_individual[2], mae_individual[0], mae_individual[1], mae_individual[2]))
        
        return individual_results, overall_results
 