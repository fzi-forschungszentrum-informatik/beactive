"""SVM model architecture"""
import csv
import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR


class SVMModel:
    """SVM architecture for raw acc -> ema"""
    def __init__(self) -> None:
        self.svm_model_EA = LinearSVR(random_state=0, tol=1e-5,verbose=0)
        self.svm_model_V = LinearSVR(random_state=0, tol=1e-5,verbose=0)
        self.svm_model_C = LinearSVR(random_state=0, tol=1e-5,verbose=0)

    def train(self, dataset,epochs=1000,split=0.8):
        """Train the SVM model"""
        X,y = dataset.get_all()
        X = np.reshape(X,(X.shape[0],-1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-split), random_state=42)

        self.svm_model_EA.set_params(max_iter=epochs)
        self.svm_model_V.set_params(max_iter=epochs)
        self.svm_model_C.set_params(max_iter=epochs)

        logging.info(f"Training EA SVM model for {epochs} epochs.")
        self.svm_model_EA.fit(X_train, y_train[:,0])
        logging.info(f"Training V SVM model for {epochs} epochs.")
        self.svm_model_V.fit(X_train, y_train[:,1])
        logging.info(f"Training C SVM model for {epochs} epochs.")
        self.svm_model_C.fit(X_train, y_train[:,2])

        logging.info(f"Finished training, validating...")
        y_pred_EA = self.svm_model_EA.predict(X_test)
        y_pred_V = self.svm_model_V.predict(X_test)
        y_pred_C = self.svm_model_C.predict(X_test)

        test_rmse_EA = root_mean_squared_error(y_pred_EA,y_test[:,0])
        test_rmse_V = root_mean_squared_error(y_pred_V,y_test[:,1])
        test_rmse_C = root_mean_squared_error(y_pred_C,y_test[:,2])

        test_mae_EA = mean_absolute_error(y_pred_EA,y_test[:,0])
        test_mae_V = mean_absolute_error(y_pred_V,y_test[:,1])
        test_mae_C = mean_absolute_error(y_pred_C,y_test[:,2])

        logging.info("SVM performance: RMSE(EA) %.4f, RMSE(V) %.4f, RMSE(C) %.4f, MAE(EA) %.4f, MAE(V) %.4f, MAE(C) %.4f," % ( test_rmse_EA,test_rmse_V,test_rmse_C,test_mae_EA,test_mae_V,test_mae_C))

    def validate(self, validation_dataset, silent=False):
        """Validate the SVM model"""
        X, y = validation_dataset.get_all()
        ids = validation_dataset.id_list  # Assumed to exist
        X = np.reshape(X, (X.shape[0], -1))
        
        # Initialize error accumulators
        sse_EA, sse_V, sse_C = 0.0, 0.0, 0.0  # Sum of squared errors
        sae_EA, sae_V, sae_C = 0.0, 0.0, 0.0  # Sum of absolute errors
        total_count = len(X)
        
        # Grouped results per ID
        id_results = {}
        
        for i in range(total_count):
            id_val = ids[i]
            if id_val not in id_results:
                id_results[id_val] = {
                    'sse_EA': 0.0, 'sse_V': 0.0, 'sse_C': 0.0,
                    'sae_EA': 0.0, 'sae_V': 0.0, 'sae_C': 0.0,
                    'count': 0
                }

            x_input = X[i].reshape(1, -1)
            y_true = y[i]

            y_pred_EA = self.svm_model_EA.predict(x_input)[0]
            y_pred_V = self.svm_model_V.predict(x_input)[0]
            y_pred_C = self.svm_model_C.predict(x_input)[0]

            # Individual errors
            err_EA = y_pred_EA - y_true[0]
            err_V = y_pred_V - y_true[1]
            err_C = y_pred_C - y_true[2]

            # Accumulate global errors
            sse_EA += err_EA**2
            sse_V += err_V**2
            sse_C += err_C**2

            sae_EA += abs(err_EA)
            sae_V += abs(err_V)
            sae_C += abs(err_C)

            # Accumulate per-ID errors
            id_results[id_val]['sse_EA'] += err_EA**2
            id_results[id_val]['sse_V'] += err_V**2
            id_results[id_val]['sse_C'] += err_C**2

            id_results[id_val]['sae_EA'] += abs(err_EA)
            id_results[id_val]['sae_V'] += abs(err_V)
            id_results[id_val]['sae_C'] += abs(err_C)

            id_results[id_val]['count'] += 1

        # Final overall RMSE and MAE
        test_rmse_EA = np.sqrt(sse_EA / total_count)
        test_rmse_V = np.sqrt(sse_V / total_count)
        test_rmse_C = np.sqrt(sse_C / total_count)
        test_rmse = np.sqrt((sse_EA + sse_V + sse_C) / (3 * total_count))

        test_mae_EA = sae_EA / total_count
        test_mae_V = sae_V / total_count
        test_mae_C = sae_C / total_count
        test_mae = (sae_EA + sae_V + sae_C) / (3 * total_count)

        overall_results = [test_rmse, test_mae, [test_rmse_EA, test_rmse_V, test_rmse_C], [test_mae_EA, test_mae_V, test_mae_C]]

        individual_results = []

        # calculate per-ID results
        for id_val, result in id_results.items():
            c = result['count']
            if c == 0:
                continue
            rmse = np.sqrt((result['sse_EA'] + result['sse_V'] + result['sse_C']) / (3 * c))
            rmse_EA = np.sqrt(result['sse_EA'] / c)
            rmse_V = np.sqrt(result['sse_V'] / c)
            rmse_C = np.sqrt(result['sse_C'] / c)
            mae = (result['sae_EA'] + result['sae_V'] + result['sae_C']) / (3 * c)
            mae_EA = result['sae_EA'] / c
            mae_V = result['sae_V'] / c
            mae_C = result['sae_C'] / c
            individual_results.append([id_val, rmse, mae, rmse_EA, rmse_V, rmse_C, mae_EA, mae_V, mae_C, c])

        if not silent:
            logging.info("Test RMSE %.4f, test MAE %.4f" % ( test_rmse, test_mae))
            logging.info("Individual values: RMSE(EA) %.4f, RMSE(V) %.4f, RMSE(C) %.4f, MAE(EA) %.4f, MAE(V) %.4f, MAE(C) %.4f," % (test_rmse_EA, test_rmse_V, test_rmse_C, test_mae_EA, test_mae_V, test_mae_C))

        return individual_results, overall_results
    