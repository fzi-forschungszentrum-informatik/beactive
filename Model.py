"Module for managing different machine learning models, including training and validation."
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Dataset import Dataset
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.svm import SVMModel
from models.xgboost import XGBmodel

logging.basicConfig(level=logging.INFO)

class Model:
    """Class to manage different machine learning models, including training and validation."""

    def __init__(self, model_type, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Get model based on available implementations
        if model_type == "lstm":
            self.model = LSTMModel(input_shape, output_shape)
        elif model_type == "gru":
            self.model = GRUModel(input_shape, output_shape)
        elif model_type == "xgb":
            self.model = XGBmodel()
        elif model_type == "svm":
            self.model = SVMModel()
        else:
            raise ValueError("No model implementation available for %s", model_type)

        # Get things needed for training later
        self.performance = {}
        self.model_type = model_type

    def __call__(self, input):
        if input.shape == self.input_shape:
            return self.model(input)
        else:
            logging.error("Input shape does not match expected input shape of model")
            return None

    def train(self,dataset,epochs,split=0.8):
        """Train the model using the provided dataset."""
        if self.model_type == 'lstm':
            self.train_nn(dataset,epochs,split)
        elif self.model_type == 'gru':
            self.train_nn(dataset,epochs,split)
        else:
            self.model.train(dataset,epochs,split)

    def validate(self, validation_dataset):
        """Validate the model using the provided dataset and parameters."""
        if self.model_type == 'lstm':
            return self.validate_nn(validation_dataset)
        elif self.model_type == 'gru':
            return self.validate_nn(validation_dataset)
        else:
            return self.model.validate(validation_dataset)

    def train_nn(self, dataset: Dataset,epochs,split=0.8):
        """Train a neural network model using the provided dataset."""
        self.optimizer = optim.Adam(self.model.parameters())
        loss_fn = nn.MSELoss()

        loader_train, loader_test = dataset.create_dataloader_test_train(batch_size=1,
                                                                         test_train_ratio=split)

        if torch.cuda.is_available():
            self.model.cuda()
            #torch.backends.cudnn.enabled = False
            logging.info("Using cuda library for training")
        else:
            logging.info("Cuda library is not available. Check your installation and hardware")

        for epoch in tqdm(range(epochs), desc ="Training: "):
            self.model.train()
            train_se = 0.0  # Squared error for total RMSE
            train_ae = 0.0  # Absolute error for total MAE
            train_se_individual = np.zeros(3)
            train_ae_individual = np.zeros(3)
            count = 0

            # Training of the model, i.e. perform forward
            # and backward pass with training data, then adjust weights
            for X_batch, y_batch in loader_train:
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate train losses, no need to track the gradients here
                with torch.no_grad():
                    se = ((y_pred - y_batch) ** 2).sum().item()
                    ae = torch.abs(y_pred - y_batch).sum().item()
                    train_se += se
                    train_ae += ae

                    for i in range(3):
                        se_i = ((y_pred[:, i] - y_batch[:, i]) ** 2).sum().item()
                        ae_i = torch.abs(y_pred[:, i] - y_batch[:, i]).sum().item()
                        train_se_individual[i] += se_i
                        train_ae_individual[i] += ae_i

                count += y_batch.size(0)

            train_rmse = math.sqrt(train_se / (count * 3))
            train_mae = train_ae / (count * 3)

            # Testing of the model, i.e. perform forward pass
            # with test data, then calculate losses
            logging.info("Validating...")
            self.model.eval()
            test_se = 0.0
            test_ae = 0.0
            test_se_individual = np.zeros(3)
            test_ae_individual = np.zeros(3)
            count = 0

            with torch.no_grad():
                for X_batch, y_batch in loader_test:
                    if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                        continue

                    y_pred = self.model(X_batch)

                    se = ((y_pred - y_batch) ** 2).sum().item()
                    ae = torch.abs(y_pred - y_batch).sum().item()
                    test_se += se
                    test_ae += ae

                    for i in range(3):
                        se_i = ((y_pred[:, i] - y_batch[:, i]) ** 2).sum().item()
                        ae_i = torch.abs(y_pred[:, i] - y_batch[:, i]).sum().item()
                        test_se_individual[i] += se_i
                        test_ae_individual[i] += ae_i

                    count += y_batch.size(0)

            test_rmse = math.sqrt(test_se / (count * 3))
            test_mae = test_ae / (count * 3)
            test_rmse_individual = np.sqrt(test_se_individual / count)
            test_mae_individual = test_ae_individual / count

            # logging.info(f"Epoch {epoch}: Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f} | Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            logging.info(f"Epoch {epoch}: RMSE(EA): {test_rmse_individual[0]:.4f}, RMSE(V): {test_rmse_individual[1]:.4f}, RMSE(C): {test_rmse_individual[2]:.4f}, "
                        f"MAE(EA): {test_mae_individual[0]:.4f}, MAE(V): {test_mae_individual[1]:.4f}, MAE(C): {test_mae_individual[2]:.4f}")

        self.performance["train_RMSE"] = train_rmse
        self.performance["train_MAE"] = train_mae
        self.performance["test_RMSE"] = test_rmse
        self.performance["test_MAE"] = test_mae


    def get_performance(self):
        """Get the performance metrics of the model."""
        return self.performance

    def validate_nn(self, validation_dataset: Dataset, silent=False):
        """Validate a neural network model using the provided dataset and parameters."""
        loss_fn = nn.MSELoss(reduction='sum')  # Sum of squared errors per batch
        loss2 = nn.L1Loss(reduction='sum')     # Sum of absolute errors per batch
        loader_test = validation_dataset.create_dataloader_with_ids(batch_size=1)

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            pass

        self.model.eval()
        with torch.no_grad():
            count = 0  # Total number of samples

            # Accumulate sum of squared errors and absolute errors for total and per output
            sse_total = 0.0
            sae_total = 0.0
            sse_individual = np.zeros(3)
            sae_individual = np.zeros(3)

            id_results = {}

            for X_batch, y_batch, id_batch in loader_test:
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                id_val = int(id_batch.item())
                if id_val not in id_results:
                    id_results[id_val] = {
                        'sse': 0.0,
                        'sae': 0.0,
                        'sse_individual': np.zeros(3),
                        'sae_individual': np.zeros(3),
                        'count': 0
                    }

                y_pred = self.model(X_batch)
                # Calculate squared and absolute errors for the batch
                sse_batch = loss_fn(y_pred.cpu(), y_batch.cpu()).item()
                sae_batch = loss2(y_pred.cpu(), y_batch.cpu()).item()

                sse_total += sse_batch
                sae_total += sae_batch

                # Per output errors
                for i in range(3):
                    sse_i = nn.MSELoss(reduction='sum')(y_pred.cpu()[:, i], y_batch.cpu()[:, i]).item()
                    sae_i = nn.L1Loss(reduction='sum')(y_pred.cpu()[:, i], y_batch.cpu()[:, i]).item()
                    sse_individual[i] += sse_i
                    sae_individual[i] += sae_i

                    id_results[id_val]['sse_individual'][i] += sse_i
                    id_results[id_val]['sae_individual'][i] += sae_i

                id_results[id_val]['sse'] += sse_batch
                id_results[id_val]['sae'] += sae_batch
                id_results[id_val]['count'] += y_pred.numel() / 3
                count += y_pred.numel()

            # Compute final RMSE and MAE for total and per output
            test_rmse = np.sqrt(sse_total / count)
            test_mae = sae_total / count

            test_rmse_individual = np.sqrt(sse_individual / (count / 3))
            test_mae_individual = sae_individual / (count / 3)

            overall_results = [test_rmse, test_mae, test_rmse_individual, test_mae_individual]
            
            individual_results = []

            for id_val, result in id_results.items():
                c = result['count']
                if c == 0:
                    continue
                rmse_id = np.sqrt(result['sse'] / (3*c))
                mae_id = result['sae'] / (3*c)
                rmse_ind = np.sqrt(result['sse_individual'] / c)
                mae_ind = result['sae_individual'] / c
                individual_results.append([id_val, rmse_id, mae_id, rmse_ind[0], rmse_ind[1], rmse_ind[2], mae_ind[0], mae_ind[1], mae_ind[2], c])

        if not silent:
            # logging.info("Test RMSE %.4f, Test MAE %.4f" % ( test_rmse, test_mae))
            logging.info("Individual values: RMSE(EA) %.4f, RMSE(V) %.4f, RMSE(C) %.4f, MAE(EA) %.4f, MAE(V) %.4f, MAE(C) %.4f," % (test_rmse_individual[0],test_rmse_individual[1], test_rmse_individual[2], test_mae_individual[0],test_mae_individual[1], test_mae_individual[2]))


        return individual_results, overall_results
    