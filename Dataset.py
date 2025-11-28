"""Module for handling datasets using PyTorch."""
import logging

import torch
import torch.utils.data as data

logging.basicConfig(level=logging.INFO)


class Dataset:
    """Class for handling datasets using PyTorch."""

    def __init__(self,data,labels,input_shape,output_shape, id_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.Tensor(data).to(self.device)
        self.labels = torch.Tensor(labels).to(self.device)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.id_list = id_list

    def __getitem__(self,id):
        return self.data[id], self.labels[id]

    def __setitem__(self, id):
        logging.warning("Setting items is not permitted in dataset, please create a new instance to change the data")
        return

    def get_io(self):
        """Get input and output shapes."""
        return self.input_shape, self.output_shape

    def create_dataloader(self, batch_size):
        """Create a DataLoader for the dataset."""
        return data.DataLoader(data.TensorDataset(self.data, self.labels), shuffle=True, batch_size=batch_size)
    
    def create_dataloader_with_ids(self, batch_size):
        """Create a DataLoader for the dataset with the id list."""
        return data.DataLoader(data.TensorDataset(self.data, self.labels, torch.tensor(self.id_list)), shuffle=True, batch_size=batch_size)


    def create_dataloader_test_train(self, batch_size, test_train_ratio):
        """Create DataLoaders for training and testing datasets."""
        train_size = int(self.data.shape[0] * test_train_ratio)
        X_train = self.data[:train_size,:,:]
        X_test = self.data[train_size:,:,:]
        y_train = self.labels[:train_size,:]
        y_test = self.labels[train_size:,:]

        loader_train = data.DataLoader(data.TensorDataset(X_train, y_train),
                                       shuffle=True,
                                       batch_size=batch_size)
        loader_test = data.DataLoader(data.TensorDataset(X_test, y_test),
                                      shuffle=True,
                                      batch_size=batch_size)

        return loader_train, loader_test

    def get_all(self):
        """Get all data and labels as numpy arrays."""
        return self.data.cpu().numpy(), self.labels.cpu().numpy()

    def get_all_with_ids(self):
        """Get all data, labels, and IDs as numpy arrays."""
        return self.data.cpu().numpy(), self.labels.cpu().numpy(), self.id_list
