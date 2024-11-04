import glob
import logging

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.nn import CrossEntropyLoss, Linear, Sequential, Tanh
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from utils import load_model_weights

logging.basicConfig(
    level=logging.INFO,
    filename="MLP.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


class SklearnMLPWrapper(BaseEstimator):
    def __init__(
        self,
        model_option,
        num_of_MLP,
        hidden_units,
        hidden_size,
        output_size,
        activation_function,
        epochs,
        loss_fn,
        optimizer_name,
        adaptive_lr,
        lr,
    ):
        """
        A wrapper class for multi-layer perceptron (MLP) models that extends Scikit-Learn's BaseEstimator. This class
        allows for the integration of single or multiple MLP models, with options for parallel concatenation
        or a basic ensemble approach. The model can be configured with various parameters for fine-tuning.

        Parameters:
            model_option (str): Specifies the type of MLP model ('singleMLP', 'parallelMMLP', or 'basicMMLP').
            num_of_MLP (int): Number of MLP models used when model_option is set to 'parallelMMLP' or 'basicMMLP'.
            hidden_units (list): List of integers specifying the number of hidden units in each hidden layer.
            hidden_size (int): Number of hidden layers in the MLP.
            output_size (int): Size of the output layer.
            activation_function (torch.nn.Module): Activation function used in the hidden layers.
            epochs (int): Number of training epochs.
            loss_fn (callable): Loss function for training the model.
            optimizer_name (str): Optimizer type ('adam' or 'sgd') for training.
            adaptive_lr (bool): Flag indicating whether to use adaptive learning rate.
            lr (float): Learning rate for the optimizer.
        """

        self.model_option = model_option
        self.num_of_MLP = num_of_MLP
        self.lr = lr
        self.hidden_units = hidden_units
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.adaptive_lr = adaptive_lr
        self.loss_fn = loss_fn  # lub inna funkcja strat, w zależności od problemu
        self.selected_features_ = None
        self.model = None

    def fit(self, X, y, selected_features=None):
        """
        Fits the model to the provided data.

        Args:
            X (numpy.ndarray): Input data matrix (features).
            y (numpy.ndarray): Target data vector (labels).
            selected_features (list, optional): List of selected feature indices. If specified, the model will use only these features.
        """
        input_size = X.shape[1]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if selected_features is not None:
            self.selected_features_ = selected_features
            X = X[:, self.selected_features_]
        if self.model_option == "singleMLP":
            print(f"Single MLP")
            self.model = MLP(
                input_size=input_size,
                hidden_units=self.hidden_units,
                hidden_layers=self.hidden_size,
                output_size=self.output_size,
                activation_function=self.activation_function,
                is_MMLP=False,
            )
        elif self.model_option == "parallelMMLP":
            print(f"parallelMMLP")
            models_list = []
            for _ in range(self.num_of_MLP):
                models_list.append(
                    MLP(
                        input_size=input_size,
                        hidden_units=self.hidden_units,
                        hidden_layers=self.hidden_size,
                        output_size=self.output_size,
                        activation_function=self.activation_function,
                        is_MMLP=False,
                    )
                )
            self.model = Parallel_Concatenation_MMLP(
                models_list, self.num_of_MLP, self.output_size
            )
        elif self.model_option == "basicMMLP":
            print(f"basicMMLP")
            models_list = []
            for _ in range(self.num_of_MLP):
                models_list.append(
                    MLP(
                        input_size=input_size,
                        hidden_units=self.hidden_units,
                        hidden_layers=self.hidden_size,
                        output_size=self.output_size,
                        activation_function=self.activation_function,
                        is_MMLP=False,
                    )
                )
            self.model = BasicMMLP(models_list)
        else:
            print("Error")
        if self.optimizer_name.lower() == "adam":
            print(f" Model in opimizer {self.model}")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if self.adaptive_lr:
            scheduler = ExponentialLR(optimizer, gamma=0.9)

        print(f"MODEL {self.model}")
        for epoch in range(int(self.epochs)):

            if 3 in torch.unique(y_tensor):
                y_tensor = y_tensor - 1
            self.model.train()
            if (y_tensor < 0).any():  # Check for any negative labels
                print("Warning: Found negative labels in y_batch!")
            x, y = X_tensor.to(torch.float32), y_tensor.to(torch.long)
            y_pred = self.model(x).squeeze()
            y = y.squeeze()
            loss = self.loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.adaptive_lr:
                scheduler.step()

    def predict(self, X):
        """
        Predicts the labels for the provided input data.

        Args:
            X (numpy.ndarray): Input data matrix (features).

        Returns:
            numpy.ndarray: Predicted labels.
        """
        if self.selected_features_ is not None:
            X = X[:, self.selected_features_]
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def score(self, X, y):
        """
        Calculates the accuracy of the model on the provided dataset.

        Args:
            X (numpy.ndarray): Input data matrix (features).
            y (numpy.ndarray): True labels for evaluation.

        Returns:
            float: Mean accuracy of the model on the provided dataset.
        """
        predictions = self.predict(X)
        return (predictions == y).mean()

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (numpy.ndarray): Input data matrix (features).

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        if self.selected_features_ is not None:
            X = X[:, self.selected_features_]
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(torch.tensor(X, dtype=torch.float32))
        return probabilities.numpy()

    def transform(self, X):
        """
        Transforms the input data using the selected features.

        Args:
            X (numpy.ndarray): Input data matrix (features).

        Returns:
            numpy.ndarray: Transformed data with only the selected features.

        Raises:
            RuntimeError: If the model has not been fitted before calling this function.
        """
        if self.selected_features_ is None:
            raise RuntimeError("You must fit the model before calling transform.")

        return X[:, self.selected_features_]

    @property
    def coef_(self):
        """
        Extracts feature importances from the trained model based on the absolute values of the first layer weights.

        Returns:
            numpy.ndarray: Feature importances derived from the model weights.
        """
        if self.model_option == "singleMLP":
            first_layer_weights = self.model.layers[0].weight.detach().numpy()
            feature_importances = np.mean(np.abs(first_layer_weights), axis=0)
        elif self.model_option == "parallelMMLP" or self.model_option == "basicMMLP":
            print("Model coef", self.model.MLP_list)
            feature_importances_model = []
            for model in self.model.MLP_list:
                first_layer_weights = model.layers[0].weight.detach().numpy()
                feature_importances_model.append(first_layer_weights)
            feature_importances = np.mean(feature_importances_model, axis=0)
        return feature_importances


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.

    Parameters:
        input_size (int): The number of input features.
        hidden_layers (int): Number of hidden layers in the model.
        hidden_units (list): List containing the number of units in each hidden layer.
        output_size (int): Number of output units.
        activation_function (torch.nn.Module): Activation function applied after each hidden layer.
        is_MMLP (bool): Determines if the model is part of an MMLP structure, adjusting the output layer accordingly.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: int,
        hidden_units: list,
        output_size: int,
        activation_function: torch.nn.Module,
        is_MMLP: bool,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Linear(input_size, hidden_units[0]))
        self.layers.append(activation_function)
        if hidden_layers > 1:
            for num_layer in range(hidden_layers - 1):
                self.layers.append(
                    Linear(hidden_units[num_layer], hidden_units[num_layer + 1])
                )
                self.layers.append(activation_function)
        if is_MMLP == False:
            self.layers.append(Linear(hidden_units[-1], output_size))
        self.enc_red = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Performs a forward pass through the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """

        return self.enc_red(x)


class BasicMMLP(nn.Module):

    def __init__(self, models_list: list):
        """
        Basic MMLP model that consists of multiple MLP models averaged at the output layer.

        Parameters:
            models_list (list): List of MLP models to be used in the MMLP.
        """
        super(BasicMMLP, self).__init__()
        self.MLP_list = nn.ModuleList(models_list)
        self.outputs = []
        self.sum = None

    def forward(self, x):
        """
        Performs a forward pass, averaging outputs from all MLP models in the list.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Averaged output tensor from all MLP models.
        """
        self.outputs.clear()

        for idx, model in enumerate(self.MLP_list):
            self.outputs.append(model(x))
        self.sum = torch.zeros_like(self.outputs[0])
        for output in self.outputs:
            self.sum += output
        self.sum = sum(self.outputs)  # Sumuj wyjścia
        self.avg_output = self.sum / len(self.outputs)
        logging.info(f"avg_output concatenating {self.avg_output.shape}")
        return self.avg_output


class Parallel_Concatenation_MMLP(nn.Module):

    def __init__(self, models_list: list, size: int, output_size: int):
        """
        MMLP model that concatenates outputs from multiple MLP models.

        Parameters:
            models_list (list): List of MLP models to be concatenated.
            size (int): Number of MLP models.
            output_size (int): Size of the output layer for each model.
        """
        super(Parallel_Concatenation_MMLP, self).__init__()
        self.MLP_list = nn.ModuleList(models_list)
        self.outputs = []
        self.size = size
        self.output_size = output_size
        print("Params", self.size, self.output_size)
        self.output = Linear(
            int(self.output_size) * int(self.size), int(self.output_size)
        )

    def forward(self, x):
        """
        Concatenates outputs from each MLP model in the list and applies a final transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated and transformed output tensor.
        """
        self.outputs = []
        logging.info(
            f"Params input size {self.output_size* self.size}, num of MLP: {len(self.MLP_list)}"
        )
        for idx, model in enumerate(self.MLP_list):
            logging.info(
                f"Model MLP {idx} architecture {model}, output shape: {model(x).shape}"
            )
            self.outputs.append(model(x))
        logging.info(
            f"Shapes from MLP {[output.shape for output in self.outputs]}",
        )
        concatenated_output = torch.cat(self.outputs, dim=1)
        logging.info(f"Shape after concatenating {concatenated_output.shape}")
        return self.output(concatenated_output)
