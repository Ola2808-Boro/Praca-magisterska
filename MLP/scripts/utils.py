import json
import logging
import os
import shutil
import sys

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
import torch
from lime.lime_text import LimeTextExplainer
from pydicom import dcmread
from scipy.ndimage import convolve
from setup_data import (
    balancing_dataset,
    convert_to_tensors,
    create_dataloder,
    create_dataset,
    preprocessing_data,
    read_csv_data,
    select_features,
)
from sklearn.linear_model import Lasso
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)
from torchviz import make_dot

logging.basicConfig(
    level=logging.INFO,
    filename="MLP.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

path_images = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/images/"
path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"

columns_names = [
    "Wiek",
    "Płeć",
    "Na leku tyroks.",
    "Zap. o tyroksynę",
    "Na lekach przeciw.",
    "Chory",
    "W ciąży",
    "Operacja tarczycy",
    "Leczenie I131",
    "Zap. o niedoczy. tarczycy",
    "Zap. o nadczy. tarczycy",
    "Lit",
    "Wole",
    "Guz",
    "Niedoczy. przys.",
    "Psych",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "Klasa",
]


class Sin(nn.Module):
    """
    Custom sinusoidal activation function for a neural network layer.

    Inherits from:
    - nn.Module: Base class for all neural network modules in PyTorch.

    Methods:
    - forward(x): Applies the sine function element-wise to the input tensor x.

    Parameters:
    - x (torch.Tensor): Input tensor to which the sine function is applied.

    Returns:
    - torch.Tensor: Output tensor with the sine function applied element-wise.
    """

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Logarithmic(nn.Module):
    """
    Custom logarithmic activation function for a neural network layer.

    Inherits from:
    - nn.Module: Base class for all neural network modules in PyTorch.

    Methods:
    - forward(x): Applies a modified logarithmic function to the input tensor x.

    Parameters:
    - x (torch.Tensor): Input tensor to which the logarithmic function is applied.

    Returns:
    - torch.Tensor: Output tensor where the logarithmic transformation is applied
      element-wise, handling positive and negative values differently.
    """

    def __init__(self):
        super(Logarithmic, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, torch.log(x + 1), -torch.log(-x + 1))


class Neural(nn.Module):
    """
    Custom neural-based activation function that combines sigmoid and sine.

    Inherits from:
    - nn.Module: Base class for all neural network modules in PyTorch.

    Methods:
    - forward(x): Applies a neural transformation using sine and sigmoid.

    Parameters:
    - x (torch.Tensor): Input tensor to apply the transformation.

    Returns:
    - torch.Tensor: Output tensor after the transformation.
    """

    def __init__(self):
        super(Neural, self).__init__()

    def forward(self, x):
        return 1 / 1 + torch.exp(-torch.sin(x))


def LIME_explainer(model, df, path, idx):
    """
    Generates LIME explanations for a given model and dataset.

    Parameters:
    - model (nn.Module): Trained model to generate explanations for.
    - df (tuple of DataFrames): Tuple with training and test DataFrames.
    - path (str): Directory path to save the LIME explanation plots.
    - idx (int or list): Index/indices of the test samples to explain.

    Returns:
    - None: Saves LIME explanation plots to the specified path.
    """

    def predict_fn(X):
        model.eval()
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_tensor)
            y_pred_softmax = torch.softmax(y_pred, dim=1)
            predicted = torch.argmax(y_pred_softmax, dim=1)
        return y_pred_softmax.numpy()

    train_data, test_data = df[0], df[1]
    logging.info(f"Zbiór treningowy w LIME: {train_data.columns}")
    X_train = train_data.values[:, :-1]
    X_test = test_data.values[:, :-1]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=[f"{columns_names[i]}" for i in range(X_train.shape[1])],
        class_names=["Klasa 0", "Klasa 1", "Klasa 2"],
        discretize_continuous=True,
    )

    if isinstance(idx, list):
        for index in idx:
            if index >= len(test_data):
                continue
            logging.info(
                "Wartości cech w zbiorze testowym:", test_data.iloc[index].values
            )
            explanation = explainer.explain_instance(X_test[index], predict_fn)

            # Wyjaśnienie dla Klasy 0
            if 0 in explanation.local_exp.keys():
                fig1 = explanation.as_pyplot_figure(label=0)
                fig1.suptitle("")  # Usunięcie domyślnego tytułu
                plt.title(f"Wyjaśnienie dla Klasy 0 - Indeks {index}", fontsize=16)
                fig1.subplots_adjust(left=0.3, right=0.99)
                plt.xlabel("Wartość cechy", fontsize=12)
                plt.ylabel("Wpływ na predykcję", fontsize=12)
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=9)
                fig1.savefig(f"{path}/LIME_{index}_klasa0.jpg")

            # Wyjaśnienie dla Klasy 1
            if 1 in explanation.local_exp.keys():
                fig2 = explanation.as_pyplot_figure(label=1)
                fig2.suptitle("")  # Usunięcie domyślnego tytułu
                plt.title(f"Wyjaśnienie dla Klasy 1 - Indeks {index}", fontsize=16)
                fig2.subplots_adjust(left=0.3, right=0.99)
                plt.xlabel("Wartość cechy", fontsize=12)
                plt.ylabel("Wpływ na predykcję", fontsize=12)
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=9)
                fig2.savefig(f"{path}/LIME_{index}_klasa1.jpg")

            # Wyjaśnienie dla Klasy 2
            if 2 in explanation.local_exp.keys():
                fig3 = explanation.as_pyplot_figure(label=2)
                fig3.suptitle("")  # Usunięcie domyślnego tytułu
                plt.title(f"Wyjaśnienie dla Klasy 2 - Indeks {index}", fontsize=16)
                fig3.subplots_adjust(left=0.3, right=0.99)
                plt.xlabel("Wartość cechy", fontsize=12)
                plt.ylabel("Wpływ na predykcję", fontsize=12)
                plt.xticks(fontsize=56)
                plt.yticks(fontsize=9)
                fig3.savefig(f"{path}/LIME_{index}_klasa2.jpg")


def SHAP_explainer(
    model,
    df,
    path,
    idx=None,
    dependence_feature=None,
):
    """
    Generates SHAP explanations for a given model and dataset.

    Parameters:
    - model (nn.Module): Trained model to generate explanations for.
    - df (tuple of DataFrames): Tuple with training and test DataFrames.
    - path (str): Directory path to save SHAP summary and individual plots.
    - idx (int or list, optional): Index/indices of test samples for local explanations.
    - dependence_feature (str, optional): Feature for dependence plots (if applicable).

    Returns:
    - None: Saves SHAP summary and local explanation plots to the specified path.
    """
    model.eval()
    train_data, test_data = df[0], df[1]
    X_train = torch.tensor(train_data.values[:, :-1]).to(torch.float32)
    X_test = torch.tensor(train_data.values[:100, :-1]).to(torch.float32)
    # global
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    fig = shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"{path}/SHAP_summary.png")
    plt.close()
    # local
    if idx:
        shap.initjs()
        if isinstance(idx, list):
            for index in idx:
                if index >= len(X_test):
                    logging.info(f"Index {index} is out of bounds.")
                    continue
                num_classes = len(shap_values)
                logging.info(f"Number of classes: {num_classes}")

                for class_idx in range(num_classes):
                    shap_value = shap_values[class_idx][index].astype(float)
                    expected_value = float(explainer.expected_value[class_idx])
                    instance = X_test[index].numpy().astype(float)
                    fig = shap.force_plot(
                        expected_value,
                        shap_value,
                        instance,
                        show=False,
                    )
                    shap.save_html(f"{path}/SHAP_{index}_class_{class_idx}.html", fig)


def define_activation_function(
    activation_function_name: str, input_size: int
) -> nn.Module:
    """
    Defines an activation function based on a given name.

    Parameters:
    - activation_function_name (str): Name of the activation function ('sigmoid', 'neural', etc.).
    - input_size (int): Input dimension for any customized activation functions.

    Returns:
    - nn.Module: PyTorch module with the selected activation function.
    """
    logging.info(f"Passed activation function {activation_function_name.lower()}")
    if activation_function_name.lower() == "sigmoid":  # sigmoidalna
        activation_function = nn.Sigmoid()
    elif activation_function_name.lower() == "neural":  # neuronalna
        activation_function = Neural()
    elif activation_function_name.lower() == "exponential":  # wykładnicza
        activation_function = nn.ELU()
    elif activation_function_name.lower() == "log":  # logarytmiczna
        activation_function = Logarithmic()
    elif activation_function_name.lower() == "sin":  # sinusoidalna
        activation_function = Sin()
    elif activation_function_name.lower() == "relu":  # sinusoidalna
        activation_function = nn.ReLU()
    else:
        logging.warning("Default activation function")
        activation_function = nn.Tanh()
    logging.info(f"Returned activation function {activation_function}")
    return activation_function


def define_scheduler(scheduler_name: str, optimizer: torch.optim) -> nn.Module:
    """
    Defines a learning rate scheduler based on the given name.

    Parameters:
    - scheduler_name (str): Name of the scheduler ('lambdalr', 'steplr', etc.).
    - optimizer (torch.optim.Optimizer): Optimizer for which the scheduler will adjust the learning rate.

    Returns:
    - nn.Module: Scheduler module that adjusts learning rate according to the specified strategy.
    """
    logging.info(f"Passed scheduler {scheduler_name.lower()}")
    if scheduler_name.lower() == "lambdalr":
        lambda1 = lambda epoch: 0.95**epoch
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
    elif scheduler_name.lower() == "steplr":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name.lower() == "constantlr":
        scheduler = ConstantLR(optimizer=optimizer, factor=0.5)
    elif scheduler_name.lower() == "exponential":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
    elif scheduler_name.lower() == "chainedscheduler":
        scheduler = ExponentialLR(optimizer=optimizer, gamma=0.1)
    elif scheduler_name.lower() == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min")
    logging.info(f"Returned scheduler {scheduler}")
    return scheduler


def define_optimizer(optimizer_name: str, model: nn.Module, lr: float) -> nn.Module:
    """
    Defines an optimizer for training the model based on the given name.

    Parameters:
    - optimizer_name (str): Name of the optimizer ('adam' or 'sgd').
    - model (nn.Module): Model containing parameters to be optimized.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - nn.Module: Configured optimizer for training the model.
    """

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        logging.warning(f"Enter the correct name of the optimizer")
    return optimizer


def plot_correlation(df):
    """
    Plots a heatmap showing the correlation matrix of the given DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    """
    Corr_Matrix = round(df.corr(), 2)
    axis_corr = sns.heatmap(
        Corr_Matrix,
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(50, 500, n=500),
        square=True,
    )

    plt.show()


def save_model_weights(
    model: nn.Module,
    optimizer: torch.optim,
    epoch: int,
    loss: int,
    dir_name: str,
    model_name: str,
    BASE_DIR: str,
):
    """
    Saves the model's weights, optimizer state, and training metadata to a file.

    Parameters:
    - model (nn.Module): The neural network model to save.
    - optimizer (torch.optim): The optimizer associated with the model.
    - epoch (int): The current epoch of training.
    - loss (int): The current loss value.
    - dir_name (str): Directory name to save the weights.
    - model_name (str): Name of the model file.
    - BASE_DIR (str): Base directory path.

    Returns:
    - None
    """
    path_dir = create_experiments_dir(dir_name=dir_name, BASE_DIR=BASE_DIR)
    path = path_dir + "/" + model_name + ".pth"
    logging.info(f"Path to save model {path}, model {model_name}")
    for name, param in model.named_parameters():
        logging.info(f"Name layer default save {name}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def plot_charts(result: dict, model_name: str, BASE_DIR: str, dir_name: str):
    """
    Plots line charts of training metrics and saves them as images.

    Parameters:
    - result (dict): Dictionary containing epoch and metric data.
    - model_name (str): Name of the model.
    - BASE_DIR (str): Base directory path.
    - dir_name (str): Subdirectory name to save the plots.

    Returns:
    - None
    """
    epoch = result["epoch"]
    for idx, key in enumerate(result.keys()):
        data = result[key]

        if "loss" in key:
            metrics = "funkcja straty"
        elif "acc" in key:
            metrics = "dokładność"
        elif "precision" in key:
            metrics = "precyzja"
        elif "recall" in key:
            metrics = "czułość"
        elif "f1-score" in key:
            metrics = "miara F1"
        else:
            continue
        if isinstance(data, (list, np.ndarray)):
            fig = px.line(x=epoch, y=data, title=f"Metryka: {metrics}")
            fig.write_image(f"{BASE_DIR}/{dir_name}/{model_name}_{key}.png")


def save_results(train_result: dict, test_result: dict, model_name: str, path: str):
    """
    Saves the training and test results to JSON files.

    Parameters:
    - train_result (dict): Training results.
    - test_result (dict): Test results.
    - model_name (str): Name of the model.
    - path (str): Directory path to save results.

    Returns:
    - None
    """
    logging.info(f"Path to save results {path}")
    train_data = json.dumps(train_result, indent=6)
    test_data = json.dumps(test_result, indent=6)
    data = [{"name": "train", "data": train_data}, {"name": "test", "data": test_data}]
    logging.info(f"Results train: {train_data}")
    logging.info(f"Results test: {test_data}")

    for itm in data:
        with open(f"{path}/{model_name}_{itm['name']}.json", "w") as file:
            file.write(itm["data"])


def save_model(model: nn.Module, path: str):
    """
    Saves the entire model to a specified path.

    Parameters:
    - model (nn.Module): The neural network model to save.
    - path (str): File path to save the model.

    Returns:
    - None
    """
    torch.save(model, path)


def load_model(path: str):
    """
    Loads a saved model from the specified path.

    Parameters:
    - path (str): Path to the saved model file.

    Returns:
    - model (nn.Module): Loaded neural network model.
    """
    model = torch.load(path)
    return model


def load_model_weights(model: nn.Module, path: str, optimizer: torch.optim):
    """
    Loads model weights and optimizer state from a checkpoint.

    Parameters:
    - model (nn.Module): The model to load the weights into.
    - path (str): Path to the saved weights.
    - optimizer (torch.optim): The optimizer to load the state into.

    Returns:
    - model (nn.Module): Model with loaded weights.
    - optimizer (torch.optim): Optimizer with loaded state.
    - epoch (int): The epoch from the checkpoint.
    - loss (int): Loss value from the checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logging.info(f"Loading weights")

    return model, optimizer, epoch, loss


def visualization_dataset(
    df_list: list, titles: list[str], xasix_name: str, yaxis_name: str, path: str
):
    """
    Generates and saves histogram plots for a list of DataFrames.

    Parameters:
    - df_list (list): List of DataFrames to plot.
    - titles (list[str]): List of titles for each histogram.
    - xasix_name (str): X-axis label.
    - yaxis_name (str): Y-axis label.
    - path (str): Directory path to save the plots.

    Returns:
    - None
    """

    for idx, df in enumerate(df_list):
        fig = px.histogram(df, x="Klasa", title=titles[idx])
        fig.update_layout(xaxis_title=xasix_name, yaxis_title=yaxis_name, bargap=0.2)
        fig.write_image(f"{path}/{titles[idx]}.png")


def analyze_results():
    """
    Analyzes results in a CSV file, filters high-performance rows,
    and sorts experiment directories by accuracy.

    Returns:
    - None
    """
    results = pd.read_csv(
        "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results_all.csv",
    )
    mask = results.filter(items=["accuracy", "f1 score", "recall"]).gt(0.90).all(axis=1)
    indexes = results[mask].index
    for i in range(1, 10):
        os.makedirs(
            f"C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments sorted/{i/10}",
            exist_ok=True,
        )
    dir_names = os.listdir(
        f"C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments"
    )
    for dir_name in dir_names:
        try:
            with open(
                f"C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments/{dir_name}/MLP_test.json"
            ) as f:
                file_contents = json.load(f)
            recall = file_contents["test_recall"][-1]
            acc = file_contents["test_acc"][-1]
            f1_score = file_contents["test_f1_score"][-1]
            shutil.copytree(
                f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments/{dir_name}",
                f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments sorted/{round(acc, 1)}/{dir_name}",
            )
        except FileNotFoundError:
            print(f"Plik nie został znaleziony")
        except json.JSONDecodeError:
            print(f"Błąd dekodowania JSON w pliku")
        except PermissionError:
            print(f"Brak uprawnień do kopiowania folderu: {dir_name}")
        except Exception as e:
            print(f"Wystąpił błąd: {e}")


def create_experiments_dir(dir_name: str, BASE_DIR: str):
    """
    Creates a directory for experiment outputs.

    Parameters:
    - dir_name (str): Name of the directory to create.
    - BASE_DIR (str): Base directory path.

    Returns:
    - path (str): Path of the created directory.
    """

    path = BASE_DIR + "/" + str(dir_name).replace("\\", "/")
    try:
        os.mkdir(path)

    except OSError as error:
        print(error)

    return path


def visaulization_models(mmlp_option, model, input_data):
    """
    Generates and saves a computational graph visualization of the model.

    Parameters:
    - mmlp_option (str): Option identifier for directory naming.
    - model (nn.Module): Model to visualize.
    - input_data (torch.Tensor): Sample input data for model.

    Returns:
    - None
    """
    path = f"C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{mmlp_option}"
    output = model(input_data)
    dot = make_dot(output, params=dict(model.named_parameters()))
    os.makedirs(path, exist_ok=True)

    dot.render(f"{path}/model_graph", format="png")
