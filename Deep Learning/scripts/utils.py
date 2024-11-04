import json
import logging
import os

import numpy as np
import plotly.express as px
import torch
from torch import nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)
from torchvision.models import (
    AlexNet_Weights,
    DenseNet161_Weights,
    EfficientNet_V2_S_Weights,
    GoogLeNet_Weights,
    Inception_V3_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    SqueezeNet1_0_Weights,
    alexnet,
    densenet161,
    efficientnet_v2_s,
    googlenet,
    inception_v3,
    resnet18,
    resnet50,
    squeezenet1_0,
)
from torchviz import make_dot

logging.basicConfig(
    level=logging.INFO,
    filename="trained_models.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save if validation loss decreases.
        """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save.
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def define_scheduler(scheduler_name: str, optimizer: torch.optim) -> nn.Module:
    """
    Define a learning rate scheduler based on the specified name.

    Args:
        scheduler_name (str): The name of the scheduler to be used.
        optimizer (torch.optim): The optimizer to which the scheduler will be applied.

    Returns:
        nn.Module: The defined learning rate scheduler.
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


def visaulization_models(model_name, model, image, text):
    """
    Visualize the architecture of a model by creating a graph.

    Args:
        model_name (str): Name of the model to visualize.
        model (nn.Module): The model to visualize.
        image (torch.Tensor): Sample image input for the model.
        text (torch.Tensor): Sample text input for the model.

    Returns:
        None: The function saves the model graph as a PNG image in the specified directory.
    """
    path = f"C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/architectures/{model_name}"
    output = model(image, text)
    dot = make_dot(output, params=dict(model.named_parameters()))
    os.makedirs(path, exist_ok=True)

    dot.render(f"{path}/model_graph", format="png")


def plot_charts(train_result: dict, model_name: str, BASE_DIR: str):
    """
    Plot and save charts for various metrics based on training results.

    Args:
        train_result (dict): A dictionary containing training metrics for each epoch.
        model_name (str): The name of the model used for training.
        BASE_DIR (str): The base directory where the charts will be saved.

    Returns:
        None: The function generates and saves plots for each metric in the specified directory.
    """
    epoch = train_result["epoch"]
    for idx, key in enumerate(train_result.keys()):
        data = train_result[key]
        print(f"key {key} data {data}")
        if key == "train_loss":
            metrics = "funkcja straty"
        elif key == "train_acc":
            metrics = "dokładność"
        elif key == "train_precision":
            metrics = "precyzja"
        elif key == "train_recall":
            metrics = "czułość"
        elif key == "train_f1-score":
            metrics = "miara F1"
        else:
            continue
        if isinstance(data, (list, np.ndarray)):
            fig = px.line(x=epoch, y=data, title=f"Metryka: {metrics}")
            fig.write_image(f"{BASE_DIR}/{model_name}_{key}.png")


def save_results(train_result: dict, test_result: dict, model_name: str, path: str):
    """
    Save training and testing results to JSON files.

    Args:
        train_result (dict): A dictionary containing training results.
        test_result (dict): A dictionary containing testing results.
        model_name (str): The name of the model associated with the results.
        path (str): The directory path where the results will be saved.

    Returns:
        None: The function writes the results to JSON files in the specified directory.
    """
    train_data = json.dumps(train_result, indent=6)
    test_data = json.dumps(test_result, indent=6)
    data = [{"name": "train", "data": train_data}, {"name": "test", "data": test_data}]
    for itm in data:
        with open(f"{path}/{model_name}_{itm['name']}.json", "w") as file:
            file.write(itm["data"])


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
    Save the model weights along with the optimizer state and training epoch.

    Args:
        model (nn.Module): The model whose weights are to be saved.
        optimizer (torch.optim): The optimizer whose state is to be saved.
        epoch (int): The current epoch number.
        loss (int): The current loss value.
        dir_name (str): The name of the directory where the model will be saved.
        model_name (str): The name of the model.
        BASE_DIR (str): The base directory where the model weights will be saved.

    Returns:
        None: The function saves the model weights in the specified path.
    """
    path_dir = create_experiments_dir(dir_name=dir_name, BASE_DIR=BASE_DIR)
    path = path_dir + "/" + model_name + ".pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def create_experiments_dir(dir_name: str, BASE_DIR: str):
    """
    Create a directory for saving experiment results.

    Args:
        dir_name (str): The name of the directory to create.
        BASE_DIR (str): The base directory where the new directory will be created.

    Returns:
        str: The path to the created directory.
    """
    path = BASE_DIR + "/" + str(dir_name).replace("\\", "/")
    print("Path", path)
    try:
        os.mkdir(path)

    except OSError as error:
        print(error)

    return path


def define_model(model_name):
    """
    Define and initialize a model based on the specified name.

    Args:
        model_name (str): The name of the model to instantiate.

    Returns:
        nn.Module: The initialized model corresponding to the specified name.
    """
    if model_name == "GoogLeNet":
        model = googlenet(GoogLeNet_Weights.DEFAULT)
    elif model_name == "ResNet":
        model = resnet50(ResNet50_Weights.DEFAULT)

    elif model_name == "ResNet18":
        model = resnet18(ResNet18_Weights.DEFAULT)

    elif model_name == "SqueezeNet1_0":
        model = squeezenet1_0(SqueezeNet1_0_Weights.DEFAULT)
    elif model_name == "DenseNet161":
        model = densenet161(DenseNet161_Weights.DEFAULT)

    elif model_name == "Inception_V3":
        model = inception_v3(Inception_V3_Weights.DEFAULT)

    elif model_name == "EfficientNet_V2":
        model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)

    elif model_name == "AlexNet":
        model = alexnet(AlexNet_Weights.DEFAULT)

    return model
