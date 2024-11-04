import csv
import datetime
import json
import sys
from timeit import default_timer as timer

import torch.optim.lr_scheduler as lr_scheduler

sys.path.insert(
    1,
    "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/scripts",
)
import logging
import os

import torch
from scripts.data_setup import prepare_data, prepare_data_multimodel_dataset
from scripts.engine import train
from scripts.multimodal_model import MultiModalClassifier, train_multimodal
from scripts.utils import (
    EarlyStopping,
    create_experiments_dir,
    define_model,
    plot_charts,
    save_model_weights,
    save_results,
)
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    filename="trained_models.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)
BASE_DIR_DATBASE = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/"
BASE_DIR = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def unfreeze_layers_multimodal(model, layers_to_unfreeze):
    """
    Unfreezes specified layers in a multimodal model.

    Parameters:
    - model: Model object
        The multimodal model with distinct components for images and captions.
    - layers_to_unfreeze: List[str]
        Names of model layers to unfreeze. Possible values include "images_encoder" and "caption_model".

    Returns:
    None
    """
    if "caption_model" in layers_to_unfreeze or "images_encoder" in layers_to_unfreeze:
        if "images_encoder" in layers_to_unfreeze:
            for param in model.images_encoder.parameters():
                param.requires_grad = True
            print("Unfroze images_encoder layers.")

        if "caption_model" in layers_to_unfreeze:
            for param in model.caption_model.parameters():
                param.requires_grad = True
            print("Unfroze caption_model layers.")
    elif not layers_to_unfreeze:
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_unfreeze):
                print(f"Odmrazamy {name, param}")
                param.requires_grad = True
            else:
                param.requires_grad = False


def load_tests_from_json(file_path: str):
    """
    Loads test configurations from a JSON file.

    Parameters:
    - file_path: str
        Path to the JSON file containing test configurations.

    Returns:
    - tests: dict
        Parsed test configurations.
    """
    with open(file_path, "r") as f:
        tests = json.load(f)
    return tests


def unfreeze_last_layers(model, num_layers=5):
    """
    Selects the last specified number of layers to unfreeze in a model.

    Parameters:
    - model: Model object
        The neural network model from which layers will be unfrozen.
    - num_layers: int
        Number of last layers to unfreeze. Defaults to 5.

    Returns:
    - layers_to_unfreeze: list
        List of last layers selected to unfreeze.
    """
    layers_to_unfreeze = []
    for name, layer in reversed(list(model.named_children())):
        layers_to_unfreeze.append(layer)
        if len(layers_to_unfreeze) >= num_layers + 1:
            break
    print(f"layers_to_unfreeze: {layers_to_unfreeze}")
    return layers_to_unfreeze


def apply_test_config(test):
    """
    Applies a test configuration to train a CNN model.

    Parameters:
    - test: dict
        Test configuration containing model and training parameters.

    Returns:
    None
    """
    logging.info(f"Running Test ID: {test['test_id']}")
    model_name = test["model"]["model_name"]
    fine_tunning = test["model"]["fine-tunning"]
    freeze_all = test["model"]["freez_all"]
    epochs = test["training"]["epochs"]
    lr = test["training"]["lr"]
    augmentation = test["training"]["augmentation"]
    name = test["training"]["dir_name"]
    valid = test["training"]["valid"]
    train_CNN(
        fine_tunning, model_name, epochs, lr, freeze_all, augmentation, name, valid
    )


def train_CNN(
    fine_tunning, model_name, epochs, lr, freeze_all, augmentation, name, valid
):
    """
    Trains a CNN model with given parameters and saves results.

    Parameters:
    - fine_tunning: bool
        Indicates if fine-tuning should be applied.
    - model_name: str
        Name of the CNN model to be trained.
    - epochs: int
        Number of training epochs.
    - lr: float
        Learning rate for the optimizer.
    - freeze_all: bool
        Whether to freeze all layers in the model.
    - augmentation: bool
        Indicates if data augmentation is applied.
    - name: str
        Directory name for saving experiments.
    - valid: bool
        Indicates if a validation dataset is used.

    Returns:
    None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir_name = f'experiments_trained_models_{name}/{model_name}/{ datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")}'
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    path_experiments = create_experiments_dir(dir_name, BASE_DIR)
    model = define_model(model_name=model_name)
    if augmentation:
        path = f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images — GUMED  Augmentation"
    else:
        path = f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images — GUMED"

    if model_name in ["GoogLeNet", "ResNet", "Inception_V3", "ResNet18"]:
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 4)

    elif model_name in ["DenseNet161", "AlexNet"]:
        if hasattr(model, "classifier"):
            if model_name == "AlexNet":
                in_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(in_features, 4)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 4)

    elif model_name == "SqueezeNet1_0":
        if hasattr(model, "classifier"):
            in_features = model.classifier[1].in_channels
            model.classifier[1] = nn.Conv2d(
                in_features, 4, kernel_size=(1, 1), stride=(1, 1)
            )

    elif model_name == "EfficientNet_V2":
        if hasattr(model, "classifier"):
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 4)

    if freeze_all:
        for name, layer in model.named_modules():
            for param in layer.parameters():
                param.requires_grad = False

    elif fine_tunning:
        layers_to_unfreeze = unfreeze_last_layers(model)
        for name, layer in model.named_modules():
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

    model.to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader, valid_dataloader, test_dataloader = prepare_data(
        model_name=model_name,
        path=path,
        augmentation=augmentation,
        valid=valid,
        freeze_all=freeze_all,
    )
    early_stopping = EarlyStopping(patience=10, verbose=True)
    start_time_training = timer()
    result_train, result_valid, result_test = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=epochs,
        path_experiments=path_experiments,
        early_stopping=early_stopping,
        device=device,
        fine_tunning=fine_tunning,
        freeze_all=freeze_all,
        valid=valid,
    )
    end_time_training = timer()
    print(f"Total training time: {end_time_training-start_time_training:.3f} seconds")
    if freeze_all:
        save_results(
            train_result=[],
            test_result=result_test,
            model_name=model_name,
            path=path_experiments,
        )
    else:
        save_results(
            train_result=result_train,
            test_result=result_test,
            model_name=model_name,
            path=path_experiments,
        )

        plot_charts(
            train_result=result_train, model_name=model_name, BASE_DIR=path_experiments
        )

    data
    plot_charts(
        train_result=result_valid, model_name=model_name, BASE_DIR=path_experiments
    )

    save_model_weights(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        loss=result_train["train_loss"][-1],
        dir_name=f"{model_name}_{result_train['train_loss'][-1]}",
        model_name=model_name,
        BASE_DIR=path_experiments,
    )


def train_multimodal_model(test_config):
    """
    Trains a multimodal model based on a specific test configuration.

    Parameters:
    - test_config: dict
        Configuration dictionary containing model and training settings, including fine-tuning,
        dropout, number of epochs, learning rate, and other model parameters.

    Returns:
    None
    """
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    print(f"Running test: {test_config['test_name']}")
    fine_tune = test_config["fine_tune"]
    layers_to_unfreeze = test_config["layers_to_unfreeze"]
    learning_rate = test_config["learning_rate"]
    dropout = test_config["dropout"]
    epochs = test_config["epochs"]
    use_scheduler = test_config["use_scheduler"]
    data_augmentation = test_config["data augmentation"]
    freeze_entire_model = test_config["freeze_entire_model"]

    model_name = "multimodal"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir_name = f'experiments_trained_models_multimodal/{model_name}/{ datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")}'
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    path_experiments = create_experiments_dir(dir_name, BASE_DIR)
    train_dataloader, test_dataloader = prepare_data_multimodel_dataset(
        IMAGE_SIZE=IMAGE_SIZE, BATCH_SIZE=BATCH_SIZE
    )
    model = MultiModalClassifier(
        num_classes=NUM_CLASSES, device=device, dropout=dropout
    )
    model.to(device=device)
    if fine_tune:
        unfreeze_layers_multimodal(model, layers_to_unfreeze)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    start_time_training = timer()
    results_train, results_valid, results_test = train_multimodal(
        NUM_EPOCHS=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        early_stopping=early_stopping,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        path_experiments=path_experiments,
        layers_to_unfreeze=layers_to_unfreeze,
        use_scheduler=use_scheduler,
    )
    end_time_training = timer()
    print(f"Total training time: {end_time_training-start_time_training:.3f} seconds")
    save_results(
        train_result=results_train,
        test_result=results_test,
        model_name=model_name,
        path=path_experiments,
    )
    results = []
    results.append(
        {
            "Train Loss": round(results_train["train_loss"][-1], 4),
            "Train Accuracy": round(results_train["train_acc"][-1], 4),
            "Train Precision": round(results_train["train_precision"][-1], 4),
            "Train Recall": round(results_train["train_recall"][-1], 4),
            "Train F1 Score": round(results_train["train_f1_score"][-1], 4),
            "Validation Loss": round(results_valid["valid_loss"][-1], 4),
            "Validation Accuracy": round(results_valid["valid_acc"][-1], 4),
            "Validation Precision": round(results_valid["valid_precision"][-1], 4),
            "Validation Recall": round(results_valid["valid_recall"][-1], 4),
            "Validation F1 Score": round(results_valid["valid_f1_score"][-1], 4),
            "Test Loss": round(results_test["test_loss"][-1], 4),
            "Test Accuracy": round(results_test["test_acc"][-1], 4),
            "Test Precision": round(results_test["test_precision"][-1], 4),
            "Test Recall": round(results_test["test_recall"][-1], 4),
            "Test F1 Score": round(results_test["test_f1_score"][-1], 4),
            "Learning Rate": learning_rate,
            "Unfrozen Layers": layers_to_unfreeze,
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "use_scheduler": use_scheduler,
            "dropout": dropout,
            "Data Augmentation": data_augmentation,
            "freeze_entire_model": freeze_entire_model,
            "fine_tune": fine_tune,
            "experiments_name": dir_name,
            "test_name": test_config["test_name"],
        }
    )
    save_results_to_csv("results_all_multimodal.csv", results)

    plot_charts(
        train_result=results_train, model_name=model_name, BASE_DIR=path_experiments
    )


def save_results_to_csv(filename, results):
    """
    Appends training results to a CSV file.

    Parameters:
    - filename: str
        Name of the CSV file to save results.
    - results: list[dict]
        List of result dictionaries containing training and validation metrics.

    Returns:
    None
    """

    fieldnames = [
        "Train Loss",
        "Train Accuracy",
        "Train Precision",
        "Train Recall",
        "Train F1 Score",
        "Validation Loss",
        "Validation Accuracy",
        "Validation Precision",
        "Validation Recall",
        "Validation F1 Score",
        "Test Loss",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1 Score",
        "Learning Rate",
        "Unfrozen Layers",
        "batch_size",
        "epochs",
        "use_scheduler",
        "dropout",
        "Data Augmentation",
        "freeze_entire_model",
        "fine_tune",
        "experiments_name",
        "test_name",
    ]

    file_exists = False
    try:
        with open(filename, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)


def main(multimodal: bool):
    """
    Main function to run tests based on model type (multimodal or CNN).

    Parameters:
    - multimodal: bool
        Specifies if the multimodal model tests should be run.

    Returns:
    None
    """
    if not multimodal:
        tests = load_tests_from_json(f"{BASE_DIR}/tests_plan.json")
        for test in tests["tests"]:
            apply_test_config(test)
    else:
        tests_config = load_tests_from_json(f"{BASE_DIR}/multimodal_test.json")
        for test_config in tests_config:
            train_multimodal_model(test_config)


multimodal = True
main(multimodal=multimodal)
