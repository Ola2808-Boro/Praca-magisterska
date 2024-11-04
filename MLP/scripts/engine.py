import os
import sys
import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ignite.handlers.early_stopping import EarlyStopping
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import ROC, Accuracy, F1Score, Precision, Recall

sys.path.insert(
    1,
    "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/scripts",
)
import logging

import numpy as np
import pandas as pd
import plotly.express as px
from model import MLP, BasicMMLP, Parallel_Concatenation_MMLP, SklearnMLPWrapper
from setup_data import (
    balancing_dataset,
    convert_to_tensors,
    create_dataloder,
    create_dataset,
    preprocessing_data,
    read_csv_data,
    select_features,
)
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from utils import (
    LIME_explainer,
    SHAP_explainer,
    create_experiments_dir,
    define_activation_function,
    define_optimizer,
    define_scheduler,
    plot_charts,
    save_model_weights,
    save_results,
    visaulization_models,
    visualization_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    filename="MLP.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def read_data_as_arrays():
    """
    Reads training and test data from specified paths and returns them as arrays.

    Returns:
        Tuple: Arrays containing the training and test data.
    """
    path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
    path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"
    paths = [path_train, path_test]
    df = read_csv_data(paths)
    train_data = df[0].values
    test_data = df[1].values
    return train_data, test_data


def train_step(
    model: nn.Module,
    epoch: int,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adaptive_lr: bool,
    scheduler,
    loss_fn: nn.Module,
    accuracy,
    precision,
    recall,
    f1_score,
    roc,
    device: torch.device,
    model_name: str,
):
    """
    Executes one training step on the model.

    Args:
        model (nn.Module): The neural network model.
        epoch (int): Current epoch number.
        dataloader (DataLoader): Dataloader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        adaptive_lr (bool): Whether to use an adaptive learning rate.
        scheduler: Scheduler for learning rate adjustments.
        loss_fn (nn.Module): Loss function for training.
        accuracy, precision, recall, f1_score, roc: Metrics for performance evaluation.
        device (torch.device): Device for computation.
        model_name (str): Name of the model.
    """
    logging.info(f"Model train {model_name} epoch {epoch} model {model}")
    model.train()
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    patience = 10
    acc_test = 0
    best_loss = float("inf")
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(torch.float32), y.to(torch.long)
        if 3 in torch.unique(y):
            y = y - 1
        y_pred = model(x).squeeze()
        y = y.squeeze()
        y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
        loss = loss_fn(y_pred, y)
        loss_avg = loss_avg + loss.item()
        acc = accuracy(y_pred_class, y)
        recall_result = recall(y_pred_class, y)
        prec = precision(y_pred_class, y)
        if torch.isnan(prec):
            prec = torch.tensor(0.0)
        f1_score_result = f1_score(y_pred_class, y)
        acc_test_item = (y_pred_class == y).float().mean().item()
        acc_test = acc_test + acc_test_item
        acc_avg = acc_avg + acc
        prec_avg = prec_avg + prec
        recall_avg = recall_avg + recall_result
        f1_score_avg = f1_score_avg + f1_score_result
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if adaptive_lr:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()
    loss_avg = loss_avg / len(dataloader)
    acc_avg = acc_avg / len(dataloader)
    prec_avg = prec_avg / len(dataloader)
    f1_score_avg = f1_score_avg / len(dataloader)
    recall_avg = recall_avg / len(dataloader)
    acc_test = acc_test / len(dataloader)

    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    y_pred_np = (
        y_pred_class.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor)
        else y_pred
    )
    report = classification_report(y_np, y_pred_np)
    logging.info(f"Train {classification_report(y, y_pred_class)}")
    conf_matrix = confusion_matrix(y_np, y_pred_np)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} |{loss_avg:.5f}, Test Acc: {acc_test:.2f}%")

    unique_classes = np.unique(np.concatenate([y_np, y_pred_np]))
    return (
        loss_avg,
        acc_avg.item(),
        prec_avg.item(),
        f1_score_avg.item(),
        recall_avg.item(),
        report,
        conf_matrix,
        model,
        unique_classes,
    )


def test_step(
    model: nn.Module,
    epoch: int,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy,
    precision,
    recall,
    f1_score,
    roc,
    device: torch.device,
    model_name: str,
):
    """
    Performs one testing step on the model.

    Args:
        model (nn.Module): The neural network model to be tested.
        epoch (int): The current epoch number.
        dataloader (DataLoader): The testing data loader for loading the test dataset.
        loss_fn (nn.Module): The loss function to calculate the loss.
        accuracy: Function to compute accuracy metric.
        precision: Function to compute precision metric.
        recall: Function to compute recall metric.
        f1_score: Function to compute F1 score metric.
        roc: Function to compute ROC curve metric.
        device (torch.device): The device to perform operations on (CPU or GPU).
        model_name (str): The name of the model.

    Returns:
        Tuple: A tuple containing:
            - loss_avg (float): Average loss over the test set.
            - acc_avg (float): Average accuracy over the test set.
            - prec_avg (float): Average precision over the test set.
            - f1_score_avg (float): Average F1 score over the test set.
            - recall_avg (float): Average recall over the test set.
            - report (str): Classification report for the test set.
            - conf_matrix (ndarray): Confusion matrix for the test set.
            - unique_classes (ndarray): Unique classes present in the test set.
    """
    model.eval()
    with torch.inference_mode():
        loss_avg = 0
        acc_avg = 0
        prec_avg = 0
        recall_avg = 0
        f1_score_avg = 0
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(torch.float32), y.to(torch.long).squeeze()
            if 3 in torch.unique(y):
                y = y - 1
            y_pred = model(x).squeeze()
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            # logging.info(f"Shape y {y.shape},y_pred:{y_pred.shape}")
            loss = loss_fn(y_pred, y)
            loss_avg = loss_avg + loss.item()
            acc = accuracy(y_pred_class, y)
            recall_result = recall(y_pred_class, y)
            prec = precision(y_pred_class, y)
            f1_score_result = f1_score(y_pred_class, y)
            acc_avg = acc_avg + acc
            prec_avg = prec_avg + prec
            recall_avg = recall_avg + recall_result
            f1_score_avg = f1_score_avg + f1_score_result
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        y_pred_np = (
            y_pred_class.detach().cpu().numpy()
            if isinstance(y_pred, torch.Tensor)
            else y_pred
        )
        report = classification_report(y_np, y_pred_np)
        conf_matrix = confusion_matrix(y_np, y_pred_np)
        logging.info(f"Test {classification_report(y, y_pred_class)}")
        loss_avg = loss_avg / len(dataloader)
        acc_avg = acc_avg / len(dataloader)
        prec_avg = prec_avg / len(dataloader)
        f1_score_avg = f1_score_avg / len(dataloader)
        recall_avg = recall_avg / len(dataloader)
        if epoch == 19:
            data = {
                "loss": [loss_avg],
                "accuracy": [acc_avg.item()],
                "f1 score": [f1_score_avg.item()],
                "recall": [recall_avg.item()],
            }
            df_results = pd.DataFrame(
                data=data,
                columns=["loss", "accuracy", "f1 score", "recall"],
            )

            df_results.to_csv(
                "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results_all_MLP_hidden.csv",
                mode="a",
                index=False,
                header=False,
            )

        unique_classes = np.unique(np.concatenate([y_np, y_pred_np]))
        return (
            loss_avg,
            acc_avg.item(),
            prec_avg.item(),
            f1_score_avg.item(),
            recall_avg.item(),
            report,
            conf_matrix,
            unique_classes,
        )


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    optimizer: torch.optim,
    adaptive_lr: bool,
    model_name: str,
    dir_name: str,
    class_num: int,
    BASE_DIR: str,
    scheduler_name: str,
):
    """
    Trains the model.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_dataloader (DataLoader): The training data loader.
        test_dataloader (DataLoader): The testing data loader.
        epochs (int): The number of epochs to train the model for.
        optimizer (torch.optim): The optimizer to be used for training the model.
        adaptive_lr (bool): Whether to use an adaptive learning rate or not.
        model_name (str): The name of the model.
        dir_name (str): The directory name for saving results.
        class_num (int): The number of classes in the dataset.
        BASE_DIR (str): The base directory for saving files.
        scheduler_name (str): The name of the learning rate scheduler.

    Returns:
        Tuple: A tuple containing:
            - result_train (dict): Dictionary containing training results.
            - result_test (dict): Dictionary containing testing results.
    """
    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=class_num, average="weighted")
    precision = Precision(task="multiclass", num_classes=class_num, average="weighted")
    f1_score = F1Score(task="multiclass", num_classes=class_num, average="weighted")
    recall = Recall(task="multiclass", num_classes=class_num, average="weighted")
    roc = ROC(task="multiclass", num_classes=class_num)
    if adaptive_lr:
        scheduler = define_scheduler(scheduler_name, optimizer)
    else:
        scheduler = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_train = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1_score": [],
        "training_time": [],
    }

    result_test = {
        "epoch": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1_score": [],
        "testing_time": [],
    }
    start_training_time = time.time()
    for epoch in range(int(epochs)):
        (
            train_loss,
            train_acc,
            train_precision,
            train_f1_score,
            train_recall,
            train_report,
            train_conf_matrix,
            model,
            unique_classes_train,
        ) = train_step(
            model=model,
            epoch=epoch,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc=roc,
            optimizer=optimizer,
            adaptive_lr=adaptive_lr,
            scheduler=scheduler,
            device=device,
            model_name=model_name,
        )

        logging.info(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} |"
            f"train_precision: {train_precision:.4f} |"
            f"train_f1_score: {train_f1_score:.4f} |"
            f"train_recall: {train_recall:.4f} |"
        )
        result_train["epoch"].append(epoch)
        result_train["train_loss"].append(round(train_loss, 5))
        result_train["train_acc"].append(round(train_acc, 5))
        result_train["train_recall"].append(round(train_recall, 5))
        result_train["train_f1_score"].append(round(train_f1_score, 5))

        os.makedirs(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/train/{epoch}/",
            exist_ok=True,
        )
        with open(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/train/{epoch}/classification_report.txt",
            "w",
        ) as f:
            f.write(train_report)

        conf_matrix_df = pd.DataFrame(
            train_conf_matrix, index=unique_classes_train, columns=unique_classes_train
        )
        fig = px.imshow(conf_matrix_df)
        fig.write_image(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/train/{epoch}/confusion_matrix.png"
        )
        conf_matrix_df.to_csv(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/train/{epoch}/confusion_matrix.csv"
        )

        start_testing_time = time.time()
    (
        test_loss,
        test_acc,
        test_precision,
        test_f1_score,
        test_recall,
        test_report,
        test_conf_matrix,
        unique_classes_test,
    ) = test_step(
        model=model,
        epoch=epoch,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        roc=roc,
        device=device,
        model_name=model_name,
    )
    end_testing_time = time.time()
    testing_time = end_testing_time - start_testing_time
    logging.info(
        f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}, test_prec: {test_precision:.4f}, test_f1_score: {test_f1_score:.4f} test_recall: {test_recall:.4f}"
    )
    result_test["epoch"].append(epoch)
    result_test["test_loss"].append(round(test_loss, 5))
    result_test["test_acc"].append(round(test_acc, 5))
    result_test["test_recall"].append(round(test_recall, 5))
    result_test["test_f1_score"].append(round(test_f1_score, 5))
    result_test["testing_time"].append(round(testing_time, 5))
    os.makedirs(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/test/{epoch}/",
        exist_ok=True,
    )
    with open(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/test/{epoch}/classification_report.txt",
        "w",
    ) as f:
        f.write(test_report)
    conf_matrix_df = pd.DataFrame(
        test_conf_matrix, index=unique_classes_test, columns=unique_classes_test
    )
    fig = px.imshow(conf_matrix_df)
    fig.write_image(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/test/{epoch}/confusion_matrix.png"
    )
    conf_matrix_df.to_csv(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/{dir_name}/test/{epoch}/confusion_matrix.csv"
    )
    end_training_time = time.time()

    training_time = end_training_time - start_training_time
    result_train["training_time"] = round(training_time, 5)
    save_model_weights(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=test_loss,
        dir_name=dir_name,
        model_name=model_name,
        BASE_DIR=BASE_DIR,
    )

    return result_train, result_test


def train_MLP(
    input_size: int,
    hidden_size: int,
    hidden_units: list,
    output_size: int,
    feature_selection_method: str,
    optimizer_name: str,
    lr: float,
    adaptive_lr: bool,
    activation_function_name: str,
    epochs: int,
    balanced_database: bool,
    batch_size: int,
    num_features: int,
    option: str,
    mmlp_option: dict,
    dir_name: str,
    BASE_DIR: str,
    idx: list | int,
    weighted_loss_function: bool,
    scheduler_name: str,
):
    """
    Trains a Multi-Layer Perceptron (MLP) model using the specified configurations.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The number of hidden layers.
        hidden_units (list): The number of units in each hidden layer.
        output_size (int): The number of output classes.
        feature_selection_method (str): The method used for feature selection.
        optimizer_name (str): The name of the optimizer to be used.
        lr (float): The learning rate for the optimizer.
        adaptive_lr (bool): Whether to use an adaptive learning rate.
        activation_function_name (str): The activation function to be used in the model.
        epochs (int): The number of epochs to train the model.
        balanced_database (bool): Whether to balance the dataset or not.
        batch_size (int): The size of each training batch.
        num_features (int): The number of features to select.
        option (str): The option for dataset balancing.
        mmlp_option (dict): Configuration options for the MMLP model.
        dir_name (str): The directory name for saving results.
        BASE_DIR (str): The base directory for saving files.
        idx (list | int): Indices for SHAP or LIME explanations.
        weighted_loss_function (bool): Whether to use a weighted loss function.
        scheduler_name (str): The name of the learning rate scheduler.

    Returns:
        None: The function does not return any value. Instead, it saves the training results,
        plots, and explanations to the specified directories.
    """
    path_experiments = create_experiments_dir(dir_name, BASE_DIR)
    with open(f"{path_experiments}/configuration.txt", "w") as f:
        f.write(
            f"""input_size:{input_size}\n
                hidden_size:{hidden_size}\n
                hidden_units:{hidden_units}\n
                output_size: {output_size}\n
                feature_selection_method:{feature_selection_method}\n
                optimizer_name: {optimizer_name}\n
                lr: {lr}\n
                adaptive_lr:{adaptive_lr}\n
                activation_function_name:{activation_function_name}\n
                epochs:{epochs}\n
                balanced_database:{balanced_database}\n
                batch_size:{batch_size}\n
                num_features:{num_features}\n
                option:{option}\n
                mmlp_option:{mmlp_option}\n
                dir_name:{dir_name}\n
                BASE_DIR:{BASE_DIR}\n"""
        )

    activation_function = define_activation_function(
        activation_function_name=activation_function_name, input_size=input_size
    )
    models_list = []
    optimizer = ""

    # print(f" mmlp_option['concatenation_option'] : {mmlp_option['concatenation_option']}\n model: {model}")
    path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
    path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"
    paths = [path_train, path_test]
    df = read_csv_data(paths)
    logging.info("Read data")
    visualization_dataset(
        df,
        ["Rozkład klas dla danych treningowych", "Rozkład klas dla danych testowych"],
        "Klasy",
        "Liczba przypadków",
        path_experiments,
    )
    if balanced_database != False:
        logging.info(f"Resampled data, method: {option}")
        resampled_data = balancing_dataset(option, df)
        df = [
            resampled_data[0],
            df[1],
        ]  # TODO: tutaj ewentualnie można suatwić, czy testowany też ma byc na jakos przerobionym zbiorze
        visualization_dataset(
            df,
            [
                f"Rozkład klas dla danych treningowych po wykonaniu operacji {option}",
                f"Rozkład klas dla danych testowych po wykonaniu operacji {option}",
            ],
            "Klasy",
            "Liczba przypadków",
            path_experiments,
        )
    logging.info("Processing data")
    print("Before", df[0].shape, df[1].shape)
    df = preprocessing_data(df)
    print("After", df[0].shape, df[1].shape)
    if weighted_loss_function:
        class_counts = df[0]["Klasa"].value_counts().tolist()
        class_weights = [sum(class_counts) / c for c in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    df = select_features(
        feature_selection_method=feature_selection_method,
        model=SklearnMLPWrapper(
            model_option=mmlp_option["concatenation_option"],
            num_of_MLP=mmlp_option["num_of_MLP"],
            hidden_units=hidden_units,
            hidden_size=hidden_size,
            output_size=output_size,
            activation_function=activation_function,
            epochs=epochs,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            adaptive_lr=adaptive_lr,
            lr=lr,
        ),
        df_list=df,
        num=num_features,
    )
    logging.info(f"Select data {df[0].shape,df[1].shape,df[0].columns}")
    if mmlp_option["concatenation_option"] == "singleMLP":
        logging.info(
            f"Model architecture {input_size,hidden_units,hidden_size,output_size}"
        )
        model = MLP(
            input_size=input_size,
            hidden_units=hidden_units,
            hidden_layers=hidden_size,
            output_size=output_size,
            activation_function=activation_function,
            is_MMLP=False,
        )
        print("1")
    elif mmlp_option["concatenation_option"] == "parallelMMLP":
        print("2")
        for _ in range(mmlp_option["num_of_MLP"]):
            models_list.append(
                MLP(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    hidden_layers=hidden_size,
                    output_size=output_size,
                    activation_function=activation_function,
                    is_MMLP=False,
                )
            )
            # models_list.append(MLP(input_size,hidden_units,hidden_size,output_size,activation_function,remove_output_layer=False))
        model = Parallel_Concatenation_MMLP(
            models_list, mmlp_option["num_of_MLP"], mmlp_option["output_size"]
        )
        print("models_list", models_list)
    else:
        print("3")
        for _ in range(mmlp_option["num_of_MLP"]):
            models_list.append(
                MLP(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    hidden_layers=hidden_size,
                    output_size=output_size,
                    activation_function=activation_function,
                    is_MMLP=False,
                )
            )
        model = BasicMMLP(models_list)
    print(f"Model {model}")
    # visaulization_models(
    #     mmlp_option=mmlp_option["concatenation_option"],
    #     model=model,
    #     input_data=torch.randn(1, 12),
    # )
    optimizer = define_optimizer(optimizer_name=optimizer_name, model=model, lr=lr)
    print(f"Optimizer {optimizer}")
    logging.info("Converts data to tensors")
    tensors = convert_to_tensors(df)  # [[X,y],[X,y]]
    logging.info("Creating datasets")
    datasets = create_dataset(tensors)
    logging.info("Creating dataloaders")
    train_dataloader, test_dataloader = create_dataloder(
        batch_size=batch_size, datasets=datasets
    )
    result_train, result_test = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        optimizer=optimizer,
        adaptive_lr=adaptive_lr,
        model_name=f"MLP",
        dir_name=dir_name,
        class_num=output_size,
        BASE_DIR=BASE_DIR,
        scheduler_name=scheduler_name,
    )
    save_results(
        train_result=result_train,
        test_result=result_test,
        model_name=f"MLP",
        path=path_experiments,
    )
    plot_charts(
        result_train, mmlp_option["concatenation_option"], path_experiments, "train"
    )
    plot_charts(
        result_test, mmlp_option["concatenation_option"], path_experiments, "test"
    )
    SHAP_explainer(
        model=model,
        df=df,
        path=path_experiments,
        idx=idx,
        dependence_feature="TSH",
    )
    LIME_explainer(model=model, df=df, path=path_experiments, idx=idx)
