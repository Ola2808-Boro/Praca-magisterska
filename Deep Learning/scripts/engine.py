import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.optim.lr_scheduler as lr_scheduler
from ignite.handlers.early_stopping import EarlyStopping
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torchmetrics.classification import ROC, Accuracy, F1Score, Precision, Recall
from torchvision.models.inception import InceptionOutputs
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    filename="trained_models.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
class_num = 4
accuracy = Accuracy(task="multiclass", num_classes=class_num, average="weighted").to(
    device
)
precision = Precision(task="multiclass", num_classes=class_num, average="weighted").to(
    device
)
f1_score = F1Score(task="multiclass", num_classes=class_num, average="weighted").to(
    device
)
recall = Recall(task="multiclass", num_classes=class_num, average="weighted").to(device)
roc = ROC(task="multiclass", num_classes=class_num)


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device,
):
    """
    Performs a single training step of the model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device: The device to perform computations on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - loss_avg (float): Average loss for the epoch.
            - acc_avg (float): Average accuracy for the epoch.
            - prec_avg (float): Average precision for the epoch.
            - f1_score_avg (float): Average F1 score for the epoch.
            - recall_avg (float): Average recall for the epoch.
            - report (str): Classification report as a string.
            - conf_matrix (ndarray): Confusion matrix.
            - present_labels (ndarray): Unique labels present in the dataset.
    """
    model.train()
    model.to(device)

    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        logging.info(f"Batch size train: {X.size(0)}")
        X, y = X.to(device), y.to(device)
        y_true.append(y)
        y_pred = model(X)
        if isinstance(y_pred, InceptionOutputs):
            y_pred = y_pred.logits
        loss = loss_fn(y_pred, y)

        loss_avg = loss_avg + loss.item()
        # print(f"Loss {loss_avg}")
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_pred_class_list.append(y_pred_class)
        acc = accuracy(y_pred_class.to(device), y.to(device))
        recall_result = recall(y_pred_class.to(device), y.to(device))
        prec = precision(y_pred_class.to(device), y.to(device))
        f1_score_result = f1_score(y_pred_class.to(device), y.to(device))
        acc_avg = acc_avg + acc
        prec_avg = prec_avg + prec
        recall_avg = recall_avg + recall_result
        f1_score_avg = f1_score_avg + f1_score_result
        logging.info(f"Train {y_pred_class}, {y}")
    loss_avg = loss_avg / len(dataloader)
    acc_avg = acc_avg / len(dataloader)
    prec_avg = prec_avg / len(dataloader)
    f1_score_avg = f1_score_avg / len(dataloader)
    recall_avg = recall_avg / len(dataloader)

    y_true_tensor = torch.cat(y_true).cpu()
    y_pred_class_tensor = torch.cat(y_pred_class_list).cpu()
    y_np = y_true_tensor.cpu().numpy()
    y_pred_np = y_pred_class_tensor.cpu().numpy()

    report = classification_report(y_np, y_pred_np, zero_division=0)
    conf_matrix = confusion_matrix(y_np, y_pred_np)
    present_labels = np.unique(np.concatenate([y_np, y_pred_np]))
    return (
        loss_avg,
        acc_avg.item(),
        prec_avg.item(),
        f1_score_avg.item(),
        recall_avg.item(),
        report,
        conf_matrix,
        present_labels,
    )


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device,
):
    """
    Performs a single testing step of the model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): Dataloader for the test data.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        device: The device to perform computations on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - loss_avg (float): Average loss for the test set.
            - acc_avg (float): Average accuracy for the test set.
            - prec_avg (float): Average precision for the test set.
            - f1_score_avg (float): Average F1 score for the test set.
            - recall_avg (float): Average recall for the test set.
            - report (str): Classification report as a string.
            - conf_matrix (ndarray): Confusion matrix.
            - present_labels (ndarray): Unique labels present in the dataset.
    """
    model.eval()
    model.to(device)
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            logging.info(f"Batch size test: {X.size(0)}")
            X, y = X.to(device), y.to(device)
            y_true.append(y)
            test_pred_logits = model(X)
            if isinstance(test_pred_logits, InceptionOutputs):
                test_pred_logits = test_pred_logits.logits
            loss = loss_fn(test_pred_logits, y)
            loss_avg = loss_avg + loss.item()
            y_pred_class = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
            y_pred_class_list.append(y_pred_class)
            acc = accuracy(y_pred_class.to(device), y.to(device))
            recall_result = recall(y_pred_class.to(device), y.to(device))
            prec = precision(y_pred_class.to(device), y.to(device))
            f1_score_result = f1_score(y_pred_class.to(device), y.to(device))

            acc_avg = acc_avg + acc
            prec_avg = prec_avg + prec
            recall_avg = recall_avg + recall_result
            f1_score_avg = f1_score_avg + f1_score_result
            logging.info(f"Test {y_pred_class,y}")
        y_true_tensor = torch.cat(y_true).cpu()
        y_pred_class_tensor = torch.cat(y_pred_class_list).cpu()
        y_np = y_true_tensor.cpu().numpy()
        y_pred_np = y_pred_class_tensor.cpu().numpy()
        report = classification_report(y_np, y_pred_np, zero_division=0)
        conf_matrix = confusion_matrix(y_np, y_pred_np)
        loss_avg = loss_avg / len(dataloader)
        acc_avg = acc_avg / len(dataloader)
        prec_avg = prec_avg / len(dataloader)
        f1_score_avg = f1_score_avg / len(dataloader)
        recall_avg = recall_avg / len(dataloader)
        present_labels = np.unique(np.concatenate([y_np, y_pred_np]))
        return (
            loss_avg,
            acc_avg.item(),
            prec_avg.item(),
            f1_score_avg.item(),
            recall_avg.item(),
            report,
            conf_matrix,
            present_labels,
        )


def valid_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device,
):
    """
    Performs a single validation step of the model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): Dataloader for the validation data.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        device: The device to perform computations on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - loss_avg (float): Average loss for the validation set.
            - acc_avg (float): Average accuracy for the validation set.
            - prec_avg (float): Average precision for the validation set.
            - f1_score_avg (float): Average F1 score for the validation set.
            - recall_avg (float): Average recall for the validation set.
            - report (str): Classification report as a string.
            - conf_matrix (ndarray): Confusion matrix.
            - present_labels (ndarray): Unique labels present in the dataset.
    """
    model.eval()
    model.to(device)
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            logging.info(f"Batch size valid: {X.size(0)}")
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            if isinstance(test_pred_logits, InceptionOutputs):
                test_pred_logits = test_pred_logits.logits
            loss = loss_fn(test_pred_logits, y)
            loss_avg = loss_avg + loss.item()
            # print(f"Loss {loss_avg}")
            y_true.append(y)
            y_pred_class = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
            y_pred_class_list.append(
                torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
            )
            acc = accuracy(y_pred_class.to(device), y.to(device))
            recall_result = recall(y_pred_class.to(device), y.to(device))
            prec = precision(y_pred_class.to(device), y.to(device))
            f1_score_result = f1_score(y_pred_class.to(device), y.to(device))

            acc_avg = acc_avg + acc
            prec_avg = prec_avg + prec
            recall_avg = recall_avg + recall_result
            f1_score_avg = f1_score_avg + f1_score_result
            logging.info(f"Valid {y_pred_class,y}")
        y_true_tensor = torch.cat(y_true).cpu()
        y_pred_class_tensor = torch.cat(y_pred_class_list).cpu()
        y_np = y_true_tensor.cpu().numpy()
        y_pred_np = y_pred_class_tensor.cpu().numpy()
        report = classification_report(y_np, y_pred_np, zero_division=0)
        conf_matrix = confusion_matrix(y_np, y_pred_np)
        loss_avg = loss_avg / len(dataloader)
        acc_avg = acc_avg / len(dataloader)
        prec_avg = prec_avg / len(dataloader)
        f1_score_avg = f1_score_avg / len(dataloader)
        recall_avg = recall_avg / len(dataloader)
        present_labels = np.unique(np.concatenate([y_np, y_pred_np]))
        return (
            loss_avg,
            acc_avg.item(),
            prec_avg.item(),
            f1_score_avg.item(),
            recall_avg.item(),
            report,
            conf_matrix,
            present_labels,
        )


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    path_experiments: str,
    early_stopping,
    device,
    fine_tunning,
    freeze_all,
    valid,
):
    """
    Trains a PyTorch model over a specified number of epochs, validating it at each epoch
    if validation data is provided. The function logs training and validation metrics,
    generates confusion matrices, and saves results to specified paths.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        loss_fn (torch.nn.Module): Loss function used to calculate the training and validation loss.
        epochs (int): Number of epochs to train the model.
        path_experiments (str): Directory path to save experiment results and metrics.
        early_stopping: Early stopping mechanism to prevent overfitting.
        device: Device to run the model (CPU or GPU).
        fine_tunning: Indicates if fine-tuning should be applied.
        freeze_all: Boolean indicating whether to freeze model layers during training.
        valid: Boolean indicating whether to validate the model during training.

    Returns:
        tuple: Contains three dictionaries with training, validation, and test metrics.
            - results_train (dict): Metrics for the training phase including loss, accuracy,
              precision, recall, and F1 score.
            - results_valid (dict): Metrics for the validation phase including loss, accuracy,
              precision, recall, and F1 score.
            - results_test (dict): Metrics for the test phase including loss, accuracy,
              precision, recall, and F1 score.
    """

    results_train = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1_score": [],
        "training_time": 0,
    }
    results_valid = {
        "valid_loss": [],
        "valid_acc": [],
        "valid_precision": [],
        "valid_recall": [],
        "valid_f1_score": [],
    }

    results_test = {
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1_score": [],
        "testing_time": [],
    }

    # scheduler = lr_scheduler.LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.3, total_iters=10
    # )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )
    if not freeze_all:
        for epoch in tqdm(range(epochs)):
            (
                train_loss,
                train_acc,
                train_precision,
                train_f1_score,
                train_recall,
                train_report,
                train_conf_matrix,
                present_labels_train,
            ) = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            if valid:
                (
                    valid_loss,
                    valid_acc,
                    valid_precision,
                    valid_f1_score,
                    valid_recall,
                    valid_report,
                    valid_conf_matrix,
                    present_labels_valid,
                ) = valid_step(
                    model=model,
                    dataloader=valid_dataloader,
                    loss_fn=loss_fn,
                    device=device,
                )
                os.makedirs(
                    f"{path_experiments}/valid/{epoch}",
                    exist_ok=True,
                )
                with open(
                    f"{path_experiments}/valid/{epoch}/classification_report.txt",
                    "w",
                ) as f:
                    f.write(valid_report)
                conf_matrix_df_valid = pd.DataFrame(
                    valid_conf_matrix,
                    index=present_labels_valid,
                    columns=present_labels_valid,
                )
                fig = px.imshow(conf_matrix_df_valid)
                fig.write_image(
                    f"{path_experiments}/valid/{epoch}/confusion_matrix.png"
                )
                conf_matrix_df_valid.to_csv(
                    f"{path_experiments}/valid/{epoch}/classification_report.csv"
                )
                results_valid["valid_loss"].append(
                    valid_loss.item()
                    if isinstance(valid_loss, torch.Tensor)
                    else valid_loss
                )
                results_valid["valid_acc"].append(
                    valid_acc.item()
                    if isinstance(valid_acc, torch.Tensor)
                    else valid_acc
                )
                results_valid["valid_precision"].append(
                    valid_precision.item()
                    if isinstance(valid_precision, torch.Tensor)
                    else valid_precision
                )
                results_valid["valid_recall"].append(
                    valid_recall.item()
                    if isinstance(valid_recall, torch.Tensor)
                    else valid_recall
                )
                results_valid["valid_f1_score"].append(
                    valid_f1_score.item()
                    if isinstance(valid_f1_score, torch.Tensor)
                    else valid_f1_score
                )

            if valid:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"valid_loss: {valid_loss:.4f} | "
                    f"valid_acc: {valid_acc:.4f}"
                )
            else:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"train_recall: {train_recall:.4f} | "
                    f"train_f1: {train_f1_score:.4f} | "
                    f"train_precision: {train_precision:.4f} | "
                )
            os.makedirs(
                f"{path_experiments}/train/{epoch}",
                exist_ok=True,
            )
            with open(
                f"{path_experiments}/train/{epoch}/classification_report.txt",
                "w",
            ) as f:
                f.write(train_report)
            conf_matrix_df_train = pd.DataFrame(
                train_conf_matrix,
                index=present_labels_train,
                columns=present_labels_train,
            )
            fig = px.imshow(conf_matrix_df_train)
            fig.write_image(f"{path_experiments}/train/{epoch}/confusion_matrix.png")
            conf_matrix_df_train.to_csv(
                f"{path_experiments}/train/{epoch}/confusion_matrix.csv"
            )

            results_train["train_loss"].append(
                train_loss.item()
                if isinstance(train_loss, torch.Tensor)
                else train_loss
            )
            results_train["epoch"].append(epoch)
            results_train["train_acc"].append(
                train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
            )
            results_train["train_precision"].append(
                train_precision.item()
                if isinstance(train_precision, torch.Tensor)
                else train_precision
            )
            results_train["train_recall"].append(
                train_recall.item()
                if isinstance(train_recall, torch.Tensor)
                else train_recall
            )
            results_train["train_f1_score"].append(
                train_f1_score.item()
                if isinstance(train_f1_score, torch.Tensor)
                else train_f1_score
            )

            # scheduler.step(valid_loss)
            # logging.info(f"Lr {scheduler.get_last_lr()}")
            # if valid:
            #     early_stopping(valid_loss, model)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break
            torch.cuda.empty_cache()
            logging.info(f"Epoch :{epoch}")
    (
        test_loss,
        test_acc,
        test_precision,
        test_f1_score,
        test_recall,
        test_report,
        test_conf_matrix,
        present_labels_test,
    ) = test_step(
        model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )

    os.makedirs(
        f"{path_experiments}/test/",
        exist_ok=True,
    )
    with open(
        f"{path_experiments}/test/classification_report.txt",
        "w",
    ) as f:
        f.write(test_report)
    conf_matrix_df = pd.DataFrame(
        test_conf_matrix, index=present_labels_test, columns=present_labels_test
    )
    fig = px.imshow(conf_matrix_df)
    fig.write_image(f"{path_experiments}/test/confusion_matrix.png")
    conf_matrix_df.to_csv(f"{path_experiments}/test/confusion_matrix.csv")

    results_test["test_loss"].append(
        test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
    )
    results_test["test_acc"].append(
        test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
    )
    results_test["test_precision"].append(
        test_precision.item()
        if isinstance(test_precision, torch.Tensor)
        else test_precision
    )
    results_test["test_recall"].append(
        test_recall.item() if isinstance(test_recall, torch.Tensor) else test_recall
    )
    results_test["test_f1_score"].append(
        test_f1_score.item()
        if isinstance(test_f1_score, torch.Tensor)
        else test_f1_score
    )

    # with open(f"{path_experiments}/options.txt", "w") as f:
    #     f.write(
    #         f"fine-tunning:{fine_tunning}\n\
    #             epochs : {epoch}"
    #     )

    return results_train, results_valid, results_test
