import logging
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
)
from timm import create_model
from torch import nn
from torch.nn.functional import relu
from torchmetrics.classification import ROC, Accuracy, F1Score, Precision, Recall
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from utils import visaulization_models

logging.basicConfig(
    level=logging.INFO,
    filename="multimodal.log",
    filemode="w",
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

from torchvision.models import densenet161  # Dodaj ten import


class MultiModalClassifier(nn.Module):
    """
    A multi-modal classifier that combines features from images and text captions for classification tasks.

    Attributes:
        device (torch.device): The device on which the model will run (CPU or GPU).
        images_encoder (nn.Module): A feature extractor for images using a Vision Transformer model.
        images_features (nn.Linear): A linear layer to transform the image features.
        caption_encoder (AutoTokenizer): A tokenizer for processing text captions.
        caption_model (AutoModel): A transformer model for obtaining text features.
        fc1 (nn.Linear): A fully connected layer that combines image and text features.
        dropout (nn.Dropout): A dropout layer for regularization.
        fc2 (nn.Linear): The final fully connected layer for classification.

    Args:
        num_classes (int): The number of output classes.
        device (torch.device): The device to run the model on.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(self, num_classes, device, dropout):

        self.device = device

        super().__init__()
        # ekstrakcja cech z obrazu
        self.images_encoder = create_model(
            "vit_small_r26_s32_224", pretrained=True, num_classes=0, global_pool=""
        )
        self.images_features = nn.Linear(self.images_encoder.num_features, 512)

        # tekst
        self.caption_encoder = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-small"
        )  # reprezentacja numeryczna słów
        self.caption_model = AutoModel.from_pretrained(
            "microsoft/deberta-v3-small"
        )  # przetwarza sekwencje tokenów i zwraca wektor cech dla każdego tokena
        self.caption_model.resize_token_embeddings(
            len(self.caption_encoder)
        )  # ustawia nowe wymiary macierzy embeddingów w zależności od wielkości tokenizera

        for param in self.caption_model.parameters():
            param.requires_grad = False

        for param in self.images_encoder.parameters():
            param.requires_grad = False

        # zdjęcia + tekst
        self.fc1 = nn.Linear(
            512 + self.caption_model.config.hidden_size, 512
        )  # łączenie cech z obrazów i tesktów
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)  # finalna klasyfikacja

    def forward(self, image, caption):
        """
        Forward pass for the multi-modal classifier.

        Args:
            image (torch.Tensor): The input image tensor.
            caption (torch.Tensor): The input caption tensor.

        Returns:
            torch.Tensor: The logits for each class based on the combined features from images and captions.
        """
        # zdjęcia
        image_features = self.images_encoder(image)
        image_features = self.images_features(image_features)
        image_features = relu(image_features)
        image_features = self.dropout(image_features)

        image_features = image_features.mean(dim=1)

        # tekst
        caption = [
            self.caption_encoder.convert_ids_to_tokens(caption[i].tolist())
            for i in range(len(caption))
        ]
        caption = [" ".join(c) for c in caption]

        encoding = self.caption_encoder(
            caption, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        caption_features = self.caption_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        caption_features = caption_features.mean(dim=1)
        # Połączenie cech obrazu i tekstu
        # caption_features = caption_features.mean(dim=1)
        # print(image_features.shape)
        # print(caption_features.shape)
        features = torch.cat((image_features, caption_features), dim=-1)
        features = self.fc1(features)
        features = self.dropout(features)
        logits = self.fc2(features)
        return logits


def train_step(model, dataloader, optimizer, criterion):
    """
    Executes a single training step.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        criterion (nn.Module): The loss function to calculate the loss.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, F1 score, recall,
               classification report, confusion matrix, and present labels for the current batch.
    """
    model.train()
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    for batch_idx, (images, captions, labels) in enumerate(dataloader):
        # move data to GPU if available
        # print(images, captions, labels)
        # print(type(images), type(captions), type(labels))
        images = images.to(device)
        captions = captions.to(device)
        y = labels.to(device)
        y_true.append(y)

        optimizer.zero_grad()

        # forward pass
        y_pred = model(images, captions)

        # visaulization_models("multimodal", model, images, captions)
        # return
        # calculate loss
        loss = criterion(y_pred, y)
        loss_avg += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_pred_class_list.append(y_pred_class)
        logging.info(f"Test y_pred{y_pred_class} y {y}")
        acc = accuracy(y_pred_class.to(device), y.to(device))
        recall_result = recall(y_pred_class.to(device), y.to(device))
        prec = precision(y_pred_class.to(device), y.to(device))
        f1_score_result = f1_score(y_pred_class.to(device), y.to(device))
        acc_avg = acc_avg + acc
        prec_avg = prec_avg + prec
        recall_avg = recall_avg + recall_result
        f1_score_avg = f1_score_avg + f1_score_result

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


def valid_step(model, dataloader, criterion):
    """
    Executes a validation step.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The DataLoader providing the validation data.
        criterion (nn.Module): The loss function to calculate the loss.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, F1 score, recall,
               classification report, confusion matrix, and present labels for the validation set.
    """
    model.eval()
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    with torch.no_grad():

        for batch_idx, (images, captions, labels) in enumerate(dataloader):

            images = images.to(device)
            captions = captions.to(device)
            y = labels.to(device)
            y_true.append(y)
            y_pred = model(images, captions)

            loss = criterion(y_pred, y)

            loss_avg = loss_avg + loss.item()
            # print(f"Loss {loss_avg}")
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            logging.info(f"Test y_pred{y_pred_class} y {y}")
            acc = accuracy(y_pred_class.to(device), y.to(device))
            recall_result = recall(y_pred_class.to(device), y.to(device))
            prec = precision(y_pred_class.to(device), y.to(device))
            f1_score_result = f1_score(y_pred_class.to(device), y.to(device))

            acc_avg = acc_avg + acc
            prec_avg = prec_avg + prec
            recall_avg = recall_avg + recall_result
            f1_score_avg = f1_score_avg + f1_score_result
            y_pred_class_list.append(y_pred_class)
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


def test_step(model, dataloader, criterion):
    """
    Executes a testing step.

    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): The DataLoader providing the test data.
        criterion (nn.Module): The loss function to calculate the loss.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, F1 score, recall,
               classification report, confusion matrix, and present labels for the test set.
    """
    model.eval()
    loss_avg = 0
    acc_avg = 0
    prec_avg = 0
    recall_avg = 0
    f1_score_avg = 0
    y_pred_class_list = []
    y_true = []

    with torch.no_grad():

        for batch_idx, (images, captions, labels) in enumerate(dataloader):

            images = images.to(device)
            captions = captions.to(device)
            y = labels.to(device)
            y_true.append(y)
            y_pred = model(images, captions)

            loss = criterion(y_pred, y)

            loss_avg = loss_avg + loss.item()
            # print(f"Loss {loss_avg}")
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            logging.info(f"Test y_pred{y_pred_class} y {y}")
            acc = accuracy(y_pred_class.to(device), y.to(device))
            recall_result = recall(y_pred_class.to(device), y.to(device))
            prec = precision(y_pred_class.to(device), y.to(device))
            f1_score_result = f1_score(y_pred_class.to(device), y.to(device))

            acc_avg = acc_avg + acc
            prec_avg = prec_avg + prec
            recall_avg = recall_avg + recall_result
            f1_score_avg = f1_score_avg + f1_score_result
            y_pred_class_list.append(y_pred_class)
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


def train_multimodal(
    NUM_EPOCHS,
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    early_stopping,
    criterion,
    device,
    path_experiments,
    layers_to_unfreeze,
    use_scheduler,
):
    """
    Train a multimodal model using the provided dataloaders, optimizer, and criterion.

    Args:
        NUM_EPOCHS (int): The number of epochs for training.
        train_dataloader (DataLoader): Dataloader for the training dataset.
        val_dataloader (DataLoader): Dataloader for the validation dataset.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        early_stopping (EarlyStopping): Early stopping mechanism to prevent overfitting.
        criterion (callable): Loss function to optimize.
        device (torch.device): Device to run the model on (CPU or GPU).
        path_experiments (str): Path to save training and validation results.
        layers_to_unfreeze (list): List of layers to unfreeze during training (if applicable).
        use_scheduler (bool): Flag indicating whether to use a learning rate scheduler.

    Returns:
        tuple: Three dictionaries containing training, validation, and test results.
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )
    # if  layers_to_unfreeze:
    for epoch in tqdm(range(NUM_EPOCHS)):
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
            criterion=criterion,
            optimizer=optimizer,
        )
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
            dataloader=val_dataloader,
            criterion=criterion,
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
        fig.write_image(f"{path_experiments}/valid/{epoch}/confusion_matrix.png")
        conf_matrix_df_valid.to_csv(
            f"{path_experiments}/valid/{epoch}/classification_report.csv"
        )
        results_valid["valid_loss"].append(
            valid_loss.item() if isinstance(valid_loss, torch.Tensor) else valid_loss
        )
        results_valid["valid_acc"].append(
            valid_acc.item() if isinstance(valid_acc, torch.Tensor) else valid_acc
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

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
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
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
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

        if use_scheduler:
            scheduler.step(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        torch.cuda.empty_cache()
        logging.info(f"Epoch :{epoch}")
        torch.cuda.empty_cache()
    (
        test_loss,
        test_acc,
        test_precision,
        test_f1_score,
        test_recall,
        test_report,
        test_conf_matrix,
        present_labels_test,
    ) = test_step(model=model, dataloader=val_dataloader, criterion=criterion)

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
