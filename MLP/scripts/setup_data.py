import datetime
import glob
import logging
import math
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import wandb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from torch import nn
from torch.nn import CrossEntropyLoss, Linear, Sequential, Tanh
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import ROC, Accuracy, F1Score, Precision, Recall
from torchsummary import summary

path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"
paths = [path_train, path_test]
logging.basicConfig(
    level=logging.INFO,
    filename="MLP.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)
columns_names = [
    "Wiek",
    "Płeć",
    "Na leku tyroksynowym",
    "Zapytanie o tyroksynę",
    "Na lekach przeciwtarczycowych",
    "Chory",
    "W ciąży",
    "Operacja tarczycy",
    "Leczenie I131",
    "Zapytanie o niedoczynność tarczycy",
    "Zapytanie o nadczynność tarczycy",
    "Lit",
    "Wole",
    "Guz",
    "Niedoczynność przysadki",
    "Psych",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "Klasa",
]


def permute_feature(model, X, y, feature_idx):
    """
    Permutates a specified feature in the dataset and returns predicted probabilities.

    Parameters:
    - model: Model used for predictions.
    - X (ndarray): Input features.
    - y (ndarray): Target values.
    - feature_idx (int): Index of the feature to permute.

    Returns:
    - probas (ndarray): Predicted probabilities after feature permutation.
    """
    X_permuted = X.copy()
    np.random.shuffle(X_permuted[:, feature_idx])
    probas = model.predict_proba(X_permuted)
    return probas


def plot_feature_importance_rfe_plotly(rfe, df_train_data, importances):
    """
    Plots feature importance using Recursive Feature Elimination (RFE) with Plotly.

    Parameters:
    - rfe: Trained RFE object.
    - df_train_data (DataFrame): Training data.
    - importances (ndarray): Importance scores for features.

    Returns:
    - fig: Plotly bar plot figure of sorted feature importances.
    """
    # feature_ranking = rfe.ranking_

    feature_names = df_train_data.columns

    # Posortowanie cech na podstawie rankingu
    sorted_idx = np.argsort(importances)[::-1]
    print(
        f"feature_names: {feature_names},importances: {importances}, sorted_idx :{sorted_idx } "
    )
    sorted_features = feature_names[sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig = px.bar(
        x=sorted_importance,
        y=sorted_features,
        orientation="h",
        labels={"x": "Ranking", "y": "Cechy"},
        title="Ważność cech - RFE",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_feature_importance_lasso_plotly(lasso_model, df_train_data):
    """
    Plots feature importance based on LASSO coefficients with Plotly.

    Parameters:
    - lasso_model: Trained LASSO regression model.
    - df_train_data (DataFrame): Training data.

    Returns:
    - fig: Plotly bar plot figure of sorted LASSO feature importances.
    """
    feature_importance = np.abs(lasso_model.coef_)
    print(f" lass feature_importance: {feature_importance}")
    feature_names = df_train_data.columns  # Ostatnia kolumna to target ('Klasa')

    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    fig = px.bar(
        x=sorted_importance,
        y=sorted_features,
        orientation="h",
        labels={"x": "Wspołczynniki LASSO", "y": "Cechy"},
        title="Ważność cech - LASSO",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_feature_importance_mlp_fspp_rfe_plotly(
    feature_importances, remaining_features
):
    """
    Plots feature importance for MLP model with FSPP RFE using Plotly.

    Parameters:
    - feature_importances (list): Importance scores for features.
    - remaining_features (list): Names of features after selection.

    Returns:
    - fig: Plotly bar plot figure of sorted MLP FSPP RFE feature importances.
    """
    print("Impotnaces", feature_importances, "Remianing", remaining_features)
    print(len(feature_importances), len(remaining_features))
    feature_importances = np.array(feature_importances)
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(remaining_features)[sorted_idx]
    sorted_importance = feature_importances[sorted_idx]
    print("Sorted indices:", sorted_idx)
    print("Type of sorted_idx:", type(sorted_idx), "Shape:", sorted_idx.shape)

    fig = px.bar(
        x=sorted_importance,
        y=sorted_features,
        orientation="h",
        labels={"x": "Współczynniki MLP FSPP RFE", "y": "Cechy"},
        title="Ważność cech - MLP FSPP RFE",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def compute_feature_importance(model, X, y):
    """
    Computes feature importance based on model predictions and permutation.

    Parameters:
    - model: Model used for predictions.
    - X (ndarray): Input features.
    - y (ndarray): Target values.

    Returns:
    - feature_importances (ndarray): Array of computed importance scores for each feature.
    """
    n_features = X.shape[1]
    original_proba = model.predict_proba(X)
    feature_importances = []
    for feature_idx in range(n_features):
        permuted_proba = permute_feature(model, X, y, feature_idx)
        feature_importances.append(
            np.abs(original_proba - permuted_proba).mean(axis=0).sum()
        )
        print(
            f"idx: {feature_idx} difference : {np.abs(original_proba - permuted_proba).mean(axis=0).sum()}"
        )
    return np.array(feature_importances)


def save_plot_with_counter_plotly(fig, counter):
    """
    Saves Plotly figure with a specified counter in the filename.

    Parameters:
    - fig: Plotly figure to save.
    - counter (int): Counter used in filename.

    Returns:
    - None
    """
    filename = f"feature_importance_experiment_{counter}.png"
    fig.write_image(filename)
    print(f"Saved plot as {filename}")


def plot_feature_importance_corr_plotly(df_train_data, target_column="Klasa"):
    """
    Plots feature correlation with target column using Plotly.

    Parameters:
    - df_train_data (DataFrame): Training data.
    - target_column (str): Target column name.

    Returns:
    - fig: Plotly bar plot figure of feature correlations with the target.
    """
    print(f"Columns {df_train_data.columns}")
    correlation_matrix = df_train_data.corr()
    print(f"correlation_matrix: {correlation_matrix.columns}")
    correlations = correlation_matrix[target_column].drop(target_column)

    sorted_idx = np.argsort(np.abs(correlations))[::-1]
    sorted_features = correlations.index[sorted_idx]
    sorted_correlations = correlations.values[sorted_idx]

    fig = px.bar(
        x=sorted_correlations,
        y=sorted_features,
        orientation="h",
        labels={"x": "Korelacja z Klasą", "y": "Cechy"},
        title="Ważomość cech - Korelacja",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def read_csv_data(paths: list):
    """
    Reads CSV files, removes columns with all NaN values, and sets column names.

    Parameters:
    - paths (list): List of file paths to CSV files.

    Returns:
    - df_list (list): List of DataFrames from the CSV files.
    """

    df_list = []
    for path in paths:
        df = pd.read_csv(path, sep=" ")
        df = df.dropna(axis=1, how="all")
        df.columns = columns_names
        df_list.append(df)
    return df_list


def feature_selection_lasso(X, y, alpha, num):
    """
    Selects features using LASSO regularization.

    Parameters:
    - X (ndarray): Input features.
    - y (ndarray): Target values.
    - alpha (float): Regularization strength.
    - num (int): Number of features to select.

    Returns:
    - selected_features (ndarray): Indices of selected features.
    - lasso: Trained LASSO model.
    """
    lasso = Lasso(alpha=alpha).fit(X, y)
    selected_features = np.argsort(np.abs(lasso.coef_))[:num]
    return selected_features, lasso


def read_data_as_arrays():
    """
    Reads training and testing data as arrays.

    Returns:
    - train_data (ndarray): Training data array.
    - test_data (ndarray): Testing data array.
    """
    path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
    path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"
    paths = [path_train, path_test]
    df = read_csv_data(paths)
    train_data = df[0].values
    test_data = df[1].values
    return train_data, test_data


def LASSO(X_train, y_train, model, num):
    """
    Selects features using LASSO and returns the selected features and trained model.

    Parameters:
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training labels.
    - model: Model to fit the LASSO.
    - num (int): Number of features to select.

    Returns:
    - selected_features (ndarray): Selected feature indices.
    - lasso: Trained LASSO model.
    """
    alpha = 0
    selected_features, lasso = feature_selection_lasso(X_train, y_train, alpha, num)
    return selected_features, lasso


def RFE_method(X_train, y_train, X_test, df_train_data, model, num: int):
    """
    Performs feature selection using Recursive Feature Elimination (RFE).

    Parameters:
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training labels.
    - X_test (ndarray): Testing features.
    - df_train_data (DataFrame): Data containing feature names.
    - model: Model to be used in RFE.
    - num (int): Number of features to select.

    Returns:
    - selected_features (Index): Selected features from DataFrame.
    - rfe: Trained RFE model.
    - importances (ndarray or None): Importance scores of selected features.
    """
    print(f"RFE num {num} {X_train.shape[1]}")
    rfe = RFE(estimator=model, n_features_to_select=num)
    X_train_selected = rfe.fit_transform(X_train, y_train)
    # selected_features = rfe.support_  # Boolean mask of selected features
    selected_features_mask = rfe.support_
    model.fit(X_train_selected, y_train)
    X_test_selected = rfe.transform(X_test)
    df_train_data_copy = df_train_data.copy()
    df_train_data_copy.drop("Klasa", axis=1, inplace=True)
    selected_features = df_train_data_copy.columns[selected_features_mask]
    logging.info(f"Wybrane cechy: {selected_features}")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_[
            selected_features_mask
        ]  # Dla modeli z feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(
            model.coef_[0][selected_features_mask]
        )  # Dla modeli liniowych
    else:
        importances = None  # Brak obsługi dla tego modelu

    return selected_features, rfe, importances


def balancing_dataset(option: str, df: pd.DataFrame):
    """
    Balances the dataset using specified resampling techniques.

    Parameters:
        option (str): The resampling technique to apply. Options include "naive random over-sampling", "smote",
                      "adasyn", "random under-sampling", "smotetomek", and "smoteenn".
        df (pd.DataFrame): A DataFrame containing the dataset, with a target column named 'Klasa'.

    Returns:
        list: A list of resampled DataFrames, each balanced according to the chosen resampling technique.

    For each DataFrame in df, the function applies the specified resampling technique to balance the classes
    in the target variable 'Klasa' and logs the changes in class distribution.
    """
    resampled_data = []
    algorithm = ""
    if option.lower() == "naive random over-sampling":
        algorithm = RandomOverSampler(random_state=0)
    elif option.lower() == "smote":
        algorithm = SMOTE(random_state=0)
    elif option.lower() == "adasyn":
        algorithm = ADASYN(random_state=0)
    elif option.lower() == "random under-sampling":
        algorithm = RandomUnderSampler(random_state=0)
    elif option.lower() == "smotetomek":
        algorithm = SMOTETomek(random_state=0)
    elif option.lower() == "smoteenn":
        algorithm = SMOTEENN(random_state=0)
    for data in df:
        X_resampled, y_resampled = algorithm.fit_resample(
            data.drop("Klasa", axis=1), data["Klasa"]
        )
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
        logging.info(f'Classes {data["Klasa"].value_counts()}')
        logging.info(f'Classes after resampled {df_resampled["Klasa"].value_counts()}')
        resampled_data.append(df_resampled)
    return resampled_data


def select_features(feature_selection_method, model, df_list, num):
    """
    Selects features from each DataFrame based on the specified feature selection method.

    Parameters:
        feature_selection_method (str): The feature selection method to apply. Options include "Corr", "RFE", "LASSO",
                                        and "MLP_FSPP_RFE".
        model: A machine learning model used for feature selection methods that require it.
        df_list (list): A list of DataFrames [training_data, test_data] to perform feature selection on.
        num (int): The number of top features to retain.

    Returns:
        list: A list of DataFrames containing the selected features for both training and test sets.

    This function performs feature selection on the provided DataFrames using correlation-based selection,
    Recursive Feature Elimination (RFE), Lasso regression, or MLP-based feature selection and plots the
    importance of selected features.
    """
    df_train_data, df_test_data = df_list[0], df_list[1]
    X_train = df_train_data.values[:, :-1]
    y_train = df_train_data.values[:, -1]
    X_test = df_test_data.values[:, :-1]
    y_test = df_test_data.values[:, -1]
    if feature_selection_method == "Corr":

        dropped_labels = []
        df_corr = df_train_data.corr()
        no_dropped_labels = (
            df_corr["Klasa"].sort_values(ascending=False)[1 : num + 1].index
        )
        logging.info(f"Dropped lables {dropped_labels}")
        fig = plot_feature_importance_corr_plotly(df_train_data, target_column="Klasa")
        save_plot_with_counter_plotly(fig, "Corr")
        df_train_data = df_train_data[no_dropped_labels]
        df_test_data = df_test_data[no_dropped_labels]
        plot_feature_importance_corr_plotly(df_train_data, target_column="Klasa")

    elif feature_selection_method == "RFE":
        selected_features, rfe, importances = RFE_method(
            X_train, y_train, X_test, df_train_data, model, num
        )
        df_test_data = df_test_data.loc[:, selected_features]
        df_train_data = df_train_data.loc[:, selected_features]
        logging.info(
            f"Shape {df_train_data.shape} {df_test_data.shape} X_train {df_train_data.columns}"
        )
        fig = plot_feature_importance_rfe_plotly(rfe, df_train_data, importances)
        save_plot_with_counter_plotly(fig, "RFE")

    elif feature_selection_method == "LASSO":
        selected_features, lasso = LASSO(X_train, y_train, model, num)
        df_test_data = df_test_data.iloc[:, selected_features]
        df_train_data = df_train_data.iloc[:, selected_features]
        print(
            f"Shape {df_train_data.shape} {df_test_data.shape},num {num} selected_features {selected_features},df_list[0] {df_list[0].columns}"
        )
        fig = plot_feature_importance_lasso_plotly(lasso, df_train_data)
        save_plot_with_counter_plotly(fig, "LASSO")

    elif feature_selection_method == "MLP_FSPP_RFE":  # Pamietaj, przerobić model
        ranking_list = []
        feature_importance_list = []
        remaining_features = list(df_list[0].columns[:-1])
        i = 0
        while remaining_features:
            print(
                f"remaining_features {remaining_features}, i: {i}, feature_num == 1 {len(remaining_features) == 1}"
            )
            if len(remaining_features) == 1:
                print(f"Last feture {remaining_features[0]}")
                ranking_list.append(remaining_features[0])
                model.fit(X_train, y_train, selected_features=None)
                feature_importances = compute_feature_importance(model, X_test, y_test)
                ranked_features = np.argsort(feature_importances)[::-1]
                feature_importance_list.append(feature_importances[ranked_features[-1]])
                remaining_features = []
            else:
                model.fit(X_train, y_train, selected_features=None)
                feature_importances = compute_feature_importance(model, X_test, y_test)
                ranked_features = np.argsort(feature_importances)[::-1]
                most_important_feature = remaining_features[ranked_features[0]]
                feature_importance_list.append(feature_importances[ranked_features[-1]])
                least_important_feature = remaining_features[ranked_features[-1]]
                remaining_features.remove(least_important_feature)
                ranking_list.append(least_important_feature)
                X_train = np.delete(X_train, ranked_features[-1], axis=1)
                logging.info(
                    f"Usunięto cechę: {least_important_feature}, pozostałe cechy: {remaining_features}"
                )
                X_test = np.delete(X_test, ranked_features[-1], axis=1)
                i = +1
        df_test_data = df_test_data.loc[
            :, ranking_list[df_list[0].shape[1] - num - 1 :]
        ]
        df_train_data = df_train_data.loc[
            :, ranking_list[df_list[0].shape[1] - num - 1 :]
        ]
        fig = plot_feature_importance_mlp_fspp_rfe_plotly(
            feature_importance_list, ranking_list
        )
        save_plot_with_counter_plotly(fig, "MLP_FSPP_RFE")
    return [
        pd.concat([df_train_data, df_list[0]["Klasa"]], axis=1),
        pd.concat([df_test_data, df_list[1]["Klasa"]], axis=1),
    ]


def preprocessing_data(df_list: list):
    """
    Preprocesses each DataFrame by standardizing numeric features.

    Parameters:
        df_list (list): A list of DataFrame objects.

    Returns:
        list: A list of preprocessed DataFrame objects with standardized numeric features.

    For each DataFrame in df_list, this function standardizes numeric features using StandardScaler.
    """
    df_preprocessed_list = []
    for df in df_list:
        scaler = StandardScaler()
        numeric_data = df.select_dtypes(include="float64")
        df[numeric_data.columns] = scaler.fit_transform(numeric_data)
        df_preprocessed_list.append(df)
    return df_preprocessed_list


def convert_to_tensors(df_list):
    """
    Converts each DataFrame in the list to PyTorch tensors.

    Parameters:
        df_list (list): A list of DataFrame objects.

    Returns:
        list: A list containing tuples of PyTorch tensors (X, y).

    For each DataFrame in df_list, this function converts features and target variable to PyTorch tensors.
    """
    tensors = []
    for df in df_list:
        print(f"Df {df.shape}")
        print(f"df column {df.columns}")
        X = torch.tensor(df.drop("Klasa", axis=1).values, dtype=torch.double)
        y = torch.tensor(df["Klasa"].values)
        print(f"X shape {X.shape} y shape {y.shape}")
        tensors.append([X, y])
    print("tesnor shape", tensors[0][0].shape, tensors[0][1].shape)
    return tensors


class ThyroidGarvanDataset(Dataset):
    """
    PyTorch dataset class for the Thyroid Garvan dataset.

    Parameters:
        X (Tensor): Input features.
        y (Tensor): Target labels.

    Returns:
        Tuple: A tuple containing input features and target labels.

    This class creates a PyTorch dataset from the Thyroid Garvan dataset.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.targets = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataset(data: list):
    """
    Creates a list of ThyroidGarvanDataset objects from a list of data.

    Parameters:
        data (list): A list of tuples, where each tuple contains input features and target labels.

    Returns:
        list: A list containing ThyroidGarvanDataset objects.

    This function creates a list of PyTorch datasets from a list of data, where each dataset is created using the
    ThyroidGarvanDataset class.
    """
    datasets = []
    for itm in data:
        dataset = ThyroidGarvanDataset(itm[0], itm[1])
        datasets.append(dataset)
    return datasets


def create_dataloder(batch_size: int, datasets: list):
    """
    Creates a list of PyTorch DataLoader objects from a list of datasets.

    Parameters:
        batch_size (int): The batch size for each DataLoader.
        datasets (list): A list of PyTorch Dataset objects.

    Returns:
        list: A list containing PyTorch DataLoader objects.

    This function creates a list of PyTorch DataLoader objects from a list of datasets.
    """
    dataloaders = []
    shuffle = True
    for idx, dataset in enumerate(datasets):
        class_counts = np.bincount(dataset.targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in dataset.targets]
        print(class_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(dataset), replacement=True
        )
        shuffle = False
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )
        logging.info(
            f"Dataloader {idx}: Size {len(dataset)}, Batch size {batch_size}, Number of batches {len(dataloader)}"
        )
        dataloaders.append(dataloader)
    return dataloaders
