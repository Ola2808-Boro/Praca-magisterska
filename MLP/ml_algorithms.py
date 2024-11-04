import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
from scripts.setup_data import preprocessing_data, read_csv_data
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=10000),
        {"C": [0.1, 1, 10], "solver": ["liblinear"]},
    ),
    "SVC": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.5],
            "max_depth": [3, 5, 7],
        },
    ),
    "AdaBoost": (
        AdaBoostClassifier(),
        {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1]},
    ),
}

features_selection_method = {
    "RFE": {
        "n_features_to_select": [1, 3, 5, 7, 9, 11, 12, 14, 16, 18],
    },
    "LASSO": {
        "n_features_to_select": [1, 3, 5, 7, 9, 11, 12, 14, 16, 18],
        #'alpha_values' : [0.001, 0.01, 0.1, 1, 10]
    },
}


def train_model(
    model_name,
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    n_features,
    results_dir,
    selected_features,
):
    """
    Trains a classification model using GridSearchCV and evaluates its performance.

    Args:
        model_name (str): The name of the model.
        model: The model object to train.
        params (dict): The parameters for GridSearchCV.
        X_train (ndarray): The training set features.
        y_train (ndarray): The training set labels.
        X_test (ndarray): The test set features.
        y_test (ndarray): The test set labels.
        n_features (int): The number of selected features.
        results_dir (str): The directory for saving results.
        selected_features (list): The list of selected features.

    Returns:
        None
    """
    print(f"Model name {model_name}")
    grid_search = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)

    start_train_time = time.time()
    grid_search.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    best_model = grid_search.best_estimator_
    print(f"Najlepsze parametry: {grid_search.best_params_}")

    start_pred_time = time.time()
    y_pred = best_model.predict(X_test)
    end_pred_time = time.time()
    prediction_time = end_pred_time - start_pred_time
    print(f" Pred: {y_pred} y true {y_test}")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_labels = np.unique(y_test)
    df_results = pd.DataFrame(
        {
            "Model name": [model_name],
            "Selection feature method": [results_dir],
            "Selected features": [selected_features],
            "Num features": [21 - n_features],
            "Accuracy": [acc],
            "Precision": [precision],
            "Recall": [recall],
            "F1-score": [f1],
            "Training time": [training_time],
            "Prediction time": [prediction_time],
        }
    )
    os.makedirs(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results/feature selections/{results_dir}/{model_name}",
        exist_ok=True,
    )
    header = (
        True
        if not os.path.exists(
            f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results/feature selections/{results_dir}/results_all.csv"
        )
        else False
    )
    df_results.to_csv(
        f"results/feature selections/{results_dir}/results_all.csv",
        mode="a",
        index=False,
        header=header,
    )
    save_results(
        model_name,
        report,
        conf_matrix,
        grid_search.best_params_,
        class_labels,
        training_time,
        prediction_time,
        n_features,
        results_dir,
        selected_features,
    )


def save_results(
    model_name,
    report,
    conf_matrix,
    best_params,
    class_labels,
    training_time,
    prediction_time,
    n_features,
    results_dir,
    selected_features,
):
    """
    Saves the model results to files.

    Args:
        model_name (str): The name of the model.
        report (str): The classification report.
        conf_matrix (ndarray): The confusion matrix.
        best_params (dict): The best parameters of the model.
        class_labels (list): The list of class labels.
        training_time (float): The training time of the model.
        prediction_time (float): The prediction time of the model.
        n_features (int): The number of selected features.
        results_dir (str): The directory for saving results.
        selected_features (list): The list of selected features.

    Returns:
        None
    """
    with open(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results/feature selections/{results_dir}/{model_name}/{model_name}_classification_report_{n_features}.txt",
        "w",
    ) as f:
        f.write(report)
    with open(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results/feature selections/{results_dir}/{model_name}/{model_name}_best_params_{n_features}.txt",
        "w",
    ) as f:
        f.write(str(best_params))
    class_labels = ["Klasa 0", "Klasa 1", "Klasa 2"]
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
    print(model_name, training_time, prediction_time)
    fig = px.imshow(conf_matrix_df)
    fig.write_image(
        f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/results/feature selections/{results_dir}/{model_name}/{model_name}_confusion_matrix_{n_features}.png"
    )
    conf_matrix_df.to_csv(
        f"results/feature selections/{results_dir}/{model_name}/{model_name}_confusion_matrix_{n_features}.csv"
    )


def main():
    """
    Main function to load data, preprocess it, train models using different feature selection methods,
    and evaluate their performance.

    This function performs the following steps:
    1. Loads training and test datasets from specified file paths.
    2. Preprocesses the data.
    3. Trains a Multi-layer Perceptron (MLP) classifier on the training data.
    4. Evaluates the MLP model's performance using accuracy, F1 score, precision, and recall metrics.
    5. Generates classification report and confusion matrix.
    6. Iterates through defined models and feature selection methods, applies LASSO or RFE for feature selection,
       and trains the models on the selected features.

    Returns:
        None
    """
    path_train = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data"
    path_test = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data"
    paths = [path_train, path_test]
    df_train_data, df_test_data = read_csv_data(paths)
    df_train_data, df_test_data = preprocessing_data([df_train_data, df_test_data])
    mlp = MLPClassifier(hidden_layer_sizes=(6,))
    X_train = df_train_data.values[:, :-1]
    y_train = df_train_data.values[:, -1]
    X_test = df_test_data.values[:, :-1]
    y_test = df_test_data.values[:, -1]
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"conf_matrix {conf_matrix}")
    print(f"Results {report}")
    for model_name, (model, params) in models.items():
        for select_features_method in list(features_selection_method.keys()):
            print(select_features_method)
            if select_features_method == "LASSO":
                for n_features in features_selection_method["LASSO"][
                    "n_features_to_select"
                ]:
                    X_train = df_train_data.values[:, :-1]
                    y_train = df_train_data.values[:, -1]
                    X_test = df_test_data.values[:, :-1]
                    y_test = df_test_data.values[:, -1]
                    lasso = Lasso(alpha=0.01)
                    lasso.fit(X_train, y_train)
                    # selected_features = np.argsort(np.abs(lasso.coef_))
                    # print(f'selected_features {selected_features}')
                    selected_features = np.argsort(np.abs(lasso.coef_))[:-n_features]
                    X_train = X_train[:, selected_features]
                    X_test = X_test[:, selected_features]
                    print(
                        f"X_train {X_train.shape}, X_test {X_test.shape}, n_features {n_features}"
                    )
                    results_dir = select_features_method
                    train_model(
                        model_name,
                        model,
                        params,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        n_features,
                        results_dir,
                        selected_features,
                    )

            else:
                for n_features in features_selection_method["RFE"][
                    "n_features_to_select"
                ]:
                    X_train = df_train_data.values[:, :-1]
                    print(df_train_data.columns, X_train.shape)
                    y_train = df_train_data.values[:, -1].astype(int)
                    X_test = df_test_data.values[:, :-1]
                    y_test = df_test_data.values[:, -1].astype(int)
                    rfe = RFE(estimator=model, n_features_to_select=n_features)
                    print("Before RFE:", X_train.shape, y_train.shape)
                    X_train = rfe.fit_transform(X_train, y_train)
                    print("After RFE:", X_train.shape)
                    X_test = rfe.transform(X_test)
                    df_train_data_copy = df_train_data.copy()
                    df_train_data_copy.drop("Klasa", axis=1, inplace=True)
                    selected_features = df_train_data_copy.columns[rfe.support_]
                    results_dir = select_features_method
                    train_model(
                        model_name,
                        model,
                        params,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        n_features,
                        results_dir,
                        list(selected_features),
                    )


main()
