import os
import time

import numpy as np
import pandas as pd
import plotly.express as px

# from scripts.setup_data import preprocessing_data, read_csv_data
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASIC_PATH = f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/results ml/"
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
    # 'LASSO':{
    #     'n_features_to_select':[1,3,5,7,9,11,12,14,16,18],
    #     #'alpha_values' : [0.001, 0.01, 0.1, 1, 10]
    # }
}

columns = {
    "age": "continuous",
    "sex": ["M", "F"],
    "on thyroxine": "boolean",
    "query on thyroxine": "boolean",
    "on antithyroid medication": "boolean",
    "sick": "boolean",
    "pregnant": "boolean",
    "thyroid surgery": "boolean",
    "I131 treatment": "boolean",
    "query hypothyroid": "boolean",
    "query hyperthyroid": "boolean",
    "lithium": "boolean",
    "goitre": "boolean",
    "tumor": "boolean",
    "hypopituitary": "boolean",
    "psych": "boolean",
    "TSH measured": "boolean",
    "TSH": "continuous",
    "T3 measured": "boolean",
    "T3": "continuous",
    "TT4 measured": "boolean",
    "TT4": "continuous",
    "T4U measured": "boolean",
    "T4U": "continuous",
    "FTI measured": "boolean",
    "FTI": "continuous",
    "TBG measured": "boolean",
    "TBG": "continuous",
    "referral source": ["WEST", "STMW", "SVHC", "SVI", "SVHD", "other"],
    "diagnosis": {
        "hyperthyroid conditions": {
            "A": "hyperthyroid",
            "B": "T3 toxic",
            "C": "toxic goitre",
            "D": "secondary toxic",
        },
        "hypothyroid conditions": {
            "E": "hypothyroid",
            "F": "primary hypothyroid",
            "G": "compensated hypothyroid",
            "H": "secondary hypothyroid",
        },
        "binding protein": {
            "I": "increased binding protein",
            "J": "decreased binding protein",
        },
        "general health": {"K": "concurrent non-thyroidal illness"},
        "replacement therapy": {
            "L": "consistent with replacement therapy",
            "M": "underreplaced",
            "N": "overreplaced",
        },
        "antithyroid treatment": {
            "O": "antithyroid drugs",
            "P": "I131 treatment",
            "Q": "surgery",
        },
        "miscellaneous": {
            "R": "discordant assay results",
            "S": "elevated TBG",
            "T": "elevated thyroid hormones",
        },
    },
}

mapping_diagnosis = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "-": 20,
}


def data_standardisation():
    """
    Standardizes the numerical features of the thyroid dataset.

    The function reads a CSV file containing preprocessed thyroid data, standardizes the numerical features
    using StandardScaler from sklearn, and saves the standardized data to a new CSV file.

    The numerical features considered for standardization are:
    - age
    - TSH
    - T3
    - TT4
    - T4U
    - FTI

    Returns:
        None
    """
    thyroid_data = pd.read_csv(f"{BASIC_PATH}/preporcessed.csv")
    # num_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    num_features = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    scaler = StandardScaler()
    df_num_standardized = pd.DataFrame(
        scaler.fit_transform(thyroid_data[num_features]),
        columns=num_features,
        index=thyroid_data.index,
    )
    df_non_num = thyroid_data.drop(columns=num_features)
    df_standardized = pd.concat([df_non_num, df_num_standardized], axis=1)

    # Uporządkowanie kolumn w oryginalnym układzie
    df_standardized = df_standardized[thyroid_data.columns]
    df_standardized.to_csv(f"{BASIC_PATH}/thyroid_standardized.csv", index=False)


def frequency_discretisation_of_data():
    """
    Discretizes the numerical features of the thyroid dataset using quantile-based binning.

    The function reads the standardized thyroid data from a CSV file, fills missing values with -9999,
    and applies KBinsDiscretizer to discretize the specified numerical columns into 10 bins.
    The discretized data is then saved to a new CSV file.

    The numerical features considered for discretization are:
    - age
    - TSH
    - T3
    - TT4
    - T4U
    - FTI

    Returns:
        None
    """
    thyroid_data = pd.read_csv(f"{BASIC_PATH}/thyroid_standardized.csv")
    # numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI','TBG']
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    print(thyroid_data.info())
    thyroid_data = thyroid_data.fillna(-9999)
    print(thyroid_data.info())
    print(f"numeric_cols {numeric_cols}")
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    thyroid_data[numeric_cols] = discretizer.fit_transform(thyroid_data[numeric_cols])
    thyroid_data.to_csv(f"{BASIC_PATH}/discretisation.csv")


def train_model(
    model_name,
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Trains a specified machine learning model using Grid Search for hyperparameter tuning.

    The function fits the model on the training data, evaluates it on the test data, and
    calculates various performance metrics. It saves the results in a CSV file and a classification report.

    Parameters:
        model_name (str): The name of the model.
        model (object): The machine learning model to be trained.
        params (dict): The hyperparameters to be tuned using GridSearchCV.
        X_train (array-like): The training feature set.
        y_train (array-like): The training target values.
        X_test (array-like): The test feature set.
        y_test (array-like): The test target values.

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
            "Accuracy": [acc],
            "Precision": [precision],
            "Recall": [recall],
            "F1-score": [f1],
            "Training time": [training_time],
            "Prediction time": [prediction_time],
            "y_train unique": [np.unique(y_train)],
        }
    )
    os.makedirs(
        f"{BASIC_PATH}/{model_name}",
        exist_ok=True,
    )
    header = True if not os.path.exists(f"{BASIC_PATH}/results_all.csv") else False
    df_results.to_csv(
        f"{BASIC_PATH}/results_all.csv",
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
    )


def save_results(
    model_name,
    report,
    conf_matrix,
    best_params,
    class_labels,
    training_time,
    prediction_time,
):
    """
    Saves the results of model evaluation including classification report,
    confusion matrix, and the best hyperparameters.

    The function creates a directory for the model if it doesn't exist,
    saves the classification report and best parameters to text files,
    and saves the confusion matrix as both an image and a CSV file.

    Parameters:
        model_name (str): The name of the model.
        report (str): The classification report.
        conf_matrix (array-like): The confusion matrix.
        best_params (dict): The best hyperparameters from the grid search.
        class_labels (array-like): The class labels.
        training_time (float): The time taken to train the model.
        prediction_time (float): The time taken to make predictions.

    Returns:
        None
    """
    with open(
        f"{BASIC_PATH}/{model_name}/{model_name}_classification_report.txt",
        "w",
    ) as f:
        f.write(report)
    with open(
        f"{BASIC_PATH}/{model_name}/{model_name}_best_params.txt",
        "w",
    ) as f:
        f.write(str(best_params))

    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
    print(model_name, training_time, prediction_time)
    fig = px.imshow(conf_matrix_df)
    fig.write_image(f"{BASIC_PATH}/{model_name}/{model_name}_confusion_matrix.png")
    conf_matrix_df.to_csv(
        f"{BASIC_PATH}/{model_name}/{model_name}_confusion_matrix.csv"
    )


def main():
    """
    Main function to run the entire data preprocessing and model training workflow.

    This function creates necessary directories, reads the raw thyroid data,
    preprocesses it, standardizes and discretizes the features, splits the data
    into training and testing sets, and trains various models defined in the 'models' dictionary.

    Returns:
        None
    """
    os.makedirs(BASIC_PATH, exist_ok=True)
    path_data = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/thyroid0387.data"
    thyroid_data = pd.read_csv(path_data)
    thyroid_data.columns = list(columns.keys())
    new_data = pd.DataFrame(columns=thyroid_data.columns)
    thyroid_data.replace("?", np.nan, inplace=True)
    thyroid_data["diagnosis"] = thyroid_data["diagnosis"].apply(
        lambda x: x.split("[")[0]
    )
    print("Unique", thyroid_data["diagnosis"].unique())
    thyroid_data["sex"].fillna(thyroid_data["sex"].mode().iloc[0], inplace=True)
    thyroid_data["age"].fillna(thyroid_data["age"].mode().iloc[0], inplace=True)
    for idx in thyroid_data.index:
        if "|" in thyroid_data.loc[idx]["diagnosis"]:
            print(
                f'Before {thyroid_data.loc[idx]["diagnosis"]},{thyroid_data.loc[idx]["diagnosis"].split("|")[0]}'
            )
            data = thyroid_data.loc[idx]
            thyroid_data.iat[idx, len(thyroid_data.loc[idx]) - 1] = thyroid_data.loc[
                idx
            ]["diagnosis"].split("|")[0]
            data["diagnosis"] = data["diagnosis"].split("|")[1]
            print(
                f"After data {data['diagnosis']} thyroid data {thyroid_data.loc[idx]['diagnosis']}"
            )
            new_data = pd.concat([new_data, data.to_frame().T], ignore_index=True)
    added_thyroid_data = pd.concat([new_data, thyroid_data])
    print("Unique  f", added_thyroid_data["diagnosis"].unique())
    for column in thyroid_data.columns:
        print("Columns", column, thyroid_data[column].unique())
        if (
            "F" in list(thyroid_data[column].unique())
            and len(list(thyroid_data[column].unique())) == 2
        ):
            thyroid_data[column].replace({"F": 0, "M": 1}, inplace=True)
        elif "t" in list(thyroid_data[column].unique()) or "f" in list(
            thyroid_data[column].unique()
        ):
            thyroid_data[column].replace({"f": 0, "t": 1}, inplace=True)
        else:
            if column == "referral source":
                thyroid_data[column].replace(
                    {"WEST": 0, "STMW": 1, "SVHC": 2, "SVI": 3, "SVHD": 4, "other": 5},
                    inplace=True,
                )

            elif column == "diagnosis":
                print("Change")
                for idx in thyroid_data.index:
                    if thyroid_data.loc[idx][column] not in list(
                        mapping_diagnosis.keys()
                    ):
                        thyroid_data.drop([idx], inplace=True, axis=0)
                thyroid_data[column].replace(mapping_diagnosis, inplace=True)
            # new_data = pd.concat([new_data, data.to_frame().T], ignore_index=True)

    print(thyroid_data["diagnosis"].unique())
    thyroid_data.to_csv(f"{BASIC_PATH}/preporcessed.csv", index=False)
    data_standardisation()
    frequency_discretisation_of_data()
    data = pd.read_csv(f"{BASIC_PATH}/discretisation.csv")
    print(data.shape)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    for model_name, (model, params) in models.items():
        train_model(
            model_name,
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
        )


main()
