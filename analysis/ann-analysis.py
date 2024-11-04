import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

MAIN_PATH_DB = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/"
MAIN_PATH_DATA_ANALYSES = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/analyses/ann/"
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


def compute_variance(paths: list[str]):
    """
    Computes the variance for each column in the provided CSV files and saves the results to separate CSV files.

    Args:
        paths (list[str]): A list of file paths to the CSV files.

    Returns:
        None
    """
    variance_data = []
    data_frames = read_csv_data(paths)
    for idx, df in enumerate(data_frames):
        for column in df.columns:
            variance_data.append([column, df[column].var()])
        name = "variance_train.csv" if idx == 1 else "variance_test.csv"
        pd.DataFrame(variance_data, columns=["column", "var"]).to_csv(
            f"{MAIN_PATH_DATA_ANALYSES}{name}", index=False
        )


def visualization_dataset_class(
    paths: list[str], titles: list[str], xasix_name: str, yaxis_name: str
):
    """
    Visualizes the distribution of classes in the specified datasets using histograms and saves the plots as images.

    Args:
        paths (list[str]): A list of file paths to the CSV files.
        titles (list[str]): A list of titles for the histogram plots.
        xasix_name (str): The name for the x-axis of the plots.
        yaxis_name (str): The name for the y-axis of the plots.

    Returns:
        None
    """
    data_frames = read_csv_data(paths)
    for idx, df in enumerate(data_frames):
        fig = px.histogram(df, x="Klasa", title=titles[idx])
        fig.update_layout(
            xaxis_title=xasix_name,
            yaxis_title=yaxis_name,
            bargap=0.2,
            xaxis=dict(dtick=1),
        )
        fig.write_image(f"{MAIN_PATH_DATA_ANALYSES}{titles[idx]}.png")


def plot_scatter(paths: list[str]):
    """
    Creates scatter plots for all combinations of features in the specified datasets, colored by class, and saves the plots as images.

    Args:
        paths (list[str]): A list of file paths to the CSV files.

    Returns:
        None
    """
    os.makedirs(f"{MAIN_PATH_DATA_ANALYSES}scatter plots", exist_ok=True)
    data_frames = read_csv_data(paths)
    for idx, df in enumerate(data_frames):
        features = df.columns[:-1]
        pairs = itertools.combinations(features, 2)
        for feature_x, feature_y in pairs:
            fig = px.scatter(
                df,
                x=feature_x,
                y=feature_y,
                color="Klasa",
                title=f"Wykres rozrzutu: {feature_x} vs {feature_y}",
            )
            print(
                f"{feature_x.replace(' ','_')}_{feature_y.replace(' ','_')}"
                != "Na_lekach_przeciwtarczycowych_Zapytanie_o_niedoczynność_tarczycy",
                f"{feature_x.replace(' ','_')}_{feature_y.replace(' ','_')}",
                "Na_lekach_przeciwtarczycowych_Zapytanie_o_niedoczynność_tarczycy",
            )
            try:
                fig.write_image(
                    f"{MAIN_PATH_DATA_ANALYSES}scatter plots/{feature_x.replace(' ','_')}_{feature_y.replace(' ','_')}.png"
                )
            except:
                print(f"Problem")


def plot_boxplots(paths: list[str]):
    """

    Visualizes thyroid data by creating box plots for specified parameters and saving the plots as images.

    Args:
        parameters (list): List of column names to be visualized.

    Returns:
        None
    """
    data_frames = read_csv_data(paths)
    for idx, df in enumerate(data_frames):
        print(df.info())
        parameters = df.select_dtypes(include=["float64"]).columns
        data = df[parameters]
        data.drop(["Wiek"], axis=1, inplace=True)

        fig = px.box(data)
        fig.update_layout(
            xaxis_title="hormony",
            yaxis_title="wartości",
            title=f"Wykres pudełkowy dla hormonów {','.join(list(data.columns))}",
        )
        fig.write_image(f"{MAIN_PATH_DATA_ANALYSES}box_plots_all.png")

        for column in data.columns:
            print(data.columns)
            fig = px.box(data[column])
            fig.update_layout(
                xaxis_title="hormon",
                yaxis_title="wartości",
                title=f"Wykres pudełkowy dla {column}",
            )
            fig.write_image(f"{MAIN_PATH_DATA_ANALYSES}box_plot_{column}.png")


def read_csv_data(paths: list):
    """
    Reads CSV files located at the given paths and returns a list of DataFrame objects.

    Parameters:
        paths (list): A list of file paths to the CSV files.

    Returns:
        list: A list containing DataFrame objects, each representing data from a CSV file.

    Each CSV file is assumed to have 24 columns. The last two columns are dropped,
    and the second-to-last column is renamed to 'class'. Columns are named as 'column 1',
    'column 2', and so on.
    """

    df_list = []
    for path in paths:
        df = pd.read_csv(path, sep=" ")
        df = df.dropna(axis=1, how="all")
        df.columns = columns_names
        df_list.append(df)
    return df_list


def detect_missing_values(paths):
    """
    Detects and counts missing values in the specified datasets and saves the counts to separate CSV files.

    Args:
        paths (list[str]): A list of file paths to the CSV files.

    Returns:
        None
    """
    df_data = read_csv_data(paths)
    for idx, df in enumerate(df_data):
        print(df.isnull().sum())
        name = "missing_values_train.csv" if idx == 1 else "missing_values_test.csv"
        df.isnull().sum().to_csv(f"{MAIN_PATH_DATA_ANALYSES}{name}", index=False)


def descriptive_analysis(paths):
    """
    Performs descriptive statistical analysis on the datasets and saves the statistics to separate CSV files.

    Args:
        paths (list[str]): A list of file paths to the CSV files.

    Returns:
        None
    """
    df = read_csv_data(paths)
    train_data = df[0]
    test_data = df[1]
    train_stats = train_data.describe()
    test_stats = test_data.describe()
    print("Train Statistics:\n", train_stats)
    print("Test Statistics:\n", test_stats)
    train_stats.to_csv(f"{MAIN_PATH_DATA_ANALYSES}train_stats.csv")
    test_stats.to_csv(f"{MAIN_PATH_DATA_ANALYSES}test_stats.csv")


def distribution_visualization(paths):
    """
    Visualizes the distribution of features in the training and test datasets using histograms and saves the plots as images.

    Args:
        paths (list[str]): A list of file paths to the CSV files.

    Returns:
        None
    """
    df = read_csv_data(paths)
    train_data = df[0]
    test_data = df[1]
    plt.rc("axes", titlesize=16)
    plt.rc("axes", labelsize=14)
    plt.rc("legend", fontsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    for column in train_data.columns:
        sns.histplot(train_data[column], color="blue", label="Treningowy", kde=True)
        sns.histplot(test_data[column], color="red", label="Testowy", kde=True)
        plt.legend()
        plt.title(column)
        plt.ylabel("Liczba wystąpień")  # Ustawienie etykiety osi Y
        plt.savefig(f"{MAIN_PATH_DATA_ANALYSES}/features/{column}.png")
        plt.clf()


def EDA_analysis(paths, titles, xaxis_name, yaxis_name):
    """
    Conducts exploratory data analysis (EDA) on the specified datasets, including visualizations, variance computation,
    missing values detection, and descriptive statistics.

    Args:
        paths (list[str]): A list of file paths to the CSV files.
        titles (list[str]): A list of titles for visualizations.
        xaxis_name (str): The name for the x-axis of the plots.
        yaxis_name (str): The name for the y-axis of the plots.

    Returns:
        None
    """
    visualization_dataset_class(paths, titles, xasix_name, yaxis_name)
    plot_boxplots(paths)
    plot_scatter(paths)
    compute_variance(paths)
    detect_missing_values(paths)
    descriptive_analysis(paths)
    distribution_visualization(paths)


paths = [f"{MAIN_PATH_DB}ann-train.data", f"{MAIN_PATH_DB}ann-test.data"]
titles = ["Rozkład klas dla danych treningowych", "Rozkład klas dla danych testowych"]
xasix_name = "Klasy"
yaxis_name = "Liczba przypadków"
EDA_analysis(paths, titles, xasix_name, yaxis_name)
