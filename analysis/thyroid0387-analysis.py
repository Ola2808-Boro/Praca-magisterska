import itertools
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

MAIN_PATH_DB = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/"
MAIN_PATH_DATA_ANALYSES = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/analyses/thyroid0387/"
# dir_name='no_lack_of_diagnosis_TBG'
dir_name = "no_lack_of_diagnosis"
# dir_name = "no_lack_of_diagnosis_TBG_measured"
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
    "S": 19,
    "T": 20,
}
data_distribution = {
    "possible_cases": 64,
    "configuration": [
        "TSH measured",
        "T3 measured",
        "TT4 measured",
        "T4U measured",
        "FTI measured",
        "TBG measured",
    ],
    "data": [],
    "distribution": {},
    "distribution_decoded": {},
}

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
    "Pomiar TSH",
    "TSH",
    "Pomiar T3",
    "T3",
    "Pomiar TT4",
    "TT4",
    "Pomiar T4U",
    "T4U",
    "Pomiat FTI",
    "FTI",
    "Pomiar TBG",
    "TBG",
    "Klasa",
]


def read_data(filename: str):
    """
    Reads a CSV file and returns its contents as a DataFrame.

    Args:
        filename (str): The name of the CSV file to be read.

    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    return pd.read_csv(filename)


def create_csv_file(filename: str):
   """
    Creates a directory if it doesn't exist, reads a CSV file,
    renames its columns, and saves it to a new location.

    Args:
        filename (str): The name of the CSV file to be saved (without the .csv extension).

    Returns:
        None
    """
    os.makedirs(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}", exist_ok=True)
    content = pd.read_csv(f"{MAIN_PATH_DB}thyroid0387.data", header=None)
    content.columns = list(columns.keys())
    content.to_csv(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}", index=False)


def analysis_of_hormone_data(filename: str, filename_saved: str):
    """
    Analyzes thyroid diagnosis data by reading a CSV file, encoding the data,
    updating distributions, and saving the results to a JSON file.

    Args:
        filename (str): The name of the CSV file to be read (without the .csv extension).
        filename_saved (str): The name of the JSON file to save the analyzed data.

    Returns:
        None
    """

    thyroid_data = pd.read_csv(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    data = thyroid_data[data_distribution["configuration"]]
    for _, row in data.iterrows():
        code = []
        for _, item in row.items():
            if item == "t":
                code.append(1)
            else:
                code.append(0)
        if (
            "".join(str(value) for value in code)
            not in data_distribution["distribution"].keys()
        ):
            data_distribution["distribution"].update(
                {"".join(str(value) for value in code): 1}
            )
        else:
            data_distribution["distribution"][
                "".join(str(value) for value in code)
            ] += 1
        data_distribution["data"].append(code)
    for key in data_distribution["distribution"].keys():
        decode = ""
        counter = 0
        for item in str(key):
            if item == "1":
                if counter == 5:
                    decode += data_distribution["configuration"][counter]
                else:
                    decode += data_distribution["configuration"][counter] + " "
            counter += 1
        data_distribution["distribution_decoded"].update(
            {decode: data_distribution["distribution"][key]}
        )
    json_object = json.dumps(
        data_distribution, indent=len(list(data_distribution.keys()))
    )
    with open(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename_saved}", "w+") as f:
        f.write(json_object)


def delete_instance_without_diagnosis(filename: str, filename_saved: str):
    """
    Reads a CSV file containing thyroid data, removes instances without a diagnosis,
    and saves the cleaned data to a new CSV file.

    Args:
        filename (str): The name of the CSV file to be read (without the .csv extension).
        filename_saved (str): The name of the CSV file to save the cleaned data.

    Returns:
        None
    """
    thyroid_data = pd.read_csv(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    for idx in thyroid_data.index:
        if "-" in thyroid_data.loc[idx]["diagnosis"]:
            thyroid_data.drop([idx], axis=0, inplace=True)
    print(f" Len {len(thyroid_data.index)}")
    thyroid_data.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename_saved}", index=False
    )


def analysis_to_select_data(
    filename: str, filename_saved: str, image_name: str, drop_TGB: bool
):
    """
    Analyzes thyroid diagnosis data, visualizes the distribution of diagnoses,
    optionally drops TGB-related columns, and saves the selected data to a CSV file.

    Args:
        filename (str): The name of the CSV file to be analyzed (without the .csv extension).
        filename_saved (str): The name of the CSV file to save the selected data.
        image_name (str): The name of the image file to save the plot.
        drop_TGB (bool): If True, drops TGB-related columns and filters the data.

    Returns:
        None
    """

    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    thyroid_data.replace("?", np.nan, inplace=True)
    print(thyroid_data.info())
    thyroid_data["diagnosis_extracted"] = thyroid_data["diagnosis"].apply(
        lambda x: x.split("[")[0]
    )

    fig = px.bar(
        thyroid_data["diagnosis_extracted"].value_counts(),
        text_auto=True,
        title="Rozkład diagnozy",
    )
    fig.update_layout(
        xaxis_title="Symbole diagnozy", yaxis_title="Liczba wystąpień diagnozy"
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    fig.write_image(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{image_name}")

    if drop_TGB:
        thyroid_data = thyroid_data.loc[
            (thyroid_data["TSH measured"] == "t")
            & (thyroid_data["T3 measured"] == "t")
            & (thyroid_data["TT4 measured"] == "t")
            & (thyroid_data["T4U measured"] == "t")
            & (thyroid_data["FTI measured"] == "t")
        ]
        # thyroid_data.drop(['TBG','TBG measured'],axis=1,inplace=True)
        thyroid_data.drop(["TBG"], axis=1, inplace=True)
        thyroid_data = clean_data(thyroid_data)

    thyroid_data.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename_saved}", index=False
    )


def analysis_selected_data(filename: str, filename_saved: str, image_name: str):
    """

    Analyzes selected thyroid data by reading a CSV file, encoding categorical variables,
    and saving the cleaned data to a new CSV file.

    Args:
        filename (str): The name of the CSV file to be analyzed (without the .csv extension).
        filename_saved (str): The name of the CSV file to save the selected data.
        image_name (str): The name of the image file to save the plot.

    Returns:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
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

            # elif column == "diagnosis_extracted":
            #     print("Change")
            #     for idx in thyroid_data.index:
            #         if thyroid_data.loc[idx][column] not in list(
            #             mapping_diagnosis.keys()
            #         ):
            #             thyroid_data.drop([idx], inplace=True, axis=0)
            # thyroid_data[column].replace(mapping_diagnosis, inplace=True)
    print("Unique", thyroid_data["diagnosis_extracted"].unique())
    print("Unique", thyroid_data["sex"].unique())
    print("Unique", thyroid_data["referral source"].unique())
    thyroid_data.drop(["diagnosis"], axis=1, inplace=True)
    thyroid_data.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename_saved}", index=False
    )
    print(thyroid_data.columns)
    fig_selected = px.bar(
        thyroid_data["diagnosis_extracted"].value_counts(),
        text_auto=True,
        title="Rozkład diagnoz",
        labels={"index": "Diagnoza", "value": "Liczba wystąpień diagnozy"},
    )
    fig_selected.update_layout(
        xaxis_title="Symbole diagnozy",
        yaxis_title="Liczba wystąpień diagnozy",
        showlegend=False,  # Wyłączenie legendy
        margin=dict(
            l=20, r=20, t=50, b=50
        ),  # Ustaw marginesy (lewy, prawy, górny, dolny)
    )
    fig_selected.update_traces(
        textfont_size=16,  # Ustawienie tej samej wielkości czcionki dla wszystkich etykiet
        textangle=0,
        textposition="outside",
        cliponaxis=False,
        # textfont_color="black"  # Ustawienie koloru czcionki na czarny
    )
    plt.subplots_adjust(left=0.04, bottom=0.04)
    fig_selected.write_image(f"{MAIN_PATH_DATA_ANALYSES}{image_name}")
    thyroid_data.columns = [f"X{i+1}" for i in range(thyroid_data.shape[1])]
    corr = thyroid_data.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    print(corr)
    df_corr_viz = corr.mask(mask).dropna(how="all").dropna(axis="columns", how="all")
    fig = px.imshow(df_corr_viz, text_auto=False)
    fig.update_xaxes(side="top")
    fig.write_image(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/correlation.png")


def uncertain_diagnosis(filename: str, filename_saved: str, fillna: bool):
    """

    Processes thyroid diagnosis data to handle uncertain diagnoses, splits them into separate entries,
    optionally fills missing values, and saves the updated data to a new CSV file.

    Args:
        filename (str): The name of the CSV file to be analyzed (without the .csv extension).
        filename_saved (str): The name of the CSV file to save the selected data.
        fillna (bool): If True, fills missing values with the mode of the 'sex' and 'age' columns.

    Return:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    new_data = pd.DataFrame(columns=thyroid_data.columns)
    for idx in thyroid_data.index:
        if "|" in thyroid_data.loc[idx]["diagnosis_extracted"]:
            print(
                f'Before {thyroid_data.loc[idx]["diagnosis_extracted"]},{thyroid_data.loc[idx]["diagnosis_extracted"].split("|")[0]}'
            )
            data = thyroid_data.loc[idx]
            thyroid_data.iat[idx, len(thyroid_data.loc[idx]) - 1] = thyroid_data.loc[
                idx
            ]["diagnosis_extracted"].split("|")[0]
            data["diagnosis_extracted"] = data["diagnosis_extracted"].split("|")[1]
            print(
                f"After data {data['diagnosis_extracted']} thyroid data {thyroid_data.loc[idx]['diagnosis_extracted']}"
            )
            new_data = pd.concat([new_data, data.to_frame().T], ignore_index=True)
    added_thyroid_data = pd.concat([new_data, thyroid_data])
    added_thyroid_data.replace("?", np.nan, inplace=True)
    if fillna:
        added_thyroid_data["sex"].fillna(
            added_thyroid_data["sex"].mode().iloc[0], inplace=True
        )
        added_thyroid_data["age"].fillna(
            added_thyroid_data["age"].mode().iloc[0], inplace=True
        )
    added_thyroid_data.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename_saved}", index=False
    )
    print(added_thyroid_data.info())


def visualization(filename: str, parameters: list):
    """

    Visualizes thyroid data by creating box plots for specified parameters and saving the plots as images.

    Args:
        parameters (list): List of column names to be visualized.

    Returns:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    thyroid_data = thyroid_data[parameters]

    fig = px.box(thyroid_data)
    fig.write_image(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/thyroid_diagnosis_selected_box_plots_all.png"
    )

    for column in thyroid_data.columns:
        print(thyroid_data.columns)
        fig = px.box(thyroid_data[column])
        fig.update_layout(
            xaxis_title="hormon",
            yaxis_title="wartości",
            title=f"Wykres pudełkowy dla {column}",
        )
        fig.write_image(
            f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/thyroid_diagnosis_selected_box_plot_{column}.png"
        )


def clean_data(df: pd.DataFrame):
    """

    This function cleans the input DataFrame by replacing all occurrences of '?' with NaN,
    and then drops all rows containing NaN values. It prints the DataFrame info before and
    after dropping the NaN values.

    Args:
        parameters: The input DataFrame to be cleaned.

    Returns:
        thyroid data: The cleaned DataFrame with no NaN values.
    """
    df.replace("?", np.nan, inplace=True)
    print(df.info())
    thyroid_data = df.dropna(axis=0)
    print(df.info())
    return thyroid_data


def analysis_TBG(filename: str):
    """
    This function reads a CSV file containing thyroid diagnosis data, replaces all occurrences of '?' with NaN,
    and prints the DataFrame information.

    Args:
        None

    Returns:
        thyroid_data: The processed DataFrame with '?' replaced by NaN.
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    thyroid_data.replace("?", np.nan, inplace=True)
    print(thyroid_data.info())


def missing_data_analysis(filename: str, title: str, image_name: str):
    """
    This function analyzes missing data in a specified CSV file. It reads the data, replaces certain values with NaN,
    processes the 'diagnosis' column, renames the columns, and generates a matrix plot showing the missing data pattern.
    The plot is saved as a PNG file.

    Args:
        filename (str): The name of the CSV file to be analyzed (without the .csv extension).
        image_name (str): The name of the image file to save the plot.

    Returns:
        None

    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    thyroid_data.replace("?", np.nan, inplace=True)
    thyroid_data["diagnosis"] = thyroid_data["diagnosis"].apply(
        lambda x: x.split("[")[0]
    )
    thyroid_data["diagnosis"].replace("-", np.nan, inplace=True)
    thyroid_data.columns = [f"X{i+1}" for i in range(thyroid_data.shape[1])]
    fig = msno.matrix(
        thyroid_data, figsize=(22, 12), fontsize=16, color=(0, 0, 1), sparkline=False
    )
    blue_patch = mpatches.Patch(color="blue", label="Obecna wartość")
    white_patch = mpatches.Patch(color="white", label="Brak wartości")
    plt.subplots_adjust(left=0.04, bottom=0.04)
    plt.title(
        f"Wykres gęstości danych dla instancji posiadających diagnozę", fontsize=24
    )
    # plt.title(f"Wykres gęstości danych dla wszystkich instancji", fontsize=24)
    plt.legend(
        loc="upper right",
        handles=[blue_patch, white_patch],
        fontsize=13,
        bbox_to_anchor=(1.115, 1),
        borderaxespad=0.0,
    )
    plt.yticks(fontsize=16)
    g = fig.figure
    g.savefig(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{image_name}")
    # print(thyroid_data.columns)
    plt.clf()


def compute_variance(filename):
    """
    Computes the variance of each column in the thyroid dataset and saves the results to a CSV file.

    Args:
        filename (str): The name of the CSV file containing thyroid data (without the .csv extension).

    Returns:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    variance_data = []
    for column in thyroid_data.columns:
        variance_data.append([column, thyroid_data[column].var()])
    pd.DataFrame(variance_data, columns=["column", "var"]).to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/variance.csv"
    )


def data_standardisation(filename):
    """
    Standardizes the data in the thyroid dataset by scaling features to have a mean of 0 and a standard deviation of 1.

    Args:
        filename (str): The name of the CSV file containing thyroid data (without the .csv extension).

    Returns:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
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
    df_standardized.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/thyroid_standardized.csv", index=False
    )


def frequency_discretisation_of_data(filename: str, num_bins: int):
    """
    Discretizes specified numeric columns in the thyroid dataset into bins and saves the 
    discretized data to a new CSV file.

    Args:
        filename (str): The name of the CSV file containing the thyroid data (without the .csv extension).
        num_bins (int): The number of bins to discretize the data into.

    Returns:
        None
    """
    thyroid_data = read_data(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    # numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI','TBG']
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    print(thyroid_data.info())
    thyroid_data = thyroid_data.fillna(-9999)
    print(thyroid_data.info())
    print(f"numeric_cols {numeric_cols}")
    discretizer = KBinsDiscretizer(
        n_bins=num_bins, encode="ordinal", strategy="quantile"
    )
    thyroid_data[numeric_cols] = discretizer.fit_transform(thyroid_data[numeric_cols])
    thyroid_data.to_csv(
        f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/thyroid_discretized.csv", index=False
    )


def plot_scatter():
    """
    Generates scatter plots for all combinations of features in the thyroid dataset and saves them as images.
    The function handles missing values and filters the dataset based on the diagnosis.

    Returns:
        None
    """
    os.makedirs(f"{MAIN_PATH_DATA_ANALYSES}scatter plots select", exist_ok=True)
    data_frames = pd.read_csv(f"{MAIN_PATH_DATA_ANALYSES}thyroid0387.csv")
    print(data_frames.columns)
    data_frames.replace("?", np.nan, inplace=True)
    new_data = pd.DataFrame(columns=data_frames.columns)
    data_frames["diagnosis"] = data_frames["diagnosis"].apply(lambda x: x.split("[")[0])
    for idx in data_frames.index:
        if "|" in data_frames.loc[idx]["diagnosis"]:
            print(
                f'Before {data_frames.loc[idx]["diagnosis"]},{data_frames.loc[idx]["diagnosis"].split("|")[0]}'
            )
            data = data_frames.loc[idx]
            data_frames.iat[idx, len(data_frames.loc[idx]) - 1] = data_frames.loc[idx][
                "diagnosis"
            ].split("|")[0]
            data["diagnosis"] = data["diagnosis"].split("|")[1]
            print(
                f"After data {data['diagnosis']} thyroid data {data_frames.loc[idx]['diagnosis']}"
            )
            new_data = pd.concat([new_data, data.to_frame().T], ignore_index=True)
    added_thyroid_data = pd.concat([new_data, data_frames])
    added_thyroid_data.to_csv("test.csv", index=False)
    # print(added_thyroid_data[["diagnosis"]].value_counts())
    for idx in added_thyroid_data.index:
        print(added_thyroid_data.loc[idx, "diagnosis"])
        #     diagnosis_value=added_thyroid_data.loc[idx, "diagnosis"]
        #     if diagnosis_value != "-":
        #         print("Value idx", idx)
        #         diagnosis_value = added_thyroid_data.loc[idx, "diagnosis"].values[0]
        #         print("Value idx", idx, diagnosis_value)
        if isinstance(added_thyroid_data.loc[idx, "diagnosis"], pd.Series):
            print("series")
            value = added_thyroid_data.loc[idx, "diagnosis"].values[0]
        else:
            value = added_thyroid_data.loc[idx, "diagnosis"]
        if value not in mapping_diagnosis.keys() and value != "-":
            print(f"Usuwam {value}")
            added_thyroid_data.drop([idx], inplace=True, axis=0)
    print(added_thyroid_data[["diagnosis"]].value_counts())
    filtered_data = added_thyroid_data[
        added_thyroid_data["diagnosis"].isin(["-", "F", "I", "G", "K"])
    ]
    features = filtered_data.columns[:-1]
    pairs = itertools.combinations(features, 2)
    for feature_x, feature_y in pairs:
        fig = px.scatter(
            filtered_data,
            x=feature_x,
            y=feature_y,
            color="diagnosis",
            title=f"Wykres rozrzutu: X{list(features).index(feature_x)} vs X{list(features).index(feature_y)}",
        )
        print(
            f"X{list(features).index(feature_x)} vs X{list(features).index(feature_y)}"
        )

        fig.update_layout(
            xaxis_title=columns_names[list(features).index(feature_x)],
            yaxis_title=columns_names[list(features).index(feature_y)],
        )
        try:
            fig.write_image(
                f"{MAIN_PATH_DATA_ANALYSES}scatter plots select/{feature_x.replace(' ','_')}_{feature_y.replace(' ','_')}.png"
            )
        except:
            print(f"Problem")


def main():
    """
    Main function that orchestrates the data processing, analysis, and visualization for the thyroid dataset.
    It performs actions such as missing data analysis, hormone data analysis, diagnosis filtering,
    and generates various visualizations including scatter plots.

    Returns:
        None
    """
    # parameters=['TSH','T3','TT4','T4U','FTI','TBG']
    parameters = ["TSH", "T3", "TT4", "T4U", "FTI"]
    title = f"Wykres gęstości danych dla instancji posiadających diagnozę"
    filename = "thyroid0387.csv"
    filename_no_instances_without_diagnosis = "thyroid_diagnosis.csv"
    filename_selected = "thyroid_diagnosis_selected.csv"
    filename_distribution = "thyroid_diagnosis_all_diagnosed.json"
    filename_certain_diagnosis = "thyroid_diagnosis_certain diagnosis.csv"
    image_name = "distribution_of_diagnoses.png"
    image_name_distribution = "distribution_of_diagnoses_removal_missing_data.png"
    image_name_missing_data_analysis = "missing_data_pattern.png"
    filename_final = "thyroid.csv"
    # drop_TBG=False
    # fillna=True
    drop_TBG = True
    fillna = False
    num_bins = 10
    create_csv_file(filename=filename)
    missing_data_analysis(
        filename=filename_no_instances_without_diagnosis,
        title=title,
        image_name=image_name_missing_data_analysis,
    )
    delete_instance_without_diagnosis(
        filename=filename, filename_saved=filename_no_instances_without_diagnosis
    )
    analysis_of_hormone_data(
        filename=filename_no_instances_without_diagnosis,
        filename_saved=filename_distribution,
    )
    analysis_to_select_data(
        filename_no_instances_without_diagnosis, filename_selected, image_name, drop_TBG
    )
    # analysis_to_select_data(filename=filename_no_instances_without_diagnosis,filename_saved=filename_selected,image_name=image_name,drop_TBG=drop_TBG)
    uncertain_diagnosis(
        filename=filename_selected,
        filename_saved=filename_certain_diagnosis,
        fillna=fillna,
    )
    analysis_selected_data(
        filename=filename_selected,
        filename_saved=filename_final,
        image_name=image_name_distribution,
    )
    visualization(filename=filename_final, parameters=parameters)
    compute_variance(filename="thyroid.csv")
    data_standardisation(filename="thyroid.csv")
    frequency_discretisation_of_data(filename=filename_final, num_bins=num_bins)


main()

