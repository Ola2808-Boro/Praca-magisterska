import glob
import json
import os
import random
from collections import defaultdict
from random import randrange

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

MAIN_PATH_DB = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/"
MAIN_PATH_DATA_ANALYSES = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/analyses/thyroid0387/"
MAIN_PATH_KCF = "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF"
MAX_CATEGORIES = 5
coverage = 4


def define_possible_combinations(content, columns: list):
    """
    Defines possible combinations of specified columns in the DataFrame and calculates their probabilities.

    Args:
        content (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column names to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the combinations of values and their corresponding probabilities.
    """
    df = content[columns].dropna(how="any")
    freq_table = df.value_counts().reset_index(name="count")
    total_observations = len(df)
    freq_table["probability"] = freq_table["count"] / total_observations
    return freq_table


def conditional_mutual_information(content, dir_name, X, Y, Z, i):
    """
    Calculates the Conditional Mutual Information (CMI) for three variables X, Y, and Z.

    Args:
        content (pd.DataFrame): The input DataFrame containing the data.
        dir_name (str): Directory name to save results.
        X (str): The first variable.
        Y (str): The second variable.
        Z (str): The conditional variable.
        i (int): The fold index for organizing output.

    Returns:
        float: The calculated CMI value for the variables.
    """
    path_freq_table = f"{MAIN_PATH_KCF}/{dir_name}/freq table"
    print(f" X {X},Y {Y},Z {Z}")
    os.makedirs(f"{path_freq_table}/fold {i}/{X}_{Y}_{Z}", exist_ok=True)

    freq_table_xyz = define_possible_combinations(content, [X, Y, Z])
    freq_table_xyz.to_csv(
        f"{path_freq_table}/fold {i}/{X}_{Y}_{Z}/xyz.csv", index=False
    )
    xyz = freq_table_xyz["probability"].values

    freq_table_xz = define_possible_combinations(content, [X, Z])
    freq_table_xz.to_csv(f"{path_freq_table}/fold {i}/{X}_{Y}_{Z}/xz.csv", index=False)
    xz = freq_table_xz["probability"].values

    freq_table_yz = define_possible_combinations(content, [Y, Z])
    freq_table_yz.to_csv(f"{path_freq_table}/fold {i}/{X}_{Y}_{Z}/yz.csv", index=False)
    yz = freq_table_yz["probability"].values

    freq_table_z = define_possible_combinations(content, [Z])
    freq_table_z.to_csv(f"{path_freq_table}/fold {i}/{X}_{Y}_{Z}/z.csv", index=False)
    z = freq_table_z["probability"].values

    cmi_simple = calculate_cmi(xyz, xz, yz, z)
    print("CMI (Simple):", cmi_simple)
    return cmi_simple


def calculate_cmi(XYZ, XZ, YZ, Z):
    """
    Computes the Conditional Mutual Information based on the given probability distributions.

    Args:
        XYZ (np.ndarray): Probabilities of the joint distribution of X, Y, and Z.
        XZ (np.ndarray): Probabilities of the joint distribution of X and Z.
        YZ (np.ndarray): Probabilities of the joint distribution of Y and Z.
        Z (np.ndarray): Probabilities of the distribution of Z.

    Returns:
        float: The calculated CMI value.
    """
    h_xyz = entropy(XYZ)
    h_xz = entropy(XZ)
    h_yz = entropy(YZ)
    h_z = entropy(Z)
    return h_xz + h_yz - h_xyz - h_z


def preparing_data(content: pd.DataFrame, binary: bool):
    """
    Prepares the dataset for training and testing by scaling features and splitting the data.

    Args:
        content (pd.DataFrame): The input DataFrame containing the data.
        binary (bool): Indicates whether to convert the target variable to binary.

    Returns:
        tuple: A tuple containing the training and testing data (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test).
    """
    y = content[list(content.columns)[-1]]
    if binary == True:
        y.replace(21, 0, inplace=True)
        for idx in y.index:
            if y[idx] != 0:
                y[idx] = 1
        print(y.value_counts())
    X = content.drop(columns=[list(content.columns)[-1]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def predict_kcf(content, dir_name, test_instance, idx, i):
    """
    Predicts class probabilities for a given test instance using the KCF model.

    Args:
        content (pd.DataFrame): The input DataFrame containing the data.
        dir_name (str): Directory name for saving results.
        test_instance (pd.Series): The test instance to predict.
        idx (int): The index of the test instance.
        i (int): The fold index for organizing output.

    Returns:
        tuple: A tuple containing the highest average probability and the corresponding class.
    """
    print("Columns", content.columns)
    unique_classes = content[list(content.columns)[-1]].unique()
    # print(f'Predict unique diagnosis {unique_classes}')
    graphs_files = glob.glob(f"{MAIN_PATH_KCF}/{dir_name}/graphs/fold {i}/*.adjlist")
    # print(f'graphs {graphs_files}')
    for graph in graphs_files:
        averaged_probabilities = {int(c): 0.0 for c in unique_classes}
        root_name = graph.split("root_")[1].split("_node")[0]
        cpt_files = glob.glob(
            f"{MAIN_PATH_KCF}/{dir_name}/CPT/fold {i}/{root_name}/*csv"
        )
        # cpt_files=glob.glob(f"{MAIN_PATH_KCF}/{dir_name}/CPT normalized fold {i}/{root_name}/*csv")
        # print(f'cpt_files {cpt_files}')
        for cpt_file in cpt_files:
            # zawsze bedzie class w cpt -> zapewne po to było łaczenie
            print("Cpt file", cpt_file)
            cpt = pd.read_csv(cpt_file)
            prob_c = 1.0
            class_probabilities = {int(c): 0.0 for c in unique_classes}
            columns_list = list(cpt.columns[:-2])
            columns_list.remove(f"{list(content.columns)[-1]}")
            # print(f'CPT columns selectec {columns_list}')
            if len(columns_list) != 0:
                for c in unique_classes:
                    filtered_cpt = cpt[cpt[list(content.columns)[-1]] == c]
                    for parent in columns_list:
                        value = test_instance[parent]
                        prob_row = filtered_cpt[filtered_cpt[parent] == value]
                        if not prob_row.empty:
                            prob_c *= prob_row.iloc[0][filtered_cpt.columns[-1]]
                            # print('Prob row',type(prob_row),prob_row.iloc[0][filtered_cpt.columns[-1]]) # jedna bo to założenia, przeciez brało unikalne kombinacje
                    class_probabilities[c] = prob_c
                # print(f'Class prob {class_probabilities}')
                name = cpt_file.split(f"{root_name}\\")[1].replace(".csv", "")
                # print(f'Name {name}, class prob {class_probabilities}')
                os.makedirs(
                    f"{MAIN_PATH_KCF}/{dir_name}/probabilities/fold_{i}/{idx}/{root_name}",
                    exist_ok=True,
                )
                # os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/probabilities normalized/{idx}/{root_name}",exist_ok=True)
                # with open(f"{MAIN_PATH_KCF}/{dir_name}/probabilities normalized/{idx}/{root_name}/{name}_class_prob.json",'w') as f:
                with open(
                    f"{MAIN_PATH_KCF}/{dir_name}/probabilities/fold_{i}/{idx}/{root_name}/{name}_class_prob.json",
                    "w",
                ) as f:
                    json.dump(class_probabilities, f, indent=len(unique_classes))
                for c in unique_classes:
                    averaged_probabilities[c] += class_probabilities[c] / len(cpt_files)
        if len(list(averaged_probabilities.values())) != 0:
            os.makedirs(
                f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities/{idx}/{root_name}",
                exist_ok=True,
            )
            # os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities normalized/{idx}/{root_name}",exist_ok=True)
            # with open(f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities normalized/{idx}/{root_name}/avg_prob.json",'w') as f:
            with open(
                f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities/{idx}/{root_name}/avg_prob.json",
                "w",
            ) as f:
                json.dump(averaged_probabilities, f, indent=len(unique_classes))
    return compute_averaged_probabilities(unique_classes, dir_name, idx, i)


def compute_averaged_probabilities(unique_classes, dir_name, idx, i):
    """
    Computes averaged class probabilities from individual probability files.

    Args:
        unique_classes (np.ndarray): Array of unique class labels.
        dir_name (str): Directory name for saving results.
        idx (int): The index of the test instance.
        i (int): The fold index for organizing output.

    Returns:
        tuple: A tuple containing the highest average probability and the corresponding class.
    """
    averaged_probabilities_files = glob.glob(
        f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities/{idx}/*/*.json"
    )
    # averaged_probabilities_files=glob.glob(f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities normalized/{idx}/*/*.json")
    averaged_probabilities_all_KCF = {int(c): 0.0 for c in unique_classes}
    print(
        f"Avg [{averaged_probabilities_all_KCF}],len {len(averaged_probabilities_files)}"
    )
    for file_name in averaged_probabilities_files:
        with open(file_name, "r") as f:
            data = json.load(f)
        print(f"Data {data}")
        for key in list(data.keys()):
            print(f"Key {key}")
            averaged_probabilities_all_KCF[int(key)] += data.get(key)

    print(f"{averaged_probabilities_all_KCF}")
    for key in averaged_probabilities_all_KCF:
        averaged_probabilities_all_KCF[key] = averaged_probabilities_all_KCF[key] / len(
            averaged_probabilities_files
        )
    os.makedirs(
        f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities_all/fold_{i}/",
        exist_ok=True,
    )
    # os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities_normalized_all",exist_ok=True)
    # with open(f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities_normalized_all/avg_prob_{idx}.json",'w') as f:
    with open(
        f"{MAIN_PATH_KCF}/{dir_name}/averaged_probabilities_all/fold_{i}/avg_prob_{idx}.json",
        "w",
    ) as f:
        json.dump(averaged_probabilities_all_KCF, f, indent=len(unique_classes))
    sorted_avg_prob = {
        k: v
        for k, v in sorted(
            averaged_probabilities_all_KCF.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    return (
        sorted_avg_prob[list(sorted_avg_prob.keys())[0]],
        list(sorted_avg_prob.keys())[0],
    )


def calculate_cmi_all_cases(content, dir_name, i: int, binary: bool):
    """
    Calculate Conditional Mutual Information (CMI) for all cases and save the results.

    Parameters:
    content (pd.DataFrame): The DataFrame containing the data for CMI calculation.
    dir_name (str): The directory name where the results will be saved.
    i (int): The fold index for cross-validation.
    binary (bool): Whether the target variable should be treated as binary.

    This function reads a variants CSV file to determine which features to analyze, calculates the CMI between
    pairs of features conditioned on a target feature, and saves the results in a specified directory.
    """
    df = pd.read_csv(f"{MAIN_PATH_KCF}/{dir_name}/variants.csv")
    column_list = content.columns[:-1]
    print(len(column_list), column_list)
    # print(len(df.loc[0]),df.loc[0].values)
    if binary == True:
        content[list(content.columns)[-1]].replace(21, 0, inplace=True)
        for idx in content[list(content.columns)[-1]].index:
            if content[list(content.columns)[-1]][idx] != 0:
                content[list(content.columns)[-1]][idx] = 1
    Z = list(content.columns)[-1]
    cmi_data = []
    for _, row in df.iterrows():
        data = row.values
        if list(data).count(1) == 2:
            print()
            X_idx = list(data).index(1)
            Y_idx = list(data)[X_idx + 1 :].index(1) + X_idx + 1
            print(f" X idx {X_idx}, yidx {Y_idx}")
            X = column_list[X_idx]
            Y = column_list[Y_idx]

            cmi = conditional_mutual_information(content, dir_name, X, Y, Z, i)
            cmi_data.append({"cmi": cmi, "X": X, "Y": Y, "Z": Z})
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}", exist_ok=True)
    pd.DataFrame(data=cmi_data, columns=["cmi", "X", "Y", "Z"]).to_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}/fold_{i}.csv", index=False
    )


def generate_cases(content: pd.DataFrame, dir_name):
    """
    Generate combinations of features to analyze for Conditional Mutual Information (CMI).

    Parameters:
    content (pd.DataFrame): The DataFrame containing the data from which to generate feature combinations.
    dir_name (str): The directory name where the variants CSV will be saved.

    This function creates all unique combinations of features (excluding "TGB") for CMI analysis
    and saves them to a variants CSV file.
    """
    hormones = ["TSH", "T3", "TT4", "T4U", "FTI"]
    columns_length = content.shape[1] - 1
    code = []
    for i in range(columns_length):
        for j in range(columns_length):
            if i != j:
                variant = [0] * columns_length
                variant[i] = 1
                variant[j] = 1
                if variant not in code:
                    if (
                        list(content.columns)[i] != "TGB"
                        or list(content.columns)[j] != "TGB"
                    ) and (
                        list(content.columns)[j] not in hormones
                        or list(content.columns)[i]
                    ):
                        # print(f' i {i}, {list(content.columns)[i]} j {j},{list(content.columns)[j]}')
                        code.append(variant)
    pd.DataFrame(data=code).to_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/variants.csv", index=False
    )


def analyze_cmi(dir_name: str, i: int):
    """
    Analyze and sort the Conditional Mutual Information (CMI) results.

    Parameters:
    dir_name (str): The directory name where the CMI results are stored.
    i (int): The fold index for cross-validation.

    This function reads the CMI results, sorts them by CMI values in descending order,
    and saves the sorted results to a new CSV file.
    """
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}", exist_ok=True)
    df_cmi = pd.read_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}/fold_{i}.csv"
    )
    df_cmi.sort_values(by=["cmi"], ascending=False).to_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}/sorted_fold_{i}.csv",
        index=False,
    )


def build_undirected_MST(content: pd.DataFrame, dir_name: str, i: int):
    """
    Build an undirected Maximum Spanning Tree (MST) from Conditional Mutual Information (CMI) results.

    Parameters:
    content (pd.DataFrame): The DataFrame containing the data for constructing the MST.
    dir_name (str): The directory name where the results will be saved.
    i (int): The fold index for cross-validation.

    Returns:
    nx.Graph: The Maximum Spanning Tree built from the CMI results.

    This function creates a graph from CMI results, calculates the Maximum Spanning Tree, 
    saves the edge weights to a CSV file, and visualizes both the original graph and the MST.
    """
    G = nx.Graph()
    df_cmi = pd.read_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}/fold_{i}.csv"
    )
    column_list = list(content.columns)[:-1]
    print("w un", column_list)
    for _, row in df_cmi.iterrows():
        G.add_edge(row["X"], row["Y"], weight=round(row["cmi"], coverage))
    mst = nx.maximum_spanning_tree(G)

    weights = nx.get_edge_attributes(mst, "weight")
    weights_data = []
    for edge, weight in weights.items():
        print(f"{edge}: {weight}")
        weights_data.append([edge, weight])
    pd.DataFrame(weights_data, columns=["Edge", "Weight"]).to_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/mst_weights/fold_{i}.csv", index=False
    )
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    weights = nx.get_edge_attributes(mst, "weight")
    rounded_weights = {k: round(v, coverage) for k, v in weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_weights)

    draw_graph(
        G,
        f"{MAIN_PATH_KCF}/{dir_name}/undirected_graph/fold_{i}_undirected_graph.png",
        False,
        pos=pos,
    )

    # Maximum Spanning Tree
    draw_graph(
        mst,
        f"{MAIN_PATH_KCF}/{dir_name}/undirected_mst/fold_{i}_undirected_mst.png",
        False,
        pos=pos,
    )

    return mst


def add_edges(parent, graph, directed_graph):
    """
    Recursively add edges from a parent node to a directed graph.

    Parameters:
    parent: The parent node from which edges will be added.
    graph (nx.Graph): The undirected graph containing edges.
    directed_graph (nx.DiGraph): The directed graph to which edges will be added.

    This function checks if edges already exist in the directed graph and adds them if not,
    recursively processing each child of the parent node.
    """
    for child in graph.neighbors(parent):
        if directed_graph.has_edge(child, parent) or directed_graph.has_edge(
            parent, child
        ):
            continue
        weight = graph.get_edge_data(parent, child)["weight"]
        print(f"Weight {weight}")
        directed_graph.add_edge(parent, child, weight=weight)
        add_edges(child, graph, directed_graph)


def visualize_cpt(cpt_df, parents, target, name, dir_name, i):
    """
    Visualize the Conditional Probability Table (CPT) using bar plots and heatmaps.

    Parameters:
    cpt_df (pd.DataFrame): The DataFrame containing the CPT data to visualize.
    parents (list): A list of parent nodes for the target variable.
    target (str): The target variable for which the CPT is being visualized.
    name (str): The name of the visualization.
    dir_name (str): The directory name where the visualizations will be saved.
    i (int): The fold index for cross-validation.

    This function creates a bar plot and a heatmap for the CPT and saves the visualizations
    in the specified directory.
    """
    print(f"Parents {parents}")
    cpt_df["Parents_Values"] = cpt_df[parents].astype(str).agg(", ".join, axis=1)

    if cpt_df["Parents_Values"].nunique() > MAX_CATEGORIES:
        top_combinations = (
            cpt_df["Parents_Values"].value_counts().nlargest(MAX_CATEGORIES).index
        )  # top MAX_CATEGORIES parents values
        cpt_df = cpt_df[cpt_df["Parents_Values"].isin(top_combinations)]
    # Rysowanie wykresu
    fig = plt.figure(
        figsize=(min(15, max(8, len(cpt_df["Parents_Values"].unique()) * 0.75)), 6)
    )
    sns.barplot(
        x="Parents_Values",
        y=f'P({target} | {", ".join(parents)})',
        hue=target,
        data=cpt_df,
    )
    plt.title(f"Conditional Probability Table (CPT) for {target}")
    plt.xlabel(f'Parent Values {", ".join(parents)}')
    plt.ylabel("Conditional Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(
        f"{MAIN_PATH_KCF}/{dir_name}/CPT - visualization/fold {i}/{name}", exist_ok=True
    )
    plt.savefig(
        f"{MAIN_PATH_KCF}/{dir_name}/CPT - visualization/fold {i}/{name}/{target}.png"
    )
    plt.close(fig)
    plt.clf()
    plt.cla()

    heatmap_data = cpt_df.pivot_table(
        index=parents, columns=target, values=f'P({target} | {", ".join(parents)})'
    )
    mask = heatmap_data.isnull()
    plt.figure(
        figsize=(
            max(10, len(heatmap_data.columns) * 0.8),
            max(10, len(heatmap_data.index) * 0.5),
        )
    )

    sns.set_context("notebook", font_scale=1.3)
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="viridis",
        cbar_kws={"label": "Prawdopodobieństwo warunkowe"},
        mask=mask,
    )
    plt.title(f"Mapa ciepła tabeli prawdopodobieństw warunkowych (CPT) dla {target}")
    plt.ylabel("Kombinacje rodziców")
    plt.xlabel("Wartość docelowa")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(
        f"{MAIN_PATH_KCF}/{dir_name}/CPT - visualization/fold {i}/{name} heatmaps",
        exist_ok=True,
    )
    plt.savefig(
        f"{MAIN_PATH_KCF}/{dir_name}/CPT - visualization/fold {i}/{name} heatmaps/{target}.png"
    )
    plt.close(fig)
    plt.clf()
    plt.cla()


def compute_cpt(df, directed_mst, class_col, name, dir_name, i):
    """
    Computes the Conditional Probability Table (CPT) for each node in the given directed minimum spanning tree (MST).

    Args:
        df (pd.DataFrame): The dataframe containing the data used for computing the CPT.
        directed_mst (nx.DiGraph): The directed minimum spanning tree.
        class_col (str): The name of the class column in the dataframe.
        name (str): The name used for saving the CPT files.
        dir_name (str): The directory where the CPT files will be saved.
        i (int): The fold number for saving and loading the relevant files.

    Returns:
        None: This function does not return any value but saves the CPT as CSV files.
    """
    with open(f"{MAIN_PATH_KCF}/{dir_name}/parents/fold {i}.txt", "a") as f:
        cpt = {}
        for node in directed_mst.nodes:
            if not os.path.exists(
                f"{MAIN_PATH_KCF}/{dir_name}/CPT/fold {i}/{name}/{node}.csv"
            ):
                node_list = []
                node_list.append(node)
                parents = list(directed_mst.predecessors(node))
                parent_list = [parent for parent in parents]
                print(f"Parents {parents} node {node}")
                if len(parent_list) != 0 and len(parent_list):
                    # if node[0]==class_col:
                    #     node='diagnosis_extracted'
                    # elif class_col in parent_list:
                    #     parent_list[parent_list.index(class_col)]='diagnosis_extracted'
                    # print(f'Parents {parent_list} node {node}')
                    f.write(f"{parents}\n")

                    count_data = defaultdict(int)
                    for idx, row in df.iterrows():
                        key = tuple(
                            row[parent_list]
                        )  # wartości w wierszu dla kolumn rodziców
                        print("chec", row[parent_list], row[parent_list].notna().all())
                        if row[parent_list].notna().all() and not np.isnan(row[node]):
                            count_data[
                                (key, row[node])
                            ] += 1  # (wartości w wierszu dla kolumn rodziców, diagnoza)
                        else:
                            print(f" Nie dodaje {row[parent_list]}")
                    print("Cound data", count_data)

                    parent_counts = defaultdict(int)
                    for key in count_data:
                        parent_key = key[0]
                        parent_counts[parent_key] += count_data[key]

                    cpt = []
                    cpt_normalized = []
                    k = len(
                        set(target_value for key, target_value in count_data.keys())
                    )
                    probabilities = defaultdict(float)
                    alpha = 1
                    for key in count_data:
                        parent_key, target_value = key
                        probability = count_data[key] / parent_counts[parent_key]
                        # probability = (count_data[key] + alpha) / (parent_counts[parent_key] + alpha * k)
                        probabilities[key] = probability
                        row = list(parent_key) + [target_value, probability]
                        cpt.append(row)

                    # normalized_probabilities = defaultdict(float)
                    # for parent_key in set(key[0] for key in count_data.keys()):
                    #     sum_prob = sum(probabilities[(parent_key, target_value)] for target_value in set(key[1] for key in count_data.keys() if key[0] == parent_key))
                    #     for target_value in set(key[1] for key in count_data.keys() if key[0] == parent_key):
                    #         normalized_probabilities[(parent_key, target_value)] = probabilities[(parent_key, target_value)] / sum_prob

                    # for key in count_data:
                    #     parent_key, target_value = key
                    #     probability = normalized_probabilities[(parent_key, target_value)]
                    #     row = list(parent_key) + [target_value, probability]
                    #     cpt_normalized.append(row)

                    # Tworzenie DataFrame z wynikami CPT
                    cpt_columns = parents + [
                        node,
                        "P({} | {})".format(node, ", ".join(parents)),
                    ]
                    print(f"CPT COLUMNS {cpt_columns}")
                    cpt_df = pd.DataFrame(cpt, columns=cpt_columns)
                    # cpt_df_normalized = pd.DataFrame(cpt_normalized, columns=cpt_columns)
                    os.makedirs(
                        f"{MAIN_PATH_KCF}/{dir_name}/CPT/fold {i}/{name}", exist_ok=True
                    )
                    # os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/CPT normalized fold {i}/{name}",exist_ok=True)
                    cpt_df.to_csv(
                        f"{MAIN_PATH_KCF}/{dir_name}/CPT/fold {i}/{name}/{node}.csv",
                        header=cpt_columns,
                        index=False,
                    )
                    # cpt_df_normalized.to_csv(f"{MAIN_PATH_KCF}/{dir_name}/CPT normalized fold {i}/{name}/{node}.csv",header=cpt_columns,index=False)
                    visualize_cpt(
                        cpt_df=cpt_df,
                        parents=parent_list,
                        target=node,
                        name=name,
                        dir_name=dir_name,
                        i=i,
                    )
                    # visualize_cpt(cpt_df=cpt_df_normalized,parents=parent_list,target=node,name=name,dir_name=dir_name)


def count_nodes_to_root(tree, target_node):
    """
    Counts the number of nodes from the target node to the root in the given directed tree.

    Args:
        tree (nx.DiGraph): The directed tree from which to count nodes.
        target_node (str): The target node for which to count the nodes to the root.

    Returns:
        int: The count of nodes from the target node to the root.
    """
    current_node = target_node
    count = 0
    while current_node is not None:
        count += 1
        predecessors = list(tree.predecessors(current_node))
        if not predecessors:
            break
        current_node = predecessors[0]  # Assuming a single predecessor

    return count


def build_directed_MST(
    content: pd.DataFrame, dir_name: str, k: int, i: int, binary: bool
):
    """
    Builds a directed minimum spanning tree (MST) from the given content.

    Args:
        content (pd.DataFrame): The data used to build the directed MST.
        dir_name (str): The directory where the MST and related files will be saved.
        k (int): The maximum number of parents allowed for each node.
        i (int): The fold number for saving and loading the relevant files.
        binary (bool): Indicates whether the last column in content should be treated as binary.

    Returns:
        None: This function does not return any value but saves the directed MST as a graph and related files.
    """
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/directed MST/fold {i}", exist_ok=True)
    os.makedirs(
        f"{MAIN_PATH_KCF}/{dir_name}/directed MST - adding class node/fold {i}",
        exist_ok=True,
    )
    os.makedirs(
        f"{MAIN_PATH_KCF}/{dir_name}/directed MST - extra arcs//fold {i}", exist_ok=True
    )
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/graphs/fold {i}", exist_ok=True)
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/undirected_graph", exist_ok=True)
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/undirected_mst", exist_ok=True)

    # content=pd.read_csv(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
    df_cmi = pd.read_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/fold_{i}/fold_{i}.csv"
    )
    column_list = list(content.columns)[:-1]
    print(f"Column list {column_list}")
    if binary == True:
        content[list(content.columns)[-1]].replace(21, 0, inplace=True)
        for idx in content[list(content.columns)[-1]].index:
            if content[list(content.columns)[-1]][idx] != 0:
                content[list(content.columns)[-1]][idx] = 1
    if (
        not f"{MAIN_PATH_KCF}/{dir_name}/undirected_graph/fold_{i}_undirected_graph.png"
        in glob.glob(f"{MAIN_PATH_KCF}/{dir_name}/*.png")
    ):
        mst = build_undirected_MST(content, dir_name, i)
    print(f"Column list  w build directed{column_list}")
    for column in column_list:
        root = column
        print(f"Root {root}")
        directed_tree = nx.DiGraph()
        add_edges(root, mst, directed_tree)
        nodes = list(directed_tree.nodes())
        print("Num od nodes in directed MST:", nodes)
        draw_graph(
            directed_tree,
            f"{MAIN_PATH_KCF}/{dir_name}/directed MST/fold {i}/root_{root}_mst.png",
            True,
            root=root,
        )

        # adding class node
        class_node = list(content.columns)[-1]
        print(f" Is directed {nx.is_directed(directed_tree)}")
        directed_tree.add_node(class_node)
        for node in directed_tree.nodes():
            if node != list(content.columns)[-1]:
                print(f"Add class node")
                directed_tree.add_edge(u_of_edge=class_node, v_of_edge=node)

        draw_graph(
            directed_tree,
            f"{MAIN_PATH_KCF}/{dir_name}/directed MST - adding class node/fold {i}root_{root}_mst.png",
            True,
            root=root,
        )

        for node in directed_tree.nodes():
            if node != list(content.columns)[-1]:
                count = count_nodes_to_root(tree=directed_tree, target_node=node)
                m = min(count, k) - 1
                print(f"M {m}")
                cmi_selected = (
                    df_cmi[df_cmi["Y"] == node]
                    .sort_values(by=["cmi"], ascending=False)
                    .iloc[:m]
                )
                for _, row in cmi_selected.iterrows():
                    X_data = row["X"]
                    Y_data = row["Y"]
                    cmi_data = row["cmi"]
                    if not directed_tree.has_edge(
                        X_data, Y_data
                    ) and not directed_tree.has_edge(Y_data, X_data):
                        # print(f'Adding edges {X_data},{Y_data}')
                        directed_tree.add_edge(
                            X_data, Y_data, weight=round(cmi_data, 2)
                        )
        nx.write_adjlist(
            directed_tree,
            path=f"{MAIN_PATH_KCF}/{dir_name}/graphs/fold {i}/root_{root}_node_{node}_k_{k}.adjlist",
        )
        draw_graph(
            directed_tree,
            f"{MAIN_PATH_KCF}/{dir_name}/directed MST - extra arcs/fold {i}/root_{root}_mst.png",
            True,
            root=root,
        )

        compute_cpt(
            content, directed_tree, list(content.columns)[-1], root, dir_name, i
        )

    # visualize_cpt()


def draw_graph(graph, save_path, directed_graph, **kwargs):
    """
    Draws the given graph and saves it to the specified path.

    Args:
        graph (nx.Graph): The graph to be drawn.
        save_path (str): The file path where the graph will be saved as an image.
        directed_graph (bool): Indicates whether the graph is directed.
        **kwargs: Additional arguments for customizing the drawing.

    Returns:
        None: This function does not return any value.
    """
    if directed_graph == True:
        levels = bfs_levels(graph, kwargs.get("root"))
        max_level = max(levels.values())
        level_count = {i: 0 for i in range(max_level + 1)}
        for level in levels.values():
            level_count[level] += 1
        pos = {}
        x_gap = 1  # Odstęp poziomy między węzłami
        y_gap = 1  # Odstęp pionowy między poziomami

        for level in range(max_level + 1):
            num_nodes = level_count[level]
            x_positions = [x_gap * (i - num_nodes / 2) for i in range(num_nodes)]
            nodes_at_level = [node for node in levels if levels[node] == level]
            print(f"Nodes at level {nodes_at_level}")
            for i, node in enumerate(nodes_at_level):
                pos[node] = (x_positions[i], -y_gap * level)

        print(
            f"Pos {pos} len pos {len(pos)} nodes {graph.nodes} len nodes {len(graph.nodes)}, leveles {levels}"
        )
        for node in graph.nodes:
            if node in pos:
                graph.nodes[node]["pos"] = pos[node]
            else:
                # Dodanie pozycji dla węzła klasy
                graph.nodes[node]["pos"] = (5, 0)
            # graph.nodes[node]['pos'] = pos[node]
    else:
        pos = kwargs.get("pos")
        for node in graph.nodes:
            graph.nodes[node]["pos"] = list(pos[node])
    edge_x = []
    edge_y = []
    edge_weight_text = []
    edge_weight_text_pos_x = []
    edge_weight_text_pos_y = []
    for edge in graph.edges(data="weight"):
        # print(f'Edge {edge}')
        x0, y0 = graph.nodes[edge[0]]["pos"]
        x1, y1 = graph.nodes[edge[1]]["pos"]
        edge_weight_text_pos_x.append((x0 + x1) / 2)
        edge_weight_text_pos_y.append((y0 + y1) / 2)
        weight = edge[2]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        if weight == None:
            edge_weight_text.append(None)
        else:
            edge_weight_text.append(round(weight, 4))
    # print(f'Edges f{edge_weight_text}')

    node_x = []
    node_y = []
    node_text = []
    for node in graph.nodes():
        x, y = graph.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        textposition="bottom center",
        hoverinfo="text",
        text=node_text,
        marker=dict(showscale=False, color="lightblue", size=20, line_width=2),
    )

    data = []
    if None not in edge_weight_text:
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            mode="lines",
            text=edge_weight_text,
        )

        edge_labels_trace = go.Scatter(
            x=edge_weight_text_pos_x,
            y=edge_weight_text_pos_y,
            line=dict(width=0.5, color="#888"),
            mode="text",
            text=edge_weight_text,
            hoverinfo="none",
        )
        data = [edge_trace, node_trace, edge_labels_trace]
    else:
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            mode="lines",
            text=[],
        )
        data = [node_trace, edge_trace]

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            width=1200,
            height=1000,
            showlegend=False,
            hovermode="closest",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        ),
    )

    fig.write_image(f"{save_path}")


def bfs_levels(graph, start):
    """
    Performs a breadth-first search (BFS) to determine the levels of nodes in the graph starting from the given node.

    Args:
        graph (nx.Graph): The graph to perform BFS on.
        start (str): The starting node for the BFS.

    Returns:
        dict: A dictionary mapping nodes to their corresponding levels in the graph.
    """
    levels = {}
    queue = [(start, 0)]
    print(f"Queue {queue}")
    visited = set()
    while queue:
        node, level = queue.pop(0)
        print(f"Node {node}")
        if node not in visited:
            visited.add(node)
            levels[node] = level
            for neighbor in graph.successors(node):
                queue.append((neighbor, level + 1))
    print(f"Visited node {visited}")
    return levels


def predict(content: pd.DataFrame, dir_name: str, i: int):
    """
    Predicts the class labels and probabilities for the given instances in the content DataFrame.

    Args:
        content (pd.DataFrame): The DataFrame containing the instances to predict.
        dir_name (str): The directory where the results will be saved.
        i (int): The current fold index used for saving results.

    Returns:
        dict: A dictionary containing accuracy, precision, and recall metrics for the predictions.
    """
    content.columns = [f"X{i+1}" for i in range(content.shape[1])]
    results = []
    # indexes=random.sample(list(content.index), 100)
    for idx in content.index:
        test_instance = content.loc[idx]
        prob, class_idx = predict_kcf(content, dir_name, test_instance, idx, i)
        results.append([test_instance[list(content.columns)[-1]], class_idx, prob])
    df_results = pd.DataFrame(results, columns=["y true", "y pred", "prob"])
    os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/results_cases", exist_ok=True)
    df_results.to_csv(
        f"{MAIN_PATH_KCF}/{dir_name}/results_cases/results_{i}.csv", index=False
    )
    metrics = {
        "acc": accuracy_score(
            df_results["y true"].to_numpy(), df_results["y pred"].to_numpy()
        ),
        "precision": precision_score(
            df_results["y true"].to_numpy(),
            df_results["y pred"].to_numpy(),
            average="macro",
            labels=list(df_results["y true"].unique()),
        ),
        "recall": recall_score(
            df_results["y true"].to_numpy(),
            df_results["y pred"].to_numpy(),
            average="macro",
        ),
    }
    return metrics


def main():
     """
    Main function that orchestrates the data processing, k-fold cross-validation, and prediction.

    It performs the following steps:
    1. Defines the directory names and filenames for input data.
    2. Reads the dataset and prepares the content for processing.
    3. Generates cases and k-fold splits if they do not already exist.
    4. Iterates over the k-fold splits to build models, compute metrics, and save results.

    Returns:
        None: This function does not return any value but saves metrics and results to files.
    """
    # dir_names=['no_lack_of_diagnosis_standardized']
    dir_names = ["no_lack_of_diagnosis_TBG_10_fold_vis"]
    filename_data_to_predict = "thyroid.csv"
    filename = "thyroid_discretized.csv"
    k = 2
    results = []
    for dir_name in dir_names:
        # if not os.path.exists(f"{MAIN_PATH_KCF}/{dir_name}/metrics.csv"):
        os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/cmi_results/", exist_ok=True)
        os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/mst_weights/", exist_ok=True)
        os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/parents/", exist_ok=True)
        os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/results/", exist_ok=True)
        os.makedirs(f"{MAIN_PATH_KCF}/{dir_name}/metrics/", exist_ok=True)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        content = pd.read_csv(f"{MAIN_PATH_KCF}/{dir_name}/{filename}")
        # content=pd.read_csv(f"{MAIN_PATH_DATA_ANALYSES}{dir_name}/{filename}")
        # if dir_name!='no_lack_of_diagnosis_TBG':
        original_columns_name = list(content.columns)
        content.columns = [f"X{i+1}" for i in range(content.shape[1])]
        os.makedirs(f'{MAIN_PATH_KCF}/{dir_name}',exist_ok=True)
        with open(f'{MAIN_PATH_KCF}/{dir_name}/columns.txt','w') as f:
                for i in range(len(original_columns_name)):
                    f.write(f'{original_columns_name[i]} = {list(content.columns)[i]} \n')

        generate_cases(content, dir_name)
        if not os.path.exists(f"{MAIN_PATH_KCF}/{dir_name}/k_fold_indexes.csv"):
            kf_data = []
            for i, (train_index, test_index) in enumerate(kf.split(content)):
                kf_data.append([i, list(train_index), list(test_index)])
            pd.DataFrame(
                kf_data, columns=["k-fold", "train index", "test index"]
            ).to_csv(f"{MAIN_PATH_KCF}/{dir_name}/k_fold_indexes.csv", index=False)

        kf_data = pd.read_csv(f"{MAIN_PATH_KCF}/{dir_name}/k_fold_indexes.csv")
        kf_data["train index"] = kf_data["train index"].apply(
            lambda x: list(map(int, x.strip("[]").split(",")))
        )
        kf_data["test index"] = kf_data["test index"].apply(
            lambda x: list(map(int, x.strip("[]").split(",")))
        )
        print(f" Len {len(kf_data.index)}")
        for idx in range(9, 10):
            print(f"Index {idx}, type {type(idx)}, {kf_data.columns}")
            train_index = kf_data.loc[idx, "train index"]
            test_index = kf_data.loc[idx, "test index"]
            i = kf_data.loc[idx, "k-fold"]
            df_train, df_test = content.iloc[train_index], content.iloc[test_index]
            calculate_cmi_all_cases(df_train, dir_name, i, binary=False)
            analyze_cmi(dir_name, i)
            build_directed_MST(df_train, dir_name, k, i, False)

            metrics = predict(df_test, dir_name, i)
            results.append([metrics["acc"], metrics["precision"], metrics["recall"]])
            with open(f"{MAIN_PATH_KCF}/{dir_name}/results/fold_{i}.json", "w") as f:
                json.dump(metrics, f, indent=len(metrics.keys()))
            metrics = pd.DataFrame(results, columns=["acc", "precision", "recall"])
            metrics.to_csv(f"{MAIN_PATH_KCF}/{dir_name}/metrics/metrics_fold_{i}.csv")


main()
