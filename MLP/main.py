import datetime
import json
import logging

import wandb
from scripts.engine import train_MLP

logging.basicConfig(
    level=logging.INFO,
    filename="MLP.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

BASE_DIR = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP"


def load_tests_from_json(file_path: str):
    """
    Load test configurations from a JSON file.

    Args:
        file_path (str): Path to the JSON file with test configurations.

    Returns:
        dict: Dictionary containing test configurations.
    """
    with open(file_path, "r") as f:
        tests = json.load(f)
    return tests


def apply_test_config(test):
    """
    Apply and run a specific test configuration for the MLP model.

    Args:
        test (dict): A dictionary containing the test configuration.
    """
    logging.info(f"Running Test ID: {test['test_id']}")
    model_architecture = test["model_architecture"]
    mmlp_option = test["mmlp_option"]
    training = test["training"]
    feature_selection = test["feature_selection"]
    dataset = test["dataset"]
    idx = test["idx"]
    dir_name = f'experiments/{ datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")}'

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="MLP",
    #     name="run 1",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": training["lr"],
    #         "architecture": f"MLP({model_architecture['input_size']},{model_architecture['hidden_size']},{model_architecture['hidden_units']},{model_architecture['output_size']})",
    #         "dataset": f"Garvan Institute - features {dataset['num_features']}",
    #         "epochs": training["epochs"],
    #         "activation": model_architecture["activation"],
    #         "num_classes": model_architecture["output_size"],
    #     },
    # )

    train_MLP(
        input_size=model_architecture["input_size"],
        hidden_size=len(model_architecture["hidden_units"]),
        hidden_units=model_architecture["hidden_units"],
        output_size=model_architecture["output_size"],
        feature_selection_method=feature_selection["method"],
        optimizer_name=training["optimizer"],
        lr=training["lr"],
        adaptive_lr=training["adaptive_lr"],
        activation_function_name=model_architecture["activation"],
        epochs=training["epochs"],
        balanced_database=dataset["balanced_database"],
        batch_size=dataset["batch_size"],
        num_features=dataset["num_features"],
        option=dataset["option"],
        mmlp_option=mmlp_option,
        dir_name=dir_name,
        BASE_DIR=BASE_DIR,
        idx=idx,
        weighted_loss_function=training["weighted_loss_function"],
        scheduler_name=training["scheduler_name"],
    )

    # wandb.finish()


tests = load_tests_from_json(f"{BASE_DIR}/tests_plan_MLP_hidden.json")
print(tests)
tests_name = ["tests_BasicMMLP"]
for test_name in tests_name:
    for test in tests[test_name]:
        apply_test_config(test)
