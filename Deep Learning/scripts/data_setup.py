import codecs
import glob
import json
import logging
import os
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nltk

# nltk.download("punkt_tab")
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Criminisi_algorithm import CriminisiAlgorithm
from numpy import asarray
from PIL import Image
from PIL import Image as im
from PIL import ImageDraw
from pydicom import dcmread
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage import color, io
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import pil_to_tensor

BASE_DIR_DATBASE = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/"
logging.basicConfig(
    level=logging.INFO,
    filename="trained_models.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
PATH_XML_FILES = f"{BASE_DIR_DATBASE}DDTI Thyroid Ultrasound Images/*.xml"
PATH_IMAGES = f"{BASE_DIR_DATBASE}DDTI Thyroid Ultrasound Images/*.jpg"
PATH_DICOM_IMAGES = f"{BASE_DIR_DATBASE}GUMED"

tirads = {"5": 5, "4a": 4, "4b": 4, "4c": 4, "3": 3, "2": 2}
data = {"xml": {"num": 0, "paths": []}, "img": {"num": 0, "paths": []}}

mapping_names = {
    "age": "wiek",
    "sex": "płeć",
    "composition": "skład",
    "echogenicity": "echogenniczność",
    "margins": "marinesy",
    "calcifications": "zwapnienia",
    "tirads": "tirads",
}
labels = {
    "sex": {"F": "kobieta", "M": "meżczyzna", "null": "nie podano"},
    "composition": {
        "null": "nie podano",
        "solid": "stały",
        "predominantly solid": "głównie stały",
        "spongiform": "gąbczasty",
        'predominantly solid"': "głównie torbielowaty",
        "dense": "gęsty",
        "cystic": "torbielowaty",
    },
    "echogenicity": {
        "hyperechogenicity": "hiperechogeniczność",
        "marked hypoechogenicity": "oznaczona hipoechogeniczność",
        "hypoechogenicity": "hipoechogeniczność",
        "isoechogenicity": "izoechogeniczność",
        "null": "nie podano",
    },
    "calcifications": {
        "null": "nie podano",
        "non": "brak",
        "microcalcifications": "mikrozwapnienia",
        "microcalcification": "mikrozwapnienie",
        "macrocalcifications": "makrozwapnienia",
        "macrocalcification": "makrozwapnienie",
    },
    "tirads": {
        "null": "nie podano",
    },
    "margins": {
        "well defined smooth": "dobrze zdefiniowany gładki",
        "well defined": "dobrze zdefiniowany",
        "ill defined": "żle zdefiniowany",
        "ill- defined": "żle zdefiniowany",
        "microlobulated": "mikrolobulowany",
        "macrolobulated": "makrolobulowany",
        "spiculated": "spiczasty",
        "null": "nie podano",
    },
}


class MultiModalDataset(Dataset):
    """
    A dataset class for loading multimodal data including images, captions, and labels.

    Attributes:
        dataframe (pd.DataFrame): The DataFrame containing image paths, captions, and labels.
        transform (callable): Transformations to be applied on images.
        word2idx (dict): A dictionary mapping words to indices for tokenization.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns a tuple (image, caption_tensor, labels) for a given index.
    """

    def __init__(self, dataframe, transform, word2idx):
        self.df = dataframe
        self.transform = transform
        self.word2idx = word2idx

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_id = self.df.loc[index, "Image ID"]
        image_path = self.df.loc[index, "Image path"]
        caption = self.df.loc[index, "Caption"]

        labels = self.df.loc[index, "Tirads"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption_tensor = torch.zeros((100,), dtype=torch.long)
        caption_tokens = nltk.word_tokenize(caption.lower(), language="polish")

        for i, token in enumerate(caption_tokens):
            if token in self.word2idx:
                caption_tensor[i] = self.word2idx[token]
            else:
                caption_tensor[i] = self.word2idx["<unk>"]

        return image, caption_tensor, labels


class DDTIThyroidUltrasoundImagesDataset(Dataset):
    """
    A dataset class for loading thyroid ultrasound images along with their labels.

    Attributes:
        X (list): List of image paths or images.
        y (list): List of corresponding labels for each image.
        transform (callable): Transformations to be applied on images.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns a tuple (transformed_image, label) for a given index.
    """

    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]


def create_multimodal_dataset():
    """
    Creates a multimodal dataset by processing metadata and images,
    applies augmentations, and saves results as a CSV file.

    Iterates over metadata to extract age, gender, TIRADS class, and other
    patient-related data to create a comprehensive dataset with original and
    augmented images. Saves the final dataset to 'mutlimodal_dataset_aug.csv'.
    """
    data = []
    pattern = r"EU-TIRADS[\w\s\-.:]*"
    metadata = pd.read_csv(f"{PATH_DICOM_IMAGES}/usg_out/metadane.csv", sep=",")
    patients_data = load_data()
    augmented_counter = 0
    for _, info in metadata.iterrows():
        if info["usg_tirads_class"] != 1:
            age = info["pacjent_wiek"]
            gender = info["pacjent_plec"]
            gender = "kobieta" if info["pacjent_plec"] == "K" else "mężczyzna"
            image_id = info["filename"]
            cleaned_text = re.sub(pattern, "", info["usg_opis"])
            cleaned_text = " ".join(cleaned_text.split())
            descrption = cleaned_text
            tirads_value = info["usg_tirads_class"]
            images_path = glob.glob(
                f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images Augmentation/{tirads_value}/{image_id}/*.png"
            )
            text = (
                f"Wiek pacjenta to {age} lat, płec pacjenta to {gender}. {descrption}"
            )
            print(
                f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images Augmentation/{tirads_value}/{image_id}, {images_path}"
            )
            for path in images_path:

                transform = Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
                    ]
                )
                image = Image.open(path)
                augmented_image = transform(image)
                augmented_image = transforms.ToPILImage()(augmented_image)
                aug_path = f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/MultiModal/augmented_{augmented_counter}.jpg"
                augmented_image.save(
                    f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/MultiModal/augmented_{augmented_counter}.jpg"
                )
                augmented_counter += 1
                data.append([image_id, text, path, tirads_value - 2])
                data.append([image_id, text, aug_path, tirads_value - 2])

    for _, patient in patients_data.iterrows():
        if patient["tirads"]:
            tirads_value = tirads[patient["tirads"]]
            image_id = patient["number"]
            images_path = glob.glob(
                f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images Augmentation/{tirads_value}/{image_id}/*.png"
            )
            age = patient["age"] if patient["age"] else "nie podano"
            sex = labels["sex"].get(patient["sex"], "nie podano")
            composition = labels["composition"].get(
                patient["composition"], "nie podano"
            )
            echogenicity = labels["echogenicity"].get(
                patient["echogenicity"], "nie podano"
            )
            margins = labels["margins"].get(patient["margins"], "nie podano")
            calcifications = labels["calcifications"].get(
                patient["calcifications"], "nie podano"
            )
            print(
                f"{BASE_DIR_DATBASE}Thyroid Ultrasound Images Augmentation/{tirads_value}/{image_id}, {images_path}"
            )
            text = f"Wiek pacjenta to {age} lat, płeć pacjenta to {sex}. Kompozycja {composition}, echogenniczność {echogenicity}, marginesy {margins}, zwapnienia {calcifications}"
            for path in images_path:

                transform = Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
                    ]
                )
                image = Image.open(path)
                augmented_image = transform(image)
                augmented_image = transforms.ToPILImage()(augmented_image)
                aug_path = f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/MultiModal/augmented_{augmented_counter}.jpg"
                augmented_image.save(
                    f"C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/MultiModal/augmented_{augmented_counter}.jpg"
                )
                augmented_counter += 1
                data.append([image_id, text, path, tirads_value - 2])
                data.append([image_id, text, aug_path, tirads_value - 2])

    pd.DataFrame(data, columns=["Image ID", "Caption", "Image path", "Tirads"]).to_csv(
        f"{BASE_DIR_DATBASE}/MultiModal/mutlimodal_dataset_aug.csv", index=False
    )


def read_data():
    """
    Reads and processes DICOM files from a directory, retrieving specific attributes.

    Collects paths of DICOM files, extracts a set of non-image attributes, and prints
    the list of extracted DICOM attributes.
    """

    dicom_files = [
        path.replace("\\", "/")
        for path in glob.glob(f"{PATH_DICOM_IMAGES}/*")
        if ".csv" not in path
        and ".jpg" not in path
        and ".png" not in path
        and ".jpeg" not in path
    ]
    dicom_files += [
        path.replace("\\", "/")
        for path in glob.glob(f"{PATH_DICOM_IMAGES}/*/*")
        if ".csv" not in path
        and ".jpg" not in path
        and ".png" not in path
        and ".jpeg" not in path
    ]
    dicom_attributes = set()
    meta_data = []
    for path in dicom_files:
        if not Path(path).is_dir():
            dicom = dcmread(path)
            dicom_attributes.update(dicom.dir())
    dicom_attributes = list(dicom_attributes)
    dicom_attributes.remove("PixelData")
    dicom_attributes.remove("PatientName")
    print(dicom_attributes)


def count_data(paths: str | list[str]) -> dict:
    """
    Counts XML and JPG files in the given directories and stores the paths.

    Args:
        paths (str or list of str): Paths to search for XML and JPG files.

    Returns:
        dict: A dictionary containing counts and paths for XML and image files.
    """
    for path in paths:
        files_path = glob.glob(path)
        if "xml" in path:
            data["xml"]["num"] = len(files_path)
            data["xml"]["paths"] = files_path
        elif "jpg" in path:
            data["img"]["num"] = len(files_path)
            data["img"]["paths"] = files_path
        else:
            logging.warning(f"Not expected data type {path}")

    logging.info(
        f"Num of xml files : {data['xml']['num']}, num of jpg files: {data['img']['num']}"
    )
    return data


def generate_dataset(data):
    """
    Generates and saves a dataset by combining XML and image data.

    Processes XML files to extract patient data, associates it with relevant images,
    and saves the combined data into a CSV file.

    Args:
        data (dict): Dictionary containing paths to XML and image files.
    """
    data = []
    xml_files = data["xml"]["paths"]
    img_files = data["img"]["paths"]
    for xml_file in xml_files:
        patient_data = {}
        # logging.info(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data_svg = []
        for child in root:
            logging.info(f"{child.tag}:{child.text}")
            if child.tag == "mark":
                images_num = []
                svg = []
                # data=[]
                for itm in child:
                    if itm.tag == "image":
                        images_num.append(itm.text)
                        # print(xml_file,images_num)
                    else:
                        svg.append(itm.text)
                data_svg.append(
                    [
                        {"num": images_num[idx], "svg": svg[idx]}
                        for idx in range(len(images_num))
                    ]
                )
                # print(data)
                # print('----------------------------------------------')
                # patient_data.update({child.tag:data})
            else:
                patient_data.update({child.tag: child.text})
        patient_data.update({child.tag: data_svg})
        img_path_regexp = re.findall(r"\b\d+", xml_file)[0]
        logging.info(
            f"Regex {img_path_regexp}, patient number{patient_data['number']}_"
        )
        # img_paths_patient=[ codecs.decode(img_file,'unicode_escape')for img_file in img_files if patient_data['number'] == re.findall(r'\b\d+',img_file)[0]]
        img_paths_patient = [
            img_file.replace("/", "\\")
            for img_file in img_files
            if patient_data["number"] == re.findall(r"\b\d+", img_file)[0]
        ]
        mask_paths_patient = [
            img_file.split(".jpg")[0] + "_masks.jpg" for img_file in img_paths_patient
        ]
        patient_data.update({"masks_path": mask_paths_patient})
        patient_data.update({"images_path": img_paths_patient})
        logging.info(f"Adding data {patient_data}")
        data.append(patient_data)
    pd.DataFrame(data, columns=list(data[0].keys())).to_csv(
        "patient_data.csv", index=False
    )

    # json_object = json.dumps(patient_data)
    # with open("patient_data.json", "a") as f:
    #     f.write(f"{json_object} \n")


def removal_artefacts(img_path: str) -> Image:
    """
    Crops unwanted areas from the input image to remove artifacts.

    Args:
        img_path (str): The path to the image file.

    Returns:
        np.array: Cropped image array.
    """
    img = Image.open(fp=img_path)
    (left, upper, right, lower) = (120, 10, 430, 300)
    img_crop = img.crop((left, upper, right, lower))
    return np.array(img_crop)


def create_masks(data_frame: pd.DataFrame):
    """
    Creates masks for each image based on SVG annotations in XML files.

    Args:
        data_frame (pd.DataFrame): DataFrame containing image paths and SVG mask data.
    """
    for idx in data_frame.index:
        images_path = data_frame["images_path"][idx]
        masks_path = data_frame["masks_path"][idx]
        logging.info(f"Index {idx}")
        for index, img_path in enumerate(images_path):
            logging.info(img_path)
            for mask in data_frame["mark"][idx]:
                regex = re.findall(r"_\d+", img_path)[0]
                logging.info(f"Tag {mask}")
                if mask[0]["num"] == regex.replace("_", ""):
                    print(f"img {img_path}, regex {regex}")
                    img = Image.open(img_path)
                    img_size = img.size
                    new_img = Image.new(mode="1", size=img_size)
                    # new_img.show()
                    draw = ImageDraw.Draw(new_img)
                    points = mask[0]["svg"]
                    logging.info(points)
                    x_coordinates = [
                        int(itm) for itm in re.findall(r'"x":\s*(\d+)', points)
                    ]
                    y_coordinates = [
                        int(itm) for itm in re.findall(r'"y":\s*(\d+)', points)
                    ]
                    coordinates = list(zip(x_coordinates, y_coordinates))
                    logging.info(f"Coordinates num :{mask[0]['num']} {coordinates}")
                    draw.polygon(coordinates, fill="white", outline="white")
                    # new_img.save(masks_path[idx])
                    new_img.show()


def analyze_data(data_frame: pd.DataFrame):
    """
    Analyzes the DataFrame for missing values and logs counts by column.

    Args:
        data_frame (pd.DataFrame): DataFrame containing dataset information.
    """
    logging.info(f"Nan values in data frame {data_frame.isna().sum().sum()}")
    for item in data.columns:
        logging.info(
            f"Number of Nan values in column {item} {data_frame[item].isna().sum().sum()}"
        )


def load_data() -> pd.DataFrame:
    """
    Loads patient data from a JSON file into a DataFrame.

    Returns:
        pd.DataFrame: The DataFrame containing patient data.
    """
    df_read_json = pd.read_json(
        "C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/patient_data.json",
        lines=True,
    )
    logging.info("DataFrame using pd.read_json() method:")
    logging.info(df_read_json)
    return df_read_json


def create_dataset(data: pd.DataFrame):
    """
    Creates a dataset with images and labels using a specified transformation pipeline.

    Args:
        data (pd.DataFrame): DataFrame containing image paths and labels.

    Returns:
        Dataset: A PyTorch dataset with transformed images and labels.
    """
    transform_basic = Compose(
        [
            ToTensor(),
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data = [
        {"class": tirads[data["tirads"][idx]], "img_path": data["images_path"][idx]}
        for idx in data["tirads"].index
        if data["tirads"][idx] != None
    ]

    x_data = []
    y_data = []
    for itm in data:
        x_data.append(Image.open(itm["img_path"][0]))
        y_data.append(torch.tensor(itm["class"]))
    # for itm in data:
    datasets = DDTIThyroidUltrasoundImagesDataset(x_data, y_data, transform_basic)
    #     datasets.append(dataset)
    return datasets


def split_datset(dataset: list) -> list[Subset]:
    """
    Splits the dataset into training and testing subsets.

    Args:
        dataset (list): The full dataset to be split.

    Returns:
        list[Subset]: The training and testing subsets.
    """
    generator1 = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(dataset, [0.8, 0.2], generator=generator1)
    logging.info(f"Train set {len(train_set)} test set {len(test_set)}")
    return train_set, test_set


def create_dataloaders(
    train_set: Subset, test_set: Subset, BATCH_SIZE: int
) -> list[DataLoader]:
    """
    Creates DataLoader instances for the training and test datasets.

    Args:
        train_set (Subset): The training dataset subset.
        test_set (Subset): The testing dataset subset.
        BATCH_SIZE (int): The batch size for data loading.

    Returns:
        list[DataLoader]: The DataLoaders for training and test data.
    """
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, test_dataloader


def setup_data(BATCH_SIZE: int) -> list[DataLoader]:
    """
    Sets up data loading by reading data, creating datasets, splitting,
    and creating DataLoaders.

    Args:
        BATCH_SIZE (int): The batch size for data loading.

    Returns:
        list[DataLoader]: DataLoaders for training and testing.
    """
    df = load_data()
    dataset = create_dataset(df)
    train_set, test_set = split_datset(dataset=dataset)
    train_dataloader, test_dataloader = create_dataloaders(
        train_set=train_set, test_set=test_set, BATCH_SIZE=BATCH_SIZE
    )
    return train_dataloader, test_dataloader


def removal_of_artefacts():
    """
    Removes blue and yellow artifacts from ultrasound images using a color mask,
    saves the processed images, and logs the results.
    """
    img_files = glob.glob(f"{BASE_DIR_DATBASE}/GUMED/usg_out/*.png")
    print(img_files)
    os.makedirs(f"{BASE_DIR_DATBASE}/usg_removal_of_artefacts_test/", exist_ok=True)
    for img_file in img_files:
        if "3k" in img_file:
            img_name = img_file.split("usg_out\\")[1]
            if not os.path.exists(
                f"{BASE_DIR_DATBASE}/usg_removal_of_artefacts/{img_name}"
            ):

                image = imread(img_file)
                image = image[:400, :, :3].copy()
                hsv_image = color.rgb2hsv(image)

                blue_lower = np.array([0.2, 0.2, 0.2])  # (H, S, V)
                blue_upper = np.array([0.8, 1.0, 1.0])

                yellow_lower = np.array([0.05, 0.3, 0.3])
                yellow_upper = np.array([0.2, 1.0, 1.0])

                blue_mask = (
                    (hsv_image[:, :, 0] >= blue_lower[0])
                    & (hsv_image[:, :, 0] <= blue_upper[0])
                    & (hsv_image[:, :, 1] >= blue_lower[1])
                    & (hsv_image[:, :, 1] <= blue_upper[1])
                    & (hsv_image[:, :, 2] >= blue_lower[2])
                    & (hsv_image[:, :, 2] <= blue_upper[2])
                )

                yellow_mask = (
                    (hsv_image[:, :, 0] >= yellow_lower[0])
                    & (hsv_image[:, :, 0] <= yellow_upper[0])
                    & (hsv_image[:, :, 1] >= yellow_lower[1])
                    & (hsv_image[:, :, 1] <= yellow_upper[1])
                    & (hsv_image[:, :, 2] >= yellow_lower[2])
                    & (hsv_image[:, :, 2] <= yellow_upper[2])
                )

                combined_mask = np.zeros_like(hsv_image[:, :, 0])
                combined_mask[blue_mask | yellow_mask] = 1
                combined_mask = binary_dilation(combined_mask, iterations=3)
                print(
                    f"combined_mask shape {combined_mask.shape}, image shape {image[:,:,:3].shape}"
                )
                Image.fromarray(
                    CriminisiAlgorithm(
                        img_name=img_name,
                        image=image[:, :, :3],
                        mask=combined_mask,
                        patch_size=9,
                        plot_progress=False,
                    ).inpaint()
                ).save(
                    f"{BASE_DIR_DATBASE}/usg_removal_of_artefacts/{img_name}",
                    quality=100,
                )
                print(f"End")


def split_data(db: str, path: str):
    """
    Splits ultrasound images into left and right parts based on the specified database.

    Args:
        db (str): The database name, either "GUMED" or "DDTI".
        path (str): Path pattern to locate the images.

    Returns:
        None. Saves split images in designated folders.
    """
    images = glob.glob(path)
    print(images)
    if db == "GUMED":
        for image_file in images:
            print(image_file)
            image = imread(image_file)
            _, height = image.shape[0], image.shape[1]
            left_part = image[:, : height // 2, :]
            right_part = image[:, height // 2 :, :]
            dir_name = image_file.split("artefacts\\")[1].replace(".png", "")
            path_save = f"{BASE_DIR_DATBASE}/usg/{dir_name}"
            os.makedirs(f"{BASE_DIR_DATBASE}/usg/{dir_name}", exist_ok=True)
            io.imsave(f"{ path_save}/left_side.png", left_part)
            io.imsave(f"{ path_save}/right_side.png", right_part)
    elif db == "DDTI":
        for image_file in images:
            print(image_file)
            img = removal_artefacts(image_file)
            dir_name = image_file.split("Images\\")[1].replace(".jpg", "").split("_")[0]
            path_save = (
                f"{BASE_DIR_DATBASE}/DDTI Thyroid Ultrasound Images/usg DDTI/{dir_name}"
            )
            os.makedirs(
                f"{BASE_DIR_DATBASE}/DDTI Thyroid Ultrasound Images/usg DDTI/{dir_name}",
                exist_ok=True,
            )
            print(
                glob.glob(
                    f"{BASE_DIR_DATBASE}/DDTI Thyroid Ultrasound Images/usg DDTI/{dir_name}/*.png"
                )
            )
            name_save = (
                "left_side"
                if glob.glob(
                    f"{BASE_DIR_DATBASE}/DDTI Thyroid Ultrasound Images/usg DDTI/{dir_name}/*.png"
                )
                else "right_side"
            )
            print(f"Save name {name_save}")
            io.imsave(f"{ path_save}/{name_save}.png", img)


def group_images_by_TIRADS(db: str, path: str):
    """
    Groups thyroid ultrasound images based on TIRADS classification.

    Args:
        db (str): The database name, either "DDTI" or "GUMED".
        path (str): Path pattern to locate the images.

    Returns:
        None. Organizes and copies images into folders according to TIRADS class.
    """
    tirads = {"5": 5, "4a": 4, "4b": 4, "4c": 4, "3": 3, "2": 2}
    for i in range(1, 6):
        os.makedirs(f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{i}", exist_ok=True)
    if db == "DDTI":
        patients_data = load_data()
        images_file = glob.glob(path)
        DDTI_data = []
        for image_file in images_file:
            print(image_file)
            dir_name = image_file.split("artefacts\\")[1].split("\\")[0]
            # print(f"{dir_name}_")
            for idx, path in enumerate(list(patients_data["images_path"].values)):
                if f"Images\\{dir_name}_" in path[0]:
                    print(patients_data["tirads"][idx])
                    if patients_data["tirads"][idx]:
                        DDTI_data.append(
                            [
                                dir_name,
                                patients_data["tirads"][idx],
                                tirads[patients_data["tirads"][idx]],
                            ]
                        )
                        print(
                            f"'path',{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads[patients_data['tirads'][idx]]}/{dir_name}"
                        )
                        os.makedirs(
                            f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads[patients_data['tirads'][idx]]}/{dir_name}",
                            exist_ok=True,
                        )

                        shutil.copy(
                            image_file,
                            f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads[patients_data['tirads'][idx]]}/{dir_name}/",
                        )
        pd.DataFrame(
            DDTI_data, columns=["dir_name", "tirads_original", "tirads"]
        ).to_csv(
            f"{BASE_DIR_DATBASE}/DDTI Thyroid Ultrasound Images/tirads.csv",
            index=False,
        )
    elif db == "GUMED":
        data_GUMED = []
        filename = f"{BASE_DIR_DATBASE}GUMED/usg_out/metadane.csv"
        patients_data = pd.read_csv(filename)
        dir_list = os.listdir(f"{BASE_DIR_DATBASE}GUMED/usg/")
        for dir_name in dir_list:
            print('dir_name.split("_3k")[0]', dir_name.split("_3k")[0])
            data = patients_data[patients_data["filename"] == dir_name.split("_3k")[0]]
            print(data)
            tirads = data["usg_tirads_class"].values[0]
            for image in glob.glob(f"{BASE_DIR_DATBASE}GUMED/usg/{dir_name}/*.png"):
                print(
                    f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads}/{ dir_name.split('_3k')[0]}/"
                )
                os.makedirs(
                    f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads}/{ dir_name.split('_3k')[0]}",
                    exist_ok=True,
                )
                shutil.copy(
                    image,
                    f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images/{tirads}/{ dir_name.split('_3k')[0]}/",
                )
                data_GUMED.append([dir_name.split("_3k")[0], tirads])
        pd.DataFrame(data_GUMED, columns=["dir_name", "tirads"]).to_csv(
            f"{BASE_DIR_DATBASE}/GUMED/tirads.csv",
            index=False,
        )


def define_transform(model_name: str):
    """
    Defines the image transformation pipeline based on the model.

    Args:
        model_name (str): The name of the model for which the transformations are to be applied.

    Returns:
        transform_basic: Image transformation pipeline (torchvision.transforms.Compose).
    """
    if model_name in [
        "GoogLeNet",
        "ResNet",
        "ResNet18",
        "SqueezeNet1_0",
        "DenseNet161",
        "AlexNet",
    ]:
        transform_basic = Compose(
            [
                ToTensor(),
                Resize((256, 256), antialias=True),
                CenterCrop(224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif model_name == "Inception_V3":
        transform_basic = Compose(
            [
                ToTensor(),
                Resize((342, 342), antialias=True),
                CenterCrop(299),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif model_name == "EfficientNet_V2":
        transform_basic = Compose(
            [
                ToTensor(),
                Resize((480, 480), antialias=True),
                CenterCrop(480),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transform_basic


def split_data_images(path: str, ratio: int, ratio_valid: int, augmentation: bool):
    """
    Splits images into training, validation, and testing sets and manages augmentation if specified.

    Args:
        path (str): The base path containing images by class.
        ratio (int): Proportion of images for training.
        ratio_valid (int): Proportion of images for validation.
        augmentation (bool): Whether to use augmented images in the training set.

    Returns:
        None. Creates directories and organizes images into sets.
    """
    for class_name in range(1, 6):
        dirs_name = glob.glob(f"{path}/{class_name}/*")
        print(dirs_name)
        augmented_images = [
            img for img in dirs_name if "augmented" in os.path.basename(img)
        ]
        regular_images = [
            img for img in dirs_name if "augmented" not in os.path.basename(img)
        ]
        print(augmented_images)
        print(regular_images)

        num_augmented_images = len(augmented_images)
        num = len(regular_images)
        train = int(num * ratio)
        valid = int(num * ratio_valid)
        print(train, valid)

        all_indices = list(range(num))
        train_indices = random.sample(all_indices, train)

        remaining_indices = list(set(all_indices) - set(train_indices))
        valid_indices = random.sample(remaining_indices, valid)

        test_indices = list(set(remaining_indices) - set(valid_indices))
        # train_indexes = random.sample(range(num), train)
        # test_indexes = list(set(np.arange(num)).difference(set(train_indexes)))

        for idx in train_indices:
            images = glob.glob(f"{regular_images[idx]}/*png")
            print(f"Path create {path}/train/{class_name}/{idx}/")
            os.makedirs(f"{path}/train/{class_name}/{idx}/", exist_ok=True)
            print("images", images)
            if augmentation:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED  Augmentation/train/{class_name}/{idx}/"

            else:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED/train/{class_name}/{idx}/"

            for image in images:
                shutil.copy(
                    image,
                    db_path,
                )
        print(f"End train")
        if augmentation:
            for idx, augmented_image in enumerate(augmented_images):
                os.makedirs(f"{path}/train/{class_name}/{idx}_aug/", exist_ok=True)
                shutil.copy(
                    augmented_image,
                    f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED  Augmentation/train/{class_name}/{idx}_aug/",
                )
        print(f"End aug")
        for idx in valid_indices:
            images = glob.glob(f"{regular_images[idx]}/*png")
            os.makedirs(f"{path}/valid/{class_name}/{idx}/", exist_ok=True)
            if augmentation:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED  Augmentation/valid/{class_name}/{idx}/"

            else:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED/valid/{class_name}/{idx}/"

            for image in images:
                shutil.copy(
                    image,
                    db_path,
                )
        print(f"End valid")
        for idx in test_indices:
            os.makedirs(f"{path}/test/{class_name}/{idx}/", exist_ok=True)
            images = glob.glob(f"{regular_images[idx]}/*png")
            if augmentation:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED  Augmentation/test/{class_name}/{idx}/"
            else:
                db_path = f"{BASE_DIR_DATBASE}/Thyroid Ultrasound Images — GUMED/test/{class_name}/{idx}/"
            for image in images:
                shutil.copy(
                    image,
                    db_path,
                )
        print(f"End test")


def augmentation_dataset(data_dir: str):
    """
    Applies augmentations to images in the dataset.

    Args:
        data_dir (str): The directory containing the dataset images.

    Returns:
        None. Saves augmented images to specified directories.
    """
    transform = Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(data_dir)

    for i, (image, label) in enumerate(dataset):
        print(f"label {label}")
        augmented_image = transform(image)

        augmented_image = transforms.ToPILImage()(augmented_image)

        augmented_image.save(
            os.path.join(f"{data_dir}/{label+2}", f"augmented_{i}.jpg")
        )


def count_class_instances(dataloader):
    """
    Counts the number of instances for each class in the dataloader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the dataset.

    Returns:
        dict: A dictionary mapping each class label to its instance count.
    """
    class_count = defaultdict(int)

    for _, labels in dataloader:
        for label in labels:
            class_count[label.item()] += 1

    return dict(class_count)


def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    """
    Visualizes class distribution per batch from the DataLoader.

    Args:
        dl (DataLoader): PyTorch DataLoader containing the dataset.
        id_to_label (dict, optional): A dictionary to map class IDs to human-readable labels.
        with_outputs (bool, optional): Whether to display the visualization.

    Returns:
        tuple: Counts of each class in batches, unique images seen, and indices.
    """
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []
    class_2_batch_counts = []
    class_3_batch_counts = []
    class_4_batch_counts = []

    for i, batch in enumerate(dl):

        idxs = batch[0][:, 0].tolist()
        classes = batch[0][:, 1]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
            class_2_batch_counts.append(class_counts[1])
            class_3_batch_counts.append(class_counts[2])
            class_4_batch_counts.append(class_counts[3])

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 15))

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            label=(id_to_label[0] if id_to_label is not None else "0"),
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.bar(
            ind + width,
            class_2_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.bar(
            ind + width,
            class_3_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.bar(
            ind + width,
            class_4_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.set_xticks(ind, ind + 1)
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        plt.show()

        num_images_seen = len(idxs_seen)

        print(
            f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
        )
        print("=============")
        print(f"Num. unique images seen: {len(set(idxs_seen))}/{total_num_images}")
    return class_0_batch_counts, class_1_batch_counts, idxs_seen


def get_targets(dataset):
    """
    Retrieves the target labels from a dataset, including concatenated datasets.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.ConcatDataset): The dataset
        or concatenated dataset from which to extract targets.

    Returns:
        np.ndarray: Array of target labels from the dataset.

    Raises:
        AttributeError: If the dataset or any sub-dataset does not contain the 'targets' attribute.
    """
    targets = []
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for sub_dataset in dataset.datasets:
            if hasattr(sub_dataset, "targets"):
                targets.extend(sub_dataset.targets)
            else:
                raise AttributeError(
                    f"Sub-dataset {sub_dataset} does not have 'targets' attribute."
                )
    else:
        if hasattr(dataset, "targets"):
            targets.extend(dataset.targets)
        else:
            raise AttributeError("Dataset does not have 'targets' attribute.")

    return np.array(targets)


def create_dataloder(dataset: ImageFolder, batch_size: int, shuffle: bool, valid: bool):
    """
    Creates a DataLoader with WeightedRandomSampler for balancing class distributions in the dataset.

    Args:
        dataset (ImageFolder): The dataset to load data from.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        valid (bool): Indicator for validation phase to handle class counting.

    Returns:
        DataLoader: A DataLoader with a weighted sampler.
    """
    if valid:
        class_counts = np.bincount(get_targets(dataset))
    else:
        class_counts = np.bincount(get_targets(dataset))

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in get_targets(dataset)]
    print(class_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset), replacement=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler
    )
    class_counts = count_class_instances(dataloader)
    print("Liczba przypadków dla każdej klasy:", class_counts)
    # visualise_dataloader(dataloader)
    return dataloader


def prepare_data(
    path: str, model_name: str, augmentation: bool, valid: bool, freeze_all: bool
):
    """
    Prepares the datasets for training, validation, and testing, with optional dataset concatenation.

    Args:
        path (str): The base path where the data is located.
        model_name (str): The name of the model for defining transformations.
        augmentation (bool): Whether data augmentation is applied.
        valid (bool): Indicator for validation phase.
        freeze_all (bool): Whether to concatenate all data for frozen models.

    Returns:
        tuple: DataLoaders for training, validation, and test datasets.
    """
    transform_basic = define_transform(model_name)
    train_dataset = ImageFolder(root=f"{path}/train", transform=transform_basic)
    test_dataset = ImageFolder(root=f"{path}/test", transform=transform_basic)
    valid_dataset = ImageFolder(root=f"{path}/valid", transform=transform_basic)
    print(f"Freeze all:{freeze_all}")
    if not valid and not freeze_all:
        train_dataset = ConcatDataset([train_dataset, valid_dataset])
    elif freeze_all:
        print(f"Freeze all: {len(test_dataset)}")
        test_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        print(f"Freeze all: {len(test_dataset)}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    # dataloaders
    train_dataloder = create_dataloder(train_dataset, 16, True, valid)
    test_dataloder = create_dataloder(test_dataset, 16, False, valid)
    valid_dataloder = create_dataloder(valid_dataset, 16, False, valid)
    return train_dataloder, valid_dataloder, test_dataloder


def prepare_data_multimodel_dataset(IMAGE_SIZE, BATCH_SIZE):
    """
    Prepares multimodal datasets with weighted samplers for balanced training and testing.

    Args:
        IMAGE_SIZE (int): The size of the images after resizing.
        BATCH_SIZE (int): The batch size for the DataLoader.

    Returns:
        tuple: DataLoaders for training and testing datasets.
    """
    tfms = Compose(
        [
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data = pd.read_csv(f"{BASE_DIR_DATBASE}/MultiModal/mutlimodal_dataset_aug.csv")

    data["is_augmented"] = data["Image path"].str.contains("augmented")

    original_data = data[~data["is_augmented"]]
    augmented_data = data[data["is_augmented"]]

    train_original_df, test_original_df = train_test_split(
        original_data, test_size=0.2, stratify=original_data["Tirads"], random_state=42
    )

    train_df = pd.concat([train_original_df, augmented_data]).reset_index(drop=True)
    test_df = test_original_df.reset_index(drop=True)

    train_df = train_df.explode(["Caption", "Image path"]).reset_index(drop=True)
    test_df = test_df.explode(["Caption", "Image path"]).reset_index(drop=True)

    captions = train_df["Caption"].tolist()
    caption_tokens = [nltk.word_tokenize(caption.lower()) for caption in captions]
    word_counts = Counter(token for caption in caption_tokens for token in caption)
    word_counts = {k: v for k, v in word_counts.items()}
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab = ["<pad>", "<unk>"] + vocab
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    lengths = [
        len(nltk.word_tokenize(caption.lower(), language="polish"))
        for caption in train_df["Caption"]
    ]
    max_length = max(lengths)
    print(f"Maksymalna długość napisu: {max_length}")
    print(f"Średnia długość napisu: {np.mean(lengths)}")

    print("Zbiór treningowy:")
    print(train_df["Tirads"].value_counts())
    print("\nZbiór testowy:")
    print(test_df["Tirads"].value_counts())

    train_labels = train_df["Tirads"].values
    test_labels = test_df["Tirads"].values
    class_counts_train = np.bincount(train_labels)
    class_counts_test = np.bincount(test_labels)
    class_weights_train = np.where(class_counts_train != 0, 1.0 / class_counts_train, 0)
    class_weights_test = np.where(class_counts_test != 0, 1.0 / class_counts_test, 0)
    sample_weights_train = np.array([class_weights_train[t] for t in train_labels])
    sample_weights_test = np.array([class_weights_test[t] for t in test_labels])
    sampler_train = WeightedRandomSampler(
        weights=sample_weights_train,
        num_samples=len(sample_weights_train),
        replacement=True,
    )
    sampler_test = WeightedRandomSampler(
        weights=sample_weights_test,
        num_samples=len(sample_weights_test),
        replacement=True,
    )

    train_dataset = MultiModalDataset(train_df, transform=tfms, word2idx=word2idx)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler_train
    )

    test_dataset = MultiModalDataset(test_df, transform=tfms, word2idx=word2idx)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, sampler=sampler_test
    )

    return train_dataloader, test_dataloader


def analyze_time(file_path):
    """
    Analyzes time logs in a file to compute average, minimum, and maximum processing times.

    Args:
        file_path (str): Path to the log file containing time measurements.

    Returns:
        tuple: Average, minimum, and maximum times if available; otherwise, returns None for each.
    """
    times = []

    with open(file_path, "r") as file:
        for line in file:
            if "Took" in line and "seconds" in line:
                time_str = line.split("Took")[1].split("seconds")[0].strip()
                try:
                    time_value = float(time_str)
                    times.append(time_value)
                except ValueError:
                    print(f"Nie udało się przekonwertować czasu w linii: {line}")

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return avg_time, min_time, max_time
    else:
        return None, None, None
