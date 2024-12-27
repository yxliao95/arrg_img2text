import argparse
import datetime
import json
import logging
import os
import random
import re
import time

import datasets
import requests
import torch
import yaml
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
)

CONFIG = None
LOGGER = None
DEVICE = None


def test_vision_model(ds_dev):

    model_name_or_path = "/home/yuxiang/liao/resources/downloaded_models/clip-vit-base-patch32"
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = CLIPVisionModel.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = ds_dev[0]["images"][0]
    print(image.size)  # (640, 480)

    inputs = processor(images=image, return_tensors="pt")
    print(inputs["pixel_values"].shape)  # torch.Size([1, 3, 224, 224])

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state.shape)  # torch.Size([1, 50, 768])
    pooled_output = outputs.pooler_output  # pooled CLS states
    print(pooled_output.shape)  # torch.Size([1, 768])


class ImageTextDataset(Dataset):
    def __init__(self, img_dataset, text_dataset, processor):
        # column_names: ['source', 'images_path', 'images', 'section_text', 'doc_key', 'split_sents', 'split_sent_toks', 'sent_idx_split_idx', 'radlex', 'cxrgraph_ent', 'cxrgraph_attr', 'cxrgraph_rel']
        self.processor = processor
        self.samples = self._process_data(img_dataset, text_dataset)

    def _process_data(self, img_dataset, text_dataset):
        filtered_dataset = self._align_img_text(img_dataset, text_dataset)

        def _process_img(batch_samples):
            batch_samples["pixel_values"] = []

            for images in batch_samples["images"]:
                # Each sample may have multiple images
                piexl_values = self.processor(images=images, return_tensors="pt").pixel_values
                batch_samples["pixel_values"].append(piexl_values)

            # 这里的key是表示列; value是iterable (list,tensor都行)，最外层的每个元素都会被视为一行
            return batch_samples

        # The transform function is applied on-the-fly on batches when hf_dataset.__getitem__ is called.
        filtered_dataset.set_transform(transform=_process_img)

        return filtered_dataset

    def _align_img_text(self, img_dataset, text_dataset):
        ds_indices_text_img = []
        for textDs_row_idx, doc_key in enumerate(text_dataset["doc_key"]):
            data_split, imgDs_row_idx, section_name = doc_key.split("#")
            ds_indices_text_img.append((int(textDs_row_idx), int(imgDs_row_idx)))

        sorted_ds_indices_text_img = sorted(ds_indices_text_img, key=lambda x: x[1])

        filtered_img_ds = img_dataset.select([x[1] for x in sorted_ds_indices_text_img])
        filtered_text_ds = text_dataset.select([x[0] for x in sorted_ds_indices_text_img])

        filtered_dataset = concatenate_datasets([filtered_img_ds, filtered_text_ds], axis=1)

        return filtered_dataset

    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def load_datasets(data_paths):

    dataset_interpret = load_from_disk(data_paths["interpret"])
    LOGGER.debug("%s loaded from interpret_cxr", [f"{split}:{len(ds)}" for split, ds in dataset_interpret.items()])
    dataset_mimic = load_from_disk(data_paths["mimic"])
    LOGGER.debug("%s loaded from mimic-cxr", [f"{split}:{len(ds)}" for split, ds in dataset_mimic.items()])

    # Concat both
    dataset_train_dev = DatasetDict({"train": concatenate_datasets([dataset_interpret["train"], dataset_mimic["train"]]), "validation": concatenate_datasets([dataset_interpret["validation"], dataset_mimic["validation"]])})
    dataset_test = load_from_disk(data_paths["interpret-test-public"])

    ds_img = DatasetDict({"train": dataset_train_dev["train"], "validation": dataset_train_dev["validation"], "test": dataset_test["test"]})
    LOGGER.debug("Final image-report dataset: %s", ds_img)

    ds_text = load_from_disk(data_paths["custom_text"])
    LOGGER.debug("Final custom split_text dataset: %s", ds_text)

    return ds_img, ds_text


def load_proj_config(file_name):
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(proj_dir, "config", file_name), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dirs = config["output_dir"]
    output_dirs["result"] = os.path.join(output_dirs["result"], config["output_name"])
    output_dirs["checkpoint"] = os.path.join(output_dirs["checkpoint"], config["output_name"])
    output_dirs["log"] = os.path.join(output_dirs["log"], config["output_name"])
    os.makedirs(output_dirs["result"], exist_ok=True)
    os.makedirs(output_dirs["checkpoint"], exist_ok=True)
    os.makedirs(output_dirs["log"], exist_ok=True)

    return config


def init_logger(log_file_mode="w", log_level=logging.DEBUG, root_log_level=logging.INFO):
    curr_file_name = os.path.basename(os.path.abspath(__file__))
    log_file_path = os.path.join(CONFIG["output_dir"]["result"], f"{curr_file_name}.log")

    file_handler = logging.FileHandler(log_file_path, log_file_mode)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler.setFormatter(stream_formatter)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    logger = logging.getLogger(curr_file_name)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)  # This logger's level
    logger.root.setLevel(root_log_level)  # Other libraries' loggers will inherit this level

    # LOGGER = MultiProcessAdapter(logger, {})
    return logger


def main(img_dataset, text_dataset):
    model_name_or_path = CONFIG["model_name_or_path"]["clip"]
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    ds_dev = ImageTextDataset(img_dataset["validation"], text_dataset["validation"], processor=processor)
    image = ds_dev[0]["images"][0]

    # test_vision_model(ds_dev)


if __name__ == "__main__":
    config_file_name = "exp1_imgcls.yaml"
    CONFIG = load_proj_config(file_name=config_file_name)
    LOGGER = init_logger(log_file_mode="w")
    LOGGER.debug(CONFIG)

    img_dataset, text_dataset = load_datasets(data_paths=CONFIG["data_path"])

    if CONFIG["target_section"] == "findings":
        img_dataset = img_dataset.remove_columns("impression")
        img_dataset = img_dataset.rename_column("findings", "section_text")
    elif CONFIG["target_section"] == "impression":
        img_dataset = img_dataset.remove_columns("findings")
        img_dataset = img_dataset.rename_column("impression", "section_text")
    else:
        raise ValueError(f"Invalid target_section from {config_file_name}, expected 'findings' or 'impression'")

    main(img_dataset, text_dataset)
