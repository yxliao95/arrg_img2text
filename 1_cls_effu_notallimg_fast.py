#############################################
# 基于 0_img_cls_effusion_notallimg.py
# 改进：使用A100时比V100慢，考虑动态调整图像数量时的遍历导致的性能问题，改用torch的批处理操作
#############################################
import argparse
import datetime
import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import datasets
import imagehash
import mlflow
import numpy as np
import requests
import torch
import transformers
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import GradientAccumulationPlugin, gather, gather_object, set_seed
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    PretrainedConfig,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

CONFIG = None
LOGGER = None
TENSORBOARD = None
DEVICE = None
ACCELERATOR = None
STATUS_INFO = None
MLFLOW_TRACKER = None

#############################################
# Model Design
#############################################


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_model"

    def __init__(self, base_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        self.base_config = base_config

        # 当从检查点恢复模型时，需要主动选择 PretrainedConfig 的实现类
        if vision_config and not isinstance(vision_config, PretrainedConfig):
            if self.base_config["vision_backbone"] == "clip":
                from transformers import CLIPVisionConfig

                vision_config = CLIPVisionConfig(**vision_config)

        self.vision_config = vision_config
        self.num_labels = self.base_config["num_classes"] if self.base_config else 2


class CustomModel(PreTrainedModel):
    config_class = CustomModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_encoder = CLIPVisionModel(config.vision_config)
        self.classifier = torch.nn.Linear(self.config.vision_config.hidden_size, config.num_labels)

    def reshape_img_features(self, last_hidden_state, image_indices_map):
        # image_indices_map 是一个嵌套list，每个样本对应一个list，list中的元素是图像在 last_hidden_state 中的索引
        # e.g. [[0], [1], [2, 3], ...]
        # 更新后的索引为 [[0, 0], [1, 1], [2, 3], ...] 方便使用select和reshape进行批量处理
        updated_image_indices_map = []
        for img_indices in image_indices_map:
            if len(img_indices) == 1:
                updated_image_indices_map.append([img_indices[0], img_indices[0]])
            else:
                updated_image_indices_map.append([img_indices[0], img_indices[1]])

        select_indices = torch.tensor(updated_image_indices_map, device=DEVICE)
        reshaped_features = last_hidden_state[select_indices]  # (bsz, 2, num_features, dim)

        return reshaped_features

    def forward(self, input_dict, return_loss=False):
        outputs = self.vision_encoder(pixel_values=input_dict["pixel_values"])
        last_hidden_state = outputs.last_hidden_state

        # (bsz, 2, num_features, dim)
        img_features = self.reshape_img_features(last_hidden_state[:, 1:, :], input_dict["image_indices_map"])
        # (bsz, dim)
        pooled_features = torch.mean(img_features, dim=[1, 2])

        logits = self.classifier(pooled_features)

        if return_loss:
            labels = input_dict["effusion_labels"]
            num_labels = labels.size(-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

            return {
                "logits": logits,
                "loss": loss,
            }

        else:
            return {
                "logits": logits,
            }


class ImageTextDataset(Dataset):
    def __init__(self, img_dataset, text_dataset, processor):
        # column_names: ['source', 'images_path', 'images', 'section_text', 'doc_key', 'split_sents', 'split_sent_toks', 'sent_idx_split_idx', 'radlex', 'cxrgraph_ent', 'cxrgraph_attr', 'cxrgraph_rel']
        self.src_path = os.path.dirname(img_dataset.cache_files[0]["filename"]) if img_dataset.cache_files else ""
        self.processor = processor
        self.samples = self._process_data(img_dataset, text_dataset)
        self.label_counter = None

    def _process_data(self, img_dataset, text_dataset):
        # filtered_dataset = self._align_img_text(img_dataset, text_dataset)
        new_ds = self._concat_img_to_text(img_dataset, text_dataset)
        self.label_counter = Counter([tuple(i) for i in new_ds["effusion_label"]])
        self.print_label_distribution()

        def _process_img(batch_samples):
            batch_samples["selected_pixel_values"] = []
            batch_samples["selected_image_indices"] = []

            for images in batch_samples["images"]:
                # Each sample may have multiple images
                # 根据相似度选择2张图片：如果小于等于2张图片，就直接使用；否则，选择最不相似的两张图片
                selected_images = []
                selected_image_indices = []
                if len(images) <= 2:
                    selected_images = images
                    selected_image_indices = list(range(len(images)))  # [0] / [0, 1]
                else:

                    n = len(images)
                    max_hash_distance = -1
                    for i in range(0, n):
                        for j in range(i + 1, n):
                            # phash值越小，表示两张图片越相似，
                            # 默认的 hash_size=8 会生成一个 8x8=64 位的哈希值（位字符串）。
                            # 更大的 hash_size 会提取图像中更多的细节，因此更能区分复杂图像，但对噪声和微小变化的鲁棒性会降低。
                            # 较小的 hash_size 更关注图像整体的结构和轮廓信息，因此对噪声或轻微变化更鲁棒，但可能忽略细节差异。
                            # 可能出现不同的 images 的 hash_distance=0 的情况。
                            hash_distance = abs(imagehash.phash(images[i]) - imagehash.phash(images[j]))
                            if hash_distance > max_hash_distance:
                                max_hash_distance = hash_distance
                                selected_images = [images[i], images[j]]
                                selected_image_indices = [i, j]

                piexl_values = self.processor(images=selected_images, return_tensors="pt").pixel_values
                batch_samples["selected_pixel_values"].append(piexl_values)
                batch_samples["selected_image_indices"].append(selected_image_indices)

            # 这里的key是表示列; value是iterable (list,tensor都行)，最外层的每个元素都会被视为一行
            return batch_samples

        # The transform function is applied on-the-fly on batches when hf_dataset.__getitem__ is called.
        new_ds.set_transform(transform=_process_img)

        return new_ds

    def _concat_text_to_img(self, img_dataset, text_dataset):
        rowId_img2text_map = {}
        for textDs_row_idx, doc_key in enumerate(text_dataset["doc_key"]):
            data_split, img_id, section_name = doc_key.split("#")
            rowId_img2text_map[int(img_id)] = int(textDs_row_idx)

        # 因为要添加了额外的分量任务，所以text中没有的数据，在img中也要保留
        textDs_column_names = text_dataset.column_names

        def map_func(example):
            # 将 text_ds 的数据拼接到 img_ds 的数据中
            # 由于可能在测试时使用裁剪后的 img_ds 数据集，所以使用 img_id 列的值来作为 key
            img_id = example["img_id"]
            if img_id in rowId_img2text_map:
                textDs_row_idx = rowId_img2text_map[img_id]
                textDS_row = text_dataset[textDs_row_idx]
            else:
                textDS_row = {col: None for col in textDs_column_names}
            example.update(textDS_row)

            example["effusion_label"] = self._get_effusion_label(col_cxrgraph_ent=textDS_row["cxrgraph_ent"], col_radlex=textDS_row["radlex"])  # [present, absent, uncertain]
            return example

        new_dataset = img_dataset.map(map_func)

        return new_dataset

    def _concat_img_to_text(self, img_dataset, text_dataset):
        ds_textRowId_imgId = []
        for textDs_row_idx, doc_key in enumerate(text_dataset["doc_key"]):
            data_split, img_id, section_name = doc_key.split("#")
            ds_textRowId_imgId.append((int(textDs_row_idx), int(img_id)))

        # 按照 img_id 排序
        sorted_ds_textRowId_imgId = sorted(ds_textRowId_imgId, key=lambda x: x[1])

        # 如果传入的是裁剪后的 img_ds 数据集，那么 img_id 与 img_row_id 不一定是一一对应的
        ds_imgId_imgRowId = {img_id: img_row_id for img_row_id, img_id in enumerate(img_dataset["img_id"])}

        # 按照 img_id 的顺序，将 img_ds 的数据拼接到 text_ds 的数据中
        filtered_img_ds = img_dataset.select([ds_imgId_imgRowId[img_id] for _, img_id in sorted_ds_textRowId_imgId if img_id in ds_imgId_imgRowId])
        filtered_text_ds = text_dataset.select([text_row_id for text_row_id, _ in sorted_ds_textRowId_imgId])

        filtered_dataset = concatenate_datasets([filtered_img_ds, filtered_text_ds], axis=1)

        def map_func(example):
            example["effusion_label"] = self._get_effusion_label(col_cxrgraph_ent=example["cxrgraph_ent"], col_radlex=example["radlex"])  # [present, absent, uncertain]
            return example

        new_dataset = filtered_dataset.map(map_func)

        return new_dataset

    # effusion 的分类任务标签 one-hot: [present, absent, uncertain]
    def _get_effusion_label(self, col_cxrgraph_ent, col_radlex):
        is_effusion_uncertain = False
        if not col_cxrgraph_ent:
            return [0, 1, 0]  # 默认为absent

        for sent_ents, sent_radlexes in zip(col_cxrgraph_ent, col_radlex):
            # 获取radlex中的effusion和pleural effusion
            candidate_nodes = []
            for radlex in sent_radlexes:
                if any([rid == radlex["radlex_id"] for rid in ["http://radlex.org/RID/RID4872", "http://radlex.org/RID/RID38588", "http://radlex.org/RID/RID34539"]]):
                    candidate_nodes.append(radlex)

            # 如果ent的start和end，被candidate_radlex覆盖，就返回True
            def has_matched_candidate(tok_start, tok_end):
                for start, end in [radlex["tok_indices"] for radlex in candidate_nodes]:
                    if tok_start >= start and tok_end <= end:
                        return True
                return False

            # 通过radlex来选择cxrgraph的ent
            # 如果没有与radlex匹配的ent，那么就默认没有effusion；
            # 如果有匹配的ent：
            # 1. 只要有一个 effusion ent 被预测为 Present，就视为有effusion
            # 2. 对于plueral，其 type 为 Anatomy，不会对默认值产生影响 （默认没有 effusion）
            for ent in sent_ents:
                tok_start, tok_end = ent["tok_indices"]
                if has_matched_candidate(tok_start, tok_end):
                    if ent["ent_type"] == "Observation-Present":
                        # 只要有一个 effusion ent 被预测为 present, 就直接返回 present
                        return [1, 0, 0]
                    elif ent["ent_type"] == "Observation-Uncertain":
                        # 如果有一个 effusion ent 被预测为 Uncertain，且没有其他被预测为 present 的 effusion ent，就视为 Uncertain
                        is_effusion_uncertain = True

        return [0, 0, 1] if is_effusion_uncertain else [0, 1, 0]

    def print_label_distribution(self):
        key_map = {(1, 0, 0): "present", (0, 1, 0): "absent", (0, 0, 1): "uncertain"}
        LOGGER.info("Effusion label distribution: %s", {key_map[k]: v for k, v in sorted(self.label_counter.items(), key=lambda x: x[0])})
        LOGGER.info("  of dataset: %s", self.src_path)

    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch_data):

    # 1. 图像数据，转换为tensor。对于数量大于2的图像，只取前两个
    # TODO 通过相似度判断，选择最不相似的两个图像
    example_img = batch_data[0]["selected_pixel_values"][0]  # torch.Size([3, 224, 224])
    pixel_values = []
    image_indices_map = []  # e.g. [[0], [1], [2, 3], ...]
    img_count = 0
    for item_idx, batch_item in enumerate(batch_data):
        num_images = batch_item["selected_pixel_values"].size(0)
        pixel_values.extend(batch_item["selected_pixel_values"])
        image_indices_map.append(list(range(img_count, img_count + num_images)))
        img_count += num_images

    pixel_val_tensors = torch.stack(pixel_values)

    effusion_labels = torch.tensor([i["effusion_label"] for i in batch_data])

    return {
        "pixel_values": pixel_val_tensors.to(DEVICE),  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "effusion_labels": effusion_labels.float().to(DEVICE),  # [bsz, 3]
        "image_indices_map": image_indices_map,  # [[0], [1], [2, 3], ...]
        "data_id_list": [i["doc_key"] for i in batch_data],
    }


#############################################
# Training and Evaluation
#############################################


@dataclass
class StatusInfo:
    curr_epoch: int = field(default=0)
    curr_batch_iter: int = field(default=0)
    curr_check_point: str = field(default="")  # "epoch" or "batch"
    curr_eval_split: str = field(default="")  # "validation" or "test"

    global_batch_iters: int = field(default=0)
    global_updates: int = field(default=0)
    dev_best: dict = field(default_factory=lambda: {"score": 0.0, "at_epoch": 0, "at_iter": 0, "at_global_iter": 0, "check_at": ""})

    batch_loss: int = field(default=0)
    batch_trained_examples: int = field(default=0)

    run_id: dict = field(default="")

    def update_batch_info(self, curr_epoch=None, curr_iter=None, curr_check_point=None, curr_eval_split=None):
        if curr_epoch is not None:
            self.curr_epoch = curr_epoch
        if curr_iter is not None:
            self.curr_batch_iter = curr_iter
        if curr_check_point is not None:
            self.curr_check_point = curr_check_point
        if curr_eval_split is not None:
            self.curr_eval_split = curr_eval_split

    def add_batch_loss(self, loss, bsz):
        self.batch_trained_examples += bsz
        self.batch_loss += loss * bsz

    def finish_batch(self):
        self.global_batch_iters += 1

    def finish_update(self):
        self.global_updates += 1

    def do_print(self, print_loss_per_n_steps):
        if self.global_updates == 0:
            return False
        if self.global_updates == 1:
            return True
        if self.global_updates % print_loss_per_n_steps == 0:
            return True
        return False

    def print_avg_loss_and_clear(self):
        avg_loss = self.batch_loss / self.batch_trained_examples
        LOGGER.debug(
            "Train proc=%s, epoch=%d, iter=%d, steps=%d, loss=%.9f",
            ACCELERATOR.process_index,
            self.curr_epoch,
            self.curr_batch_iter,
            self.global_updates,
            avg_loss,
            main_process_only=False,
        )
        # TENSORBOARD.add_scalar(f"{CONFIG['output_name']}/loss", avg_loss, self.global_batch_iters)
        self.clear_batch_loss_info()

    def clear_batch_loss_info(self):
        self.batch_loss, self.batch_trained_examples = 0, 0

    def is_achieving_best_dev_score(self, score):
        if score >= self.dev_best["score"]:
            self.dev_best["score"] = score
            self.dev_best["at_iter"] = self.curr_batch_iter
            self.dev_best["at_epoch"] = self.curr_epoch
            self.dev_best["at_global_epoch"] = self.global_batch_iters
            self.dev_best["check_at"] = self.curr_check_point
            return True
        return False

    def state_dict(self):
        return asdict(self)

    def draw_eval_details(self, eval_field, p, r, f1):
        # TENSORBOARD.add_scalar(f"{CONFIG['output_name']}_{self.curr_eval_split}_{eval_field}/pecision", p * 100, self.global_batch_iters)
        # TENSORBOARD.add_scalar(f"{CONFIG['output_name']}_{self.curr_eval_split}_{eval_field}/recall", r * 100, self.global_batch_iters)
        # TENSORBOARD.add_scalar(f"{CONFIG['output_name']}_{self.curr_eval_split}_{eval_field}/f1", f1 * 100, self.global_batch_iters)
        pass

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(self, k):
                self.__setattr__(k, v)


class MLflowTracker:
    def __init__(self, url=None, exp_name=None, run_id=None, run_name=None):
        self.run = None
        self.run_id = None
        self.run_name = None

        if ACCELERATOR.is_main_process:
            mlflow.set_tracking_uri(uri=url)
            exp_name = exp_name if exp_name else os.path.basename(__file__)
            mlflow.set_experiment(exp_name)
            if run_id:
                mlflow.start_run(run_id=run_id, log_system_metrics=True)
            elif run_name:
                mlflow.start_run(run_name=f"{run_name}", log_system_metrics=True)
            else:
                raise ValueError(f"Either run_id or run_name should be provided.")

            self.run = mlflow.last_active_run()
            self.run_id = self.run.info.run_id
            self.run_name = self.run.info.run_name

    def set_tag(self, key, value):
        if ACCELERATOR.is_main_process:
            mlflow.set_tag(key, value)

    def log_params(self, kv_dict: dict):
        if ACCELERATOR.is_main_process:
            mlflow.log_params(kv_dict)

    def log(self, kv_dict, step):
        if ACCELERATOR.is_main_process:
            for k, v in kv_dict.items():
                k = re.sub(r"[\W\s]", "_", k)  # replace all symbols and spaces by _
                mlflow.log_metric(k, v, step)

    def finish(self):
        if ACCELERATOR.is_main_process:
            mlflow.end_run()

    @staticmethod
    def launch_tracker():
        if CONFIG["resume_from_checkpoint"]:
            mlf_tracker = MLflowTracker(url=CONFIG["mlflow_url"], run_id=STATUS_INFO.run_id)
            mlf_tracker.set_tag(str(CONFIG["jobid"]), f"resume:{resume_epoch_start},{resume_iter_start}")
        else:
            # Start a new mlflow run
            mlf_tracker = MLflowTracker(url=CONFIG["mlflow_url"], run_name=CONFIG["output_name"])
            mlf_tracker.set_tag(str(CONFIG["jobid"]), "scratch")
            STATUS_INFO.run_id = mlf_tracker.run_id
            # should only log params once, otherwise 500 errors
            mlf_tracker.log_params({k: v for k, v in CONFIG.items() if k[0] != "_"})
            mlf_tracker.log_params({"use_distributed": ACCELERATOR.use_distributed, "num_process": ACCELERATOR.num_processes, "mixed_precision": ACCELERATOR.mixed_precision, "tracking_process_idx": ACCELERATOR.process_index})
        return mlf_tracker


def train(model, train_dataloader, valid_dataloader):
    train_cfg = CONFIG["train"]

    model_params = list(model.named_parameters())
    assert model_params[0][0].startswith("vision_encoder")  # check the layer name
    assert model_params[-1][0].startswith("classifier")
    vis_enc_params = [(n, p) for n, p in model_params if n.startswith("vision_encoder")]
    classifier_params = [(n, p) for n, p in model_params if n.startswith("classifier")]

    no_decay_names = ["bias", "layer_norm1.weight", "layer_norm2.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in vis_enc_params if any(nd_name in n for nd_name in no_decay_names)], "lr": (train_cfg["lr"]), "weight_decay": 0.0},
        {"params": [p for n, p in vis_enc_params if all(nd_name not in n for nd_name in no_decay_names)], "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n, p in classifier_params], "lr": train_cfg["mlc_lr"], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_num_steps = len(train_dataloader) // train_cfg["grad_accum_steps"] * train_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * train_cfg["warmup_proportion"]), num_training_steps=total_num_steps)

    # Prepare for multi GPUs. All prepared and registered objs will be checkpointed automatically
    model, optimizer, train_dataloader, valid_dataloader, scheduler = ACCELERATOR.prepare(model, optimizer, train_dataloader, valid_dataloader, scheduler)
    ACCELERATOR.register_for_checkpointing(STATUS_INFO)

    # Checkpoint resume
    resume_epoch_start, resume_iter_start = check_and_resume()
    MLFLOW_TRACKER = MLflowTracker.launch_tracker()
    LOGGER.info("p=%d, resume_epoch_start %d, resume_iter_start %d", ACCELERATOR.process_index, resume_epoch_start, resume_iter_start, main_process_only=False)

    LOGGER.info("****************************** Training ******************************")
    LOGGER.info("Total samples = %d, batch size = %d", len(train_dataloader.dataset), train_cfg["batch_size"])
    LOGGER.info("Total epochs = %d, total iterations per epoch = %d", train_cfg["num_epochs"], len(train_dataloader))
    LOGGER.info("Total optimization steps = %d", total_num_steps)
    LOGGER.info("Gradient accumulation steps = %d", train_cfg["grad_accum_steps"])
    LOGGER.info("Resume from epoch %d, iter %d", resume_epoch_start, resume_iter_start)
    check_memory()

    model.zero_grad()
    for curr_epoch in range(resume_epoch_start, train_cfg["num_epochs"]):
        # Ensure dataloader batches is reproducable
        train_dataloader.set_epoch(curr_epoch)

        # We need to skip steps until we reach the resumed step
        # After the first iteration, we need to go back to the original dataloader
        if CONFIG["resume_from_checkpoint"] and curr_epoch == resume_epoch_start and resume_iter_start > 0:
            active_dataloader = ACCELERATOR.skip_first_batches(train_dataloader, resume_iter_start)
        else:
            active_dataloader = train_dataloader

        start = time.time()
        for curr_iter, batch_inputs_dict in enumerate(active_dataloader, start=resume_iter_start if curr_epoch == resume_epoch_start else 0):
            # LOGGER.debug("Train proc=%s, epoch=%s, iter=%s, global_updates=%s, data_ids=%s", ACCELERATOR.process_index, curr_epoch, curr_iter, STATUS_INFO.global_batch_iters, batch_inputs_dict["data_id_list"], main_process_only=False)

            model.train()
            model_accessor = get_model_accessor(model)
            out = model_accessor(input_dict=batch_inputs_dict, return_loss=True)
            loss = out["loss"]

            ACCELERATOR.backward(loss)

            STATUS_INFO.update_batch_info(curr_epoch=curr_epoch, curr_iter=curr_iter)
            STATUS_INFO.add_batch_loss(loss=loss.item(), bsz=batch_inputs_dict["effusion_labels"].size(0))

            if train_cfg["clip_grad_norm"] > 0:
                ACCELERATOR.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            MLFLOW_TRACKER.log(
                {
                    "epoch": curr_epoch,
                    "loss": STATUS_INFO.batch_loss / STATUS_INFO.batch_trained_examples,
                    "lr": scheduler.get_last_lr()[0],
                    "global_updates": STATUS_INFO.global_updates + 1,
                },
                step=STATUS_INFO.global_batch_iters,
            )

            STATUS_INFO.finish_update()
            if STATUS_INFO.do_print(train_cfg["print_loss_per_n_steps"]):
                STATUS_INFO.print_avg_loss_and_clear()

            STATUS_INFO.finish_batch()

            # checkpoint, eval at specific steps:
            if train_cfg["eval_per_steps"] > 0 and STATUS_INFO.global_updates % train_cfg["eval_per_steps"] == 0:
                STATUS_INFO.update_batch_info(curr_check_point="batch", curr_eval_split="validation")
                eval_result_dict = evaluate(model, target_dataloader=valid_dataloader, STATUS_INFO=STATUS_INFO)
                check_and_save(model, eval_result_dict, STATUS_INFO)
                check_memory()

        end = time.time()
        LOGGER.info("Batch training time: %d minutes (including in_batch eval)", (end - start) / 60)
        STATUS_INFO.update_batch_info(curr_check_point="epoch", curr_eval_split="validation")
        eval_result_dict = evaluate(model, target_dataloader=valid_dataloader, STATUS_INFO=STATUS_INFO)
        check_and_save(model, eval_result_dict, STATUS_INFO)

    LOGGER.info("Best achieved: %s", STATUS_INFO.dev_best)


def check_and_save(model, eval_result_dict, STATUS_INFO):
    # Check
    score = 0
    num_metrics = 0
    for metric_key in ["present", "absent", "uncertain"]:
        if metric_key in eval_result_dict:
            score += eval_result_dict[metric_key]
            num_metrics += 1
    score = score / num_metrics
    # TENSORBOARD.add_scalar(f"{CONFIG['output_name']}/{STATUS_INFO.curr_eval_split}_avg_f1", score * 100, STATUS_INFO.global_batch_iters)

    achieved_best = STATUS_INFO.is_achieving_best_dev_score(score)

    # Save the best
    if achieved_best:
        save_model(model, CONFIG["output_dir"]["model"])
        LOGGER.info("Model saved to %s", CONFIG["output_dir"]["model"])

    # checkpointing
    if STATUS_INFO.curr_check_point:
        LOGGER.info("****************************** Checkpoint ******************************")
        LOGGER.info("Current [dev] f1: %.3f, at epoch %d, iter %d (%s)", score * 100, STATUS_INFO.curr_epoch, STATUS_INFO.curr_batch_iter, STATUS_INFO.curr_check_point)
        LOGGER.info("Best [dev] f1: %.3f, at epoch %d, iter %d", STATUS_INFO.dev_best["score"] * 100, STATUS_INFO.dev_best["at_epoch"], STATUS_INFO.dev_best["at_iter"])
        save_model(model, CONFIG["output_dir"]["checkpoint"])
        LOGGER.info("Model checkpointed to %s", CONFIG["output_dir"]["checkpoint"])


def evaluate(model, target_dataloader, STATUS_INFO=None):
    eval_results = {
        "present": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "absent": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "uncertain": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
    }

    LOGGER.info("****************************** Evaluation ******************************")
    LOGGER.info("Source = %s", target_dataloader.dataset.src_path)
    LOGGER.info("Batch size = %d", CONFIG["eval"]["batch_size"])
    LOGGER.info("Num samples = %d", len(target_dataloader.dataset))

    start = time.time()
    model.eval()
    with torch.no_grad():
        for input_tensors_dict in target_dataloader:
            # Model inference
            out = model(input_dict=input_tensors_dict)
            logits = out["logits"]

            _, predicted_labels = logits.max(dim=-1)
            _, gold_labels = input_tensors_dict["effusion_labels"].max(dim=-1)
            preds = predicted_labels.cpu().numpy()  # (8,)
            golds = gold_labels.cpu().numpy()

            key_map = {0: "present", 1: "absent", 2: "uncertain"}
            for gold, pred in zip(golds, preds):
                eval_results[key_map[gold]]["num_gold_label"] += 1
                eval_results[key_map[pred]]["num_pred_label"] += 1

                if gold == pred:
                    eval_results[key_map[gold]]["num_correct_label"] += 1

    # Evaluate the results
    task_f1 = {}
    for eval_field, result_dict in eval_results.items():
        num_corr = result_dict["num_correct_label"]
        num_pred = result_dict["num_pred_label"]
        num_gt = result_dict["num_gold_label"]
        p = num_corr / num_pred if num_corr > 0 else 0.0
        r = num_corr / num_gt if num_corr > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if num_corr > 0 else 0.0
        LOGGER.info("[%s]: P: %.3f, R: %.3f, 【F1: %.3f】", eval_field, p, r, f1 * 100)
        task_f1[eval_field] = f1

        if STATUS_INFO:
            STATUS_INFO.draw_eval_details(eval_field, p, r, f1)

    end = time.time()
    LOGGER.info("Evaluation time: %d minutes", (end - start) / 60)
    check_memory()
    return task_f1


def check_and_resume():
    resume_epoch_start = 0
    resume_iter_start = 0
    resume_from_checkpoint = CONFIG["resume_from_checkpoint"]

    if resume_from_checkpoint:
        checkpoint_dir = resume_from_checkpoint if isinstance(resume_from_checkpoint, str) and os.path.exists(resume_from_checkpoint) else CONFIG.output_checkpoint_dir

        # STATUS_INFO will be loaded automatically in load_state as we have already registered it via ACCELERATOR.register_for_checkpointing(STATUS_INFO)
        ACCELERATOR.load_state(checkpoint_dir)
        LOGGER.info("p=%d, Resumed status info %s", ACCELERATOR.process_index, STATUS_INFO.state_dict(), main_process_only=False)

        # Prepare the next epoch and iter indices for continue training
        if STATUS_INFO.curr_check_point == "epoch":
            resume_epoch_start = STATUS_INFO.curr_epoch + 1
        elif STATUS_INFO.curr_check_point == "batch":
            resume_epoch_start = STATUS_INFO.curr_epoch
            resume_iter_start = STATUS_INFO.curr_batch_iter + 1
        else:
            raise ValueError(f"Invaild STATUS_INFO.curr_check_point: {STATUS_INFO.curr_check_point}")

    return resume_epoch_start, resume_iter_start


def get_model_accessor(model):
    """使用Accelerator后，model 会作为 DistributedDataParallel 的一个attribute（名为module的变量）"""
    if isinstance(model, DistributedDataParallel):
        model_accessor = model.module
    else:
        model_accessor = model
    return model_accessor


#############################################
# Utils
#############################################


def check_memory():
    if not torch.cuda.is_available():
        return
    # 获取当前 GPU 设备的属性
    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    # 获取 GPU 总显存
    total_memory = device_properties.total_memory / 1024**3  # 转换为 GB
    # 获取Torch总占用显存
    total_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    LOGGER.info(f"Memory reserved: {total_reserved:.2f} / {total_memory:.2f} GB")


def seconds_to_time_str(seconds):
    hours = seconds // 3600  # 1小时 = 3600秒
    minutes = (seconds % 3600) // 60  # 除去小时部分后，再计算分钟
    seconds = seconds % 60  # 剩余的秒数

    return f"{hours:.0f}h {minutes:.0f}min {seconds:.1f}s"


def load_model(model_path):
    model = CustomModel.from_pretrained(model_path)
    LOGGER.info("Pre-trained model loaded from %s", model_path)
    return model


def save_model(model, output_dir):
    model.save_pretrained(output_dir)
    LOGGER.info("Model saved to %s", output_dir)


def set_seed(seed):
    random.seed(seed)  # Python的随机性
    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


#############################################


def get_dataloaders(img_dataset, text_dataset, processor, use_debug_subset=False):
    train_cfg = CONFIG["train"]
    eval_cfg = CONFIG["eval"]
    # select是dataset caching 操作，主进程优先或许能快一点
    with ACCELERATOR.main_process_first():
        if use_debug_subset:
            # train_dataset = ImageTextDataset(img_dataset["train"].select(range(550295, 550395)), text_dataset["train"], processor=processor)
            # vaild_dataset = ImageTextDataset(img_dataset["validation"].select(range(14011, 14111)), text_dataset["validation"], processor=processor)
            # test_dataset = ImageTextDataset(img_dataset["test"].select(range(3577, 3677)), text_dataset["test"], processor=processor)

            train_dataset = ImageTextDataset(img_dataset["train"], text_dataset["train"].select(range(170701, 170801)), processor=processor)
            vaild_dataset = ImageTextDataset(img_dataset["validation"], text_dataset["validation"].select(range(4253, 4353)), processor=processor)
            test_dataset = ImageTextDataset(img_dataset["test"], text_dataset["test"].select(range(2036, 2136)), processor=processor)
        else:
            train_dataset = ImageTextDataset(img_dataset["train"], text_dataset["train"], processor=processor)
            vaild_dataset = ImageTextDataset(img_dataset["validation"], text_dataset["validation"], processor=processor)
            test_dataset = ImageTextDataset(img_dataset["test"], text_dataset["test"], processor=processor)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=lambda batch: collate_fn(batch), batch_size=train_cfg["batch_size"], drop_last=True)
    valid_dataloader = DataLoader(vaild_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch), batch_size=eval_cfg["batch_size"], drop_last=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch), batch_size=eval_cfg["batch_size"], drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader


def init_model(model_name_or_path, model_base_cfg):
    LOGGER.info("Initializing model of %s", model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_config = CustomModelConfig(vision_config=config.vision_config, base_config=model_base_cfg)
    model = CustomModel(config=model_config)
    return model


def load_datasets(data_paths):

    dataset_interpret = load_from_disk(data_paths["interpret"])
    LOGGER.debug("%s loaded from interpret_cxr", [f"{split}:{len(ds)}" for split, ds in dataset_interpret.items()])
    dataset_mimic = load_from_disk(data_paths["mimic"])
    LOGGER.debug("%s loaded from mimic-cxr", [f"{split}:{len(ds)}" for split, ds in dataset_mimic.items()])

    # Concat both
    dataset_train_dev = DatasetDict({"train": concatenate_datasets([dataset_interpret["train"], dataset_mimic["train"]]), "validation": concatenate_datasets([dataset_interpret["validation"], dataset_mimic["validation"]])})
    dataset_test = load_from_disk(data_paths["interpret-test-public"])

    ds_img = DatasetDict({"train": dataset_train_dev["train"], "validation": dataset_train_dev["validation"], "test": dataset_test["test"]})
    for split in ds_img:
        ds_img[split] = ds_img[split].add_column("img_id", range(len(ds_img[split])))
    LOGGER.debug("Loaded image-report dataset: %s", ds_img)

    ds_text = load_from_disk(data_paths["custom_text"])

    for split in ds_text:
        ds_text[split] = ds_text[split].add_column("text_id", range(len(ds_text[split])))
    LOGGER.debug("Loaded custom split_text dataset: %s", ds_text)

    if CONFIG["target_section"] == "findings":
        ds_img = ds_img.remove_columns("impression")
        ds_img = ds_img.rename_column("findings", "section_text")
    elif CONFIG["target_section"] == "impression":
        ds_img = ds_img.remove_columns("findings")
        ds_img = ds_img.rename_column("impression", "section_text")
    else:
        raise ValueError(f"Invalid target_section from {config_file_name}, expected 'findings' or 'impression'")

    return ds_img, ds_text


def init_accelerator():
    global ACCELERATOR, DEVICE

    # However, if no_sync is disabled via sync_each_batch=True, then the memory consumption for gradient_accumulation_steps=16 reverts to that of gradient_accumulation_steps=1.
    dataloader_cfg = DataLoaderConfiguration(use_seedable_sampler=True)
    # gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=CONFIG.grad_accum_steps, sync_each_batch=True)
    ACCELERATOR = Accelerator(dataloader_config=dataloader_cfg)
    DEVICE = ACCELERATOR.device

    if ACCELERATOR.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    LOGGER.info("Available cuda: %d", torch.cuda.device_count())
    LOGGER.info("Accelerator state: %s", ACCELERATOR.state)
    LOGGER.info("Accelerator process idx: %d, device: %s", ACCELERATOR.process_index, ACCELERATOR.device)
    LOGGER.info([i for i in CONFIG.items() if i[0][0] != "_"])


def init_logger(log_level=logging.DEBUG, root_log_level=logging.INFO):
    global LOGGER

    log_file_mode = "w"
    if CONFIG["resume_from_checkpoint"]:
        log_file_mode = "a"

    curr_file_name = os.path.basename(os.path.abspath(__file__))
    log_file_path = os.path.join(CONFIG["output_dir"]["result"], f"{curr_file_name}.log")

    file_handler = logging.FileHandler(log_file_path, log_file_mode)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler.setFormatter(stream_formatter)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    LOGGER = logging.getLogger(curr_file_name)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(log_level)  # This logger's level
    LOGGER.root.setLevel(root_log_level)  # Other libraries' loggers will inherit this level

    LOGGER = MultiProcessAdapter(LOGGER, {})


def init_proj_config():
    global CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--config_file", type=str, help=f".yaml file path")

    parser.add_argument("--output_name", type=str)
    parser.add_argument("--jobid", type=int)

    args = parser.parse_args()

    if args.from_bash:
        file_path = args.config_file
    else:
        file_path = "/home/yuxiang/liao/workspace/arrg_img2text/config/0_imgcls.yaml"

    with open(file_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    if args.from_bash:
        CONFIG["output_name"] = args.output_name
        CONFIG["jobid"] = args.jobid
    else:
        CONFIG["jobid"] = "00000"

    output_dirs = CONFIG["output_dir"]
    output_dirs["result"] = os.path.join(output_dirs["result"], CONFIG["output_name"])
    output_dirs["model"] = os.path.join(output_dirs["model"], CONFIG["output_name"])
    output_dirs["checkpoint"] = os.path.join(output_dirs["checkpoint"], CONFIG["output_name"])
    output_dirs["log"] = os.path.join(output_dirs["log"], CONFIG["output_name"])
    os.makedirs(output_dirs["result"], exist_ok=True)
    os.makedirs(output_dirs["model"], exist_ok=True)
    os.makedirs(output_dirs["checkpoint"], exist_ok=True)
    os.makedirs(output_dirs["log"], exist_ok=True)

    for key in CONFIG["train"]:
        if "lr" in key:
            CONFIG["train"][key] = float(CONFIG["train"][key])


#############################################


def main(img_dataset, text_dataset):
    train_cfg = CONFIG["train"]
    eval_cfg = CONFIG["eval"]
    model_base_cfg = CONFIG["model"]

    model_name_or_path = CONFIG["model_name_or_path"][model_base_cfg["vision_backbone"]]
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(img_dataset, text_dataset, processor, use_debug_subset=True)

    model = init_model(model_name_or_path, model_base_cfg)
    model.to(DEVICE)

    start = time.time()
    train(model, train_dataloader, valid_dataloader)
    end = time.time()
    LOGGER.info("Total training time: %d minutes", (end - start) / 60)

    model = load_model(CONFIG["output_dir"]["model"])
    model.to(DEVICE)

    start = time.time()
    evaluate(model, test_dataloader)
    end = time.time()
    LOGGER.info("Final evaluation time: %d minutes", (end - start) / 60)


if __name__ == "__main__":

    init_proj_config()
    init_logger()
    init_accelerator()
    LOGGER.debug(CONFIG)
    set_seed(CONFIG["train"]["seed"])
    img_dataset, text_dataset = load_datasets(data_paths=CONFIG["data_path"])
    STATUS_INFO = StatusInfo()
    # TENSORBOARD = SummaryWriter(log_dir=CONFIG["output_dir"]["log"])

    start0 = time.time()
    main(img_dataset, text_dataset)
    end0 = time.time()
    LOGGER.info("Total time: %d ", seconds_to_time_str(end0 - start0))

    # TENSORBOARD.close()