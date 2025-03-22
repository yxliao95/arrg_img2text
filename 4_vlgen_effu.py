#############################################
# 改用生成模型
#############################################
import argparse
import datetime
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Tuple, Union

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
from scorers.scores import compute_scores
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    Dinov2Config,
    Dinov2Model,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
    Swinv2Config,
    Swinv2Model,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

CONFIG = None
LOGGER = None
TENSORBOARD = None
DEVICE = None
ACCELERATOR = None
STATUS_INFO = None
MLFLOW_TRACKER = None

SPECIAL_TOKENS_MAP = {
    "<|image_token|>": "<|reserved_special_token_1|>",
    "<|image_start|>": "<|reserved_special_token_2|>",
    "<|image_end|>": "<|reserved_special_token_3|>",
}

#############################################
# Model Design
#############################################


@dataclass
class Vision2LanguageOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class VisionLanguageAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.encoder_hidden_size, config.decoder_hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Vision2LanguageModel(VisionEncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):

        super().__init__(config=config, encoder=encoder, decoder=decoder)
        self.config.encoder_hidden_size = self.encoder.config.hidden_size
        self.config.decoder_hidden_size = self.decoder.config.hidden_size

        # replace enc_to_dec_proj with VisionLanguageAdaptor
        self.adaptor = VisionLanguageAdaptor(self.config)
        if hasattr(self, "enc_to_dec_proj"):
            del self.enc_to_dec_proj  # 移除投影层

    def _inject_image_features(self, input_ids, inputs_embeds, image_features):
        # image_indices_map 是一个嵌套list，每个样本对应一个list，list中的元素是图像在 last_hidden_state 中的索引
        # e.g. [[0], [1], [2, 3], ...]

        # replace img features with the <|image_token|> placeholder token in the input text
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        # 保证所有 image_features 都能够被复制到 inputs_embeds 中
        assert special_image_mask.sum() == image_features.numel(), f"special_image_mask.sum()={special_image_mask.sum()}, image_features.numel()={image_features.numel()}, should be equal to guarantee that all image features are copied to inputs_embeds"

        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_loss: Optional[bool] = False,
        assistant_masks: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **kwargs,
    ) -> Union[Tuple, Vision2LanguageOutputWithPast]:
        """Additional args:
        `inputs_embeds`: should represent the text embeddings with image features injected.
        `encoder_outputs`: in inference statge, we encode `pixel_values` and get `encoder_outputs` outside this forward method. This is because the `pixel_values` and `input_ids` have different batch sizes, which cause error in generate().

        If `output_loss` is True, by default we use `input_ids` as `labels`.
        And the `assistant_masks` should be provided to compute the loss.
        `assistant_masks` is provided by `tokenizer.apply_chat_template`.
        `assistant_masks` is a tensor with the same shape as input_ids, and the value is 0 or 1. 0: system/user tokens, 1: assistant tokens, which is the tokens that need to be generated.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if (pixel_values is not None) and (encoder_outputs is not None):
            # train时，传入 pixel_values
            # inference时，第一轮生成，encoder_outputs 由 do_generate 传入；后续生成则都不需要
            raise ValueError("You must not specify both pixel_values and encoder_outputs, choose one of them or leave them None (for cache generation).")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or encoder_outputs is not None) and inputs_embeds is not None:
            raise ValueError("You cannot specify both `pixel_values`/`encoder_outputs` and `inputs_embeds` at the same time, and must specify either one")

        if inputs_embeds is None:
            # get text embeddings
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

        # 如果有encoder_outputs，就不需要再次 encode pixel_values
        if (pixel_values is not None) and (encoder_outputs is None):
            # get img features
            encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)

        if encoder_outputs is not None:
            image_features = encoder_outputs.last_hidden_state  # torch.Size([4, 1370, enc_dim])
            # project image features
            image_features = self.adaptor(image_features)
            # inject image features into text embeddings
            inputs_embeds = self._inject_image_features(input_ids, inputs_embeds, image_features)

        # Text generation. inputs_embeds is used in replace of input_ids on decoder in all cases.
        # In train statge, input_ids is encoded into inputs_embeds and then merged with image features.
        # In inference stage, inputs_embeds is passed from generate(), where the encoding and merging are done in model.do_generate(). We do this in do_generate() as the image_features and input_ids have different batch sizes.
        decoder_outputs = self.decoder(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        logits = decoder_outputs.logits

        # text loss
        loss = None
        if output_loss:
            labels = labels if labels is not None else input_ids

            # Shift so that tokens < n predict n
            if assistant_masks is not None:
                shift_label_mask = assistant_masks[:, 1:]  # torch.Size([bsz, seq_len - 1])
            elif attention_mask is not None:
                shift_label_mask = attention_mask[:, 1:]
            else:
                raise ValueError("assistant_masks or attention_mask should be provided")

            shift_logits = logits[:, :-1, :]  # torch.Size([bsz, seq_len - 1, vocab_size])
            shift_labels = labels[:, 1:]  # torch.Size([bsz, seq_len - 1])
            active_shift_logits = shift_logits[shift_label_mask != 0].contiguous()  # torch.Size([num_acitve_labels, vocab_size])
            active_shift_labels = shift_labels[shift_label_mask != 0].contiguous()  # torch.Size([num_acitve_labels])

            ce_loss_fct = nn.CrossEntropyLoss()
            loss = ce_loss_fct(active_shift_logits, active_shift_labels)

        return Vision2LanguageOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def do_generate(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        # As the batch size between input_ids and pixel_values may be different,
        # we construct a dummy_inputs to ensure generate() method can use the correct batch size,
        # which should be equals to input_ids's batch size.
        dummy_inputs = torch.ones((input_ids.size(0), 1), dtype=torch.long, device=DEVICE)

        # We manually encode the pixel_values here, otherwize generate() will create a wrong encoder output (or error) with the dummy_inputs.
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)

        # self.main_input_name = "inputs_embeds"
        outputs = self.generate(
            inputs=dummy_inputs,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            pad_token_id=kwargs["pad_token_id"],
            bos_token_id=kwargs["bos_token_id"],
            eos_token_id=kwargs["eos_token_id"],
            max_new_tokens=20,
            do_sample=False,
            num_beams=3,
            return_dict_in_generate=True,
            output_logits=True,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        encoder_outputs=None,
        pixel_values=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        """
        Copy from LLaVA.
        Overwritten -- in specific circumstances we don't want to forward image inputs to the model.
        """

        # At the first round, we need encoder_outputs (encoded from pixel_values) and input_ids
        # At the following rounds, we need input_ids (the generated ones) and past_key_values (represent the previous input_embeds for decoder)

        # If we're in cached decoding stage (after the first round), pixel values should be None
        # because input ids do not contain special image token anymore
        model_inputs = self.decoder.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # At the first generation round, we need encoder_outputs to be passed to model. pixel_values has encoded into encoder_outputs in do_generate().
        # e.g.
        # at the first round, cache_position == tensor([   0,    1,    2,  ..., 2805, 2806, 2807])
        # at second round, cache_position == tensor([2808])
        if cache_position[0] == 0:
            # model_inputs["pixel_values"] = pixel_values
            model_inputs["encoder_outputs"] = encoder_outputs

        return model_inputs


class ImageTextDataset(Dataset):
    def __init__(self, hf_dataset, img_processor, tokenizer, split):
        # column_names: ['source', 'images_path', 'images', 'section_text', 'doc_key', 'split_sents', 'split_sent_toks', 'sent_idx_split_idx', 'radlex', 'cxrgraph_ent', 'cxrgraph_attr', 'cxrgraph_rel']
        self.split = split
        self.src_path = os.path.dirname(hf_dataset.cache_files[0]["filename"]) if hf_dataset.cache_files else ""
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.samples = hf_dataset

    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch_data, img_processor, tokenizer, do_inference=False):

    # 处理图像，因为每个样本的图像数量不一样，所以需要image_indices_map来记录每个样本的图像在batch中的索引
    nested_images = [i["images"] for i in batch_data]  # nested list of imgs: [[img1, img2], [img1], ...]
    piexl_values_tensor = img_processor(images=[img for imgs in nested_images for img in imgs], return_tensors="pt", do_convert_rgb=True).pixel_values

    img_count = 0
    image_indices_map = []  # e.g. [[0], [1], [2, 3], ...]
    for item_idx, item_images in enumerate(nested_images):
        num_images = len(item_images)
        assert num_images <= 2, f"num_images should be less equal than 2, but got {num_images}"
        image_indices_map.append(list(range(img_count, img_count + num_images)))
        img_count += num_images

    # 处理对话数据
    conversations = []
    num_image_tokens = tokenizer.num_image_tokens
    for idx, item in enumerate(batch_data):
        num_images = len(image_indices_map[idx])
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiology assistant tasked with interpreting a chest X-ray study."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "num_images": num_images, "num_image_tokens": num_image_tokens},
                    {"type": "text", "text": "Given the chest X-ray images, generate a description of the findings."},
                ],
            },
        ]
        if not do_inference:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": " ".join(item["split_sents"])}],
                }
            )
        conversations.append(conversation)

    # See descriptions for assistant_tokens_mask
    # Assistant tokens are the tokens that need to be generated, we use these tokens to compute the loss
    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask

    tokenizer_kwargs = {"pad_to_multiple_of": 8}

    if do_inference:
        add_generation_prompt = True
        return_assistant_tokens_mask = False
        tokenizer_kwargs["padding_side"] = "left"
    else:
        add_generation_prompt = False
        return_assistant_tokens_mask = True
        tokenizer_kwargs["padding_side"] = "right"

    input_text_tensor_dict = tokenizer.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, tokenize=True, padding=True, return_dict=True, return_tensors="pt", tokenizer_kwargs=tokenizer_kwargs, return_assistant_tokens_mask=return_assistant_tokens_mask)

    assistant_masks = None
    if "assistant_masks" in input_text_tensor_dict:
        assistant_masks = input_text_tensor_dict.assistant_masks.to(DEVICE)

    gold_text_list = None
    if do_inference:
        gold_text_list = [" ".join(i["split_sents"]) for i in batch_data]

    return {
        "pixel_values": piexl_values_tensor.to(DEVICE),  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map,  # [[0], [1], [2, 3], ...]
        "input_ids": input_text_tensor_dict.input_ids.to(DEVICE),
        "attention_mask": input_text_tensor_dict.attention_mask.to(DEVICE),
        "assistant_masks": assistant_masks,
        "data_id_list": [i["doc_key"] for i in batch_data],
        "gold_text_list": gold_text_list,
    }


#############################################
# Training and Evaluation
#############################################


@dataclass
class StatusInfo:
    curr_epoch: int = field(default=0)
    curr_batch_iter: int = field(default=0)
    curr_checkpoint_at: str = field(default="")  # "epoch" or "batch"
    curr_eval_split: str = field(default="")  # "validation" or "test"

    global_iters: int = field(default=0)
    global_update_steps: int = field(default=0)
    dev_best: dict = field(default_factory=lambda: {"text_score": 0.0, "at_epoch": 0, "at_iter": 0, "check_at": ""})

    batch_loss: int = field(default=0)
    batch_trained_examples: int = field(default=0)

    run_id: dict = field(default="")

    def is_achieving_best_dev_score(self, text_score):
        if text_score >= self.dev_best["text_score"]:
            self.dev_best["text_score"] = text_score
            self.dev_best["at_iter"] = self.curr_batch_iter
            self.dev_best["at_epoch"] = self.curr_epoch
            self.dev_best["check_at"] = self.curr_checkpoint_at
            return True
        return False

    def get_resumed_epoch_and_iter(self):
        epoch_resumed = 0
        iter_resumed = 0
        if self.curr_checkpoint_at == "":
            return epoch_resumed, iter_resumed

        # Prepare the next epoch and iter indices for continue training
        if self.curr_checkpoint_at == "epoch":
            epoch_resumed = self.curr_epoch + 1
        elif self.curr_checkpoint_at == "batch":
            epoch_resumed = self.curr_epoch
            iter_resumed = self.curr_batch_iter + 1
        else:
            raise ValueError(f"Invaild STATUS_INFO.curr_checkpoint_at: {self.curr_checkpoint_at}")
        return epoch_resumed, iter_resumed

    def state_dict(self):
        return asdict(self)

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
                mlflow.start_run(run_id=run_id, log_system_metrics=False)
            elif run_name:
                mlflow.start_run(run_name=f"{run_name}", log_system_metrics=False)
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
            epoch_resumed, iter_resumed = STATUS_INFO.get_resumed_epoch_and_iter()
            mlf_tracker.set_tag(str(CONFIG["jobid"]), f"resume:{epoch_resumed},{iter_resumed}")
        else:
            # Start a new mlflow run
            mlf_tracker = MLflowTracker(url=CONFIG["mlflow_url"], run_name=CONFIG["output_name"])
            mlf_tracker.set_tag(str(CONFIG["jobid"]), "scratch")
            STATUS_INFO.run_id = mlf_tracker.run_id
            # should only log params once, otherwise 500 errors
            mlf_tracker.log_params({k: v for k, v in CONFIG.items() if k[0] != "_"})
            mlf_tracker.log_params({"use_distributed": ACCELERATOR.use_distributed, "num_process": ACCELERATOR.num_processes, "mixed_precision": ACCELERATOR.mixed_precision, "tracking_process_idx": ACCELERATOR.process_index})
        return mlf_tracker


def evaluate(model, target_dataloader, output_result=False):
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
    tokenizer = target_dataloader.dataset.tokenizer

    data_ids = []
    pred_seqs = []
    gold_seqs = []

    start = time.time()
    model.eval()
    with torch.no_grad():
        for input_tensors_dict in target_dataloader:
            # Model inference, check args in https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin
            outputs = model.do_generate(
                pixel_values=input_tensors_dict["pixel_values"],
                input_ids=input_tensors_dict["input_ids"],
                attention_mask=input_tensors_dict["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            pred_seq_start_ids = input_tensors_dict["input_ids"].size(1)  # 生成的序列的起始位置
            pred_sequences_ids = outputs.sequences[:, pred_seq_start_ids:]
            pred_sequences = tokenizer.batch_decode(pred_sequences_ids, skip_special_tokens=True)
            gold_sequences = input_tensors_dict["gold_text_list"]

            # Gathers input_data and potentially drops duplicates in the last batch if on a distributed system.
            data_ids.extend(ACCELERATOR.gather_for_metrics(input_tensors_dict["data_id_list"], use_gather_object=True))
            pred_seqs.extend(ACCELERATOR.gather_for_metrics(pred_sequences, use_gather_object=True))
            gold_seqs.extend(ACCELERATOR.gather_for_metrics(gold_sequences, use_gather_object=True))

    assert len(data_ids) == len(set(data_ids)), f"Duplicated data exists, please check {data_ids}"
    assert len(data_ids) == target_dataloader.total_dataset_length, f"Gathered data size ({len(data_ids)}) should equals to dataset size ({target_dataloader.total_dataset_length})"
    # LOGGER.debug("p=%s, len=%s, data_ids: %s", ACCELERATOR.process_index, len(data_ids), data_ids)
    # LOGGER.debug("p=%s, len=%s, pred_seqs: %s", ACCELERATOR.process_index, len(pred_seqs), pred_seqs)
    # LOGGER.debug("p=%s, len=%s, gold_seqs: %s", ACCELERATOR.process_index, len(gold_seqs), gold_seqs)
    if output_result:
        with open(f"{CONFIG['output_dir']['result']}/{target_dataloader.dataset.split}_{ACCELERATOR.process_index}.json", "w") as f:
            f.write(json.dumps({"gold_seqs": [i.tolist() for i in gold_seqs], "pred_seqs": [i.tolist() for i in pred_seqs]}))

    # Evaluate the results
    text_scores_dict = compute_generation_score(gold_text_list=gold_texts, pred_text_list=pred_seqs)
    LOGGER.info("[TextGen]: %s", json.dumps(text_scores_dict))

    if STATUS_INFO:
        for metric_name, metric_val in text_scores_dict.items():
            k = f"{STATUS_INFO.curr_eval_split}_{metric_name}"
            MLFLOW_TRACKER.log({k: metric_val}, step=STATUS_INFO.global_iters)

    end = time.time()
    LOGGER.info("Evaluation time: %s", seconds_to_time_str(end - start))
    check_memory()
    return text_scores_dict


def compute_generation_score(gold_text_list, pred_text_list):
    """Based on the script from https://vilmedic.app/misc/bionlp24/leaderboard#anchor-baseline"""
    if DEVICE.type == "cpu":
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert"]
    else:
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert", "bertscore"]

    refs = [" ".join(wordpunct_tokenize(s.lower())) for s in gold_text_list]
    hyps = [" ".join(wordpunct_tokenize(s.lower())) for s in pred_text_list]

    # https://github.com/jbdel/vilmedic/blob/main/vilmedic/blocks/scorers/scores.py
    out_dict = compute_scores(use_metrics, refs=refs, hyps=hyps, split=None, seed=None, config=None, epoch=None, logger=LOGGER, dump=False)
    out_dict = {k: float(v) for k, v in out_dict.items()}
    return out_dict


# 用于存储每个值的区间
def find_interval(value, thresholds):
    # e.g. thresholds = [0.33, 0.66]
    # value = 0.5, return 1; value = 0.8, return 2; value = 0.66, return 1
    for i, t in enumerate(thresholds):
        if value <= t:
            return i  # 返回所在区间的索引
    return len(thresholds)  # 如果大于所有阈值，返回最后一个区间


def train(model, train_dataloader, valid_dataloader):
    global MLFLOW_TRACKER, STATUS_INFO

    train_cfg = CONFIG["train"]

    # hyperparameters
    unwrapped_model = get_unwrapped_model(model)
    model_params = list(unwrapped_model.named_parameters())

    encoder_params = [(n, p) for n, p in model_params if n.startswith("encoder")]
    decoder_params = [(n, p) for n, p in model_params if n.startswith("decoder")]
    adaptor_params = [(n, p) for n, p in model_params if n.startswith("adaptor")]

    no_decay_names = ["bias", "norm1.weight", "norm2.weight", "layernorm.weight", "layer_scale"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder_params if any(nd in n for nd in no_decay_names)], "lr": (train_cfg["lr"]), "weight_decay": 0.0},
        {"params": [p for n, p in encoder_params if all(nd not in n for nd in no_decay_names)], "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n, p in decoder_params if any(nd in n for nd in no_decay_names)], "lr": (train_cfg["lr"]), "weight_decay": 0.0},
        {"params": [p for n, p in decoder_params if all(nd not in n for nd in no_decay_names)], "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n, p in adaptor_params], "lr": train_cfg["lr"], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_num_steps = len(train_dataloader) // train_cfg["grad_accum_steps"] * train_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * train_cfg["warmup_proportion"]), num_training_steps=total_num_steps)

    # 1. Prepare for multi GPUs. All prepared and registered objs will be checkpointed automatically
    train_dataloader, valid_dataloader, optimizer, scheduler = ACCELERATOR.prepare(train_dataloader, valid_dataloader, optimizer, scheduler)
    STATUS_INFO = StatusInfo()
    ACCELERATOR.register_for_checkpointing(STATUS_INFO)

    # 2. Check and resume checkpoint if needed
    epoch_resumed, iter_resumed = check_status_and_resume_checkpoint()

    # 3. Launch after resuming STATUS_INFO
    MLFLOW_TRACKER = MLflowTracker.launch_tracker()

    LOGGER.info("****************************** Training ******************************")
    LOGGER.info("Total samples = %d, batch size = %d", len(train_dataloader.dataset), train_cfg["batch_size"])
    LOGGER.info("Total epochs = %d, total iterations per epoch = %d", train_cfg["num_epochs"], len(train_dataloader))
    LOGGER.info("Total optimization steps = %d", total_num_steps)
    LOGGER.info("Gradient accumulation steps = %d", train_cfg["grad_accum_steps"])
    LOGGER.info("Accelerator mixed_precision = %s", ACCELERATOR.mixed_precision)

    check_memory()

    model.zero_grad()
    for curr_epoch in range(epoch_resumed, train_cfg["num_epochs"]):
        # Ensure dataloader batches is reproducable
        train_dataloader.set_epoch(curr_epoch)

        # We need to skip steps until we reach the resumed step
        # After the first iteration, we need to go back to the original dataloader
        if CONFIG["resume_from_checkpoint"] and curr_epoch == epoch_resumed and iter_resumed > 0:
            active_dataloader = ACCELERATOR.skip_first_batches(train_dataloader, iter_resumed)
        else:
            active_dataloader = train_dataloader

        start = time.time()
        for curr_iter, batch_inputs_dict in enumerate(active_dataloader, start=iter_resumed if curr_epoch == epoch_resumed else 0):
            with ACCELERATOR.accumulate(model):
                # LOGGER.debug("Train proc=%s, epoch=%s, iter=%s, global_update_steps=%s, data_ids=%s", ACCELERATOR.process_index, curr_epoch, curr_iter, STATUS_INFO.global_iters, batch_inputs_dict["data_id_list"], main_process_only=False)
                # Not necessarily need ACCELERATOR.autocast()
                # Accelerate enables automatic mixed precision, so autocast() is only needed if there are other mixed precision operations besides those performed on loss by backward() which already handles the scaling.
                with ACCELERATOR.autocast():
                    model.train()
                    out = model.forward(output_loss=True, **batch_inputs_dict)
                    loss = out.loss

                ACCELERATOR.backward(loss)
                if train_cfg["clip_grad_norm"] > 0:
                    ACCELERATOR.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_and_update_status(curr_epoch=curr_epoch, curr_iter=curr_iter, loss=loss.item(), bsz=batch_inputs_dict["effusion_labels"].size(0), lr=scheduler.get_last_lr()[0])

                # eval and save
                validation_process(model, valid_dataloader, max_num_iters_per_epoch=len(train_dataloader))

        end = time.time()
        LOGGER.info("Batch training time: %s ", seconds_to_time_str(end - start))

    LOGGER.info("Best achieved: %s", STATUS_INFO.dev_best)
    MLFLOW_TRACKER.finish()


def log_and_update_status(curr_epoch, curr_iter, loss, bsz, lr):
    STATUS_INFO.curr_epoch = curr_epoch
    STATUS_INFO.curr_batch_iter = curr_iter
    STATUS_INFO.batch_trained_examples += bsz
    STATUS_INFO.batch_loss += loss * bsz
    STATUS_INFO.global_iters += 1

    if ACCELERATOR.sync_gradients:
        STATUS_INFO.global_update_steps += 1

    # Logging too often may slow down the process
    print_loss_per_n_steps = CONFIG["train"]["print_loss_per_n_steps"]
    if STATUS_INFO.global_update_steps == 1 or STATUS_INFO.global_update_steps % print_loss_per_n_steps == 0:
        avg_loss = STATUS_INFO.batch_loss / STATUS_INFO.batch_trained_examples

        MLFLOW_TRACKER.log(
            {
                "lr": lr,
                "avg_loss": avg_loss,
                "epoch": STATUS_INFO.curr_epoch,
                "global_update_steps": STATUS_INFO.global_update_steps,
            },
            step=STATUS_INFO.global_iters,
        )

        LOGGER.info(
            "p=%s, Epoch=%d, iter=%d, steps=%d, loss=%.9f",
            ACCELERATOR.process_index,
            STATUS_INFO.curr_epoch,
            STATUS_INFO.curr_batch_iter,
            STATUS_INFO.global_update_steps,
            avg_loss,
            main_process_only=True,
        )
        STATUS_INFO.batch_loss, STATUS_INFO.batch_trained_examples = 0, 0


def validation_process(model, valid_dataloader, max_num_iters_per_epoch):
    train_cfg = CONFIG["train"]
    # global_update_steps == 0 时，默认不评估
    do_eval = True if STATUS_INFO.global_update_steps > 0 else False

    # eval at the end of each epoch
    if STATUS_INFO.curr_batch_iter + 1 == max_num_iters_per_epoch:
        STATUS_INFO.curr_checkpoint_at = "epoch"
        STATUS_INFO.curr_eval_split = "validation"
    # eval at specific steps:
    elif train_cfg["eval_per_steps"] > 0 and STATUS_INFO.global_update_steps % train_cfg["eval_per_steps"] == 0:
        STATUS_INFO.curr_checkpoint_at = "batch"
        STATUS_INFO.curr_eval_split = "validation"
    else:
        do_eval = False

    if do_eval:
        check_memory()
        eval_result_dict = evaluate(model, target_dataloader=valid_dataloader)
        check_results_and_save_model(model, eval_result_dict)


def final_test_process(model, test_dataloader):
    model, test_dataloader = ACCELERATOR.prepare(model, test_dataloader)


def check_results_and_save_model(model, eval_result_dict):
    # Check
    text_score = 0
    num_metrics = 0
    for metric_key in ["BLEU", "ROUGEL", "chexbert-all_micro avg_f1-score", "radgraph_partial", "bertscore"]:
        if metric_key in eval_out:
            TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}/{metric_key}", eval_out[metric_key], log_info["global_steps"])
            text_score += eval_out[metric_key]
            num_metrics += 1
    text_score = text_score / num_metrics

    LOGGER.info("****************************** Checkpoint ******************************")
    LOGGER.info("Current [%s] text_avg_f1: %.3f, at epoch %d, iter %d (%s)", STATUS_INFO.curr_eval_split, text_score * 100, STATUS_INFO.curr_epoch, STATUS_INFO.curr_batch_iter, STATUS_INFO.curr_checkpoint_at)
    LOGGER.info("Best [%s] avg-f1: %.3f, at epoch %d, iter %d", STATUS_INFO.curr_eval_split, STATUS_INFO.dev_best["text_score"] * 100, STATUS_INFO.dev_best["at_epoch"], STATUS_INFO.dev_best["at_iter"])
    MLFLOW_TRACKER.log({f"{STATUS_INFO.curr_eval_split}_text_avg_f1": text_score}, step=STATUS_INFO.global_iters)

    # checkpointing
    save_checkpoint(checkpoint_dir=CONFIG["output_dir"]["checkpoint"])

    # Save the best
    if STATUS_INFO.is_achieving_best_dev_score(text_score):
        save_model(model, CONFIG["output_dir"]["model"])


def check_status_and_resume_checkpoint():
    epoch_resumed, iter_resumed = 0, 0
    resume_from_checkpoint = CONFIG["resume_from_checkpoint"]

    if resume_from_checkpoint:
        LOGGER.info("****************************** Resume checkpoint ******************************")
        # STATUS_INFO will also be loaded automatically in load_state as we have already registered it via ACCELERATOR.register_for_checkpointing(STATUS_INFO)
        checkpoint_dir = resume_from_checkpoint if isinstance(resume_from_checkpoint, str) and os.path.exists(resume_from_checkpoint) else CONFIG["output_dir"]["checkpoint"]
        load_checkpoint(checkpoint_dir)
        LOGGER.info("p=%d, Resumed status info %s", ACCELERATOR.process_index, STATUS_INFO.state_dict(), main_process_only=False)

        # Prepare the next epoch and iter indices for continue training
        epoch_resumed, iter_resumed = STATUS_INFO.get_resumed_epoch_and_iter()
        LOGGER.info("p=%d, epoch_resumed %d, iter_resumed %d", ACCELERATOR.process_index, epoch_resumed, iter_resumed, main_process_only=False)

    return epoch_resumed, iter_resumed


#############################################
# Utils
#############################################


def get_unwrapped_model(model):
    """使用Accelerator后，model 会作为 DistributedDataParallel 的一个attribute（名为module的变量）"""
    if isinstance(model, DistributedDataParallel):
        return ACCELERATOR.unwrap_model(model)
    else:
        return model


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


def load_checkpoint(checkpoint_dir):
    """training_info will be loaded automatically in load_state as we have already registered it via ACCELERATOR.register_for_checkpointing(training_info)"""
    # 如果 checkpoint_dir 是 CONFIG["output_dir"]["checkpoint"]，就选择最新的 checkpoint
    checkpoint_list = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*_iter_*")), key=os.path.getctime)
    if len(checkpoint_list) > 0:
        checkpoint_dir = checkpoint_list[-1]

    ACCELERATOR.load_state(checkpoint_dir)
    LOGGER.info("Checkpoint loaded from %s", checkpoint_dir)


def load_model(model_path):
    model = Vision2LanguageModel.from_pretrained(model_path)
    LOGGER.info("Pre-trained model loaded from %s", model_path)
    return model


def save_checkpoint(checkpoint_dir, max_to_keep=5):
    ckp_path = os.path.join(checkpoint_dir, f"epoch_{STATUS_INFO.curr_epoch}_iter_{STATUS_INFO.curr_batch_iter}")
    ACCELERATOR.save_state(ckp_path, safe_serialization=False)
    LOGGER.info("Checkpoint saved to %s", ckp_path)

    # 如果文件数量超过 max_to_keep，删除旧的 checkpoint
    max_to_keep = CONFIG["max_checkpoints_to_keep"] if CONFIG["max_checkpoints_to_keep"] else max_to_keep
    if ACCELERATOR.is_main_process:
        # os.path.getctime 按创建时间排序
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*_iter_*")), key=os.path.getctime)
        if len(checkpoint_files) > max_to_keep:
            old_checkpoints = checkpoint_files[:-max_to_keep]  # 排除最近的 max_to_keep 个
            for old_checkpoint in old_checkpoints:
                if os.path.isdir(old_checkpoint):
                    shutil.rmtree(old_checkpoint)
                LOGGER.info("Old checkpoint removed: %s", old_checkpoint)


def save_model(model, output_dir):
    unwrapped_model = ACCELERATOR.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, is_main_process=ACCELERATOR.is_main_process, save_function=ACCELERATOR.save)
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
# Data pre-processing
#############################################


def convert_label_str_to_int(label_str, label_type):
    label_map = {"onehot": {"present": [1, 0, 0], "absent": [0, 1, 0], "uncertain": [0, 0, 1]}, "regression": {"present": 1, "absent": 0, "uncertain": 0.5}}
    return label_map[label_type][label_str]


def get_effusion_label(col_cxrgraph_ent, col_radlex, label_type="onehot"):
    """
    label_type:
        onehot: [present, absent, uncertain]
        regression: 1=present, 0.5=uncertain, 0=absent
    """

    is_effusion_uncertain = False
    if not col_cxrgraph_ent:
        return convert_label_str_to_int("absent", label_type)  # 默认为absent

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
                    return convert_label_str_to_int("present", label_type)
                elif ent["ent_type"] == "Observation-Uncertain":
                    # 如果有一个 effusion ent 被预测为 Uncertain，且没有其他被预测为 present 的 effusion ent，就视为 Uncertain
                    is_effusion_uncertain = True

    if is_effusion_uncertain:
        return convert_label_str_to_int("uncertain", label_type)
    else:
        return convert_label_str_to_int("absent", label_type)


def select_images(images):
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

    return selected_images, selected_image_indices


def pre_process_dataset(img_processor, img_dataset, text_dataset, resize_h_w, convert_to_rgb=True):
    # align image_ds to text_ds
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
    LOGGER.debug("Concatenated image-text dataset dict (aligning image_ds to text_ds): \n%s", filtered_dataset)

    def map_func(examples):
        # 添加 effusion label 作为分类标签: onehot: [present, absent, uncertain]
        examples["effusion_label"] = [get_effusion_label(col_cxrgraph_ent=cxrgraph_ent, col_radlex=radlex, label_type="regression") for cxrgraph_ent, radlex in zip(examples["cxrgraph_ent"], examples["radlex"])]

        # Select images
        # 保存图像的piexl_values会占用极大硬盘空间，且极大的减慢模型训练时的数据读取速度。
        # 因此预处理只进行resize
        selected_images_list = []
        selected_indices_list = []
        for example_idx, images_per_example in enumerate(examples["images"]):
            selected_images, selected_indices = select_images(images_per_example)
            # 更适合处理含有精细细节的图像（如 X-ray 图像）。 可以更好地保留图像中高频信息。适合对病灶等微小特征的保留。
            selected_images = [img.resize(resize_h_w, resample=Image.Resampling.LANCZOS) for img in selected_images]
            selected_images_list.append(selected_images)
            selected_indices_list.append(selected_indices)

        examples["images"] = selected_images_list
        examples["selected_indices_list"] = selected_images_list
        return examples

    preprocess_cfg = CONFIG["preprocess"]
    new_dataset = filtered_dataset.map(map_func, batched=preprocess_cfg["batched"], batch_size=preprocess_cfg["batch_size"], num_proc=preprocess_cfg["num_proc"])  #
    LOGGER.debug("Preprocessed final dataset dict: \n%s", new_dataset)
    return new_dataset


def get_dataloaders(ds_dict, img_processor, tokenizer, use_debug_subset=False):
    train_cfg = CONFIG["train"]
    eval_cfg = CONFIG["eval"]
    # select是dataset caching 操作，主进程优先或许能快一点
    with ACCELERATOR.main_process_first():
        if use_debug_subset:
            train_dataset = ImageTextDataset(ds_dict["train"].select(range(len(ds_dict["train"]) - 200, len(ds_dict["train"]))), img_processor=img_processor, tokenizer=tokenizer, split="train")
            vaild_dataset = ImageTextDataset(ds_dict["validation"].select(range(len(ds_dict["validation"]) - 200, len(ds_dict["validation"]))), img_processor=img_processor, tokenizer=tokenizer, split="validation")
            test_dataset = ImageTextDataset(ds_dict["test"].select(range(len(ds_dict["test"]) - 200, len(ds_dict["test"]))), img_processor=img_processor, tokenizer=tokenizer, split="test")
        else:
            train_dataset = ImageTextDataset(ds_dict["train"], img_processor=img_processor, tokenizer=tokenizer, split="train")
            vaild_dataset = ImageTextDataset(ds_dict["validation"], img_processor=img_processor, tokenizer=tokenizer, split="validation")
            test_dataset = ImageTextDataset(ds_dict["test"], img_processor=img_processor, tokenizer=tokenizer, split="test")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer), batch_size=train_cfg["batch_size"], drop_last=True)
    valid_dataloader = DataLoader(vaild_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer, do_inference=True), batch_size=eval_cfg["batch_size"], drop_last=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer, do_inference=True), batch_size=eval_cfg["batch_size"], drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader


def load_preprocessed_dataset(ds_path):
    ds_final = load_from_disk(ds_path)
    LOGGER.info("Loaded pre_processed dataset dict: \n%s", ds_final)
    return ds_final


def load_src_datasets(data_paths):
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
    LOGGER.debug("Loaded image-report dataset: \n%s", ds_img)

    ds_text = load_from_disk(data_paths["custom_text"])

    for split in ds_text:
        ds_text[split] = ds_text[split].add_column("text_id", range(len(ds_text[split])))
    LOGGER.debug("Loaded custom split_text dataset: \n%s", ds_text)

    if CONFIG["target_section"] == "findings":
        ds_img = ds_img.remove_columns("impression")
        ds_img = ds_img.rename_column("findings", "section_text")
    elif CONFIG["target_section"] == "impression":
        ds_img = ds_img.remove_columns("findings")
        ds_img = ds_img.rename_column("impression", "section_text")
    else:
        raise ValueError(f"Invalid target_section {CONFIG['target_section']}, expected 'findings' or 'impression'")

    return ds_img, ds_text


def preprocess_dataset():
    img_dataset, text_dataset = load_src_datasets(data_paths=CONFIG["data_path"])

    # Get dataloader for training and testing
    image_processor_name = CONFIG["preprocess"]["image_processor"]
    model_name_or_path = CONFIG["model_name_or_path"][image_processor_name]
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    if image_processor_name == "clip":
        img_processor = processor.image_processor
        resize_h_w = (img_processor.size["shortest_edge"], img_processor.size["shortest_edge"])
    elif image_processor_name == "swinv2":
        img_processor = processor
        resize_h_w = (img_processor.size["height"], img_processor.size["width"])

    ds_dict = {}
    for split in ["train", "validation", "test"]:
        ds_dict[split] = pre_process_dataset(img_processor=img_processor, img_dataset=img_dataset[split], text_dataset=text_dataset[split], resize_h_w=resize_h_w, convert_to_rgb=True)
        # .select(range(len(text_dataset[split]) - 200, len(text_dataset[split])))

    pre_processed_dataset_dict = DatasetDict(ds_dict)
    pre_processed_dataset_dict.save_to_disk(CONFIG["preprocess"]["cache_path"])
    LOGGER.info("Preprocessed dataset dict saved to: %s", CONFIG["preprocess"]["cache_path"])


#############################################


def post_init_model_and_tokenizer(model, tokenizer):
    if len(tokenizer) != model.config.decoder.vocab_size:
        LOGGER.info("Resizing model decoder to match tokenizer size: %d -> %d", model.config.decoder.vocab_size, len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=True)

    # 用于在 input_ids 中查找需要替换的图像占位符 <|image_token|>
    if not hasattr(model.config, "image_token_index"):
        model.config.image_token_index = tokenizer.convert_tokens_to_ids("<|image_token|>")

    if not hasattr(tokenizer, "num_image_tokens"):
        # 计算 vision model 输出的图像特征的数量，该数量等于我们应该在 input_ids 中插入 <|image_token|> 的数量
        img_size = model.config.encoder.image_size
        dummy_img = torch.zeros((1, 3, img_size, img_size))
        num_image_tokens = model.encoder(dummy_img).last_hidden_state.size(1)
        tokenizer.num_image_tokens = num_image_tokens


def init_model(vision_model_path, language_model_path, model_base_cfg):

    LOGGER.info("Initializing vision language mode: %s, %s", vision_model_path, language_model_path)
    model = Vision2LanguageModel.from_encoder_decoder_pretrained(vision_model_path, language_model_path)

    return model


def init_processor(vision_model_path, language_model_path, model_base_cfg):
    LOGGER.info("Loading ImageProcessor from: %s", vision_model_path)
    if model_base_cfg["vision_model"] == "clip":
        img_processor = AutoProcessor.from_pretrained(vision_model_path).image_processor
    else:
        img_processor = AutoProcessor.from_pretrained(vision_model_path)

    LOGGER.info("Loading Tokenizer from: %s", language_model_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_path)

    # Add special tokens
    LOGGER.info("Adding special tokens")
    bos_token = tokenizer.bos_token if tokenizer.bos_token else "<BOS>"
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<EOS>"
    pad_token = tokenizer.pad_token if tokenizer.pad_token else "<PAD>"
    special_tokens_dict = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "additional_special_tokens": ["<|image_token|>", "<|image_start|>", "<|image_end|>"],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    # print special tokens and their ids
    LOGGER.info("Special tokens: %s", [(key, tok, tokenizer.convert_tokens_to_ids(tok)) for key, tok in tokenizer.special_tokens_map.items()])

    # set chat template
    assert tokenizer.chat_template == None, "Tokenizer has chat_template, please check whether to use the existing one or our new chat template."
    chat_template_path = model_base_cfg["chat_template"]
    LOGGER.info("Adding chat template to tokenizer from: %s", chat_template_path)
    with open(chat_template_path, "r") as f:
        chat_template = "".join([line.strip() for line in f.readlines()])
    tokenizer.chat_template = chat_template
    LOGGER.info("Chat template: %s", tokenizer.chat_template)

    return img_processor, tokenizer


def init_accelerator():
    global ACCELERATOR, DEVICE, LOGGER

    dataloader_cfg = DataLoaderConfiguration(use_seedable_sampler=True)
    # https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin
    # 如果OOM，可以尝试设置 sync_each_batch=True，但是会导致训练速度变慢
    # adjust_scheduler=False，我们在train方法中手动计算 scheduler 在使用梯度累计后的 step
    plugin = GradientAccumulationPlugin(num_steps=CONFIG["train"]["grad_accum_steps"], adjust_scheduler=False, sync_with_dataloader=True)
    ACCELERATOR = Accelerator(mixed_precision=CONFIG["train"]["mixed_precision"], dataloader_config=dataloader_cfg, gradient_accumulation_plugin=plugin)
    DEVICE = ACCELERATOR.device

    if ACCELERATOR.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    LOGGER = MultiProcessAdapter(LOGGER, {})
    LOGGER.info("Available cuda: %d", torch.cuda.device_count())
    LOGGER.info("Accelerator state: %s", ACCELERATOR.state)
    LOGGER.info("Accelerator mixed_precision: %s", ACCELERATOR.mixed_precision)
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


def init_proj_config():
    global CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--config_file", type=str, help=f".yaml file path")

    parser.add_argument("--output_name", type=str)
    parser.add_argument("--jobid", type=int)
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    parser.add_argument("--preprocess_dataset", action="store_true")
    parser.add_argument("--image_processor", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default=None)

    args = parser.parse_args()

    if args.from_bash:
        file_path = args.config_file
    else:
        file_path = "/home/yuxiang/liao/workspace/arrg_img2text/config/4_vlgen.yaml"

    with open(file_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    if args.from_bash:
        CONFIG["output_name"] = args.output_name
        CONFIG["jobid"] = args.jobid
        CONFIG["resume_from_checkpoint"] = args.resume_from_checkpoint
        CONFIG["preprocess_dataset"] = args.preprocess_dataset
        if args.preprocess_dataset:
            CONFIG["preprocess"]["image_processor"] = args.image_processor
            CONFIG["preprocess"]["cache_path"] = args.cache_path
    else:
        CONFIG["jobid"] = "00000"

    output_dirs = CONFIG["output_dir"]
    output_dirs["result"] = os.path.join(output_dirs["result"], CONFIG["output_name"])
    output_dirs["model"] = os.path.join(output_dirs["model"], CONFIG["output_name"])
    output_dirs["checkpoint"] = os.path.join(output_dirs["checkpoint"], CONFIG["output_name"])
    # output_dirs["log"] = os.path.join(output_dirs["log"], CONFIG["output_name"])
    os.makedirs(output_dirs["result"], exist_ok=True)
    os.makedirs(output_dirs["model"], exist_ok=True)
    os.makedirs(output_dirs["checkpoint"], exist_ok=True)
    # os.makedirs(output_dirs["log"], exist_ok=True)

    for key in CONFIG["train"]:
        if "lr" in key:
            CONFIG["train"][key] = float(CONFIG["train"][key])


#############################################


def main():
    model_base_cfg = CONFIG["model"]
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]

    init_accelerator()
    set_seed(CONFIG["train"]["seed"])

    img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)

    ds_final = load_preprocessed_dataset(CONFIG["preprocess"]["cache_path"])
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(ds_final, img_processor=img_processor, tokenizer=tokenizer, use_debug_subset=CONFIG["use_debug_subset"])

    if CONFIG["do_train"]:
        # Training
        model = init_model(vision_model_path, language_model_path, model_base_cfg)
        post_init_model_and_tokenizer(model, tokenizer)

        model.to(DEVICE)
        model = ACCELERATOR.prepare(model)
        check_memory()

        start = time.time()
        train(model, train_dataloader, valid_dataloader)
        end = time.time()
        LOGGER.info("Total training time: %s", seconds_to_time_str(end - start))

        img_processor.save_pretrained(CONFIG["output_dir"]["model"])
        tokenizer.save_pretrained(CONFIG["output_dir"]["model"])
        LOGGER.info("Tokenizer and Image Processor are saved to: %s", CONFIG["output_dir"]["model"])

    if CONFIG["do_test"]:
        # Testing
        # model = load_model(CONFIG["output_dir"]["model"])
        model = init_model(vision_model_path, language_model_path, model_base_cfg)
        post_init_model_and_tokenizer(model, tokenizer)

        model.to(DEVICE)
        model, test_dataloader = ACCELERATOR.prepare(model, test_dataloader)

        start = time.time()
        evaluate(model, test_dataloader, output_result=True)
        end = time.time()
        LOGGER.info("Final evaluation time: %s", seconds_to_time_str(end - start))


if __name__ == "__main__":
    init_proj_config()
    init_logger()
    LOGGER.debug(CONFIG)

    start0 = time.time()

    if CONFIG["preprocess_dataset"]:
        preprocess_dataset()
    else:
        import cProfile

        cProfile.run("main()", filename=os.path.join(CONFIG["output_dir"]["result"], "time_statistic.cprofile"))
        # main()

    end0 = time.time()
    LOGGER.info("Total time: %s ", seconds_to_time_str(end0 - start0))
