#############################################
# 基于6修改
# 6使用单个疾病：使用graph后反而影响结果，猜测是loss不合适，但目前的方法没有更好的使用loss的方式，因此改用其他更直观的loss
# 现在每个图可以用42个疾病标签表示，每个疾病有单独的特征表示方法
# 由原本的3个标签，改为2个标签，合并“”和"absent"为一个标签，称为"absent"
# 训练和评估时，移除空的文本数据，这是合理的，因为findings为空时，可能impression有数据，如果保留空数据用于训练和评估，实际上就是引入了错误数据
# 使用分类器，并将分类特征注入decoder
#############################################
import argparse
import ast
import datetime
import gc
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from accelerate.utils import (
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    gather,
    gather_object,
)
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from nltk.tokenize import wordpunct_tokenize
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners import lora
from peft.utils import AuxiliaryTrainingWrapper
from PIL import Image
from safetensors.torch import load_file
from scipy.ndimage import zoom
from scorers.scores import compute_scores
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    Dinov2Config,
    Dinov2Model,
    EosTokenCriteria,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteriaList,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
from transformers.generation import GenerationConfig
from transformers.generation import utils as tf_generation_utils
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

CONFIG = None
LOGGER = None
DEVICE = None
ACCELERATOR = None
STATUS_INFO = None
MLFLOW_TRACKER = None


@dataclass
class GlobalVariables:
    """
    Some global variables for the script.
    """

    peak_mem = 0
    num_image_tokens = 0
    additional_special_tokens = ["<|image_token|>", "<|image_start|>", "<|image_end|>", "<|label_start|>", "<|label_token|>", "<|label_end|>"]

    eot_token = "<|eot_id|>"
    eot_token_id = None

    obs_name2id_dict = {}  # e.g. {"effusion": 0, ...}
    obs_id2name_dict = {}  # e.g. {0: "effusion", ...}
    obs_cls2id_dict = {}  # e.g. {"": 0, "mentioned": 1, "absent": 2}
    obs_id2cls_dict = {}  # e.g. {0: "", 1: "mentioned", 2: "absent"}


GLOBAL_VARS = GlobalVariables()


#############################################
# Model Classes
#############################################
@dataclass
class Vision2LanguageOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None

    logits: torch.FloatTensor = None
    cls_logits: Optional[torch.FloatTensor] = None

    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class VisionLanguageProjector(nn.Module):
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


class VisionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 可学习的标签嵌入（每个标签一个向量）
        self.obs_embeddings = nn.Embedding(config.num_observations, config.encoder_hidden_size)

        # 注意力模块（标签为 query，图像为 key 和 value）
        self.attn = nn.MultiheadAttention(embed_dim=config.encoder_hidden_size, num_heads=8, batch_first=True)

        # 注意力投影层：将注意力输出的维度投影到 decoder_hidden_size
        self.hidden_proj = nn.Linear(config.encoder_hidden_size, config.decoder_hidden_size)

        # 分类器：对每个标签嵌入进行分类
        self.classifier = nn.Linear(config.decoder_hidden_size, config.num_cls_labels)

    def merge_obs_tensor_by_indices(self, obs_tensor, image_indices_map, mean_pooling=False):
        """
        等价于用 sum 和 stack 来合并 logits，但这个方法更高效
        obs_tensor: [img_bsz, num_obs, dim] 可以是 特征向量 或者是 logits
        image_indices_map: # tensor([0, 0, 1, 2, 2, ...]), img0 img1 -> sample0, img2 -> sample1 ...
        如果当前处于pred， image_indices_map 的长度会变成 img_bsz * num_beams ，此时image_indices_map应该是类似于tensor([0,1,2, 0,1,2, 3,4,5, 6,7,8, 6,7,8, ...])，因为图像特征会在dim0上，每个样本都重复 num_beams 次
        Returns:
            merged_obs_tensor: [output_bsz, num_obs, 3]
        """
        device = obs_tensor.device
        img_bsz, num_obs, dim = obs_tensor.shape
        output_bsz = int(torch.max(image_indices_map).item()) + 1

        # 初始化输出张量
        merged_obs_tensor = torch.zeros(output_bsz, num_obs, dim, dtype=obs_tensor.dtype, device=device)  # torch.Size([output_bsz, num_obs, 3])

        # scatter_add 累加对应位置
        # index_add: image_indices_map.shape == obs_tensor.shape
        # 按照 image_indices_map 的值，将 obs_tensor 累加到 merged_obs_tensor 的对应位置，比如 dim=0, image_indices_map = [0, 0, 1]
        # 则 obs_tensor[0] 和 obs_tensor[1] 累加到 merged_obs_tensor[0]，obs_tensor[2] 累加到 merged_obs_tensor[1]
        merged_obs_tensor = merged_obs_tensor.index_add(0, image_indices_map, obs_tensor)

        if mean_pooling:
            # 统计每个样本对应的图像数量，用于平均池化
            counts = torch.zeros(output_bsz, 1, 1, dtype=obs_tensor.dtype, device=device)  # [output_bsz, 1, 1]
            ones = torch.ones_like(image_indices_map, dtype=obs_tensor.dtype, device=device)  # shape: [N_total]
            counts = counts.index_add(0, image_indices_map, ones.unsqueeze(-1).unsqueeze(-1))  # broadcast to [N_total, 1, 1]

            if (counts == 0).any():
                raise ValueError("Found zero-count sample in image_indices_map")

            merged_obs_tensor = merged_obs_tensor / counts  # 平均池化

        return merged_obs_tensor

    def forward(self, img_features, image_indices_map, obs_ids, obs_labels, output_loss=False):
        """
        img_features: [img_bsz, num_fea, fea_dim] 图像 encoder 输出

        image_indices_map: 长度为 img_bsz 的列表，每个元素对应一个 img_feature，元素值为 output batch 中的样本索引
        如果某个样本对应两个图像，则需要对 self.attn 输出的特征进行融合

        obs_ids: 长度为num_obs, id取值范围为 [0, 41]， 对应 obs_embeddings 的索引
        obs_labels: 长度为num_obs, 标签取值范围为 [0, 2]，对应分类器的输出类别

        返回: logits: [output_bsz, num_obs, 3]
        """
        img_bsz = img_features.size(0)

        label_ids = obs_ids.unsqueeze(0).expand(img_bsz, -1)  # [num_obs] -> [img_bsz, num_obs]
        label_embed = self.obs_embeddings(label_ids)  # [img_bsz, num_obs, enc_dim]

        # 注意力交互（Q=label, K=img, V=img），输出 shape: [img_bsz, num_obs, enc_dim]
        attn_output, _ = self.attn(query=label_embed, key=img_features, value=img_features)
        # 注意力输出投影到 decoder_hidden_size
        attn_output = self.hidden_proj(attn_output)  # [img_bsz, num_obs, dec_dim]

        # 特征层融合 输出 shape: [output_bsz, num_obs, dec_dim]
        fused_attn_output = self.merge_obs_tensor_by_indices(obs_tensor=attn_output, image_indices_map=image_indices_map, mean_pooling=True)
        features_fused_logits = self.classifier(fused_attn_output)

        # 每个标签单独分类：[img_bsz, num_obs, 3]
        # logits = self.classifier(attn_output)
        # # 决策层融合 [output_bsz, num_obs, 3]
        # decision_fused_logits = self.merge_obs_tensor_by_indices(obs_tensor=logits, image_indices_map=image_indices_map, mean_pooling=True)

        loss = None
        if output_loss:
            # labels: [output_bsz * num_obs]
            labels_flat = obs_labels.view(-1)

            # logits: [output_bsz * num_obs, 3]
            # decision_fused_logits_flat = decision_fused_logits.view(-1, decision_fused_logits.size(-1))
            features_fused_logits_flat = features_fused_logits.view(-1, features_fused_logits.size(-1))

            # Cross-entropy loss
            ce_loss_fct = nn.CrossEntropyLoss(reduction="mean")
            # decision_loss = ce_loss_fct(decision_fused_logits_flat, labels_flat.long())
            features_loss = ce_loss_fct(features_fused_logits_flat, labels_flat.long())

            # loss = (decision_loss + features_loss) / 2
            loss = features_loss

        return ClassifierOutput(
            loss=loss,  # 如果需要计算损失，可以在调用时传入 labels
            logits=features_fused_logits,
            hidden_states=fused_attn_output,  # 如果需要，可以返回中间层的特征
        )


class Vision2LanguageModel(VisionEncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        # 通过 from_encoder_decoder_pretrained 加载模型时的额外传入参数都会被加入到config中
        super().__init__(config=config, encoder=encoder, decoder=decoder)
        # replace enc_to_dec_proj with VisionLanguageProjector
        self.config.encoder_hidden_size = self.encoder.config.hidden_size
        self.config.decoder_hidden_size = self.decoder.config.hidden_size
        assert "classification_only" in config, "config should have 'classification_only' key"

        self.v2l_projector = VisionLanguageProjector(self.config)
        if hasattr(self, "enc_to_dec_proj"):
            del self.enc_to_dec_proj  # 移除投影层

        self.obs_classifier = VisionClassifier(self.config)

    def _expand_image_indices_map_generation(self, image_indices_map, expand_size):
        # [0, 0, 1] -> [0, 0, 0, 0, 0, 0, 1, 1, 1]
        expanded_map = image_indices_map.repeat_interleave(expand_size, dim=0)
        # 每组 image 的 index 偏移为 0~expand_size-1
        local_offsets = torch.arange(expand_size, device=image_indices_map.device).repeat(len(image_indices_map))
        # [0, 0, 0, 0, 0, 0, 1, 1, 1] * 3 + [0, 1, 2, 0, 1, 2, 0, 1, 2]
        # -> [0, 1, 2, 0, 1, 2, 3, 4, 5]
        image_indices_map = expanded_map * expand_size + local_offsets
        return image_indices_map

    def _inject_features(self, input_ids, decoder_inputs_embeds, target_token_id, target_features, image_indices_map=None):
        """
        image_indices_map (torch.Tensor): 指示了每个图像特征应放置到的目标批次索引。
        """
        # replace img features with the <|image_token|> placeholder token in the input text
        special_image_mask = (input_ids == target_token_id).unsqueeze(-1)  # torch.Size([bsz, seq_len, 1])
        special_image_mask = special_image_mask.expand_as(decoder_inputs_embeds).to(decoder_inputs_embeds.device)  # torch.Size([bsz, seq_len, dec_dim])

        # 在pred阶段使用，以解决 beam search 时图像特征与文本特征的对应偏差问题
        # 在train阶段，图像特征与文本特征是按顺序对应的，只需要按顺序将图像特征注入到文本特征中即可， 如果一个文本样本对应2个图像时，文本样本中的 <|image_token|> 数量会正确反映出所需的图像特征。
        # 但是在pred阶段，图像和文本在 generate() 方法中会被扩展为多个 beam，但是扩展后的图像与文本的映射是错误的，比如原本 [img0, img1, img2] 分别对应 [sample0, sample0, sample1]
        # 扩展后变成 图像[img0beam0, img0beam1, img0beam2, img1beam0, img1beam1, img1beam2, img2beam0, img2beam1, img2beam2] 和 文本[sample0beam0, sample0beam1, sample0beam2, sample1beam0, sample1beam1, sample1beam2]
        # 默认的注入顺序会变成 img0beam0, img0beam1 -> sample0beam0，而正确的映射应该是 img0beam0, img1beam0 -> sample0beam0
        # 因此需要使用 image_indices_map 来重新映射图像特征到文本特征中
        if image_indices_map is not None:
            # 按 image_indices_map 的值排序
            sorted_indices = torch.argsort(image_indices_map)
            # 重新排序 target_features
            target_features = target_features[sorted_indices]

        # 保证所有 target_features 都能够被复制到 decoder_inputs_embeds 中
        assert special_image_mask.sum() == target_features.numel(), f"special_image_mask.sum()={special_image_mask.sum()}, target_features.numel()={target_features.numel()}, should be equal to guarantee that all image features are copied to decoder_inputs_embeds"

        target_features = target_features.to(decoder_inputs_embeds.device, decoder_inputs_embeds.dtype)
        # decoder_inputs_embeds 和 special_image_mask 的形状相同，target_features 与 special_image_mask True 的数量相同，会按顺序替换 True 位置的特征
        decoder_inputs_embeds = decoder_inputs_embeds.masked_scatter(special_image_mask, target_features)

        return decoder_inputs_embeds

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_assistant_masks: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_loss: Optional[bool] = False,
        decoder_extra_inputs: Optional[Dict[str, Any]] = None,  # 用于传递不希望被generate处理的输入参数，目前是用于动态分配图像和用于分类器的参数
        decoder_extra_outputs: Optional[Dict[str, Any]] = None,  # 用于传递分类器输出到generate方法中，作为pred输出的一部分，包含 "cls_logits" 键
        **kwargs,
    ) -> Union[Tuple, Vision2LanguageOutputWithPast]:
        """Additional args:
        `decoder_inputs_embeds`: should represent the text embeddings with image features injected.
        `encoder_outputs`: in inference statge, we encode `pixel_values` and get `encoder_outputs` outside this forward method. This is because the `pixel_values` and `decoder_input_ids` have different batch sizes, which cause error in generate().

        If `output_loss` is True, by default we use `decoder_input_ids` as `labels`.
        And the `decoder_assistant_masks` should be provided to compute the loss.
        `decoder_assistant_masks` is provided by `tokenizer.apply_chat_template`.
        `decoder_assistant_masks` is a tensor with the same shape as decoder_input_ids, and the value is 0 or 1. 0: system/user tokens, 1: assistant tokens, which is the tokens that need to be generated.

        `decoder_extra_inputs`
        `decoder_extra_outputs`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # train时，有pixel_values，没有encoder_outputs
        # inference时，没有pixel_values，有encoder_outputs；encoder_outputs只有第一轮才需要，后续需要忽略
        if (pixel_values is not None) and (encoder_outputs is not None):
            raise ValueError("You must not specify both pixel_values and encoder_outputs.")

        # 我们目前没有使用过 decoder_inputs_embeds
        if (decoder_input_ids is None) ^ (decoder_inputs_embeds is not None):
            raise ValueError("You must specify exactly one of decoder_input_ids or decoder_inputs_embeds")

        if (pixel_values is not None or encoder_outputs is not None) and decoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both `pixel_values`/`encoder_outputs` and `decoder_inputs_embeds` at the same time, and must specify either one")

        if decoder_inputs_embeds is None:
            # get text embeddings
            decoder_inputs_embeds = self.decoder.get_input_embeddings()(decoder_input_ids)

        # 如果有encoder_outputs，就不需要再次 encode pixel_values
        if (pixel_values is not None) and (encoder_outputs is None):
            # get img features
            encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)

        # train forward 以及 inference first round，需要进行这一步
        # train forward 会提供 pixel_values
        # inference all rounds 会提供 encoder_outputs，而pixel_values=None；在first round时，past_key_values=None，后续为past_key_values=DynamicCache()
        if encoder_outputs is not None and past_key_values is None:
            image_features = encoder_outputs.last_hidden_state  # torch.Size([4, 1370, enc_dim])

            image_indices_map = decoder_extra_inputs["image_indices_map"]
            if "num_beams" in decoder_extra_inputs:
                # 说明是 pred, 从 generate() 方法中调用的，需要扩展 image_indices_map，
                # 原本一个元素表示一个图像，元素值表示样本索引
                # [0, 0, 1] 表示 [img0, img0, img1]
                # img0 + img1 -> sample0,
                # img2 -> sample1
                # 扩展后，每个图像会被重复 num_beams 次, 值对应于扩展后的样本(decoder_input_ids)的索引
                # [0, 1, 2, 0, 1, 2, 3, 4, 5] 表示 [img0beam0, img0beam1, img0beam2, img1beam0, img1beam1, img1beam2, img2beam0, img2beam1, img2beam2]
                # img0beam0 + img1bema0 -> sample0beam0,
                # img0beam1 + img1beam1 -> sample0beam1,
                # img0beam2 + img1beam2 -> sample0beam2,
                # img2beam0 -> sample1beam0, img2beam1 -> sample1beam1, img2beam2 -> sample1beam2
                image_indices_map = self._expand_image_indices_map_generation(image_indices_map=decoder_extra_inputs["image_indices_map"], expand_size=decoder_extra_inputs["num_beams"])

            # img clssifier
            classifier_outputs = self.obs_classifier(
                image_features,
                image_indices_map=image_indices_map,
                obs_ids=decoder_extra_inputs["obs_ids"],
                obs_labels=decoder_extra_inputs.get("obs_labels", None),  # 如果是pred，则不会传入 obs_labels
                output_loss=output_loss,
            )
            label_features = classifier_outputs.hidden_states  # torch.Size([2, num_obs, dec_dim])，如果是pred，则是 [num_beams * bsz, num_obs, dec_dim]；在pred阶段，label_features 跟 decoder_input_ids 是一一对应的，不需要进行额外处理

            # pred时，将logits传递到 generate() 方法中
            if decoder_extra_outputs is not None:
                if "num_beams" in decoder_extra_inputs:
                    # 在pred阶段，cls_logits 是被 expand 的， 我们只需要每 expand_size 个元素取第一个
                    expand_size = decoder_extra_inputs["num_beams"]
                    cls_logits = classifier_outputs.logits
                    cls_logits = cls_logits[::expand_size, :, :]
                    decoder_extra_outputs["cls_logits"] = cls_logits
                else:
                    decoder_extra_outputs["cls_logits"] = classifier_outputs.logits  # torch.Size([2, num_obs, 3])；

            # 如果是 classification_only 模式，则只返回分类器的输出就可以结束，后面的组件已经被移除
            if self.config.classification_only:
                return Vision2LanguageOutputWithPast(
                    loss=classifier_outputs.loss,
                    logits=classifier_outputs.logits,
                    hidden_states=classifier_outputs.hidden_states,
                    past_key_values=None,  # classification_only 时不需要 past_key_values
                    image_hidden_states=None,  # 返回图像特征
                    gen_loss=None,  # for logging purposes
                    cls_loss=classifier_outputs.loss,  # for logging purposes
                )

            # project image features
            image_features = self.v2l_projector(image_features)  # torch.Size([2, 1370, 2048])

            # inject image features into text embeddings
            decoder_inputs_embeds = self._inject_features(
                input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                target_token_id=self.config.image_token_index,
                target_features=image_features,
                image_indices_map=image_indices_map,
            )

            # inject observation label features into text embeddings
            # 无论是 train 还是 pred 阶段，label_features 与 decoder_inputs_embeds 的每个样本都是对齐的，只需要按照顺序注入即可
            assert (decoder_input_ids == self.config.label_token_index).any(), f"label_token_id {self.config.label_token_index} (<|label_token|>) not found in decoder_input_ids"
            decoder_inputs_embeds = self._inject_features(
                input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                target_token_id=self.config.label_token_index,
                target_features=label_features,
            )

        # Text generation. decoder_inputs_embeds is used in replace of decoder_input_ids on decoder in all cases.
        # In train statge, decoder_input_ids is encoded into decoder_inputs_embeds and then merged with image features.
        # In inference stage, encoder_outputs is passed from generate() in replace of pixel_values.
        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
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
        loss, gen_loss, cls_loss = None, None, None
        if output_loss:
            labels = labels if labels is not None else decoder_input_ids

            # Shift so that tokens < n predict n
            if decoder_assistant_masks is not None:
                shift_label_mask = decoder_assistant_masks[:, 1:]  # torch.Size([bsz, seq_len - 1])
            elif decoder_attention_mask is not None:
                shift_label_mask = decoder_attention_mask[:, 1:]
            else:
                raise ValueError("decoder_assistant_masks or decoder_attention_mask should be provided")

            shift_logits = logits[:, :-1, :]  # torch.Size([bsz, seq_len - 1, vocab_size])
            shift_labels = labels[:, 1:]  # torch.Size([bsz, seq_len - 1])
            active_shift_logits = shift_logits[shift_label_mask != 0].contiguous()  # torch.Size([num_acitve_labels, vocab_size])
            active_shift_labels = shift_labels[shift_label_mask != 0].contiguous()  # torch.Size([num_acitve_labels])

            ce_loss_fct = nn.CrossEntropyLoss()
            gen_loss = ce_loss_fct(active_shift_logits, active_shift_labels)

            cls_loss = classifier_outputs.loss
            if cls_loss:
                loss = gen_loss + cls_loss
            else:
                loss = gen_loss

        return Vision2LanguageOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            gen_loss=gen_loss,  # for logging purposes
            cls_loss=cls_loss,  # for logging purposes
        )

    @torch.no_grad()
    def generate(
        self,
        inputs,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,  # If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with decoder_.
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (tf_generation_utils.is_deepspeed_zero3_enabled() or tf_generation_utils.is_fsdp_managed_module(self)) and tf_generation_utils.dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else tf_generation_utils.LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else tf_generation_utils.StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(tf_generation_utils.inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        # batch_size = inputs_tensor.shape[0]
        # encoder和decoder的bsz可能不一样，我们以decoder的bsz为准
        batch_size = model_kwargs["decoder_input_ids"].shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not tf_generation_utils.is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            LOGGER.warning("Should not see this warning!!! A decoder-only architecture is detected, while we are using encoder-decoder model.")
            if generation_config._pad_token_tensor is not None and batch_size > 1 and len(inputs_tensor.shape) == 2 and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0:
                LOGGER.warning("A decoder-only architecture is being used, but right-padding was detected! For correct " "generation results, please set `padding_side='left'` when initializing the tokenizer.")

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config, model_kwargs)
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, generation_config)

            # 5. Prepare `input_ids` which will be used for auto-regressive generation
            # 原始方法，当input_ids不是以decoder_start_token_id开头时，添加decoder_start_token_id
            # 更新后的方法，当input_ids不是以decoder_start_token_id 或 pad_token_id 开头时，添加decoder_start_token_id
            # 因为我们在collect_fn中，会将input_ids以8的倍数填充left padding，然后紧跟着decoder_start_token_id和正文
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                pad_token_id=torch.tensor(generation_config.pad_token_id, device=inputs_tensor.device),
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if inputs_tensor.shape[1] != input_ids_length and model_input_name == "inputs_embeds" and not self.config.is_encoder_decoder:
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError("`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.")

        if not tf_generation_utils.is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            tf_generation_utils.warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different" f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model" f" is on {self.device.type}. You may experience unexpected behaviors or slower generation." " Please make sure that you have put `input_ids` to the" f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before" " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs)

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        result = None
        if generation_mode == tf_generation_utils.GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError("num_return_sequences has to be 1 when doing assisted generate, " f"but is {generation_config.num_return_sequences}.")
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
                raise ValueError("assisted generate is not supported with Static cache classes`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}")

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == tf_generation_utils.GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}")
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == tf_generation_utils.GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}")

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (tf_generation_utils.GenerationMode.SAMPLE, tf_generation_utils.GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (tf_generation_utils.GenerationMode.BEAM_SAMPLE, tf_generation_utils.GenerationMode.BEAM_SEARCH):
            # 11. prepare beam search scorer
            beam_scorer = tf_generation_utils.BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == tf_generation_utils.GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = tf_generation_utils.BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == tf_generation_utils.GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError("`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` " f"of positive integers, but is {generation_config.force_words_ids}.")

                if not isinstance(generation_config.force_words_ids, list) or len(generation_config.force_words_ids) == 0:
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids) for token_ids in word_ids):
                            typeerror()

                        constraint = tf_generation_utils.DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = tf_generation_utils.PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = tf_generation_utils.ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if generation_config.return_legacy_cache is True and not tf_generation_utils.is_torchdynamo_compiling() and hasattr(result, "past_key_values") and getattr(result.past_key_values, "to_legacy_cache") is not None:
            result.past_key_values = result.past_key_values.to_legacy_cache()

        extra_outputs = None
        if "decoder_extra_outputs" in kwargs:
            extra_outputs = kwargs["decoder_extra_outputs"]

        return (result, extra_outputs)

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        pad_token_id: torch.Tensor,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models
        Update: if the first token is not decoder_start_token_id or pad_token_id, we need to prepend decoder_start_token_id. Because our input_ids are left padded to multiple of 8, and then followed by decoder_start_token_id and the real input_ids. It is done in the collate_fn.
        """
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. `decoder_start_token_id` must have shape (batch_size, 1)
        if device is None:
            device = self.device
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}")
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

        # 3. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token. Note that the
        # original checkpoints can't be detected through `self.__class__.__name__.lower()`, needing custom logic.
        # See: https://github.com/huggingface/transformers/pull/31470
        elif "donut" in self.__class__.__name__.lower() or (self.config.model_type == "vision-encoder-decoder" and "donut" in self.config.encoder.model_type.lower()):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        #######################################
        # !!! Update: if the first token is not decoder_start_token_id or pad_token_id, we need to prepend decoder_start_token_id
        #######################################
        elif ((decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]) & (decoder_input_ids[:, 0] != pad_token_id)).all().item():
            decoder_input_ids = torch.cat([decoder_start_token_id, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key != "cache_position" and dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


class ImageTextDataset(Dataset):
    def __init__(self, hf_dataset, img_processor, tokenizer, split, target_obs=None):
        # column_names: ['source', 'images_path', 'images', 'section_text', 'doc_key', 'split_sents', 'split_sent_toks', 'sent_idx_split_idx', 'radlex', 'cxrgraph_ent', 'cxrgraph_attr', 'cxrgraph_rel']
        self.split = split
        self.target_section = CONFIG["target_section"]
        self.src_path = os.path.dirname(hf_dataset.cache_files[0]["filename"]) if hf_dataset.cache_files else ""
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.target_observations = target_obs
        self.samples = self._process_text(hf_dataset)
        self.samples = self._add_gold_text_column(self.samples)

    def _process_text(self, hf_dataset):
        if self.target_section == "findings":
            hf_dataset = hf_dataset.remove_columns("impression")
            hf_dataset = hf_dataset.rename_column("findings", "section_text")
        elif self.target_section == "impression":
            hf_dataset = hf_dataset.remove_columns("findings")
            hf_dataset = hf_dataset.rename_column("impression", "section_text")
        else:
            raise ValueError(f"Invalid target_section {self.target_section}, expected 'findings' or 'impression'")

        # Remove empty string
        non_empty_section_indices = [idx for idx, txt in enumerate(hf_dataset["section_text"]) if txt != ""]
        filtered_dataset = hf_dataset.select(non_empty_section_indices)
        num_removed_data = len(hf_dataset) - len(non_empty_section_indices)
        LOGGER.info("Removed [%d] samples with empty [%s] section from [%s] split", num_removed_data, self.target_section, self.split)
        return filtered_dataset

    def _add_gold_text_column(self, hf_dataset):
        run_mode = CONFIG["run_mode"]

        # 临时选择与gold_text计算相关的列
        if "pretrain" in run_mode:
            temp_dataset = hf_dataset.select_columns(["split_sents"])

            def get_gold_text_pretrain(example):
                return {"gold_text": " ".join(example["split_sents"])}

            temp_dataset = temp_dataset.map(get_gold_text_pretrain)

        elif "finetune" in run_mode:
            if self.target_observations is None:
                raise ValueError("target_observations must be provided for finetune mode")

            temp_dataset = hf_dataset.select_columns(["split_sents", "radlex_to_splitsents_map"])
            target_observations = self.target_observations

            def get_gold_text_finetune(example):
                gold_text = []
                for obs_name in target_observations:
                    indices = example["radlex_to_splitsents_map"].get(obs_name, [])
                    for idx in indices:
                        gold_text.append(example["split_sents"][idx])
                return {"gold_text": " ".join(gold_text)}

            temp_dataset = temp_dataset.map(get_gold_text_finetune)

        else:
            raise ValueError(f"Invalid run_mode {run_mode}, expected 'pretrain' or 'finetune'")

        # 合并 gold_text 列回 hf_dataset
        gold_texts = temp_dataset["gold_text"]
        assert len(hf_dataset) == len(gold_texts), f"Length mismatch: {len(hf_dataset)} vs {len(gold_texts)}"
        hf_dataset = hf_dataset.add_column("gold_text", gold_texts)

        # Remove empty gold_text entries
        non_empty_indices = [i for i, txt in enumerate(gold_texts) if txt != ""]
        filtered_dataset = hf_dataset.select(non_empty_indices)
        num_removed = len(hf_dataset) - len(non_empty_indices)
        LOGGER.info("Removed [%d] samples with empty [%s] gold_text from [%s] split", num_removed, self.target_section, self.split)

        return filtered_dataset
    
    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch_data, img_processor, tokenizer, target_obs=None):
    # 这个方法只处理图像，
    # 对话数据由于在训练和推理时不同，所以分开进行处理

    # 处理图像，因为每个样本的图像数量不一样，所以需要image_indices_map来记录每个样本的图像在batch中的索引
    nested_images = [i["images"] for i in batch_data]  # nested list of imgs: [[img1, img2], [img1], ...]
    piexl_values_tensor = img_processor(images=[img for imgs in nested_images for img in imgs], return_tensors="pt", do_convert_rgb=True).pixel_values

    image_idx_to_batch_idx = []  # e.g. [0, 0, 1, ...] 位置表示每个图像在 encoder_input 中的索引，值表示该图像在 decoder_input的索引 （属于哪个样本）
    for item_idx, item_images in enumerate(nested_images):
        num_images = len(item_images)
        assert num_images <= 2, f"num_images should be less equal than 2, but got {num_images}"
        image_idx_to_batch_idx.extend([item_idx] * num_images)
    image_indices_map = torch.tensor(image_idx_to_batch_idx, dtype=torch.long)  # torch.Size([bsz < x < 2*bsz])

    # 我们使用分类器时的分类目标，预训练时使用所有类别；微调时可以指定target_obs
    # 当target_obs=['effusion', 'pneumothorax', 'opacity', 'normal']时，从每个样本的 radlex_types 中找到对应的标签
    # obs_ids = tensor([0, 1, 2, 3]) 值对应的是 CONFIG["observation_map"] 中的idx
    # obs_labels = tensor([[0, 0, 0, 0]]) 值对应的是 CONFIG["obs_classification_map"] 中的类别idx
    obs_name2id_dict = GLOBAL_VARS.obs_name2id_dict  # e.g. {'effusion': 0, ...}
    obs_id_list = []  # e.g. [0, 1, 2, 7, 8, ...] 每个值是一个类别的id
    if target_obs:
        obs_id_list = [obs_name2id_dict[obs_name] for obs_name in target_obs]
    else:
        obs_id_list = list(obs_name2id_dict.values())
    obs_ids = torch.tensor(obs_id_list, dtype=torch.long)

    # 多任务多分类标签
    obs_cls2id_dict = GLOBAL_VARS.obs_cls2id_dict  # e.g. {'mentioned': 1, 'absent': 0}
    obs_cls_merge_map = CONFIG["obs_cls_merge_map"]  # e.g. {'': 'absent'}
    obs_label_list = []
    for obs_label_dict in [data_item["radlex_types"] for data_item in batch_data]:
        obs_cls_ids = []  # [0, 2, 1, ..., 1] 每个值是一个类别的三分类标签。
        for obs_name, obs_cls_name in obs_label_dict.items():
            # 如果指定了target_obs，则只保留target_obs中的标签
            if target_obs and (obs_name not in target_obs):
                continue
            if obs_cls_name in obs_cls2id_dict or obs_cls_name in obs_cls_merge_map:
                # 如果有obs_cls_merge_map，则使用映射后的类别名，这里将 ”“标签 射为 'absent'标签，由原先的三分类任务变为二分类任务
                if obs_cls_name in obs_cls_merge_map:
                    obs_cls_name = obs_cls_merge_map[obs_cls_name]
                obs_cls_id = obs_cls2id_dict[obs_cls_name]
                obs_cls_ids.append(obs_cls_id)
            else:
                raise ValueError(f"Invalid observation label value: [{obs_cls_name}] for [{obs_name}]")
        obs_label_list.append(obs_cls_ids)
        assert len(obs_cls_ids) == len(obs_ids), f"obs_cls_ids should have the same length as obs_ids, but got {len(obs_cls_ids)} vs {len(obs_ids)}"
    obs_labels = torch.tensor(obs_label_list, dtype=torch.long)  # torch.Size([bsz, num_obs])

    label_name_list = [GLOBAL_VARS.obs_id2name_dict[idx] for idx in obs_ids.tolist()]
    gold_text_list = [i["gold_text"] for i in batch_data]

    # if "pretrain" in CONFIG["run_mode"]:
    #     # 预训练时，用所有的split_sents
    #     gold_text_list = [" ".join(item["split_sents"]) for item in batch_data]
    # elif "finetune" in CONFIG["run_mode"]:
    #     # 微调时，只使用与obs_ids对应的split_sents
    #     gold_text_list = []
    #     for item in batch_data:
    #         gold_text_peritem = []
    #         for obs_name in label_name_list:
    #             split_sent_indices = item["radlex_to_splitsents_map"][obs_name]
    #             for split_sent_idx in split_sent_indices:
    #                 gold_text_peritem.append(item["split_sents"][split_sent_idx])
    #         gold_text_list.append(" ".join(gold_text_peritem))

    #     label_name_list = [GLOBAL_VARS.obs_id2name_dict[idx] for idx in obs_ids.tolist()]
    # else:
    #     raise ValueError(f"Invalid run_mode {CONFIG['run_mode']}, expected to find either a string 'pretrain' or 'finetune'")

    return {
        "batch_data": batch_data,
        "gold_text_list": gold_text_list,
        "label_name_list": label_name_list,  # e.g. ['effusion', 'pneumothorax', 'opacity', 'normal' ...]
        "pixel_values": piexl_values_tensor.to(DEVICE),  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map.to(DEVICE),  # torch.Size([bsz < x < 2*bsz]), # e.g. [0, 0, 1, 2, 2, ...]
        "obs_labels": obs_labels.to(DEVICE),  # torch.Size([bsz, num_obs]),
        "obs_ids": obs_ids.to(DEVICE),  # torch.Size([num_obs]),
    }


def get_inputs_for_training(tokenizer, batch_data, gold_text_list, label_name_list, pixel_values, image_indices_map, obs_labels, obs_ids):
    conversations = []
    num_image_tokens = GLOBAL_VARS.num_image_tokens
    for idx, item in enumerate(batch_data):
        num_images = len(item["images"])

        assistaant_output_text = gold_text_list[idx]

        content_list = []
        content_list.append({"type": "image", "num_images": num_images, "num_image_tokens": num_image_tokens})
        content_list.append({"type": "label", "obs_labels": label_name_list})
        content_list.append({"type": "text", "text": "Based on the provided chest X-ray images and the preliminary key radiological findings, please verify the findings on the images and generate an accurate report accordingly."})

        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiology assistant tasked with interpreting a chest X-ray study."}],
                },
                {"role": "user", "content": content_list},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistaant_output_text}],
                },
            ]
        )

    # See descriptions for assistant_tokens_mask
    # Assistant tokens are the tokens that need to be generated, we use these tokens to compute the loss
    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask

    tokenizer_kwargs = {"pad_to_multiple_of": 8}

    add_generation_prompt = False
    return_assistant_tokens_mask = True
    tokenizer_kwargs["padding_side"] = "right"

    input_text_tensor_dict = tokenizer.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, tokenize=True, padding=True, return_dict=True, return_tensors="pt", tokenizer_kwargs=tokenizer_kwargs, return_assistant_tokens_mask=return_assistant_tokens_mask)

    decoder_assistant_masks = None
    if "decoder_assistant_masks" in input_text_tensor_dict:
        decoder_assistant_masks = input_text_tensor_dict.decoder_assistant_masks
        if isinstance(decoder_assistant_masks, list):  # transformers==4.47.1 will return decoder_assistant_masks in nested list
            decoder_assistant_masks = torch.tensor(decoder_assistant_masks)
        decoder_assistant_masks = decoder_assistant_masks.to(DEVICE)

    return {
        "pixel_values": pixel_values,  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map,  # [[0], [1], [2, 3], ...]
        "decoder_input_ids": input_text_tensor_dict.input_ids.to(DEVICE),
        "decoder_attention_mask": input_text_tensor_dict.attention_mask.to(DEVICE),
        "decoder_assistant_masks": decoder_assistant_masks,
        "obs_labels": obs_labels,
        "obs_ids": obs_ids,
    }


def get_inputs_for_inference(tokenizer, batch_data, gold_text_list, label_name_list, pixel_values, image_indices_map, obs_labels, obs_ids):

    conversations = []
    num_image_tokens = GLOBAL_VARS.num_image_tokens
    for idx, item in enumerate(batch_data):
        num_images = len(item["images"])

        content_list = []
        content_list.append({"type": "image", "num_images": num_images, "num_image_tokens": num_image_tokens})
        content_list.append({"type": "label", "obs_labels": label_name_list})
        content_list.append({"type": "text", "text": "Based on the provided chest X-ray images and the preliminary key radiological findings, please verify the findings on the images and generate an accurate report accordingly."})

        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiology assistant tasked with interpreting a chest X-ray study."}],
                },
                {
                    "role": "user",
                    "content": content_list,
                },
            ]
        )

    # See descriptions for assistant_tokens_mask
    # Assistant tokens are the tokens that need to be generated, we use these tokens to compute the loss
    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask
    tokenizer_kwargs = {"pad_to_multiple_of": 8}

    add_generation_prompt = True
    return_assistant_tokens_mask = False
    tokenizer_kwargs["padding_side"] = "left"

    input_text_tensor_dict = tokenizer.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, tokenize=True, padding=True, return_dict=True, return_tensors="pt", tokenizer_kwargs=tokenizer_kwargs, return_assistant_tokens_mask=return_assistant_tokens_mask)

    return {
        "batch_data": batch_data,
        "pixel_values": pixel_values,  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map,  # [0, 0, 1, 2, 2, ...]
        "decoder_input_ids": input_text_tensor_dict.input_ids.to(DEVICE),
        "decoder_attention_mask": input_text_tensor_dict.attention_mask.to(DEVICE),
        "data_id_list": [i["data_key"] for i in batch_data],
        "gold_text_list": gold_text_list,
        "obs_labels": obs_labels,
        "obs_ids": obs_ids,
    }


#############################################
# Status Class and Tracker
#############################################
@dataclass
class StatusInfo:
    curr_epoch: int = field(default=0)
    curr_batch_iter: int = field(default=0)
    curr_checkpoint_at: str = field(default="")  # "epoch" or "batch"
    curr_eval_split: str = field(default="")  # "validation" or "test"

    global_iters: int = field(default=0)
    global_update_steps: int = field(default=0)
    dev_best: dict = field(default_factory=lambda: {"text_score": 0.0, "cls_score": 0.0, "sum_score": 0.0, "at_epoch": 0, "at_iter": 0, "check_at": ""})

    batch_loss: float = field(default=0.0)
    batch_gen_loss: float = field(default=0.0)
    batch_cls_loss: float = field(default=0.0)
    batch_trained_examples: int = field(default=0)

    run_id: str = field(default="")

    grad_accum_eval_mark: int = field(default=0)
    train_print_loss_mark: int = field(default=0)

    def is_achieving_best_dev_score(self, text_score=None, cls_score=None):
        sum_score = 0
        if text_score:
            sum_score += text_score
            if text_score >= self.dev_best["text_score"]:
                self.dev_best["text_score"] = text_score
        if cls_score:
            sum_score += cls_score
            if cls_score >= self.dev_best["cls_score"]:
                self.dev_best["cls_score"] = cls_score

        if sum_score >= self.dev_best["sum_score"]:
            self.dev_best["sum_score"] = sum_score
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
                raise ValueError("Either run_id or run_name should be provided.")

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


#############################################
# Training
#############################################


def train(model, train_dataloader, train_cfg, valid_dataloader=None, test_dataloader=None):
    global MLFLOW_TRACKER, STATUS_INFO

    # hyperparameters
    model_params = list(model.named_parameters())
    optimizer_grouped_parameters = prepare_optimizer_grouped_parameters(model_params, train_cfg)
    LOGGER.info("Model trainable params:\n%s", "\n".join([n for n, p in model.named_parameters() if p.requires_grad]))

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_num_steps = len(train_dataloader) // train_cfg["grad_accum_steps"] * train_cfg["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * train_cfg["warmup_proportion"]), num_training_steps=total_num_steps)

    # 1. Prepare for multi GPUs. All prepared and registered objs will be checkpointed automatically
    model, train_dataloader, optimizer, scheduler = ACCELERATOR.prepare(model, train_dataloader, optimizer, scheduler)
    if valid_dataloader:
        valid_dataloader = ACCELERATOR.prepare(valid_dataloader)
    if test_dataloader:
        test_dataloader = ACCELERATOR.prepare(test_dataloader)
    STATUS_INFO = StatusInfo()
    ACCELERATOR.register_for_checkpointing(STATUS_INFO)
    # LOGGER.debug("Final model structure:\n%s", model)

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
    LOGGER.info("[For pretrain] Classification only = %s", train_cfg.get("classification_only", False))
    LOGGER.info("[For finetune] Use classifier = True, inject_cls_token = True")
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
                # Not necessarily need ACCELERATOR.autocast()
                # Accelerate enables automatic mixed precision, so autocast() is only needed if there are other mixed precision operations besides those performed on loss by backward() which already handles the scaling.

                # 由于训练和推理时的输入数据不同，所以需要在这里分开处理
                batch_inputs_dict = get_inputs_for_training(tokenizer=active_dataloader.dataset.tokenizer, **batch_inputs_dict)

                with ACCELERATOR.autocast():
                    model.train()
                    out = model.forward(
                        pixel_values=batch_inputs_dict["pixel_values"],
                        decoder_input_ids=batch_inputs_dict["decoder_input_ids"],
                        decoder_attention_mask=batch_inputs_dict["decoder_attention_mask"],
                        decoder_assistant_masks=batch_inputs_dict["decoder_assistant_masks"],
                        decoder_extra_inputs={
                            "image_indices_map": batch_inputs_dict["image_indices_map"],
                            "obs_labels": batch_inputs_dict["obs_labels"],
                            "obs_ids": batch_inputs_dict["obs_ids"],
                        },
                        output_loss=True,
                    )
                    loss = out.loss

                ACCELERATOR.backward(loss)
                if train_cfg["clip_grad_norm"] > 0:
                    ACCELERATOR.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                check_memory(show_only_if_peak=True)

                gen_loss = out.gen_loss.item() if out.gen_loss is not None else 0.0
                cls_loss = out.cls_loss.item() if out.cls_loss is not None else 0.0
                log_and_update_status(curr_epoch=curr_epoch, curr_iter=curr_iter, loss=loss.item(), bsz=batch_inputs_dict["decoder_input_ids"].size(0), lr=scheduler.get_last_lr()[0], train_cfg=train_cfg, gen_loss=gen_loss, cls_loss=cls_loss)

                if model.config.classification_only:
                    # 当 classification_only=True 时，使用 eval_per_steps 在 training 过程中进行验证和保存
                    validation_process(model, valid_dataloader, max_num_iters_per_epoch=len(train_dataloader), train_cfg=train_cfg)
                else:
                    # 当 classification_only=False 时，不进行验证， 使用 ckp_per_steps 保存检查点
                    check_and_save_checkpoint(max_num_iters_per_epoch=len(train_dataloader), train_cfg=train_cfg)

        end = time.time()
        LOGGER.info("Batch training time: %s ", seconds_to_time_str(end - start))

    if not model.config.classification_only:
        # 当 classification_only=False 时，不进行验证，使用 ckp_per_steps 保存检查点，并在最后保存模型
        save_model(model, CONFIG["output_dir"]["model"])
    MLFLOW_TRACKER.finish()


def prepare_optimizer_grouped_parameters(model_params, train_cfg):
    # 为了节省计算资源和显存，应将需要冻结的参数的 `requires_grad` 显式设置为 `False`，并且在优化器中过滤不可训练参数

    optimizer_grouped_parameters = []
    if CONFIG["run_mode"] == "pretrain":
        encoder_params = [(n, p) for n, p in model_params if n.startswith("encoder")]
        decoder_params = [(n, p) for n, p in model_params if n.startswith("decoder")]
        adaptor_params = [(n, p) for n, p in model_params if n.startswith("v2l_projector")]
        classifier_params = [(n, p) for n, p in model_params if n.startswith("obs_classifier")]
        assert encoder_params and decoder_params and adaptor_params and classifier_params

        # 冻结 encoder, decoder，训练 v2l_projector
        for n, p in decoder_params:
            p.requires_grad = False
        for n, p in encoder_params + adaptor_params + classifier_params:
            p.requires_grad = True

        # no_decay_names = ["bias", "norm1.weight", "norm2.weight", "layernorm.weight", "layer_scale"]
        optimizer_grouped_parameters.append({"params": [p for n, p in model_params if p.requires_grad], "lr": train_cfg["lr"], "weight_decay": 0.0})

    elif CONFIG["run_mode"] == "finetune":
        # When using peft, params requires_grad are set during initialization of PeftModel. See `apply_peft_to_model()`.
        # We only need to group them for optimizer.
        optimizer_grouped_parameters.append({"params": [p for n, p in model_params if p.requires_grad and "obs_classifier" not in n], "lr": train_cfg["lr"], "weight_decay": 0.0})
        optimizer_grouped_parameters.append({"params": [p for n, p in model_params if p.requires_grad and "obs_classifier" in n], "lr": train_cfg["cls_lr"], "weight_decay": 0.0})

    return optimizer_grouped_parameters


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


def log_and_update_status(curr_epoch, curr_iter, loss, bsz, lr, train_cfg, gen_loss=None, cls_loss=None):
    STATUS_INFO.curr_epoch = curr_epoch
    STATUS_INFO.curr_batch_iter = curr_iter
    STATUS_INFO.batch_trained_examples += bsz
    STATUS_INFO.batch_loss += loss * bsz
    STATUS_INFO.global_iters += 1
    if gen_loss is not None:
        STATUS_INFO.batch_gen_loss += gen_loss * bsz
    if cls_loss is not None:
        STATUS_INFO.batch_cls_loss += cls_loss * bsz

    if ACCELERATOR.sync_gradients:
        STATUS_INFO.global_update_steps += 1

    # Logging too often may slow down the process
    do_log = False
    print_loss_per_n_steps = train_cfg["print_loss_per_n_steps"]

    if STATUS_INFO.global_update_steps == 1 or STATUS_INFO.global_update_steps % print_loss_per_n_steps == 0:
        do_log = True
        if STATUS_INFO.train_print_loss_mark == STATUS_INFO.global_update_steps:
            # We dont want to log multiple times in the same global_update_steps (when grad_accum > 1)
            do_log = False

    if do_log:
        avg_loss = STATUS_INFO.batch_loss / STATUS_INFO.batch_trained_examples
        avg_gen_loss = STATUS_INFO.batch_gen_loss / STATUS_INFO.batch_trained_examples
        avg_cls_loss = STATUS_INFO.batch_cls_loss / STATUS_INFO.batch_trained_examples

        MLFLOW_TRACKER.log(
            {
                "lr": lr,
                "avg_loss": avg_loss,
                "avg_gen_loss": avg_gen_loss,
                "avg_cls_loss": avg_cls_loss,
                "epoch": STATUS_INFO.curr_epoch,
                "global_update_steps": STATUS_INFO.global_update_steps,
            },
            step=STATUS_INFO.global_iters,
        )

        LOGGER.info(
            "p=%s, Epoch=%d, iter=%d, steps=%d, loss=%.9f (gen=%.9f, cls=%.9f)",
            ACCELERATOR.process_index,
            STATUS_INFO.curr_epoch,
            STATUS_INFO.curr_batch_iter,
            STATUS_INFO.global_update_steps,
            avg_loss,
            gen_loss if gen_loss is not None else 0.0,
            cls_loss if cls_loss is not None else 0.0,
            main_process_only=True,
        )
        STATUS_INFO.batch_loss, STATUS_INFO.batch_trained_examples = 0, 0
        STATUS_INFO.batch_gen_loss, STATUS_INFO.batch_cls_loss = 0, 0
        STATUS_INFO.train_print_loss_mark = STATUS_INFO.global_update_steps


def check_and_save_checkpoint(max_num_iters_per_epoch, train_cfg):
    do_ckp = True
    # do_ckp at the end of each epoch
    if STATUS_INFO.curr_batch_iter + 1 == max_num_iters_per_epoch:
        STATUS_INFO.curr_checkpoint_at = "epoch"
    # do_ckp at specific steps:
    elif train_cfg["ckp_per_steps"] > 0 and STATUS_INFO.global_update_steps % train_cfg["ckp_per_steps"] == 0:
        STATUS_INFO.curr_checkpoint_at = "batch"
    else:
        do_ckp = False

    # 当 grad_accum = N > 1 时，这 N 个 iters 的 STATUS_INFO.global_update_steps 都是一样的。不做处理时，都会激活 do_eval。
    # 我们希望这 N 个 iters 只进行一次 eval。
    # 目前的逻辑是，当进入这个条件时，说明在这个 global_update_steps 中，已经进行过一次 eval 了，其余的 iters 不需要进行 eval。
    # 由于 grad_accum_eval_mark 默认值为 0，所以 global_update_steps == 0 时，也默认不评估。
    if STATUS_INFO.grad_accum_eval_mark == STATUS_INFO.global_update_steps:
        do_ckp = False

    if do_ckp:
        check_memory()
        STATUS_INFO.grad_accum_eval_mark = STATUS_INFO.global_update_steps  # this line shoud runs before save_checkpoint(), to set the correct STATUS_INFO.grad_accum_eval_mark for checkpoingting
        save_checkpoint(checkpoint_dir=CONFIG["output_dir"]["checkpoint"])


#############################################
# Validation
# Since the validation process takes a lot of time, we skip the validation process in training.
#############################################
def validation_process(model, valid_dataloader, max_num_iters_per_epoch, train_cfg):

    do_eval = True
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

    # 当 grad_accum = N > 1 时，这 N 个 iters 的 STATUS_INFO.global_update_steps 都是一样的。不做处理时，都会激活 do_eval。
    # 我们希望这 N 个 iters 只进行一次 eval。
    # 目前的逻辑是，当进入这个条件时，说明在这个 global_update_steps 中，已经进行过一次 eval 了，其余的 iters 不需要进行 eval。
    # 由于 grad_accum_eval_mark 默认值为 0，所以 global_update_steps == 0 时，也默认不评估。
    if STATUS_INFO.grad_accum_eval_mark == STATUS_INFO.global_update_steps:
        do_eval = False

    if do_eval:
        check_memory()
        eval_result_dict = evaluate(model, target_dataloader=valid_dataloader, overwrite_pred_file=True, **train_cfg)
        STATUS_INFO.grad_accum_eval_mark = STATUS_INFO.global_update_steps  # this line shoud runs before check_results_and_save_model(), to set the correct STATUS_INFO.grad_accum_eval_mark for checkpoingting
        check_results_and_save_model(model, eval_result_dict)


def check_results_and_save_model(model, eval_result_dict):
    # Check
    text_score = 0
    num_metrics = 0
    for metric_key in ["BLEU", "ROUGEL", "chexbert-all_micro avg_f1-score", "radgraph_partial", "bertscore"]:
        if metric_key in eval_result_dict:
            MLFLOW_TRACKER.log(
                {f"{STATUS_INFO.curr_eval_split}_{metric_key}": eval_result_dict[metric_key]},
                step=STATUS_INFO.global_iters,
            )
            text_score += eval_result_dict[metric_key]
            num_metrics += 1
    text_score = text_score / num_metrics if num_metrics > 0 else 0.0

    cls_score = 0
    if "overall" in eval_result_dict:
        cls_score = eval_result_dict["overall"]["micro"]["f1"]

    LOGGER.info("****************************** Checkpoint ******************************")
    LOGGER.info("Current [%s] text_avg_f1: %.3f, cls_micro_f1: %.3f, at epoch %d, iter %d (%s)", STATUS_INFO.curr_eval_split, text_score * 100, cls_score * 100, STATUS_INFO.curr_epoch, STATUS_INFO.curr_batch_iter, STATUS_INFO.curr_checkpoint_at)
    LOGGER.info("Best [%s] %s", STATUS_INFO.curr_eval_split, STATUS_INFO.dev_best)
    MLFLOW_TRACKER.log({f"{STATUS_INFO.curr_eval_split}_text_avg_f1": text_score}, step=STATUS_INFO.global_iters)
    MLFLOW_TRACKER.log({f"{STATUS_INFO.curr_eval_split}_cls_avg_f1": cls_score}, step=STATUS_INFO.global_iters)

    # checkpointing
    save_checkpoint(checkpoint_dir=CONFIG["output_dir"]["checkpoint"])

    # Save the best
    if STATUS_INFO.is_achieving_best_dev_score(text_score, cls_score):
        save_model(model, CONFIG["output_dir"]["model"])


#############################################
# Evaluation
#############################################
def evaluate(model, target_dataloader, overwrite_pred_file=False, **kwargs):
    GLOBAL_VARS.peak_mem = 0

    eval_bsz = kwargs["eval_batch_size"]
    max_new_tokens = kwargs["max_new_tokens"]
    print_pred_per_n_steps = kwargs["print_pred_per_n_steps"]
    num_beams = kwargs["num_beams"]
    classification_only = kwargs.get("classification_only", False)

    # only used during fintune mode
    target_obs = kwargs.get("target_observation", None)

    if overwrite_pred_file:
        data_split = target_dataloader.dataset.split
        output_dir = CONFIG["output_dir"]["result"]
        output_file = os.path.join(output_dir, f"{data_split}_{ACCELERATOR.process_index}.json")
        if os.path.exists(output_file):
            os.remove(output_file)
            LOGGER.info("Removed existing prediction results file: %s", output_file)

    # 由于评估时间过长，pred结果将被存放到文件中，已经pred过的数据已经提前从dataset中移除
    LOGGER.info("****************************** Evaluation ******************************")
    LOGGER.info("Source = %s", target_dataloader.dataset.src_path)
    LOGGER.info("Batch size = %d", eval_bsz)
    LOGGER.info("Num samples = %d", len(target_dataloader.dataset))
    LOGGER.info("Max new tokens = %d, Num beams = %d", max_new_tokens, num_beams)
    LOGGER.info("[For pretrain] Classification only = %s", classification_only)
    LOGGER.info("[For finetune] Use classifier = True, inject_cls_token = True")
    tokenizer = target_dataloader.dataset.tokenizer

    LOGGER.info("****************************** Model Predicting ******************************")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, input_tensors_dict in enumerate(target_dataloader):
            if len(target_dataloader.dataset) == 0 or input_tensors_dict is None:
                LOGGER.info("No data remain unpredicted, stop model inference")
                break

            input_tensors_dict = get_inputs_for_inference(tokenizer=tokenizer, **input_tensors_dict)

            data_ids = input_tensors_dict["data_id_list"]
            pred_labels, gold_labels, pred_text, gold_text = [], [], [], []

            if classification_only:
                with ACCELERATOR.autocast():
                    outputs = model.forward(
                        pixel_values=input_tensors_dict["pixel_values"],
                        decoder_input_ids=input_tensors_dict["decoder_input_ids"],
                        decoder_attention_mask=input_tensors_dict["decoder_attention_mask"],
                        decoder_extra_inputs={
                            "image_indices_map": input_tensors_dict["image_indices_map"],
                            "obs_labels": input_tensors_dict["obs_labels"],
                            "obs_ids": input_tensors_dict["obs_ids"],
                        },
                        output_loss=True,
                    )
                pred_obs_logits = outputs.logits  # torch.Size([bsz, num_obs, 3])
                pred_obs_labels = pred_obs_logits.argmax(dim=-1)
                pred_label_ids = pred_obs_labels.tolist()
                gold_label_ids = input_tensors_dict["obs_labels"].tolist()

                assert len(pred_label_ids) == len(gold_label_ids), f"All lists must have the same length: [pred_label_ids: {len(pred_label_ids)}, gold_label_ids: {len(gold_label_ids)}]"

                for i in range(len(pred_label_ids)):
                    pred_labels_inbatch, gold_labels_inbatch = {}, {}
                    for obs_id, pred_label_id, gold_label_id in zip(input_tensors_dict["obs_ids"].tolist(), pred_label_ids[i], gold_label_ids[i]):
                        obs_name = GLOBAL_VARS.obs_id2name_dict[obs_id]
                        pred_cls_label = GLOBAL_VARS.obs_id2cls_dict[pred_label_id]
                        gold_cls_label = GLOBAL_VARS.obs_id2cls_dict[gold_label_id]
                        pred_labels_inbatch[obs_name] = pred_cls_label
                        gold_labels_inbatch[obs_name] = gold_cls_label
                    pred_labels.append(pred_labels_inbatch)
                    gold_labels.append(gold_labels_inbatch)

                if (print_pred_per_n_steps > 0 and batch_idx % print_pred_per_n_steps == 0) or (batch_idx + 1 == len(target_dataloader)):
                    LOGGER.info(
                        "Eval at: p=%s, iter=%d, finished_samples=%s, pred_example: \n[obs_label] %s",
                        ACCELERATOR.process_index,
                        batch_idx,
                        batch_idx * eval_bsz,
                        {k: v for k, v in pred_labels[0].items() if v != ""},
                        main_process_only=False,
                    )
            else:
                with ACCELERATOR.autocast():
                    # https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation#transformers.GenerationConfig
                    # 防重复设置
                    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=[tokenizer.eos_token_id, GLOBAL_VARS.eot_token_id], do_sample=False, num_beams=num_beams, return_dict_in_generate=True, output_logits=True, no_repeat_ngram_size=4, temperature=0.9, top_k=50, top_p=0.9)
                    # https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation#transformers.GenerationMixin
                    # If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and forward specific kwargs should be prefixed with decoder_.
                    outputs, extra_outputs = model.generate(
                        generation_config=generation_config,
                        inputs=input_tensors_dict["pixel_values"],
                        decoder_input_ids=input_tensors_dict["decoder_input_ids"],
                        decoder_attention_mask=input_tensors_dict["decoder_attention_mask"],
                        decoder_extra_inputs={
                            "image_indices_map": input_tensors_dict["image_indices_map"],
                            "obs_ids": input_tensors_dict["obs_ids"],
                            "num_beams": num_beams,
                        },
                        decoder_extra_outputs={"cls_logits": None},
                    )
                    check_memory(show_only_if_peak=True)

                # Gathers input_data and potentially drops duplicates in the last batch if on a distributed system.
                # 没有使用 gather，而是在 load_pred_results 中对结果进行去重
                pred_seq_start_ids = input_tensors_dict["decoder_input_ids"].size(1)  # 生成的序列的起始位置
                pred_sequences_ids = outputs.sequences[:, pred_seq_start_ids:]
                pred_sequences = tokenizer.batch_decode(pred_sequences_ids, skip_special_tokens=True)
                pred_text = pred_sequences
                gold_text = input_tensors_dict["gold_text_list"]

                pred_obs_logits = extra_outputs["cls_logits"]  # torch.Size([bsz, num_obs, 3])
                pred_obs_labels = pred_obs_logits.argmax(dim=-1)
                pred_label_ids = pred_obs_labels.tolist()
                gold_label_ids = input_tensors_dict["obs_labels"].tolist()

                assert len(pred_label_ids) == len(gold_label_ids) == len(pred_text) == len(gold_text), f"All lists must have the same length: [pred_label_ids: {len(pred_label_ids)}, gold_label_ids: {len(gold_label_ids)}, pred_text: {len(pred_text)}, gold_text: {len(gold_text)}]"

                for i in range(len(pred_label_ids)):
                    pred_labels_inbatch, gold_labels_inbatch = {}, {}
                    for obs_id, pred_label_id, gold_label_id in zip(input_tensors_dict["obs_ids"].tolist(), pred_label_ids[i], gold_label_ids[i]):
                        obs_name = GLOBAL_VARS.obs_id2name_dict[obs_id]
                        pred_cls_label = GLOBAL_VARS.obs_id2cls_dict[pred_label_id]
                        gold_cls_label = GLOBAL_VARS.obs_id2cls_dict[gold_label_id]
                        pred_labels_inbatch[obs_name] = pred_cls_label
                        gold_labels_inbatch[obs_name] = gold_cls_label
                    pred_labels.append(pred_labels_inbatch)
                    gold_labels.append(gold_labels_inbatch)

                if (print_pred_per_n_steps > 0 and batch_idx % print_pred_per_n_steps == 0) or (batch_idx + 1 == len(target_dataloader)):
                    LOGGER.info(
                        "Eval at: p=%s, iter=%d, finished_samples=%s, pred_example: \n[text]: %s \n[obs_label] %s",
                        ACCELERATOR.process_index,
                        batch_idx,
                        batch_idx * eval_bsz,
                        pred_sequences[0],
                        {k: v for k, v in pred_labels[0].items() if v != "absent"} if pred_labels else "N/A",
                        main_process_only=False,
                    )

            save_pred_results_per_batch(
                data_ids=data_ids,
                pred_text=pred_text,
                pred_labels=pred_labels,
                gold_text=gold_text,
                gold_labels=gold_labels,
                data_split=target_dataloader.dataset.split,
                output_dir=CONFIG["output_dir"]["result"],
            )

    LOGGER.info("****************************** Computing Scores ******************************")
    pred_result_dict = load_pred_results(intput_dir=CONFIG["output_dir"]["result"], split=target_dataloader.dataset.split)
    data_ids = [item["data_id"] for item in pred_result_dict.values()]
    pred_text = [item["pred_text"] for item in pred_result_dict.values()]
    pred_labels = [item["pred_label"] for item in pred_result_dict.values()]
    gold_text = [item["gold_text"] for item in pred_result_dict.values()]
    gold_labels = [item["gold_label"] for item in pred_result_dict.values()]

    # Evaluate text results
    text_scores_dict = {}
    if not classification_only:
        text_scores_dict = compute_generation_score(gold_text_list=gold_text, pred_text_list=pred_text)
        LOGGER.info("[TextGen]: %s", json.dumps(text_scores_dict, indent=4))

    label_scores_dict = {}
    label_scores_dict = compute_multilabel_multiclass_scores(gold_labels_list=gold_labels, pred_labels_list=pred_labels, obs_names=list(GLOBAL_VARS.obs_name2id_dict.keys()), obs_labels=list(GLOBAL_VARS.obs_cls2id_dict.keys()))
    if target_obs:
        # 只保留指定的 observation 的结果
        label_scores_dict = {obs: label_scores_dict[obs] for obs in label_scores_dict if obs in target_obs}
    LOGGER.info("[ClsPred]: %s", custom_dumps(label_scores_dict, max_indent_level=2, indent=4))

    end = time.time()
    LOGGER.info("Evaluation time: %s", seconds_to_time_str(end - start))
    check_memory()
    return {**text_scores_dict, **label_scores_dict}


def save_pred_results_per_batch(data_ids, pred_text, pred_labels, gold_text, gold_labels, data_split, output_dir):
    """Save at each batch, so that we can use the results for further analysis or debugging."""

    if len(pred_text) != 0:
        if len(pred_labels) != 0:
            assert len(data_ids) == len(pred_labels) == len(gold_labels) == len(gold_text) == len(pred_text), f"All lists must have the same length: [data_ids: {len(data_ids)}, pred_text: {len(pred_text)}, pred_labels: {len(pred_labels)}, gold_text: {len(gold_text)}, gold_labels: {len(gold_labels)}]"
        else:
            assert len(data_ids) == len(pred_text) == len(gold_text), f"All lists must have the same length: [data_ids: {len(data_ids)}, pred_text: {len(pred_text)}, gold_text: {len(gold_text)}]"
    else:
        assert len(data_ids) == len(pred_labels) == len(gold_labels), f"All lists must have the same length: [data_ids: {len(data_ids)}, pred_labels: {len(pred_labels)}, gold_labels: {len(gold_labels)}]"

    output_file = os.path.join(output_dir, f"{data_split}_{ACCELERATOR.process_index}.json")

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, data_id in enumerate(data_ids):
            p_text = pred_text[idx] if idx < len(pred_text) else ""
            g_text = gold_text[idx] if idx < len(gold_text) else ""
            p_label = pred_labels[idx] if idx < len(pred_labels) else {}
            g_label = gold_labels[idx] if idx < len(gold_labels) else {}
            out_line = {"data_id": data_id, "pred_text": p_text, "pred_label": p_label, "gold_text": g_text, "gold_label": g_label}
            f.write(json.dumps(out_line))
            f.write("\n")


def load_pred_results(intput_dir, split):
    data_dict = {}  # key=data_id, value={"data_id": , "pred_text": , "gold_text": ,...}

    # 遍历目录中的所有文件
    for filename in os.listdir(intput_dir):
        if filename.startswith(f"{split}_") and filename.endswith(".json"):
            file_path = os.path.join(intput_dir, filename)
            LOGGER.info("Loading pred_results from file: \n%s", file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                # Load each line as a JSON object, and remove duplicate entries
                for line in f:
                    if line.strip():
                        data_item = json.loads(line.strip())
                        data_id = data_item.get("data_id")
                        if data_id not in data_dict:
                            # Only add if the data_id is not already in the dictionary
                            data_dict[data_id] = data_item
                        else:
                            # If the data_id already exists
                            # Update the existing entry with the new one
                            LOGGER.info("Duplicate data_id found: %s, \n(Use) %s \n(Drop) %s", data_id, data_item, data_dict[data_id])
                            data_dict[data_id].update(data_item)
            LOGGER.info("Loaded pred_results from file: \n%s", file_path)

    return data_dict


def compute_multilabel_multiclass_scores(gold_labels_list, pred_labels_list, obs_names, obs_labels, round_digits=5):
    assert len(gold_labels_list) == len(pred_labels_list), "Mismatched number of samples"

    scores = {"overall": {}}  # 让 overall 排在最前面，便于log打印观察

    # 用于计算总 micro
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)

    # 每个 obs 的宏平均指标（用于最终宏平均）
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for obs in obs_names:
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for gold_dict, pred_dict in zip(gold_labels_list, pred_labels_list):
            gold = gold_dict.get(obs)
            pred = pred_dict.get(obs)

            for label in obs_labels:
                if pred == label and gold == label:
                    tp[label] += 1
                    total_tp[label] += 1
                elif pred == label and gold != label:
                    fp[label] += 1
                    total_fp[label] += 1
                elif pred != label and gold == label:
                    fn[label] += 1
                    total_fn[label] += 1

        # 计算 per-label 指标
        obs_result = {}
        obs_p_list = []
        obs_r_list = []
        obs_f1_list = []

        for label in obs_labels:
            p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            obs_result[label] = {
                "precision": round(p, round_digits),
                "recall": round(r, round_digits),
                "f1": round(f1, round_digits),
                "tp": tp[label],
                "fp": fp[label],
                "fn": fn[label],
            }
            obs_p_list.append(p)
            obs_r_list.append(r)
            obs_f1_list.append(f1)

        # 每个 obs 的 micro/macro 平均
        micro_tp = sum(tp.values())
        micro_fp = sum(fp.values())
        micro_fn = sum(fn.values())
        micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        scores[obs] = {
            "per_class": obs_result,
            "micro": {
                "precision": round(micro_p, round_digits),
                "recall": round(micro_r, round_digits),
                "f1": round(micro_f1, round_digits),
            },
            "macro": {
                "precision": round(sum(obs_p_list) / len(obs_p_list), round_digits),
                "recall": round(sum(obs_r_list) / len(obs_r_list), round_digits),
                "f1": round(sum(obs_f1_list) / len(obs_f1_list), round_digits),
            },
        }

        macro_precisions.append(sum(obs_p_list) / len(obs_p_list))
        macro_recalls.append(sum(obs_r_list) / len(obs_r_list))
        macro_f1s.append(sum(obs_f1_list) / len(obs_f1_list))

    # 所有标签的 micro 总体指标
    total_tp_sum = sum(total_tp.values())
    total_fp_sum = sum(total_fp.values())
    total_fn_sum = sum(total_fn.values())

    overall_micro_p = total_tp_sum / (total_tp_sum + total_fp_sum) if (total_tp_sum + total_fp_sum) > 0 else 0.0
    overall_micro_r = total_tp_sum / (total_tp_sum + total_fn_sum) if (total_tp_sum + total_fn_sum) > 0 else 0.0
    overall_micro_f1 = 2 * overall_micro_p * overall_micro_r / (overall_micro_p + overall_micro_r) if (overall_micro_p + overall_micro_r) > 0 else 0.0

    overall_macro_p = sum(macro_precisions) / len(macro_precisions)
    overall_macro_r = sum(macro_recalls) / len(macro_recalls)
    overall_macro_f1 = sum(macro_f1s) / len(macro_f1s)

    scores["overall"] = {
        "micro": {
            "precision": round(overall_micro_p, round_digits),
            "recall": round(overall_micro_r, round_digits),
            "f1": round(overall_micro_f1, round_digits),
        },
        "macro": {
            "precision": round(overall_macro_p, round_digits),
            "recall": round(overall_macro_r, round_digits),
            "f1": round(overall_macro_f1, round_digits),
        },
    }

    return scores


def compute_generation_score(gold_text_list, pred_text_list):
    """Based on the script from https://vilmedic.app/misc/bionlp24/leaderboard#anchor-baseline"""
    if DEVICE.type == "cpu":
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert"]
    else:
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert", "bertscore"]

    refs = [" ".join(wordpunct_tokenize(s.lower())) for s in gold_text_list]
    hyps = [" ".join(wordpunct_tokenize(s.lower())) for s in pred_text_list]

    # 我们想检查一下空报告样本对eval分数的影响
    # refs, hyps = [], []
    # for ref, hyp in zip(gold_text_list, pred_text_list):
    #     if ref.strip() == "" or hyp.strip() == "":
    #         continue
    #     refs.append(" ".join(wordpunct_tokenize(ref.lower())))
    #     hyps.append(" ".join(wordpunct_tokenize(hyp.lower())))

    # https://github.com/jbdel/vilmedic/blob/main/vilmedic/blocks/scorers/scores.py
    out_dict = compute_scores(use_metrics, refs=refs, hyps=hyps, split=None, seed=None, config=None, epoch=None, logger=LOGGER, dump=False)
    out_dict = {k: float(v) for k, v in out_dict.items()}
    return out_dict


import json


def custom_dumps(obj, level=0, max_indent_level=2, indent=4):
    if isinstance(obj, dict):
        if level >= max_indent_level:
            return json.dumps(obj, separators=(",", ":"))
        items = []
        for key, value in obj.items():
            items.append(" " * indent * (level + 1) + json.dumps(key) + ": " + custom_dumps(value, level + 1, max_indent_level, indent))
        return "{\n" + ",\n".join(items) + "\n" + " " * indent * level + "}"
    elif isinstance(obj, list):
        if level >= max_indent_level:
            return json.dumps(obj, separators=(",", ":"))
        items = []
        for value in obj:
            items.append(" " * indent * (level + 1) + custom_dumps(value, level + 1, max_indent_level, indent))
        return "[\n" + ",\n".join(items) + "\n" + " " * indent * level + "]"
    else:
        return json.dumps(obj)


#############################################
# Utils
#############################################


def check_memory(show_only_if_peak=False):
    if not torch.cuda.is_available():
        return
    # 获取当前 GPU 设备的属性

    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    # 获取 GPU 总显存
    total_memory = device_properties.total_memory / 1024**3  # 转换为 GB
    # 获取Torch总占用显存
    total_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

    if show_only_if_peak:
        # 显存占用的峰值值
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
        if peak_reserved > GLOBAL_VARS.peak_mem:
            GLOBAL_VARS.peak_mem = peak_reserved
            LOGGER.info("Peak memory reached: %.2f / %.2f GB", peak_reserved, total_memory)
        # torch.cuda.reset_max_memory_reserved()  # 重置峰值值
    else:
        LOGGER.info("Memory reserved: %.2f / %.2f GB", total_reserved, total_memory)


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


def save_checkpoint(checkpoint_dir, max_to_keep=5):
    ckp_path = os.path.join(checkpoint_dir, f"epoch_{STATUS_INFO.curr_epoch}_iter_{STATUS_INFO.curr_batch_iter}")
    ACCELERATOR.save_state(ckp_path)
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
    # 在使用 FULL_SHARD 和 SHARDED_STATE_DICT 时，model.state_dict()，unwrapped_model.state_dict()， ACCELERATOR.get_state_dict(model) 获得的 state_dict 都是一样的
    ACCELERATOR.wait_for_everyone()
    unwrapped_model = ACCELERATOR.unwrap_model(model)
    if isinstance(unwrapped_model, PeftModel):
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=ACCELERATOR.is_main_process,
            save_function=ACCELERATOR.save,
            save_embedding_layers=True,
        )
    else:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=ACCELERATOR.is_main_process,
            save_function=ACCELERATOR.save,
        )
    ACCELERATOR.wait_for_everyone()
    LOGGER.info("Model saved to %s", output_dir)


def save_model_outside_train_func(model, output_dir):
    # accelerator.get_state_dict(model) 会导致 rank1 上的 state_dict 不完整，不能使用
    # get_peft_model_state_dict(unwrapped_model) 可以让所有rank都拿到完整的 state_dict
    # unwrapped_model.save_pretrained 不传入 state_dict 时，会使用 get_peft_model_state_dict(self) 来获取 state_dict。也就是说是正确的
    # 当获取state_dict时，如果程序尝试在多个rank中同步state_dict，而由使用了 if ACCELERATOR.is_main_process: 来判断是否保存模型，则可能会导致进程卡主。比如rank0需要等到rank1同步state_dict，而rank1已经跳过了这个判断，进入wait的状态。
    # 所以结论就是，unwrapped_model 是 peft model时，不需要传入 state_dict 给 unwrapped_model.save_pretrained()，让它自己在内部调用 get_peft_model_state_dict() 来获取 state_dict即可，它也会处理不同rank的情况。
    LOGGER.debug("Saving model %s", model)

    ACCELERATOR.wait_for_everyone()
    unwrapped_model = ACCELERATOR.unwrap_model(model)
    if isinstance(unwrapped_model, PeftModel):
        # accelerator.get_state_dict(model) 会导致 rank1 上的 state_dict 不完整，不能使用
        # get_peft_model_state_dict(unwrapped_model) 可以让所有rank都拿到完整的 state_dict
        # unwrapped_model.save_pretrained 不传入 state_dict 时，会使用 get_peft_model_state_dict(self) 来获取 state_dict。也就是说是正确的
        # 当获取state_dict时，如果程序尝试在多个rank中同步state_dict，而由使用了 if ACCELERATOR.is_main_process: 来判断是否保存模型，则可能会导致进程卡主。比如rank0需要等到rank1同步state_dict，而rank1已经跳过了这个判断，进入wait的状态。
        # 所以结论就是，unwrapped_model 是 peft model时，不需要传入 state_dict 给 unwrapped_model.save_pretrained()，让它自己在内部调用 get_peft_model_state_dict() 来获取 state_dict即可，它也会处理不同rank的情况。
        # 目前的测试来看，不需要state_dict=ACCELERATOR.get_state_dict(model)， 模型也能够正确的保存完整的参数。
        # 当前使用的FSDPPlugin 参数为 ：sharding_strategy = "FULL_SHARD"，state_dict_type = "SHARDED_STATE_DICT"
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=ACCELERATOR.is_main_process,
            save_function=ACCELERATOR.save,
            save_embedding_layers=True,
        )
    else:
        # 虽然Accelerate FSDP 建议使用 state_dict=ACCELERATOR.get_state_dict(model)
        # 但经过测试，使用该参数，会导致 rank1 额外保存一个 model.satetensors 文件，导致加载时出现识别错误。
        # 目前的测试来看，不需要state_dict=ACCELERATOR.get_state_dict(model)， 模型也能够正确的保存完整的参数。
        # 当前使用的FSDPPlugin 参数为 ：sharding_strategy = "FULL_SHARD"，state_dict_type = "SHARDED_STATE_DICT"
        # https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=ACCELERATOR.is_main_process,
            save_function=ACCELERATOR.save,
            # state_dict=ACCELERATOR.get_state_dict(model),  # suggested by Accelerate FSDP when using transformers
        )

    ACCELERATOR.wait_for_everyone()
    LOGGER.info("Model saved to %s", output_dir)


def save_processors(img_processor, tokenizer, output_dir):
    ACCELERATOR.wait_for_everyone()
    if ACCELERATOR.is_main_process:
        img_processor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        LOGGER.info("Image Processor and tokenizer are saved to: %s", output_dir)
    ACCELERATOR.wait_for_everyone()


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
                    selected_image_indices = [int(i), int(j)]

    return selected_images, selected_image_indices


def resize_image_with_bspline_pil(image, target_size=518):
    # 转换 PIL 图像为 numpy 数组
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # 计算缩放比例，确保较短边为 target_size
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 根据图像维度计算缩放因子
    if img_array.ndim == 2:  # 灰度图像
        zoom_factors = (new_h / h, new_w / w)
    elif img_array.ndim == 3:  # RGB/多通道图像
        zoom_factors = (new_h / h, new_w / w, 1)
    else:
        raise ValueError(f"Unsupported image dimension: {img_array.ndim}")

    # 使用 B-spline 插值 (order=3 表示 B-spline)
    resized_array = zoom(img_array, zoom_factors, order=3)

    # 转换回 PIL 图像
    resized_image = Image.fromarray(np.uint8(resized_array))

    return resized_image


def process_image(img_processor, img_dataset, data_split, shortest_edge):
    preprocess_cfg = CONFIG["preprocess"]

    # add `data_key` to image dataset as a unique identifier which is identical to the text dataset's `doc_key`
    img_dataset = img_dataset.add_column("data_key", [f"{data_split}#{idx}" for idx in range(len(img_dataset))])

    def map_func(examples):
        # Select images
        # 保存图像的piexl_values会占用极大硬盘空间，且极大的减慢模型训练时的数据读取速度。
        # 因此预处理只进行resize
        selected_images_list = []
        selected_indices_list = []
        for example_idx, images_per_example in enumerate(examples["images"]):
            selected_images, selected_indices = select_images(images_per_example)
            # LANCZOS 更适合处理含有精细细节的图像 (如 X-ray 图像), 可以更好地保留图像中高频信息。适合对病灶等微小特征的保留。
            selected_images = [resize_image_with_bspline_pil(img, target_size=shortest_edge) for img in selected_images]
            selected_images_list.append(selected_images)
            selected_indices_list.append(selected_indices)

        examples["images"] = selected_images_list
        examples["selected_indices_list"] = selected_images_list
        return examples

    new_dataset = img_dataset.map(map_func, batched=preprocess_cfg["batched"], batch_size=preprocess_cfg["batch_size"], num_proc=preprocess_cfg["num_proc"])
    LOGGER.info("Processed image dataset: \n%s", new_dataset)
    return new_dataset


def load_image_datasets(data_paths):
    LOGGER.info("Loading raw image dataset")
    dataset_interpret = load_from_disk(data_paths["interpret"])
    LOGGER.info("%s loaded from interpret-cxr", [f"{split}:{len(ds)}" for split, ds in dataset_interpret.items()])
    dataset_mimic = load_from_disk(data_paths["mimic"])
    LOGGER.info("%s loaded from mimic-cxr", [f"{split}:{len(ds)}" for split, ds in dataset_mimic.items()])

    # Concat both
    dataset_train_dev = DatasetDict({"train": concatenate_datasets([dataset_interpret["train"], dataset_mimic["train"]]), "validation": concatenate_datasets([dataset_interpret["validation"], dataset_mimic["validation"]])})

    dataset_test = load_from_disk(data_paths["interpret-test-public"])
    LOGGER.info("%s loaded from interpret-test-public", [f"{split}:{len(ds)}" for split, ds in dataset_mimic.items()])

    ds_img = DatasetDict({"train": dataset_train_dev["train"], "validation": dataset_train_dev["validation"], "test": dataset_test["test"]})
    LOGGER.info("Loaded image-report dataset: \n%s", ds_img)
    return ds_img


def preprocess_dataset():
    LOGGER.info("Preprocessing image dataset")
    img_dataset = load_image_datasets(data_paths=CONFIG["data_path"])

    # Get dataloader for training and testing
    image_processor_name = CONFIG["preprocess"]["image_processor"]
    model_name_or_path = CONFIG["model_name_or_path"][image_processor_name]
    img_processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=True)
    LOGGER.info("Loaded image processor from: \n%s", model_name_or_path)
    shortest_edge = img_processor.size["shortest_edge"]

    ds_dict = {}
    for split in ["train", "validation", "test"]:
        ds_dict[split] = process_image(img_processor=img_processor, img_dataset=img_dataset[split], data_split=split, shortest_edge=shortest_edge)

    pre_processed_dataset_dict = DatasetDict(ds_dict)
    pre_processed_dataset_dict.save_to_disk(CONFIG["preprocess"]["cache_path"])
    LOGGER.info("Preprocessed image dataset, saved to: \n%s", CONFIG["preprocess"]["cache_path"])


#############################################
def load_peft_model(base_model, peft_model_path):
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    LOGGER.info("Fine-tuned PEFT model loaded from \n%s", peft_model_path)
    log_trainable_parameters(peft_model)
    return peft_model


def load_module_state_dict_from(model_path, target_module_prefixs):
    index_file_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file_path, "r", encoding="utf-8") as f:
        sd_index = json.load(f)

    target_model_file_paths = set()
    for key, file_name in sd_index["weight_map"].items():
        if any(key.startswith(prefix) for prefix in target_module_prefixs):
            target_model_file_paths.add(os.path.join(model_path, file_name))

    target_state_dict = {}
    for model_file_path in target_model_file_paths:
        for name, param in load_file(model_file_path).items():
            if any(name.startswith(prefix) for prefix in target_module_prefixs):
                target_state_dict[name] = param

    LOGGER.info("Loaded pretrained params: [%s] from: \n%s", target_state_dict.keys(), model_path)
    return target_state_dict


def load_state_dict_to_model(base_model, target_state_dict):
    base_model.load_state_dict(target_state_dict, strict=False)

    model_named_params = dict(base_model.named_parameters())
    for n, p in target_state_dict.items():
        assert torch.equal(model_named_params[n], p), f"Model params update failed [{n}], expected: {p}, got:{model_named_params[n]}"
    LOGGER.info("Updated pretrained params to base model: [%s]", target_state_dict.keys())

    return base_model


def load_model(model_path):
    model = Vision2LanguageModel.from_pretrained(model_path)
    LOGGER.info("Fine-tuned model loaded from %s", model_path)
    return model


def load_processor(processor_path):
    img_processor = AutoImageProcessor.from_pretrained(processor_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(processor_path)
    LOGGER.info("Image_processor and tokenizer are loaded from %s", processor_path)
    return img_processor, tokenizer


def filter_dataset_by_data_id(ds, split):
    # 单次48小时可能不足以完成评估，因此我们将每个batch的预测结果保存到文件中，最后再计算分数
    # 这里获取已经存在的预测结果文件，避免重复计算
    pred_result_dict = load_pred_results(intput_dir=CONFIG["output_dir"]["result"], split=split)
    LOGGER.info("Loaded pred_results of split [%s], total: %d", CONFIG["output_dir"]["result"], len(pred_result_dict))
    LOGGER.info("Dataset [%s] size before filtering: %d", split, len(ds))
    data_ids_to_skip = pred_result_dict.keys()
    ds = ds.filter(lambda x: x["data_key"] not in data_ids_to_skip)
    LOGGER.info("Dataset [%s] size after filtering: %d", split, len(ds))
    return ds


def get_dataloaders(img_processor, tokenizer, ds_train=None, ds_valid=None, ds_test=None, train_bsz=1, eval_bsz=1, use_debug_subset=False, target_observation=None):

    train_dataloader, valid_dataloader, test_dataloader = None, None, None

    if ds_train:
        with ACCELERATOR.main_process_first():  # select是dataset caching 操作，主进程优先或许能快一点
            if use_debug_subset:
                train_dataset = ImageTextDataset(ds_train.select(range(len(ds_train) - 6, len(ds_train))), img_processor=img_processor, tokenizer=tokenizer, split="train", target_obs=target_observation)
            else:
                train_dataset = ImageTextDataset(ds_train, img_processor=img_processor, tokenizer=tokenizer, split="train", target_obs=target_observation)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer, target_observation), batch_size=train_bsz, drop_last=True)

    if ds_valid:
        with ACCELERATOR.main_process_first():  # select是dataset caching 操作，主进程优先或许能快一点
            if use_debug_subset:
                vaild_dataset = ImageTextDataset(ds_valid.select(range(len(ds_valid) - 2, len(ds_valid))), img_processor=img_processor, tokenizer=tokenizer, split="validation", target_obs=target_observation)
            else:
                ds_valid = filter_dataset_by_data_id(ds_valid, split="validation")
                vaild_dataset = ImageTextDataset(ds_valid, img_processor=img_processor, tokenizer=tokenizer, split="validation", target_obs=target_observation)
        valid_dataloader = DataLoader(vaild_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer, target_observation), batch_size=eval_bsz, drop_last=False)

    if ds_test:
        with ACCELERATOR.main_process_first():
            if use_debug_subset:
                test_dataset = ImageTextDataset(ds_test.select(range(len(ds_test) - 3, len(ds_test))), img_processor=img_processor, tokenizer=tokenizer, split="test", target_obs=target_observation)
            else:
                ds_test = filter_dataset_by_data_id(ds_test, split="test")
                test_dataset = ImageTextDataset(ds_test, img_processor=img_processor, tokenizer=tokenizer, split="test", target_obs=target_observation)
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer, target_observation), batch_size=eval_bsz, drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader


def merge_dataset(img_dataset, graph_dataset):
    imgId_2_graphRowIdx = {}
    for graph_row_idx, doc_key in enumerate(graph_dataset["doc_key"]):
        _, img_id, _ = doc_key.split("#")  # doc_key = test#2250#findings
        imgId_2_graphRowIdx[int(img_id)] = int(graph_row_idx)

    # 如果传入的是 select 后的 img_ds 数据集，那么 img_id 与 img_row_idx 不一定是一一对应的
    # data_key: test#89
    imgId_2_imgRowIdx = {}
    for img_row_idx, img_data_key in enumerate(img_dataset["data_key"]):
        _, img_id = img_data_key.split("#")  # data_key = test#89
        imgId_2_imgRowIdx[int(img_id)] = int(img_row_idx)

    # 以数量较少的数据集为基准
    img_ids_in_img_ds = set(imgId_2_imgRowIdx.keys())
    img_ids_in_graph_ds = set(imgId_2_graphRowIdx.keys())
    intersection_ids = img_ids_in_img_ds.intersection(img_ids_in_graph_ds)

    # 按照 img_id 的顺序，将 img_ds 的数据拼接到 graph_ds 的数据中
    filtered_img_ds = img_dataset.select([imgId_2_imgRowIdx[img_id] for img_id in intersection_ids])
    filtered_graph_ds = graph_dataset.select([imgId_2_graphRowIdx[img_id] for img_id in intersection_ids])
    merged_ds = concatenate_datasets([filtered_img_ds, filtered_graph_ds], axis=1)
    return merged_ds


def load_dataset(ds_img_path, ds_graph_path, target_section):
    # ds_img 是 image + report 数据集，report 包含 findings和impression
    # ds_graph 是纯文本数据集，是对应特定 target_section，即findings 或impression

    # TODO 在linux上debug时，加载原始图像数据集
    # ds_img = load_image_datasets(data_paths=CONFIG["data_path"])
    # for data_split in ["train", "validation", "test"]:
    #     img_dataset = ds_img[data_split]
    #     img_dataset = img_dataset.add_column("data_key", [f"{data_split}#{idx}" for idx in range(len(img_dataset))])
    #     ds_img[data_split] = img_dataset

    ds_img = load_from_disk(ds_img_path)
    LOGGER.info("Loaded pre_processed image dataset from: \n%s \n%s", ds_img_path, ds_img)
    ds_graph_path = os.path.join(ds_graph_path, f"interpret_disease_{target_section}")
    ds_graph = load_from_disk(ds_graph_path)
    LOGGER.info("Loaded pre_processed graph dataset from: \n%s \n%s", ds_graph_path, ds_graph)

    ds_dict = {}
    for split in ["train", "validation", "test"]:
        ds_dict[split] = merge_dataset(img_dataset=ds_img[split], graph_dataset=ds_graph[split])

    ds_final = DatasetDict(ds_dict)
    LOGGER.debug("Merged image-graph dataset: \n%s", ds_final)
    return ds_final


def post_init_model_and_tokenizer(model, tokenizer):
    if len(tokenizer) >= model.config.decoder.vocab_size:
        LOGGER.info("Decoder token_embedding [%d] and tokenizer [%d] size mismatch", model.config.decoder.vocab_size, len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=True)
        LOGGER.info("Decoder token_embedding resized to [%d] (pad_to_multiple_of=8)", model.config.decoder.vocab_size)

    # 检查 tokenizer 的特殊 token 是否存在，用于判断 eval 是否加载了正确的 tokenizer
    for spec_tok in GLOBAL_VARS.additional_special_tokens:
        assert spec_tok in tokenizer.special_tokens_map["additional_special_tokens"], f"Missing special token: {spec_tok} in tokenizer, expect: {GLOBAL_VARS.additional_special_tokens}, got: {tokenizer.special_tokens_map['additional_special_tokens']} in tokenizer."

    eot_token_id = tokenizer.encode(GLOBAL_VARS.eot_token, add_special_tokens=False)
    assert len(eot_token_id) == 1, f"Expected single token for eot_token: {GLOBAL_VARS.eot_token}, got: {eot_token_id}"
    GLOBAL_VARS.eot_token_id = eot_token_id[0]

    # 用于在 input_ids 中查找需要替换的图像占位符 <|image_token|>
    if not hasattr(model.config, "image_token_index"):
        model.config.image_token_index = tokenizer.convert_tokens_to_ids("<|image_token|>")
    if not hasattr(model.config, "label_token_index"):
        model.config.label_token_index = tokenizer.convert_tokens_to_ids("<|label_token|>")

    # 计算 vision model 输出的图像特征的数量，该数量等于我们应该在 input_ids 中插入 <|image_token|> 的数量
    img_size = model.config.encoder.image_size
    dummy_img = torch.zeros((1, 3, img_size, img_size))
    num_image_tokens = model.encoder(dummy_img).last_hidden_state.size(1)
    GLOBAL_VARS.num_image_tokens = num_image_tokens


def init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefixs=["v2l_projector"]):
    # 重新初始化模型，在后续再单独加载预训练的 img_projector，避免OOM的问题（不知道为什么会出现这个问题）
    base_model = init_model(vision_model_path, language_model_path, model_base_cfg)
    # Load only img_projector state_dict to the base model
    # 如果直接加载整个 pre_trained 模型，会导致训练时OOM，但只加载 img_projector 到base_model则不会
    # 有点担心的是 decoder embedding 重新初始化，是否会导致其与 img_projector 的不匹配
    target_state_dict = load_module_state_dict_from(model_path=pretain_model_path, target_module_prefixs=target_module_prefixs)
    model = load_state_dict_to_model(base_model=base_model, target_state_dict=target_state_dict)
    LOGGER.info("Initialized model with pretrained weights: %s", target_module_prefixs)
    return model


def init_model(vision_model_path, language_model_path, model_base_cfg, classification_only=False):
    # cls(encoder=encoder, decoder=decoder, config=config)，
    # 使用 from_encoder_decoder_pretrained 方法初始化模型时，额外的方法参数会被加入到init的config里，不需要额外定义
    model = Vision2LanguageModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=vision_model_path,
        decoder_pretrained_model_name_or_path=language_model_path,
        num_observations=len(GLOBAL_VARS.obs_id2name_dict),
        num_cls_labels=len(GLOBAL_VARS.obs_cls2id_dict),
        classification_only=classification_only,
    )
    LOGGER.info("Initialized vision language mode from: \n%s\n%s", vision_model_path, language_model_path)
    return model


def init_processor(vision_model_path, language_model_path, model_base_cfg):
    img_processor = AutoImageProcessor.from_pretrained(vision_model_path, use_fast=True)
    LOGGER.info("Loaded ImageProcessor from: %s", vision_model_path)

    tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=True)
    LOGGER.info("Loaded Tokenizer from: %s", language_model_path)

    # Add special tokens
    LOGGER.info("Adding special tokens")
    bos_token = tokenizer.bos_token if tokenizer.bos_token else "<BOS>"
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<EOS>"
    pad_token = tokenizer.pad_token if tokenizer.pad_token else "<PAD>"
    special_tokens_dict = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "additional_special_tokens": GLOBAL_VARS.additional_special_tokens,
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


def apply_peft_to_model(model):
    # https://huggingface.co/docs/peft/developer_guides/troubleshooting#bad-results-from-a-loaded-peft-model

    # The names of the modules to apply the adapter to.
    # Also check TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING: https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py
    # When the lora layers are applied to embedding layers, the corresponding base model embedding layers are also saved.
    target_modules = ["query", "value", "fc1", "fc2", "embed_tokens", "q_proj", "v_proj", "lm_head"]  # 需要注入 LoRA 的模块。
    # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    # e.g. Transformers adds a randomly initialized classification head on top of the model. If you do not add this layer to modules_to_save, the classification head won’t be saved. The next time you load the model, you’ll get a different randomly initialized classification head, resulting in completely different results.
    modules_to_save = ["v2l_projector", "obs_classifier"]  # 没注入LoRA 但又需要训练和保存的模块。添加模块后，peft会包装一个一模一样的模块，并将requires_grad 会被设置为 True。原模块不变。
    lora_config = LoraConfig(
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        init_lora_weights="pissa_niter_16",  # 不确定时：True 或 pissa 是最保险的起点；你想训练少轮就见效果：corda；做正式训练/部署，追求SOTA：eva（但初始化时要花点功夫）；想节省时间资源：pissa_niter_16；LoRA + 量化一起用：pissa / loftq；
        # task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    LOGGER.info("Applied PEFT model with config: %s", lora_config)
    log_trainable_parameters(peft_model)
    return peft_model


def log_trainable_parameters(peft_model):
    peft_model.print_trainable_parameters()
    trainable_params, all_param = peft_model.get_nb_trainable_parameters()
    LOGGER.info("PEFT model trainable params=%d, all params=%d | trainable=[%.4f]%%", trainable_params, all_param, 100 * trainable_params / all_param)


def global_init_accelerator(model, fsdp_no_shard=False, **kwargs):
    global ACCELERATOR, DEVICE, LOGGER

    grad_accum_steps = kwargs["grad_accum_steps"]
    mixed_precision = kwargs["mixed_precision"]

    if isinstance(model, PeftModel):
        # TODO 如何在不使用 ignore_modules 的情况下，让modules能够顺利的从flattened state_dict中恢复为unflattened的状态，然后保存
        ignored_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Embedding, Dinov2Model, LlamaRMSNorm, LlamaRotaryEmbedding, lora.Embedding, lora.Linear)):
                ignored_modules.append(module)
        transformer_cls_names_to_wrap = ["LlamaDecoderLayer", "Dinov2Layer", "VisionLanguageProjector", "VisionClassifier"]
        # sharding_strategy = "NO_SHARD", state_dict_type = "FULL_STATE_DICT"，用这套配置可以解决加载checkpoint后，在optimizer.step()时的报错 RuntimeError: torch function 'lerp_', with args: (ShardedTensor(ShardedTensorMetadata(sha ... and kwargs: None not supported for ShardedTensor!
        sharding_strategy = "NO_SHARD"
        state_dict_type = "FULL_STATE_DICT"

        # 关于 FSDP1 -> FSDP2 https://huggingface.co/docs/accelerate/main/en/concept_guides/fsdp1_vs_fsdp2
    else:
        ignored_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Embedding, Dinov2Model, LlamaRMSNorm, LlamaRotaryEmbedding)):
                ignored_modules.append(module)
        transformer_cls_names_to_wrap = ["LlamaDecoderLayer", "Dinov2Layer", "VisionLanguageProjector", "VisionClassifier"]
        sharding_strategy = "NO_SHARD"
        state_dict_type = "FULL_STATE_DICT"

    # 如果不使用这段代码，我们在eval时会遇到 RuntimeError: mat2 must be a matrix, got 1-D tensor
    # 可能是因为 PEFT modules_to_save 的部分与 FSDP 不兼容。目前的临时解决办法是改成 DDP
    if fsdp_no_shard:
        sharding_strategy = "NO_SHARD"
        state_dict_type = "FULL_STATE_DICT"

    # 关于 FSDP1 -> FSDP2 https://huggingface.co/docs/accelerate/main/en/concept_guides/fsdp1_vs_fsdp2
    fsdp_plugin = FullyShardedDataParallelPlugin(
        # mixed_precision_policy=mixed_precision,
        sharding_strategy=sharding_strategy,  # FULL_SHARD=ZeRO3, SHARD_GRAD_OP=ZeRO2, NO_SHARD (DDP), HYBRID_SHARD, HYBRID_SHARD_ZERO2,
        backward_prefetch="BACKWARD_PRE",  # [1] BACKWARD_PRE 中等显存/通用场景, [2] BACKWARD_POST 显存充足/极致优化, [3] NO_PREFETCH 显存紧张
        auto_wrap_policy="transformer_based_wrap",  # transformer_based_wrap, size_based_wrap, or no_wrap
        transformer_cls_names_to_wrap=transformer_cls_names_to_wrap,
        ignored_modules=ignored_modules,
        # transformer_layer_cls=int(1e6),
        state_dict_type=state_dict_type,  # [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
        use_orig_params=True,  # 设置为True才能手动调整params lr, requires_grad 等
        cpu_offload=False,  # cpu_offload=True与FULL_SHARD组合可最大化显存节省，但通信开销最高。能节省5G的peak mem，但100iter从3s下降到5s
        activation_checkpointing=False,  # A technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage. Will cause RuntimeError: The expanded size of the tensor (2896) must match the existing size (1448) at non-singleton dimension 3.  Target sizes: [2, 32, 1448, 2896].  Tensor sizes: [2, 1, 1448, 1448]
        # cpu_ram_efficient_loading=True, #If True, only the first process loads the pretrained model checkoint while all other processes have empty weights. Only applicable for Transformers. When using this, sync_module_states needs to be True.
        # sync_module_states=True,
    )

    # https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin
    # 如果OOM，可以尝试设置 sync_each_batch=True，但是会导致训练速度变慢
    # adjust_scheduler=False，我们在train方法中手动计算 scheduler 在使用梯度累计后的 step
    ga_plugin = GradientAccumulationPlugin(
        num_steps=grad_accum_steps,
        adjust_scheduler=False,
        sync_with_dataloader=True,
        sync_each_batch=False,
    )

    dataloader_cfg = DataLoaderConfiguration(use_seedable_sampler=True)

    ACCELERATOR = Accelerator(
        mixed_precision=mixed_precision,
        dataloader_config=dataloader_cfg,
        gradient_accumulation_plugin=ga_plugin,
        fsdp_plugin=fsdp_plugin,
    )
    DEVICE = ACCELERATOR.device

    if ACCELERATOR.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    LOGGER = MultiProcessAdapter(LOGGER, {})  # must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility.
    LOGGER.info("Available cuda: %d", torch.cuda.device_count())
    LOGGER.info("Accelerator state: %s", ACCELERATOR.state, main_process_only=False)
    if hasattr(ACCELERATOR.state, "fsdp_plugin"):
        LOGGER.info("FSDP sharding_strategy: %s", ACCELERATOR.state.fsdp_plugin.sharding_strategy)
        LOGGER.info("FSDP state_dict_type: %s", ACCELERATOR.state.fsdp_plugin.state_dict_type)
    LOGGER.info([i for i in CONFIG.items() if i[0][0] != "_"])


def global_init_logger(log_level=logging.DEBUG, base_log_level=logging.WARNING, fsdp_log_level=logging.ERROR):
    global LOGGER
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=base_log_level)
    logging.getLogger("torch.distributed.fsdp").setLevel(fsdp_log_level)

    log_file_mode = "w"
    if CONFIG["resume_from_checkpoint"]:
        log_file_mode = "a"

    curr_file_name = os.path.basename(os.path.abspath(__file__))
    log_file_path = os.path.join(CONFIG["output_dir"]["result"], f"{curr_file_name}_{CONFIG['run_mode']}.log")

    file_handler = logging.FileHandler(log_file_path, log_file_mode)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler.setFormatter(stream_formatter)

    LOGGER = logging.getLogger(curr_file_name)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(log_level)  # This logger's level


def global_init_proj_config():
    global CONFIG, GLOBAL_VARS

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--config_file", type=str, help=".yaml file path")

    parser.add_argument("--output_name", type=str)
    parser.add_argument("--jobid", type=int)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=None)

    parser.add_argument("--run_mode", type=str, default=None, help="Choose from [preprocess, pretrain, eval_pretrained, finetune, eval_finetuned]")
    parser.add_argument("--mlflow_port", type=str, default=None)

    parser.add_argument("--target_observation", type=str, default=None, help="A string representation of a list, e.g. \"['effusion', 'pneumothorax']\"")
    parser.add_argument("--classification_only", action="store_true", default=None)

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_per_steps", type=int, default=None)

    parser.add_argument("--use_pretrained", action="store_true", default=None)
    parser.add_argument("--pretain_model_path", type=str, default=None)

    args = parser.parse_args()

    if args.from_bash:
        file_path = args.config_file
    else:
        file_path = "/home/yuxiang/liao/workspace/arrg_img2text/config/7_1_disease_fea.yaml"

    with open(file_path, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)

    if args.from_bash:
        CONFIG["output_name"] = args.output_name
        CONFIG["jobid"] = args.jobid

        if args.run_mode:
            CONFIG["run_mode"] = args.run_mode

        if args.resume_from_checkpoint:
            CONFIG["resume_from_checkpoint"] = args.resume_from_checkpoint

        if args.mlflow_port:
            CONFIG["mlflow_port"] = args.mlflow_port

        if args.classification_only:
            CONFIG["classification_only"] = True

        run_mode = None
        if "finetune" in CONFIG["run_mode"]:
            run_mode = "finetune"
            if args.use_pretrained:
                CONFIG[run_mode]["use_pretrained"] = args.use_pretrained
            if args.pretain_model_path:
                CONFIG[run_mode]["pretain_model_path"] = args.pretain_model_path
            if args.target_observation:
                CONFIG[run_mode]["target_observation"] = ast.literal_eval(args.target_observation)

        elif "pretrain" in CONFIG["run_mode"]:
            run_mode = "pretrain"
            if args.eval_per_steps:
                CONFIG[run_mode]["eval_per_steps"] = args.eval_per_steps

        if run_mode:
            if args.num_epochs:
                CONFIG[run_mode]["num_epochs"] = args.num_epochs
            if args.batch_size:
                CONFIG[run_mode]["batch_size"] = args.batch_size
            if args.grad_accum_steps:
                CONFIG[run_mode]["grad_accum_steps"] = args.grad_accum_steps
            if args.lr:
                CONFIG[run_mode]["lr"] = args.lr

    else:
        CONFIG["jobid"] = "00000"

    CONFIG["mlflow_url"] = f"{CONFIG['mlflow_url']}:{CONFIG['mlflow_port']}"

    GLOBAL_VARS.obs_name2id_dict = {obs_name: idx for idx, obs_name in enumerate(CONFIG["observation_map"])}
    GLOBAL_VARS.obs_id2name_dict = {idx: obs_name for obs_name, idx in GLOBAL_VARS.obs_name2id_dict.items()}
    GLOBAL_VARS.obs_cls2id_dict = {cls_name: idx for idx, cls_name in enumerate(CONFIG["obs_classification_map"])}
    GLOBAL_VARS.obs_id2cls_dict = {idx: cls_name for cls_name, idx in GLOBAL_VARS.obs_cls2id_dict.items()}

    output_dirs = CONFIG["output_dir"]
    output_dirs["result"] = os.path.join(output_dirs["result"], CONFIG["output_name"])
    output_dirs["model"] = os.path.join(output_dirs["model"], CONFIG["output_name"])
    output_dirs["checkpoint"] = os.path.join(output_dirs["checkpoint"], CONFIG["output_name"])
    # output_dirs["log"] = os.path.join(output_dirs["log"], CONFIG["output_name"])
    os.makedirs(output_dirs["result"], exist_ok=True)
    os.makedirs(output_dirs["model"], exist_ok=True)
    os.makedirs(output_dirs["checkpoint"], exist_ok=True)
    # os.makedirs(output_dirs["log"], exist_ok=True)

    for cfg_key in ["pretrain", "finetune"]:
        for key in CONFIG[cfg_key]:
            if "lr" in key:
                CONFIG[cfg_key][key] = float(CONFIG[cfg_key][key])


#############################################
def pretrain_model(model_base_cfg, train_cfg):
    """pre-train the image_adaptor, freeze encoder and decoder"""
    model_base_cfg = CONFIG["model"]
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"])

    model = init_model(vision_model_path, language_model_path, model_base_cfg, classification_only=train_cfg["classification_only"])
    img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)

    post_init_model_and_tokenizer(model, tokenizer)
    global_init_accelerator(model, **train_cfg)
    check_memory()
    model.to(DEVICE)
    train_dataloader, valid_dataloader, _ = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_train=ds_final["train"], train_bsz=train_cfg["batch_size"], ds_valid=ds_final["validation"], eval_bsz=train_cfg["eval_batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()

    start = time.time()
    train(model, train_dataloader, train_cfg=train_cfg, valid_dataloader=valid_dataloader)
    end = time.time()
    LOGGER.info("Total training time: %s", seconds_to_time_str(end - start))

    save_processors(img_processor=img_processor, tokenizer=tokenizer, output_dir=CONFIG["output_dir"]["model"])


def eval_pretrained_model(train_cfg):
    pretain_model_path = CONFIG["output_dir"]["model"]
    model_base_cfg = CONFIG["model"]
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"])

    model = load_model(pretain_model_path)
    img_processor, tokenizer = load_processor(pretain_model_path)
    post_init_model_and_tokenizer(model, tokenizer)
    global_init_accelerator(model, **train_cfg)
    # TODO
    _, valid_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], eval_bsz=train_cfg["eval_batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    # train_dataloader, _, _ = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_train=ds_final["train"], train_bsz=train_cfg["batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()
    model.to(DEVICE)
    model, valid_dataloader, test_dataloader = ACCELERATOR.prepare(model, valid_dataloader, test_dataloader)
    # model, train_dataloader = ACCELERATOR.prepare(model, train_dataloader)
    check_memory()

    start = time.time()
    start2 = start
    if train_cfg["eval_valid_split"]:
        evaluate(model, valid_dataloader, **train_cfg)
        start2 = time.time()
        LOGGER.info("Valid time: %s", seconds_to_time_str(start2 - start))

    evaluate(model, test_dataloader, **train_cfg)
    # evaluate(model, train_dataloader, **train_cfg)
    end = time.time()
    LOGGER.info("Test time: %s", seconds_to_time_str(end - start2))
    LOGGER.info("Total evaluation time: %s", seconds_to_time_str(end - start))


def finetune_model(train_cfg):
    """use peft with fsdp to train image projector and decoder"""
    model_base_cfg = CONFIG["model"]
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]
    pretain_model_path = train_cfg["pretain_model_path"]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"])

    if train_cfg["use_pretrained"]:
        if CONFIG["pretrain"]["classification_only"]:
            target_module_prefixs = ["obs_classifier"]
        else:
            target_module_prefixs = ["v2l_projector", "obs_classifier"]
        model = init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefixs=target_module_prefixs)
        img_processor, tokenizer = load_processor(pretain_model_path)
    else:
        model = init_model(vision_model_path, language_model_path, model_base_cfg)
        img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)

    post_init_model_and_tokenizer(model, tokenizer)
    model = apply_peft_to_model(model)
    global_init_accelerator(model, **train_cfg)
    check_memory()
    model.to(DEVICE)

    train_dataloader, _, _ = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_train=ds_final["train"], train_bsz=train_cfg["batch_size"], use_debug_subset=CONFIG["use_debug_subset"], target_observation=train_cfg["target_observation"])
    # _, valid_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()

    start = time.time()
    train(model, train_dataloader, train_cfg=train_cfg)
    # evaluate(model, test_dataloader, **train_cfg)
    end = time.time()
    LOGGER.info("Total training time: %s", seconds_to_time_str(end - start))

    save_processors(img_processor=img_processor, tokenizer=tokenizer, output_dir=CONFIG["output_dir"]["model"])


def eval_finetuned_model(train_cfg):
    """eval the peft model with FSDP set to NO_SHARD"""
    model_base_cfg = CONFIG["model"]
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]
    pretain_model_path = train_cfg["pretain_model_path"]
    peft_model_path = CONFIG["output_dir"]["model"]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"])

    if train_cfg["use_pretrained"]:
        if CONFIG["pretrain"]["classification_only"]:
            target_module_prefixs = ["obs_classifier"]
        else:
            target_module_prefixs = ["v2l_projector", "obs_classifier"]
        model = init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefixs=target_module_prefixs)
        img_processor, tokenizer = load_processor(pretain_model_path)
    else:
        model = init_model(vision_model_path, language_model_path, model_base_cfg)
        # 理论上，加载processor和init_processor的效果是一样的，tokenizer
        # img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)
        img_processor, tokenizer = load_processor(peft_model_path)

    post_init_model_and_tokenizer(model, tokenizer)
    model = load_peft_model(base_model=model, peft_model_path=peft_model_path)
    global_init_accelerator(model, fsdp_no_shard=True, **train_cfg)
    model.to(DEVICE)

    _, valid_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], eval_bsz=train_cfg["eval_batch_size"], use_debug_subset=CONFIG["use_debug_subset"], target_observation=train_cfg["target_observation"])
    check_memory()

    model, valid_dataloader, test_dataloader = ACCELERATOR.prepare(model, valid_dataloader, test_dataloader)
    check_memory()

    start = time.time()
    start2 = start
    if train_cfg["eval_valid_split"]:
        evaluate(model, valid_dataloader, **train_cfg)
        start2 = time.time()
        LOGGER.info("Valid time: %s", seconds_to_time_str(start2 - start))

    evaluate(model, test_dataloader, **train_cfg)
    end = time.time()
    LOGGER.info("Test time: %s", seconds_to_time_str(end - start2))
    LOGGER.info("Total evaluation time: %s", seconds_to_time_str(end - start))


if __name__ == "__main__":
    global_init_proj_config()
    global_init_logger(log_level=logging.DEBUG, base_log_level=logging.WARNING, fsdp_log_level=logging.WARNING)
    LOGGER.info(CONFIG)

    if torch.cuda.is_available():
        LOGGER.info("[rank %s] CUDA_VISIBLE_DEVICES = %s", os.environ.get("RANK"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        LOGGER.info("[rank %s] Current device: %s", os.environ.get("RANK"), torch.cuda.current_device())
        LOGGER.info("[rank %s] Memory allocated: %s GB", os.environ.get("RANK"), torch.cuda.memory_allocated() / 1024**3)

    start0 = time.time()

    if CONFIG["run_mode"] == "preprocess":
        preprocess_dataset()
    elif CONFIG["run_mode"] == "pretrain":
        # import cProfile
        # cProfile.run("main()", filename=os.path.join(CONFIG["output_dir"]["result"], "time_statistic.cprofile"))
        pretrain_model(model_base_cfg=CONFIG["model"], train_cfg=CONFIG["pretrain"])
    elif CONFIG["run_mode"] == "eval_pretrained":
        eval_pretrained_model(train_cfg=CONFIG["pretrain"])
    elif CONFIG["run_mode"] == "finetune":
        finetune_model(train_cfg=CONFIG["finetune"])
    elif CONFIG["run_mode"] == "eval_finetuned":
        eval_finetuned_model(train_cfg=CONFIG["finetune"])

    if torch.distributed.is_initialized() and ACCELERATOR and ACCELERATOR.is_main_process:
        torch.distributed.destroy_process_group()

    end0 = time.time()
    LOGGER.info("Total time: %s ", seconds_to_time_str(end0 - start0))
