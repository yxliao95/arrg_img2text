#############################################
# 基于5_3修改
# 从使用完整的报告的图，改为使用单个疾病，目前是effu
# 对于独立疾病，如果radlex没有对应的node，那么该split sent就不会被引入 （但被radlex 关联的 split sent还是会被引入）。
# 在这种条件下，引入的数据集大小会更小
#############################################
import argparse
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
from collections import Counter
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
TENSORBOARD = None
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
    additional_special_tokens = ["<|image_token|>", "<|image_start|>", "<|image_end|>"]
    ent_type_tokens = ["<Anatomy>", "<Observation-Present>", "<Observation-Absent>", "<Observation-Uncertain>", "<Location-Attribute>"]
    rel_type_tokens = ["<modify>", "<located_at>", "<suggestive_of>", "<part_of>"]
    graph_tokens = ["<normal>", "<abnormal>", "<absent>", "<uncertain>", "<located_at>", "<suggestive_of>", "<ENT>", "<REL>"]

    eot_token = "<|eot_id|>"
    eot_token_id = None


GLOBAL_VARS = GlobalVariables()


#############################################
# Model Classes
#############################################
@dataclass
class Vision2LanguageOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


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


class Vision2LanguageModel(VisionEncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):

        super().__init__(config=config, encoder=encoder, decoder=decoder)
        self.config.encoder_hidden_size = self.encoder.config.hidden_size
        self.config.decoder_hidden_size = self.decoder.config.hidden_size

        # replace enc_to_dec_proj with VisionLanguageProjector
        self.v2l_projector = VisionLanguageProjector(self.config)
        if hasattr(self, "enc_to_dec_proj"):
            del self.enc_to_dec_proj  # 移除投影层

    def _inject_image_features(self, input_ids, decoder_input_ids, image_features):
        # image_indices_map 是一个嵌套list，每个样本对应一个list，list中的元素是图像在 last_hidden_state 中的索引
        # e.g. [[0], [1], [2, 3], ...]

        # replace img features with the <|image_token|> placeholder token in the input text
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(decoder_input_ids).to(decoder_input_ids.device)

        # 保证所有 image_features 都能够被复制到 decoder_input_ids 中
        assert special_image_mask.sum() == image_features.numel(), f"special_image_mask.sum()={special_image_mask.sum()}, image_features.numel()={image_features.numel()}, should be equal to guarantee that all image features are copied to decoder_input_ids"

        image_features = image_features.to(decoder_input_ids.device, decoder_input_ids.dtype)
        decoder_input_ids = decoder_input_ids.masked_scatter(special_image_mask, image_features)

        return decoder_input_ids

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
        **kwargs,
    ) -> Union[Tuple, Vision2LanguageOutputWithPast]:
        """Additional args:
        `decoder_inputs_embeds`: should represent the text embeddings with image features injected.
        `encoder_outputs`: in inference statge, we encode `pixel_values` and get `encoder_outputs` outside this forward method. This is because the `pixel_values` and `decoder_input_ids` have different batch sizes, which cause error in generate().

        If `output_loss` is True, by default we use `decoder_input_ids` as `labels`.
        And the `decoder_assistant_masks` should be provided to compute the loss.
        `decoder_assistant_masks` is provided by `tokenizer.apply_chat_template`.
        `decoder_assistant_masks` is a tensor with the same shape as decoder_input_ids, and the value is 0 or 1. 0: system/user tokens, 1: assistant tokens, which is the tokens that need to be generated.
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
            # project image features
            image_features = self.v2l_projector(image_features)
            # inject image features into text embeddings
            decoder_inputs_embeds = self._inject_image_features(decoder_input_ids, decoder_inputs_embeds, image_features)

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
        loss = None
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
            loss = ce_loss_fct(active_shift_logits, active_shift_labels)

        return Vision2LanguageOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
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
        return result

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


class ImageTextDataset(Dataset):
    def __init__(self, hf_dataset, img_processor, tokenizer, split):
        # column_names: ['source', 'images_path', 'images', 'section_text', 'doc_key', 'split_sents', 'split_sent_toks', 'sent_idx_split_idx', 'radlex', 'cxrgraph_ent', 'cxrgraph_attr', 'cxrgraph_rel']
        self.split = split
        self.target_section = CONFIG["target_section"]
        self.src_path = os.path.dirname(hf_dataset.cache_files[0]["filename"]) if hf_dataset.cache_files else ""
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.samples = self._process_text(hf_dataset)

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

    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch_data, img_processor, tokenizer):
    # 这个方法只处理图像，
    # 对话数据由于在训练和推理时不同，所以分开进行处理

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

    return {
        "batch_data": batch_data,
        "pixel_values": piexl_values_tensor.to(DEVICE),  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map,  # [[0], [1], [2, 3], ...]
    }


def get_inputs_for_training(tokenizer, batch_data, pixel_values, image_indices_map):
    # training中，多轮对话可以合并为一个input
    target_section = CONFIG["target_section"]

    gold_graph_list, gold_text_list = get_gold_labels(batch_data)

    conversations = []
    num_image_tokens = GLOBAL_VARS.num_image_tokens
    for idx, item in enumerate(batch_data):
        num_images = len(image_indices_map[idx])

        assistant_output_graph_str = gold_graph_list[idx]
        assistaant_output_text = gold_text_list[idx]

        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiology assistant tasked with interpreting a chest X-ray study."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "num_images": num_images, "num_image_tokens": num_image_tokens},
                        {"type": "text", "text": "Here's a set of chest X-ray images. Your task is to analyze these input radiological image and output your observations in a structured format. Classify each observation as one of the following categories: [<normal>, <abnormal>, <uncertain>, or <absent>]. When applicable, also include relational information using <suggestive_of> and <located_at>.\nPlease follow this output format exactly:\n\n<normal>:\n<abnormal>:\n<uncertain>:\n<absent>: \n\nNow, analyze the input image and report the findings in this format."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_output_graph_str}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "According to the given X-ray images and the structured report you just generated, please output a detailed narrative report."},
                    ],
                },
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
    }


def get_inputs_for_inference(tokenizer, batch_data, pixel_values, image_indices_map, assistant_responses, conversations_history):
    # inference 中，多轮对话需要循环处理
    # 第一轮 model_response = {"graph": [], "seq": []}
    # 第二轮 model_response = {"graph": ["graph str", "..."], "seq": []}
    target_section = CONFIG["target_section"]

    conversations = []
    num_image_tokens = GLOBAL_VARS.num_image_tokens
    for idx, item in enumerate(batch_data):
        num_images = len(image_indices_map[idx])

        if assistant_responses is None:
            # 第一轮的用户输入
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiology assistant tasked with interpreting a chest X-ray study."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "num_images": num_images, "num_image_tokens": num_image_tokens},
                        {"type": "text", "text": "Here's a set of chest X-ray images. Your task is to analyze these input radiological image and output your observations in a structured format. Classify each observation as one of the following categories: [<normal>, <abnormal>, <uncertain>, or <absent>]. When applicable, also include relational information using <suggestive_of> and <located_at>.\nPlease follow this output format exactly:\n\n<normal>:\n<abnormal>:\n<uncertain>:\n<absent>: \n\nNow, analyze the input image and report the findings in this format."},
                    ],
                },
            ]
        else:
            # 历史输入 + 第一轮的模型输出 + 第二轮的用户输入
            conversation = conversations_history[idx]
            assistant_output_graph_str = assistant_responses[idx]
            conversation.extend(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_output_graph_str}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"According to the given X-ray images and the structured report you just generated, please output a detailed narrative report."},
                        ],
                    },
                ]
            )

        conversations.append(conversation)

    # See descriptions for assistant_tokens_mask
    # Assistant tokens are the tokens that need to be generated, we use these tokens to compute the loss
    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask
    tokenizer_kwargs = {"pad_to_multiple_of": 8}

    add_generation_prompt = True
    return_assistant_tokens_mask = False
    tokenizer_kwargs["padding_side"] = "left"

    input_text_tensor_dict = tokenizer.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, tokenize=True, padding=True, return_dict=True, return_tensors="pt", tokenizer_kwargs=tokenizer_kwargs, return_assistant_tokens_mask=return_assistant_tokens_mask)

    gold_graph_list, gold_text_list = get_gold_labels(batch_data)

    return {
        "batch_data": batch_data,
        "pixel_values": pixel_values,  # torch.Size([bsz < x < 2*bsz, 3, 224, 224])
        "image_indices_map": image_indices_map,  # [[0], [1], [2, 3], ...]
        "decoder_input_ids": input_text_tensor_dict.input_ids.to(DEVICE),
        "decoder_attention_mask": input_text_tensor_dict.attention_mask.to(DEVICE),
        "data_id_list": [i["data_key"] for i in batch_data],
        "gold_graph_list": gold_graph_list,
        "gold_text_list": gold_text_list,
        "conversations_history": conversations,
    }


def get_gold_labels(batch_data):
    gold_graph_list = []
    gold_text_list = []
    for idx, item in enumerate(batch_data):
        graph_str_dict = {"normal": [], "abnormal": [], "uncertain": [], "absent": []}
        for graph_repr in item["graph_reprs3"]:
            # e.g. graph_repr = {'abnormal': ['pulmonary vasculature', 'severe pulmonary edema'], 'absent': [], 'normal': [], 'uncertain': []}
            # {'abnormal': ['<ENT> hazy opacity <REL> <located_at> <ENT> overlies <REL> <located_at> <ENT> lungs'], 'absent': [], 'normal': [], 'uncertain': []}
            for key in ["normal", "abnormal", "uncertain", "absent"]:
                if not graph_repr[key]:
                    continue
                graph_str_dict[key].extend([graph_str.lower() for graph_str in graph_repr[key]])

        normal_str = ", ".join(graph_str_dict["normal"]) if graph_str_dict["normal"] else "None"
        abnormal_str = ", ".join(graph_str_dict["abnormal"]) if graph_str_dict["abnormal"] else "None"
        uncertain_str = ", ".join(graph_str_dict["uncertain"]) if graph_str_dict["uncertain"] else "None"
        absent_str = ", ".join(graph_str_dict["absent"]) if graph_str_dict["absent"] else "None"

        assistant_output_graph_str = f"<normal>: {normal_str}\n<abnormal>: {abnormal_str}\n<uncertain>: {uncertain_str}\n<absent>: {absent_str}"
        assistaant_output_text = item["split_sents"]

        gold_graph_list.append(assistant_output_graph_str)
        gold_text_list.append(assistaant_output_text)

    return gold_graph_list, gold_text_list


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
    dev_best: dict = field(default_factory=lambda: {"text_score": 0.0, "at_epoch": 0, "at_iter": 0, "check_at": ""})

    batch_loss: int = field(default=0)
    batch_trained_examples: int = field(default=0)

    run_id: dict = field(default="")

    grad_accum_eval_mark: int = field(default=0)
    train_print_loss_mark: int = field(default=0)

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


def train(model, train_dataloader, train_cfg):
    global MLFLOW_TRACKER, STATUS_INFO

    # hyperparameters
    model_params = list(model.named_parameters())
    optimizer_grouped_parameters = prepare_optimizer_grouped_parameters(model_params, train_cfg)
    # LOGGER.debug("Model trainable params:\n%s", "\n".join([n for n, p in model.named_parameters() if p.requires_grad]))

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_num_steps = len(train_dataloader) // train_cfg["grad_accum_steps"] * train_cfg["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * train_cfg["warmup_proportion"]), num_training_steps=total_num_steps)

    # 1. Prepare for multi GPUs. All prepared and registered objs will be checkpointed automatically
    model, train_dataloader, optimizer, scheduler = ACCELERATOR.prepare(model, train_dataloader, optimizer, scheduler)
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
                    out = model.forward(output_loss=True, **batch_inputs_dict)
                    loss = out.loss

                ACCELERATOR.backward(loss)
                if train_cfg["clip_grad_norm"] > 0:
                    ACCELERATOR.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                check_memory(show_only_if_peak=True)
                log_and_update_status(curr_epoch=curr_epoch, curr_iter=curr_iter, loss=loss.item(), bsz=batch_inputs_dict["decoder_input_ids"].size(0), lr=scheduler.get_last_lr()[0], train_cfg=train_cfg)

                # we dont do validation, as it cost too much time
                check_and_save_checkpoint(max_num_iters_per_epoch=len(train_dataloader), train_cfg=train_cfg)

        end = time.time()
        LOGGER.info("Batch training time: %s ", seconds_to_time_str(end - start))

    save_model(model, CONFIG["output_dir"]["model"])
    MLFLOW_TRACKER.finish()


def prepare_optimizer_grouped_parameters(model_params, train_cfg):
    # 为了节省计算资源和显存，应将需要冻结的参数的 `requires_grad` 显式设置为 `False`，并且在优化器中过滤不可训练参数

    optimizer_grouped_parameters = []
    if CONFIG["run_mode"] == "pretrain":
        encoder_params = [(n, p) for n, p in model_params if n.startswith("encoder")]
        decoder_params = [(n, p) for n, p in model_params if n.startswith("decoder")]
        adaptor_params = [(n, p) for n, p in model_params if n.startswith("v2l_projector")]
        assert encoder_params and decoder_params and adaptor_params

        # 冻结 encoder, decoder，训练 v2l_projector
        for n, p in encoder_params + decoder_params:
            p.requires_grad = False
        for n, p in adaptor_params:
            p.requires_grad = True

        # no_decay_names = ["bias", "norm1.weight", "norm2.weight", "layernorm.weight", "layer_scale"]
        optimizer_grouped_parameters.append({"params": [p for n, p in adaptor_params], "lr": train_cfg["lr"], "weight_decay": 0.0})

    elif CONFIG["run_mode"] == "finetune":
        # When using peft, params requires_grad are set during initialization of PeftModel. See `apply_peft_to_model()`.
        # We only need to group them for optimizer.
        optimizer_grouped_parameters.append({"params": [p for n, p in model_params if p.requires_grad], "lr": train_cfg["lr"], "weight_decay": 0.0})

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


def log_and_update_status(curr_epoch, curr_iter, loss, bsz, lr, train_cfg):
    STATUS_INFO.curr_epoch = curr_epoch
    STATUS_INFO.curr_batch_iter = curr_iter
    STATUS_INFO.batch_trained_examples += bsz
    STATUS_INFO.batch_loss += loss * bsz
    STATUS_INFO.global_iters += 1

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
        eval_result_dict = evaluate(model, target_dataloader=valid_dataloader)
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


#############################################
# Evaluation
#############################################
def evaluate(model, target_dataloader, **kwargs):
    GLOBAL_VARS.peak_mem = 0

    eval_bsz = kwargs["eval_batch_size"]
    max_new_tokens = kwargs["max_new_tokens"]
    print_pred_per_n_steps = kwargs["print_pred_per_n_steps"]

    # 由于评估时间过长，pred结果将被存放到文件中，已经pred过的数据已经提前从dataset中移除
    LOGGER.info("****************************** Evaluation ******************************")
    LOGGER.info("Source = %s", target_dataloader.dataset.src_path)
    LOGGER.info("Batch size = %d", eval_bsz)
    LOGGER.info("Num samples = %d", len(target_dataloader.dataset))
    tokenizer = target_dataloader.dataset.tokenizer

    LOGGER.info("****************************** Model Predicting ******************************")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, input_tensors_dict in enumerate(target_dataloader):
            if input_tensors_dict["batch_data"]:
                LOGGER.info("No data remain unpredicted, stop model inference")
                break

            data_ids, pred_graphs, gold_graphs, pred_text, gold_text = [], [], [], [], []

            # Model inference, check args in https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin
            assistant_responses = None
            conversations_history = None
            for turn in range(2):
                # 第一轮生成graph，第二轮生成section
                input_tensors_dict = get_inputs_for_inference(tokenizer=tokenizer, batch_data=input_tensors_dict["batch_data"], pixel_values=input_tensors_dict["pixel_values"], image_indices_map=input_tensors_dict["image_indices_map"], assistant_responses=assistant_responses, conversations_history=conversations_history)

                with ACCELERATOR.autocast():
                    # https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation#transformers.GenerationConfig
                    if turn == 0:
                        # 生成graph的防重复设置
                        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=[tokenizer.eos_token_id, GLOBAL_VARS.eot_token_id], do_sample=False, num_beams=3, return_dict_in_generate=True, output_logits=True, no_repeat_ngram_size=4, temperature=0.9, top_k=50, top_p=0.9)
                    elif turn == 1:
                        # 生成text的防重复设置
                        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=[tokenizer.eos_token_id, GLOBAL_VARS.eot_token_id], do_sample=False, num_beams=3, return_dict_in_generate=True, output_logits=True, no_repeat_ngram_size=4, temperature=0.9, top_k=50, top_p=0.9)
                    # stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=[tokenizer.eos_token_id, GLOBAL_VARS.eot_token_id])])
                    # https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation#transformers.GenerationMixin
                    outputs = model.generate(
                        generation_config=generation_config,
                        inputs=input_tensors_dict["pixel_values"],
                        decoder_input_ids=input_tensors_dict["decoder_input_ids"],
                        decoder_attention_mask=input_tensors_dict["decoder_attention_mask"],
                    )
                    check_memory(show_only_if_peak=True)

                pred_seq_start_ids = input_tensors_dict["decoder_input_ids"].size(1)  # 生成的序列的起始位置
                pred_sequences_ids = outputs.sequences[:, pred_seq_start_ids:]
                pred_sequences = tokenizer.batch_decode(pred_sequences_ids, skip_special_tokens=True)

                # 用于第二轮生成
                assistant_responses = pred_sequences
                conversations_history = input_tensors_dict["conversations_history"]

                # Gathers input_data and potentially drops duplicates in the last batch if on a distributed system.
                if turn == 0:
                    pred_graphs = pred_sequences
                    gold_graphs = input_tensors_dict["gold_graph_list"]
                elif turn == 1:
                    pred_text = pred_sequences
                    gold_text = input_tensors_dict["gold_text_list"]

                if (print_pred_per_n_steps > 0 and batch_idx % print_pred_per_n_steps == 0) or (batch_idx + 1 == len(target_dataloader)):
                    LOGGER.info(
                        "Eval at: p=%s, iter=%d, finished_samples=%s, pred_example (%s): %s",
                        ACCELERATOR.process_index,
                        batch_idx,
                        batch_idx * eval_bsz,
                        "graph" if turn == 0 else "text",
                        pred_sequences[0],
                        main_process_only=False,
                    )

            data_ids = input_tensors_dict["data_id_list"]

            save_pred_results_per_batch(
                data_ids=data_ids,
                pred_text=pred_text,
                pred_graphs=pred_graphs,
                gold_text=gold_text,
                gold_graphs=gold_graphs,
                data_split=target_dataloader.dataset.split,
                output_dir=CONFIG["output_dir"]["result"],
            )

    LOGGER.info("****************************** Computing Scores ******************************")
    pred_result_dict = load_pred_results(intput_dir=CONFIG["output_dir"]["result"], split=target_dataloader.dataset.split)
    data_ids = [item["data_id"] for item in pred_result_dict.values()]
    pred_text = [item["pred_text"] for item in pred_result_dict.values()]
    pred_graphs = [item["pred_graph"] for item in pred_result_dict.values()]
    gold_text = [item["gold_text"] for item in pred_result_dict.values()]
    gold_graphs = [item["gold_graph"] for item in pred_result_dict.values()]

    # Evaluate text results
    text_scores_dict = compute_generation_score(gold_text_list=gold_text, pred_text_list=pred_text)
    LOGGER.info("[TextGen]: %s", json.dumps(text_scores_dict, indent=4))

    # Evaluate graph results
    text_scores_dict = compute_generation_score(gold_text_list=gold_graphs, pred_text_list=pred_graphs)
    LOGGER.info("[GraphTextGen]: %s", json.dumps(text_scores_dict, indent=4))

    # pred_graph_reprs = []
    # for graph_str in pred_graphs:
    #     graph_repr = parse_unquoted_graph_str(graph_str)
    #     pred_graph_reprs.append(graph_repr)
    # graph_scores_dict = compute_graph_scores(gold_graphs=gold_graphs, pred_graphs=pred_graph_reprs)
    # LOGGER.info("[GraphMatch]: %s", json.dumps(graph_scores_dict, indent=4))

    end = time.time()
    LOGGER.info("Evaluation time: %s", seconds_to_time_str(end - start))
    check_memory()
    return text_scores_dict


def save_pred_results_per_batch(data_ids, pred_text, pred_graphs, gold_text, gold_graphs, data_split, output_dir):
    """Save at each batch, so that we can use the results for further analysis or debugging."""

    output_file = os.path.join(output_dir, f"{data_split}_{ACCELERATOR.process_index}.json")

    with open(output_file, "a", encoding="utf-8") as f:
        for data_id, p_text, p_graph, g_text, g_graph in zip(data_ids, pred_text, pred_graphs, gold_text, gold_graphs):
            out_line = {"data_id": data_id, "pred_text": p_text, "pred_graph": p_graph, "gold_text": g_text, "gold_graph": g_graph}
            f.write(json.dumps(out_line))
            f.write("\n")


def load_pred_results(intput_dir, split):
    data_dict = {}  # key=data_id, value={"data_id": , "pred_text": , "pred_graph": , "gold_text": , "gold_graph": }

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


def parse_unquoted_graph_str(assistant_output_graph_str, strict=False):
    """
    预期：assistant_output_graph_str='[monitoring, <Observation-Present>], [support, <Observation-Present>], [support, <modify>, devices];\n ...'
    对于 ent 和 rel 的 type，使用 get_close_matches() 来进行模糊匹配
    目前ent默认为长度为2的列表，rel默认为长度为3的列表
    """

    graph_repr = []

    valid_ent_types = GLOBAL_VARS.ent_type_tokens
    valid_rel_types = GLOBAL_VARS.rel_type_tokens

    def get_closest_type(t, valid_set):
        matches = get_close_matches(t, valid_set, n=1, cutoff=0.5)
        return matches[0] if matches else None

    graph_sections = re.split(r"[;\n]+", assistant_output_graph_str.strip())

    for graph_str in graph_sections:
        if not graph_str.strip():
            continue

        ent_graph = []
        rel_graph = []

        items = re.findall(r"\[(.*?)\]", graph_str)
        for item in items:
            parts = [s.strip() for s in item.split(",")]
            if len(parts) == 2:
                typ = parts[1]
                if typ not in valid_ent_types:
                    if strict:
                        print(f"Entity type '{typ}' not valid, skipped.")
                        continue
                    corrected = get_closest_type(typ, valid_ent_types)
                    if corrected:
                        print(f"Entity type '{typ}' corrected to '{corrected}'")
                        typ = corrected
                    else:
                        print(f"Entity type '{typ}' not recognized, skipped.")
                        continue
                ent_graph.append([parts[0], typ, "NA", "NA", "NA"])

            elif len(parts) == 3:
                rel = parts[1]
                if rel not in valid_rel_types:
                    if strict:
                        print(f"Relation type '{rel}' not valid, skipped.")
                        continue
                    corrected = get_closest_type(rel, valid_rel_types)
                    if corrected:
                        print(f"Relation type '{rel}' corrected to '{corrected}'")
                        rel = corrected
                    else:
                        print(f"Relation type '{rel}' not recognized, skipped.")
                        continue
                rel_graph.append([parts[0], rel, parts[2]])

            else:
                print(f"Skipping badly formatted item: [{item}]")
                continue

        graph_repr.append([ent_graph, rel_graph])

    return graph_repr


def compute_ent_scores(gold_ents_list, pred_ents_list):
    assert len(gold_ents_list) == len(pred_ents_list), "Mismatched number of samples"

    def to_set(ents, mode):
        if mode == "text":
            return set(e[0].lower() for e in ents)
        elif mode == "text_and_type":
            # pred 中的type是包括尖括号的, e.g. <Anatomy>
            return set((e[0].lower(), e[1].replace("<", "").replace(">", "").lower()) for e in ents)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    scores = {}

    for mode in ["text", "text_and_type"]:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for gold_ents, pred_ents in zip(gold_ents_list, pred_ents_list):
            gold_set = to_set(gold_ents, mode)
            pred_set = to_set(pred_ents, mode)

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        scores[mode] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return scores


def compute_rel_scores(gold_rels_list, pred_rels_list):
    assert len(gold_rels_list) == len(pred_rels_list), "Mismatched number of samples"

    def to_set(rels, mode):
        if mode == "text":
            return set((r[0].lower(), r[2].lower()) for r in rels)
        elif mode == "text_and_type":
            return set((r[0].lower(), r[1].replace("<", "").replace(">", "").lower(), r[2].lower()) for r in rels)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    scores = {}

    for mode in ["text", "text_and_type"]:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for gold_rels, pred_rels in zip(gold_rels_list, pred_rels_list):
            gold_set = to_set(gold_rels, mode)
            pred_set = to_set(pred_rels, mode)

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        scores[mode] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return scores


def compute_graph_scores(gold_graphs, pred_graphs):
    # 嵌套列表，每个子列表包含一个文档的所有ent或rel
    gold_ents_list, pred_ents_list = [], []
    gold_rels_list, pred_rels_list = [], []

    for gold_graphs_docwise, pred_graphs_docwise in zip(gold_graphs, pred_graphs):
        # 原本每个文档的ent和rel是按句子分组的，现在需要将其合并，即每个文档的ent合并为一个列表，rel也合并为一个列表
        gold_ents = [ent for graph in gold_graphs_docwise for ent in graph[0]]
        gold_rels = [rel for graph in gold_graphs_docwise for rel in graph[1]]
        pred_ents = [ent for graph in pred_graphs_docwise for ent in graph[0]]
        pred_rels = [rel for graph in pred_graphs_docwise for rel in graph[1]]

        gold_ents_list.append(gold_ents)
        gold_rels_list.append(gold_rels)
        pred_ents_list.append(pred_ents)
        pred_rels_list.append(pred_rels)

    # 计算ent和rel的f1分数
    ent_scores = compute_ent_scores(gold_ents_list, pred_ents_list)
    rel_scores = compute_rel_scores(gold_rels_list, pred_rels_list)
    out_dict = {"ent_scores": ent_scores, "rel_scores": rel_scores}
    return out_dict


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


def load_module_state_dict_from(model_path, target_module_prefix):
    index_file_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file_path, "r", encoding="utf-8") as f:
        sd_index = json.load(f)

    target_model_file_paths = set()
    for key, file_name in sd_index["weight_map"].items():
        if key.startswith(target_module_prefix):
            target_model_file_paths.add(os.path.join(model_path, file_name))

    target_state_dict = {}
    for model_file_path in target_model_file_paths:
        for name, param in load_file(model_file_path).items():
            if name.startswith(target_module_prefix):
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


def get_dataloaders(img_processor, tokenizer, ds_train=None, ds_valid=None, ds_test=None, train_bsz=1, eval_bsz=1, use_debug_subset=False):

    train_dataloader, valid_dataloader, test_dataloader = None, None, None

    if ds_train:
        with ACCELERATOR.main_process_first():  # select是dataset caching 操作，主进程优先或许能快一点
            if use_debug_subset:
                train_dataset = ImageTextDataset(ds_train.select(range(len(ds_train) - 20, len(ds_train))), img_processor=img_processor, tokenizer=tokenizer, split="train")
            else:
                train_dataset = ImageTextDataset(ds_train, img_processor=img_processor, tokenizer=tokenizer, split="train")
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer), batch_size=train_bsz, drop_last=True)

    if ds_valid:
        with ACCELERATOR.main_process_first():  # select是dataset caching 操作，主进程优先或许能快一点
            if use_debug_subset:
                vaild_dataset = ImageTextDataset(ds_valid.select(range(len(ds_valid) - 2, len(ds_valid))), img_processor=img_processor, tokenizer=tokenizer, split="validation")
            else:
                ds_valid = filter_dataset_by_data_id(ds_valid, split="validation")
                vaild_dataset = ImageTextDataset(ds_valid, img_processor=img_processor, tokenizer=tokenizer, split="validation")
        valid_dataloader = DataLoader(vaild_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer), batch_size=eval_bsz, drop_last=False)

    if ds_test:
        with ACCELERATOR.main_process_first():
            if use_debug_subset:
                test_dataset = ImageTextDataset(ds_test.select(range(len(ds_test) - 3, len(ds_test))), img_processor=img_processor, tokenizer=tokenizer, split="test")
            else:
                ds_test = filter_dataset_by_data_id(ds_test, split="test")
                test_dataset = ImageTextDataset(ds_test, img_processor=img_processor, tokenizer=tokenizer, split="test")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=lambda batch: collate_fn(batch, img_processor, tokenizer), batch_size=eval_bsz, drop_last=False)

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


def load_dataset(ds_img_path, ds_graph_path, target_section, target_entity):
    # ds_img 是 image + report 数据集，report 包含 findings和impression
    # ds_graph 是纯文本数据集，是对应特定 target_section，即findings 或impression
    ds_img = load_from_disk(ds_img_path)
    LOGGER.info("Loaded pre_processed image dataset from: \n%s \n%s", ds_img_path, ds_img)
    ds_graph_path = os.path.join(ds_graph_path, target_entity, f"interpret_graph_{target_section}_{target_entity}")
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
    # 检查 tokenizer 是否存在新增的实体和关系 token
    for ent_tok in GLOBAL_VARS.graph_tokens:
        assert ent_tok in tokenizer.get_vocab(), f"Missing entity token: {ent_tok} in tokenizer, expect: {GLOBAL_VARS.graph_tokens} in tokenizer."

    eot_token_id = tokenizer.encode(GLOBAL_VARS.eot_token, add_special_tokens=False)
    assert len(eot_token_id) == 1, f"Expected single token for eot_token: {GLOBAL_VARS.eot_token}, got: {eot_token_id}"
    GLOBAL_VARS.eot_token_id = eot_token_id[0]

    # 用于在 input_ids 中查找需要替换的图像占位符 <|image_token|>
    if not hasattr(model.config, "image_token_index"):
        model.config.image_token_index = tokenizer.convert_tokens_to_ids("<|image_token|>")

    # 计算 vision model 输出的图像特征的数量，该数量等于我们应该在 input_ids 中插入 <|image_token|> 的数量
    img_size = model.config.encoder.image_size
    dummy_img = torch.zeros((1, 3, img_size, img_size))
    num_image_tokens = model.encoder(dummy_img).last_hidden_state.size(1)
    GLOBAL_VARS.num_image_tokens = num_image_tokens


def init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefix="v2l_projector"):
    # 重新初始化模型，在后续再单独加载预训练的 img_projector，避免OOM的问题（不知道为什么会出现这个问题）
    base_model = init_model(vision_model_path, language_model_path, model_base_cfg)
    # Load only img_projector state_dict to the base model
    # 如果直接加载整个 pre_trained 模型，会导致训练时OOM，但只加载 img_projector 到base_model则不会
    # 有点担心的是 decoder embedding 重新初始化，是否会导致其与 img_projector 的不匹配
    target_state_dict = load_module_state_dict_from(model_path=pretain_model_path, target_module_prefix=target_module_prefix)
    model = load_state_dict_to_model(base_model=base_model, target_state_dict=target_state_dict)
    LOGGER.info("Initialized model with pretrained weights %s", target_module_prefix)
    return model


def init_model(vision_model_path, language_model_path, model_base_cfg):
    model = Vision2LanguageModel.from_encoder_decoder_pretrained(vision_model_path, language_model_path)
    LOGGER.info("Initialized vision language mode from: \n%s\n%s", vision_model_path, language_model_path)
    return model


def init_processor(vision_model_path, language_model_path, model_base_cfg):
    img_processor = AutoImageProcessor.from_pretrained(vision_model_path, use_fast=True)
    LOGGER.info("Loaded ImageProcessor from: %s", vision_model_path)

    tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=True)
    LOGGER.info("Loaded Tokenizer from: %s", language_model_path)

    LOGGER.info("Adding new tokens for ent/rel types")
    new_tokens = GLOBAL_VARS.graph_tokens
    tokenizer.add_tokens(new_tokens)
    LOGGER.info("New tokens: %s", [(tok, tokenizer.convert_tokens_to_ids(tok)) for tok in new_tokens])

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
    modules_to_save = ["v2l_projector"]  # 没注入LoRA 但又需要训练和保存的模块。添加模块后，peft会包装一个一模一样的模块，并将requires_grad 会被设置为 True。原模块不变。
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
        transformer_cls_names_to_wrap = ["LlamaDecoderLayer", "Dinov2Layer", "VisionLanguageProjector"]
        # sharding_strategy = "NO_SHARD", state_dict_type = "FULL_STATE_DICT"，用这套配置可以解决加载checkpoint后，在optimizer.step()时的报错 RuntimeError: torch function 'lerp_', with args: (ShardedTensor(ShardedTensorMetadata(sha ... and kwargs: None not supported for ShardedTensor!
        sharding_strategy = "NO_SHARD"
        state_dict_type = "FULL_STATE_DICT"

        # 关于 FSDP1 -> FSDP2 https://huggingface.co/docs/accelerate/main/en/concept_guides/fsdp1_vs_fsdp2
    else:
        ignored_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Embedding, Dinov2Model, LlamaRMSNorm, LlamaRotaryEmbedding)):
                ignored_modules.append(module)
        transformer_cls_names_to_wrap = ["LlamaDecoderLayer", "Dinov2Layer", "VisionLanguageProjector"]
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
    global CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--config_file", type=str, help=".yaml file path")

    parser.add_argument("--output_name", type=str)
    parser.add_argument("--jobid", type=int)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=None)

    parser.add_argument("--run_mode", type=str, default=None, help="Choose from [preprocess, pretrain, eval_pretrained, finetune, eval_finetuned]")
    parser.add_argument("--mlflow_port", type=str, default=None)

    args = parser.parse_args()

    if args.from_bash:
        file_path = args.config_file
    else:
        file_path = "/home/yuxiang/liao/workspace/arrg_img2text/config/6_effu.yaml"

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
    else:
        CONFIG["jobid"] = "00000"

    CONFIG["mlflow_url"] = f"{CONFIG['mlflow_url']}:{CONFIG['mlflow_port']}"

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
    vision_model_path = CONFIG["model_name_or_path"][model_base_cfg["vision_model"]]
    language_model_path = CONFIG["model_name_or_path"][model_base_cfg["language_model"]]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"], target_entity=CONFIG["target_entity"])

    img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)
    model = init_model(vision_model_path, language_model_path, model_base_cfg)
    post_init_model_and_tokenizer(model, tokenizer)
    global_init_accelerator(model, **train_cfg)
    check_memory()
    model.to(DEVICE)
    train_dataloader, _, _ = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_train=ds_final["train"], train_bsz=train_cfg["batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()

    start = time.time()
    train(model, train_dataloader, train_cfg=train_cfg)
    end = time.time()
    LOGGER.info("Total training time: %s", seconds_to_time_str(end - start))

    save_processors(img_processor=img_processor, tokenizer=tokenizer, output_dir=CONFIG["output_dir"]["model"])


def eval_pretrained_model(train_cfg):
    pretain_model_path = CONFIG["output_dir"]["model"]

    # Train and test
    set_seed(train_cfg["seed"])
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"], target_entity=CONFIG["target_entity"])

    model = load_model(pretain_model_path)
    img_processor, tokenizer = load_processor(pretain_model_path)
    post_init_model_and_tokenizer(model, tokenizer)
    global_init_accelerator(model, **train_cfg)
    _, validation_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], eval_bsz=train_cfg["eval_batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()
    model.to(DEVICE)
    model, validation_dataloader, test_dataloader = ACCELERATOR.prepare(model, validation_dataloader, test_dataloader)
    check_memory()

    start = time.time()
    evaluate(model, validation_dataloader, **train_cfg)
    start2 = time.time()
    LOGGER.info("Valid time: %s", seconds_to_time_str(start2 - start))

    evaluate(model, test_dataloader, **train_cfg)
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
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"], target_entity=CONFIG["target_entity"])

    if train_cfg["use_pretrained"]:
        model = init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefix="v2l_projector")
        img_processor, tokenizer = load_processor(pretain_model_path)
    else:
        model = init_model(vision_model_path, language_model_path, model_base_cfg)
        img_processor, tokenizer = init_processor(vision_model_path, language_model_path, model_base_cfg)

    post_init_model_and_tokenizer(model, tokenizer)
    model = apply_peft_to_model(model)
    global_init_accelerator(model, **train_cfg)
    check_memory()
    model.to(DEVICE)

    train_dataloader, _, _ = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_train=ds_final["train"], train_bsz=train_cfg["batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    # _, validation_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], use_debug_subset=CONFIG["use_debug_subset"])
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
    ds_final = load_dataset(ds_img_path=CONFIG["preprocess"]["cache_path"], ds_graph_path=CONFIG["data_path"]["text_graph"], target_section=CONFIG["target_section"], target_entity=CONFIG["target_entity"])

    if train_cfg["use_pretrained"]:
        model = init_model_with_pretrained_weights(model_base_cfg, vision_model_path, language_model_path, pretain_model_path, target_module_prefix="v2l_projector")
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

    _, validation_dataloader, test_dataloader = get_dataloaders(img_processor=img_processor, tokenizer=tokenizer, ds_valid=ds_final["validation"], ds_test=ds_final["test"], eval_bsz=train_cfg["eval_batch_size"], use_debug_subset=CONFIG["use_debug_subset"])
    check_memory()

    model, validation_dataloader, test_dataloader = ACCELERATOR.prepare(model, validation_dataloader, test_dataloader)
    check_memory()

    start = time.time()
    start2 = start
    if train_cfg["eval_valid_split"]:
        evaluate(model, validation_dataloader, **train_cfg)
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
