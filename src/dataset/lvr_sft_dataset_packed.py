"""
    This file is adapted from InternVL [https://github.com/OpenGVLab/InternVL/tree/main]
    We adapted and simplified the PackedDataset and IterableSupervisedDataset for our LVR finetuning
"""


import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset, IterableDataset

from functools import partial

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVR_PLACEHOLDER,
    LVR_TOKEN,
    PLVR_CTX_END_TOKEN,
    PLVR_CTX_START_TOKEN,
    PLVR_FREE_END_TOKEN,
    PLVR_FREE_START_TOKEN,
    PLVR_FREE_TOKEN,
    PLVR_TGT_END_TOKEN,
    PLVR_TGT_START_TOKEN,
)
from transformers import TrainingArguments

from .data_utils import get_image_info, llava_to_openai_lvr, pad_sequence
import numpy as np
from PIL import Image
from typing import List, Tuple
import math
from src.lvr_utils import expand_bbox

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

  
def make_packed_supervised_data_module_lvr(model_id, processor, data_args, training_args: TrainingArguments,latent_end_token=False):

    """Make dataset and collator for supervised fine-tuning."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

    # we assume meta data
    meta_data = json.load(open(data_args.data_path))

    datasets = []
    total_data_len = 0
    for meta in meta_data:
        iterable_sft_dataset = IterableSupervisedDatasetLVR(
            data_path=meta['data_path'],
            image_folder=meta['image_folder'],
            ds_name=meta['ds_name'],
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode = training_args.enable_data_packing,    # set for packed dataset
            random_seed=data_args.random_seed,
            latent_end_token=latent_end_token
        )
        datasets.append(iterable_sft_dataset)
        total_data_len += len(iterable_sft_dataset)

    packed_train_dataset = PackedDataset(
        tokenizer=processor.tokenizer,
        datasets=datasets,
        # Get rank and world size from Hugging Face Trainer arguments
        data_rank=data_rank,
        data_world_size=data_world_size,
        # --- Configure your packing parameters ---
        max_packed_tokens=training_args.max_packed_tokens,
        max_buffer_size=100,
        long_seq_threshold=training_args.long_seq_threshold,
        # Limiting the number of training data per device to avoid exploded tokens_per_device when pairing long with short
        max_instance_per_batch=training_args.max_instance_per_batch,
    )

    data_collator = PackedDataCollatorForSupervisedDatasetLVR(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=packed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,), total_data_len

import torch
from torch.utils.data import IterableDataset

class IterableSupervisedDatasetLVR(Dataset):
    """
    An iterable version of your dataset that streams one processed sample at a time.
    This will be the input to PackedDataset.
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        ds_name: str,
        model_id,
        data_rank=0,
        data_world_size=1,
        distributed_mode = True,    # set for packed dataset
        random_seed=None,
        latent_end_token=False,
    ):
        super().__init__()
        if isinstance(data_path, str):
            self.raw_data = json.load(open(data_path, "r"))
        else:
            self.raw_data = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_folder = image_folder
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.ds_name = ds_name
        self.fps = data_args.fps

        self.data_world_size = data_world_size
        self.worker_id = None
        self.distributed_mode = distributed_mode
        self.worker_distributed = False
        self._state_dict = {}

        self.random_seed = None
        if random_seed:
            logger.info(f"{self.ds_name} is Shuffled!")
            self.random_seed = random_seed
            self.rng = np.random.default_rng(seed=self.random_seed)
            self.rng.shuffle(self.raw_data)

        # int latent_end_token mode, a latent end token wil be appended to the selected lvr tokens
        self.latent_end_token = latent_end_token

    def __len__(self):
        return len(self.raw_data)

    def bbox_to_token_idxs(self, bboxes, image_grid_thw):
        """
            This function is intended for Qwen-VL series only.
            The target visual tokens is computed based on image_grid_thw,
            which is the best estimation

            image_grid_thw is a 2D tensor with a single item

        """
        _, h, w = image_grid_thw[0].tolist()
        token_idxs = []
        for bbox in bboxes: 
            x0, y0, x1, y1 = bbox

            # Scale to 14by14 grid
            x0_grid = max(0, min(int(np.floor(x0 * w)), w-1))
            x1_grid = max(0, min(int(np.ceil (x1 * w)), w))
            y0_grid = max(0, min(int(np.floor(y0 * h)), h-1))
            y1_grid = max(0, min(int(np.ceil (y1 * h)), h))


            # Map to 28by28 grid
            x0_token = x0_grid // 2
            x1_token = (x1_grid + 1) // 2
            y0_token = y0_grid // 2
            y1_token = (y1_grid + 1) // 2

            H2, W2 = h // 2, w // 2

            idxs = [
                int(yy * W2 + xx)
                for yy in range(y0_token, y1_token)
                for xx in range(x0_token, x1_token)
            ]

            token_idxs.append(idxs)

        return token_idxs

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)


        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
        )

        for i in range(start_idx,len(self.raw_data)):


            sources = self.raw_data[i]

            is_video = False

            processor = self.processor

            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

            # Extract LVR tokens
            image_grid_thw = processor(text=[""], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')['image_grid_thw']
            lvr_token_idxs_list = self.bbox_to_token_idxs(sources['bboxes'],image_grid_thw)

            stage_sizes = None
            if self.data_args.plvr_mode:
                expanded_bboxes = [expand_bbox(bb, self.data_args.expansion_factor) for bb in sources['bboxes']]
                lvr_token_idxs_context = self.bbox_to_token_idxs(expanded_bboxes, image_grid_thw)

                visual_lvr_tokens = []
                all_lvr_tokens = []
                stage_sizes = []
                if self.data_args.include_free_stage:
                    n_free = self.data_args.num_free_tokens
                    for ctx, tgt in zip(lvr_token_idxs_context, lvr_token_idxs_list):
                        visual_lvr_tokens.extend([ctx, tgt])
                        all_lvr_tokens.extend([ctx, [0] * n_free, tgt])
                        stage_sizes.append([len(ctx), n_free, len(tgt)])
                    per_bbox_stage_tokens = [
                        (PLVR_CTX_START_TOKEN, LVR_TOKEN, PLVR_CTX_END_TOKEN),
                        (PLVR_FREE_START_TOKEN, PLVR_FREE_TOKEN, PLVR_FREE_END_TOKEN),
                        (PLVR_TGT_START_TOKEN, LVR_TOKEN, PLVR_TGT_END_TOKEN),
                    ]
                else:
                    for ctx, tgt in zip(lvr_token_idxs_context, lvr_token_idxs_list):
                        visual_lvr_tokens.extend([ctx, tgt])
                        all_lvr_tokens.extend([ctx, tgt])
                        stage_sizes.append([len(ctx), len(tgt)])
                    per_bbox_stage_tokens = [
                        (PLVR_CTX_START_TOKEN, LVR_TOKEN, PLVR_CTX_END_TOKEN),
                        (PLVR_TGT_START_TOKEN, LVR_TOKEN, PLVR_TGT_END_TOKEN),
                    ]

                conversations = copy.deepcopy(sources['conversations'])
                stage_tokens = per_bbox_stage_tokens * len(sources['bboxes'])
                for conv in conversations:
                    if conv['from'] == 'gpt' and LVR_PLACEHOLDER in conv['value']:
                        conv['value'] = conv['value'].replace(
                            LVR_PLACEHOLDER,
                            LVR_PLACEHOLDER * len(per_bbox_stage_tokens),
                        )
                lvr_token_idxs_list = visual_lvr_tokens
                sources = copy.deepcopy(
                    llava_to_openai_lvr(
                        conversations,
                        is_video=is_video,
                        lvr_token_idxs_list=all_lvr_tokens,
                        latent_end_token=self.latent_end_token,
                        stage_tokens=stage_tokens,
                    )
                )
            else:
                sources = copy.deepcopy(
                    llava_to_openai_lvr(
                        sources['conversations'],
                        is_video=is_video,
                        lvr_token_idxs_list=lvr_token_idxs_list,
                        latent_end_token=self.latent_end_token,
                    )
                )

            all_input_ids = [] 
            all_labels = []
            all_pixel_values = []
            all_image_grid_thw = []
            all_second_gird = []

            # Qwen2-VL uses a default system message.
            if len(SYSTEM_MESSAGE) > 0:
                system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
                system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
                
                all_input_ids.append(system_message_input_ids.squeeze(0))
                all_labels.append(system_labels.squeeze(0))

            for _, j in enumerate(range(0, len(sources), 2)):
                user_input = sources[j]
                gpt_response = sources[j + 1]

                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
                
                if DEFAULT_IMAGE_TOKEN in user_input:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                    prompt_input_ids = inputs['input_ids']
                    all_pixel_values.append(inputs[pixel_key])
                    all_image_grid_thw.append(inputs[grid_key])

                else:
                    prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                # filling the response with bboxes
                
                response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                        response_input_ids.squeeze(0),
                    ],
                    dim=0,
                )

                all_input_ids.append(input_ids)
                all_labels.append(labels)
            
            # There is no need for eos or bos tokens in the input_ids
            # Qwen2-VL does not use them
            input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
            labels = torch.cat(all_labels, dim=0).to(torch.long)

            attention_mask = (input_ids > -1000000).to(torch.long)

            '''Only 1-image-multi-area is supported.'''
            lvr_tokens = []
            for group in lvr_token_idxs_list:
                lvr_tokens.append(torch.tensor(group,dtype=torch.int))

            data_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lvr_tokens=lvr_tokens,
            )

            if stage_sizes is not None:
                data_dict['lvr_stage_sizes'] = [stage_sizes]

            if pixel_key and grid_key:
                pixel_values = torch.cat(all_pixel_values, dim=0)
                image_thw = torch.cat(all_image_grid_thw, dim=0)
                data_dict[pixel_key] = pixel_values
                data_dict[grid_key] = image_thw

            if len(all_second_gird) > 0:
                second_gird = all_second_gird
                data_dict["second_per_grid_ts"] = second_gird

            data_dict['input_lengths'] = torch.tensor([input_ids.size(0)])

            # Instead of returning, we yield the processed dictionary
            yield data_dict


"""
    Below is a adaptation of 
    https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset_packed.py
    We adopted a greedy data packing logic
"""

# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import bisect
import copy
import logging
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 4096,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        long_seq_threshold: int = 6144,      # Instance longer than this will be individually processed
        max_instance_per_batch: int = 4,     # max num of instance per device
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens
        self.long_seq_threshold = long_seq_threshold
        self.max_instance_per_batch = max_instance_per_batch

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self.lvr_token_id = self.tokenizer.convert_tokens_to_ids(LVR_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [len(d) for d in datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]


        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {d.ds_name:0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}',
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(
                f'Sampling prob for each dataset:\n{temp}'
            )

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):

        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except:
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except:
                logger.error(f'worker_id{self.worker_id} data_rank={self.data_rank} data_world_size={self.data_world_size} Unexpected error!')
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        find = False
        find_idx = -1

        # if we see a new sample > LST, we need it to be in the buffer list,
        # instead of concatenating it to any existing buffer
        if new_sample['input_ids'].size(0) >= self.long_seq_threshold:
            return None

        for buffer_idx, buffer in enumerate(buffer_list):
            num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)
            num_instance_buffer = buffer['input_lengths'].size(0)
            if num_instance_buffer + 1 <= self.max_instance_per_batch:
                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        else:
            return None

    @staticmethod
    def _num_lvr_groups_for_last_sample(buffer):
        if 'lvr_stage_sizes' in buffer:
            return 2 * len(buffer['lvr_stage_sizes'][-1])
        return 1

    @staticmethod
    def _truncate_lvr_groups(lvr_tokens, kept_visual_tokens, sample_stage_sizes=None):
        if kept_visual_tokens <= 0:
            return [], []

        truncated_tokens = []
        if sample_stage_sizes is None:
            remaining = kept_visual_tokens
            for group in lvr_tokens:
                keep = min(group.size(0), remaining)
                if keep <= 0:
                    break
                truncated_tokens.append(group[:keep])
                remaining -= keep
                if remaining <= 0:
                    break
            return truncated_tokens, []

        truncated_stage_sizes = []
        remaining = kept_visual_tokens
        group_idx = 0
        for stage_sz in sample_stage_sizes:
            if group_idx + 1 >= len(lvr_tokens):
                break

            n_ctx = stage_sz[0]
            n_tgt = stage_sz[-1]
            keep_ctx = min(n_ctx, remaining)
            remaining -= keep_ctx
            keep_tgt = min(n_tgt, remaining)
            remaining -= keep_tgt

            if keep_ctx == 0 and keep_tgt == 0:
                break

            truncated_tokens.append(lvr_tokens[group_idx][:keep_ctx])
            truncated_tokens.append(lvr_tokens[group_idx + 1][:keep_tgt])
            if len(stage_sz) == 3:
                truncated_stage_sizes.append([keep_ctx, stage_sz[1], keep_tgt])
            else:
                truncated_stage_sizes.append([keep_ctx, keep_tgt])
            group_idx += 2

            if remaining <= 0:
                break

        return truncated_tokens, truncated_stage_sizes

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        assert buffer.keys() == new_sample.keys()
        for k in buffer:
            if k == 'lvr_tokens':
                buffer[k] = buffer[k] + new_sample[k]
            elif k == 'lvr_stage_sizes':
                buffer[k].extend(new_sample[k])
            else:
                buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id,lvr_token_id,long_seq_threshold,max_instance_per_batch):
        
        if not long_seq_threshold:
            long_seq_threshold = max_tokens // 2

        def _image_is_splitted(input_ids, cut_idx):
            if cut_idx >= input_ids.size(0):
                return False
            else:
                is_image_start = input_ids[cut_idx].item() == img_start_token_id
                is_image_token = input_ids[cut_idx].item() == img_token_id
                is_image_end = input_ids[cut_idx].item() == img_end_token_id
                return is_image_start or is_image_token or is_image_end
        
        '''
            Handles long single-/multi- instance buffer differently
        '''
        # condition 1: single instance
        if buffer['data_index'][-1].item() == 0:
            # condition 1.1: single long instance
            if buffer['input_ids'].size(0) >= long_seq_threshold:

                '''cut_id is the idx of the first token to be dropped'''
                cut_id = min(max_tokens, buffer['input_ids'].size(0))

                if not _image_is_splitted(buffer['input_ids'], cut_id):
                    truncated_lvr_tokens = None
                    truncated_stage_sizes = None
                    if lvr_token_id is not None and len(buffer['lvr_tokens']) > 0:
                        kept_lvr_tokens = (buffer['input_ids'][:cut_id] == lvr_token_id).sum().item()
                        sample_stage_sizes = buffer['lvr_stage_sizes'][0] if 'lvr_stage_sizes' in buffer else None
                        truncated_lvr_tokens, truncated_stage_sizes = PackedDataset._truncate_lvr_groups(
                            buffer['lvr_tokens'],
                            kept_lvr_tokens,
                            sample_stage_sizes=sample_stage_sizes,
                        )
                        if not truncated_lvr_tokens:
                            return [], []
                    
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer[k] = buffer[k][:cut_id]
                        elif k in ['pixel_values', 'image_flags','image_grid_thw']:
                            buffer[k] = buffer[k]
                        elif k in ['lvr_tokens']:
                            buffer[k] = truncated_lvr_tokens
                        elif k in ['lvr_stage_sizes']:
                            buffer[k] = [truncated_stage_sizes]
                        elif k in ['input_lengths']:
                            pass
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    # re-assign lengths
                    buffer['input_lengths'][0] = buffer['input_ids'].size(0)
                    buffer_ready = [buffer]
                    buffer_unready = []
                else:   # if image is getting cut, discard the overlong instance
                    buffer_ready = []
                    buffer_unready = []
            
            # condition 1.2: single short instance
            else:
                buffer_ready = []
                buffer_unready = [buffer]

        # condition 2: multi instance
        else:
            # condition 2.1: < maxToken AND < max_instance_per_batch
            if (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) < max_instance_per_batch):
                buffer_ready = []
                buffer_unready = [buffer]
            # condition 2.2: < maxToken AND == max_instance_per_batch
            elif (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) == max_instance_per_batch):
                buffer_ready = [buffer]
                buffer_unready = []
            # condition 2.3: otherwise
            else:
                buffer_ready = []
                buffer_unready = []
                while buffer['input_ids'].size(0) >= max_tokens:
                    buffer_right = {}
                    cut_idx_right_size = buffer['input_lengths'][-1].item()     # number of tokens to be cut from right side
                    image_cut_idx_right_size = buffer['image_grid_thw'][-1].prod()  # number of pixels to be cut from right side
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer_right[k] = buffer[k][-cut_idx_right_size:]
                            buffer[k] = buffer[k][:-cut_idx_right_size]
                        elif k in ['pixel_values', 'image_flags']:
                            buffer_right[k] = buffer[k][-image_cut_idx_right_size:]
                            buffer[k] = buffer[k][:-image_cut_idx_right_size]
                        elif k in ['lvr_tokens']:
                            num_right_groups = PackedDataset._num_lvr_groups_for_last_sample(buffer)
                            buffer_right[k] = buffer[k][-num_right_groups:]
                            buffer[k] = buffer[k][:-num_right_groups]
                        elif k in ['lvr_stage_sizes', 'image_grid_thw', 'input_lengths']:
                            buffer_right[k] = buffer[k][-1:]
                            buffer[k] = buffer[k][:-1]
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    # if left buffer is longer
                    if buffer['input_ids'].size(0) >= buffer_right['input_ids'].size(0):
                        buffer_ready.append(buffer)
                        buffer = buffer_right
                    else:   # buffer_right is longer than the accumulated left
                        buffer_ready.append(buffer_right)

                buffer_unready.append(buffer)

        return buffer_ready, buffer_unready

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        buffer_ready, buffer_unready = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
            lvr_token_id=self.lvr_token_id,
            long_seq_threshold=self.long_seq_threshold,
            max_instance_per_batch=self.max_instance_per_batch
        )

        for each_buffer in buffer_ready:
            buffer_max_len_list.append(each_buffer)

        for each_buffer in buffer_unready:
            find_idx = len(buffer_list)
            num_tokens_new_sample = each_buffer['input_ids'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['input_ids'].size(0) < num_tokens_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        return buffer_list, buffer_max_len_list

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]

        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                buffer_list = custom_infos['buffer_list']
                if self._should_log() and worker_id == 0:
                    logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f'{self.__class__.__name__} Rank {self.data_rank} '
            f'Worker {worker_id} begin to load data'
        )

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                while len(buffer_list) > 0:
                    yield buffer_list.pop(0)
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            # it is guaranteed in self.find_buffer() that if current_sample is >= max_tokens,
            # it will not get concatenated to any existing buffer
            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            '''
                A greedy method to balance effifiency and memory safety:
                1. if buffer stacks up tp max_packed_tokens, it is poped
                2. if buffer has >= max_instance_per_batch, it is poped
                3. if a single sample is >= long_seq_thresh, it is poped as a single buffer

                This is intended to avoid a long seq paired with multiple short seqs since
                padding them will explode the memory
            '''
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                
                yield buffer_max_len_list.pop(0)

            while len(buffer_list) > self.max_buffer_size:
                
                yield buffer_list.pop(0)

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1


WARNING_CNT = defaultdict(int)

class PackedDataCollatorForSupervisedDatasetLVR(object):

    def __init__(self,pad_token_id):
        self.pad_token_id = pad_token_id 
    
    def __call__(self, features):
        # features is supposed to be a list of packed items
        # We will set batch_per_device to 1

        if not isinstance(features, list):
            features = [features]

        #  Unpack all sequences
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        all_lvr_tokens = []
        all_lvr_stage_sizes = []
        all_pixel_values = []
        all_image_grid_thw = []

        for feature in features:
            # each feature is a packed mini-batch
            all_input_ids.extend(torch.split(feature["input_ids"], feature["input_lengths"].tolist()))
            all_attention_masks.extend(torch.split(feature["attention_mask"], feature["input_lengths"].tolist()))
            all_labels.extend(torch.split(feature["labels"], feature["input_lengths"].tolist()))

            all_lvr_tokens.extend(feature['lvr_tokens'])
            if 'lvr_stage_sizes' in feature:
                all_lvr_stage_sizes.extend(feature['lvr_stage_sizes'])
            all_pixel_values.append(feature['pixel_values'])
            all_image_grid_thw.append(feature['image_grid_thw'])
        
        # pad all sequences
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            seq_len = len(all_input_ids[i])
            padding_needed = max_len - seq_len

            # Pad on the right side.
            padded_input_ids.append(torch.nn.functional.pad(
                all_input_ids[i], (0, padding_needed), value=self.pad_token_id))
            
            padded_attention_masks.append(torch.nn.functional.pad(
                all_attention_masks[i], (0, padding_needed), value=0))
            
            padded_labels.append(torch.nn.functional.pad(
                all_labels[i], (0, padding_needed), value=IGNORE_INDEX))

        data_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),

            "lvr_tokens": all_lvr_tokens,
            "pixel_values": torch.cat(all_pixel_values),
            "image_grid_thw": torch.cat(all_image_grid_thw),
            }

        if all_lvr_stage_sizes:
            lvr_group_to_batch_idx = []
            for sample_idx, sample_stages in enumerate(all_lvr_stage_sizes):
                lvr_group_to_batch_idx.extend([sample_idx] * (2 * len(sample_stages)))
            data_dict["lvr_group_to_batch_idx"] = torch.tensor(lvr_group_to_batch_idx, dtype=torch.long)
            data_dict["lvr_stage_sizes"] = [stage for sample_stages in all_lvr_stage_sizes for stage in sample_stages]
        
        return data_dict
