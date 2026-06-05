import sys
import os

import os
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_supervised_data_module_lvr, make_packed_supervised_data_module_lvr
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.constants import (
    LVR_END_TOKEN,
    LVR_LATENT_END_TOKEN,
    LVR_START_TOKEN,
    LVR_TOKEN,
    PLVR_CTX_END_TOKEN,
    PLVR_CTX_START_TOKEN,
    PLVR_FREE_END_TOKEN,
    PLVR_FREE_START_TOKEN,
    PLVR_FREE_TOKEN,
    PLVR_TGT_END_TOKEN,
    PLVR_TGT_START_TOKEN,
)

from train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

# Cloud checkpointing depends on boto3/botocore. Import lazily so the
# release runs end-to-end without the cloud deps when --online_checkpoint
# is False (the default).
def _import_oci():
    from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
    return OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.include_free_stage and not data_args.plvr_mode:
        raise ValueError("`include_free_stage` requires `plvr_mode=True`.")
    if data_args.plvr_mode and not training_args.enable_data_packing:
        raise ValueError("P-LVR is implemented on the packed data path only. Set `--enable_data_packing True`.")
    if data_args.plvr_mode and model_args.max_lvr_tokens is not None:
        raise ValueError("P-LVR is incompatible with `max_lvr_tokens`; use bbox-derived token counts instead.")
    if data_args.plvr_mode and model_args.latent_end_token:
        raise ValueError("P-LVR with `latent_end_token` is not implemented.")

    '''
        set up oci checkpointing;
        set online_checkpoint to False if you dont need
    '''
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint:
        OCIFolderCheckpointHandler, create_temp_dir = _import_oci()
        # oci keys
        access_key_id = os.environ.get('ACCESS_KEY_ID')
        secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('ENDPOINT_URL')
        bucket_name = os.environ.get('BUCKET_NAME')
        region_name = os.environ.get('REGION_NAME')

        model_name = model_args.model_id.split('/')[-1]     # "Qwen2.5-VL-7B-Instruct"
        # local cache dir and tempFile class
        cache_dir = os.getenv("CACHE_DIR")
        local_model_name_or_path = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=training_args.run_name + '-')
        temp_folder = local_model_name_or_path

        # remote dir
        remote_dir = training_args.output_dir  # output_dir is remote now; "/checkpoints"
        remote_dir = os.path.join(remote_dir,model_name,training_args.run_name)    # "/checkpoints/Qwen2.5-VL-7B-Instruct/run_name"
        training_args.remote_output_dir = remote_dir
        training_args.output_dir = local_model_name_or_path.name    # output_dir should always be local

        # oci handler
        oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name, region_name)
    

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        if training_args.online_checkpoint:
            _, create_temp_dir = _import_oci()
            local_pth_to_download_chkpt = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=f"warmed_{model_args.lvr_head_type}" + '-')
            oci_handler.load_checkpoint(training_args.checkpoint_name, local_pth_to_download_chkpt,inference_mode=True)
            
            model_pth = local_pth_to_download_chkpt.name
        else:
            model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head
    config.lvr_head_type = model_args.lvr_head_type
    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function
        replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=model_args.coconut,
                                                        lvr_head=model_args.lvr_head,
                                                        mode_switch_loss=training_args.mode_switch_loss,
                                                        latent_end_token=model_args.latent_end_token)
        
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        # init lvr_head
        if model_args.lvr_head:
            model._init_lvr_head(lvr_head_type =  model_args.lvr_head_type)

        # init latent_end_token
        if model_args.latent_end_token:
            model._init_lvr_latent_end_emb()
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)

    processor.tokenizer.add_tokens(LVR_START_TOKEN, special_tokens=True)
    processor.tokenizer.add_tokens(LVR_TOKEN, special_tokens=True)
    processor.tokenizer.add_tokens(LVR_LATENT_END_TOKEN, special_tokens=True)
    processor.tokenizer.add_tokens(LVR_END_TOKEN, special_tokens=True)

    lvr_id = processor.tokenizer.convert_tokens_to_ids(LVR_TOKEN)
    lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids(LVR_LATENT_END_TOKEN)
    lvr_start_id = processor.tokenizer.convert_tokens_to_ids(LVR_START_TOKEN)
    lvr_end_id = processor.tokenizer.convert_tokens_to_ids(LVR_END_TOKEN)

    model.config.lvr_id = lvr_id
    model.config.lvr_latent_end_id = lvr_latent_end_id
    model.config.lvr_start_id = lvr_start_id
    model.config.lvr_end_id = lvr_end_id
    model.config.plvr_mode = data_args.plvr_mode
    model.config.include_free_stage = data_args.include_free_stage
    model.config.num_free_tokens = data_args.num_free_tokens
    model.config.nlvr_mode = data_args.nlvr_mode
    model.config.nlvr_noise_scale = data_args.nlvr_noise_scale

    if data_args.plvr_mode:
        for token in [
            PLVR_CTX_START_TOKEN,
            PLVR_CTX_END_TOKEN,
            PLVR_FREE_START_TOKEN,
            PLVR_FREE_END_TOKEN,
            PLVR_TGT_START_TOKEN,
            PLVR_TGT_END_TOKEN,
            PLVR_FREE_TOKEN,
        ]:
            processor.tokenizer.add_tokens(token, special_tokens=True)

        model.config.ctx_start_id = processor.tokenizer.convert_tokens_to_ids(PLVR_CTX_START_TOKEN)
        model.config.ctx_end_id = processor.tokenizer.convert_tokens_to_ids(PLVR_CTX_END_TOKEN)
        model.config.free_start_id = processor.tokenizer.convert_tokens_to_ids(PLVR_FREE_START_TOKEN)
        model.config.free_end_id = processor.tokenizer.convert_tokens_to_ids(PLVR_FREE_END_TOKEN)
        model.config.tgt_start_id = processor.tokenizer.convert_tokens_to_ids(PLVR_TGT_START_TOKEN)
        model.config.tgt_end_id = processor.tokenizer.convert_tokens_to_ids(PLVR_TGT_END_TOKEN)
        model.config.free_id = processor.tokenizer.convert_tokens_to_ids(PLVR_FREE_TOKEN)
    else:
        model.config.free_id = None


    # there are some dummy tokens in newer hf version
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure lvr loss type
    model.config.loss_lvr_fct = training_args.loss_lvr_fct


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LVR
    '''
    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        if model_args.max_lvr_tokens is not None:
            data_module, total_data_len = make_packed_supervised_data_module_lvr_fixedToken(model_id=model_args.model_id,
                                                                                            processor=processor,
                                                                                            max_lvr_tokens=model_args.max_lvr_tokens,
                                                                                            data_args=data_args,
                                                                                            training_args=training_args,
                                                                                            latent_end_token=model_args.latent_end_token)
        else:
            data_module, total_data_len = make_packed_supervised_data_module_lvr(model_id=model_args.model_id,
                                                                                processor=processor,
                                                                                data_args=data_args,
                                                                                training_args=training_args,
                                                                                latent_end_token=model_args.latent_end_token)
        if not training_args.max_steps:
            training_args.max_steps = total_data_len // (training_args.gradient_accumulation_steps 
                                                         * training_args.world_size
                                                         * training_args.per_device_train_batch_size)
        # Very crucial or the packed data will get incorrectly sliced by the dataloader
        replace_train_dataloader()
    else:
        data_module = make_supervised_data_module_lvr(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              latent_end_token=model_args.latent_end_token)
    
    # tempFolder = temp_file class; "/path/to/local/cache/model_name/run_name-[random]"
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
