import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import get_cosine_schedule_with_warmup

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import get_current_device, print_rank_0
from colossalai.zero import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.cluster import DistCoordinator
import torch.distributed as dist
from tqdm import tqdm
from statistics import mean
import GPUtil
import psutil
from pynvml import *


def train():
    # ======== Due to experiments aren't related to distribution training ============
    # ======== Therefore, below code could be disabled =========
    # Launch ColossalAI
    # colossalai.launch_from_torch(config={})
    # coordinator = DistCoordinator()
    # world_size = coordinator.world_size

    # =========== This is a parser = ==========
    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments)
    # )
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ========== logger related, disabled ================
    # Manage loggers
    # disable_existing_loggers()
    # logger = get_dist_logger()

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    total_nvml = int(info.total / (1024 * 1024))
    used_nvml = int(info.used / (1024 * 1024))
    # print_rank_0("[0]Total nvml GPU mem: {}".format(total_nvml))
    # print_rank_0("[0]Used nvml GPU mem: {}".format(used_nvml))
    cuda_context_mem = used_nvml - GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    framework_initial_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # print_rank_0("[0]Used GPUtil GPU mem: {}".format(framework_initial_mem))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    ).to("cuda")
    model.half()  # cf. PRECISION_STR_TO_DTYPE = {'fp16': torch.half, 'bf16': torch.bfloat16}
    torch.cuda.empty_cache()

    model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.eos_token is None:  # for bert
        tokenizer.eos_token = tokenizer.pad_token  #

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Set plugin
    booster_kwargs = {}
    # plugin = GeminiPlugin(
    #     device=get_current_device(),
    #     placement_policy="cuda",
    #     precision="fp16",
    #     pin_memory=False,
    #     strict_ddp_mode=False,
    #     initial_scale=2**5,
    # )

    config = {
        "batch_size": training_args.per_device_train_batch_size,
        "lr": training_args.learning_rate,
        "epochs": int(training_args.num_train_epochs),
        "warmup_ratio": training_args.warmup_ratio,
        "weight_decay": training_args.weight_decay,
    }

    dataloader = plugin.prepare_dataloader(
        data_module["train_dataset"],
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        collate_fn=data_module["data_collator"],
    )

    for batch in dataloader:
        if batch["input_ids"].size()[1] == max_seq_len:
            test_long_input = move_to_cuda(batch, torch.cuda.current_device())
            break

    # Set lr scheduler
    total_steps = len(dataloader) * config["epochs"]
    num_warmup_steps = int(config["warmup_ratio"] * total_steps)

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config["epochs"],
    )

    # ############################## 1 ##############################
    # lm_fp32 = False
    # if ('codegen' in model_args.model_name_or_path) or ('neo' in model_args.model_name_or_path):
    #     lm_fp32 = True
    # real_bs = 0 # For batch size search mode
    # # real_bs = test_long_input["input_ids"].size()[0] # To estimate with specific batch size
    # se = SizeEstimator(model, test_long_input["input_ids"][0:2], real_bs, bytes=2, bytes_input=8,
    #                    gpu_n=world_size, tp=0, lm_fp32=lm_fp32, m_total=total_nvml)
    # torch.cuda.empty_cache()
    # prev_get_output = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # se.get_output_sizes()
    # torch.cuda.empty_cache()
    # after_get_output = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # ###############################################################

    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)
    torch.cuda.empty_cache()

    # ############################## 2 ##############################
    # booster_chunk_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # m_pbase = booster_chunk_mem + cuda_context_mem - (after_get_output - prev_get_output)
    # print_rank_0('[m_pbase]: {}'.format(m_pbase))
    # esti_mem, real_bs = se.estimate_size(m_init=m_pbase)
    # print_rank_0('Estimated memory: {0}, real bs: {1}'.format(esti_mem, real_bs))
    # import sys
    # sys.exit()
    # ###############################################################

    # with open(txt_file_name, 'a') as f:
    #     f.write('[cuda:{}] Before fine-tuning: {} -> {} -> {}\n'.format(dist.get_rank(),
    #                         framework_initial_mem, load_model_mem, booster_chunk_mem))

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config["epochs"]):
        train_epoch(
            epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator
        )

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    # booster.save_model(model, training_args.output_dir)
    # logger.info(f"Saving model checkpoint to {training_args.output_dir}")