"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...

    
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# === Local GPU ===
# Note: If you want to run on local CPU, comment out the following lines.in terminal, and run following command on terminal 
'''
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
'''
# import torch.distributed as dist

# dist.init_process_group(
#     backend='nccl',  # or 'gloo' for CPU
#     init_method='env://'
# )
# # Sane Defaults
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# === Local GPU ===


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("/scratch/work/zhangy50/RL/spot_VLA/dataset/tensorflow_datasets/")        # Path to Open-X dataset directory
    dataset_name: str = "spot_kitchen"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                              # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16  # 16 is good, 24 to big for H100                                        # Fine-tuning batch size
    max_steps: int = 10_000 # 10_000                                        # Max number of fine-tuning steps
    save_steps: int = 2000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 4   # or 4?                             # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000 # 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # Validation
    validation_interval = 100       # Validate every 500 optimizer steps
    patience = 5                    # Early stopping after 3 validation checks


    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 64 # 64 is good                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla_spot"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "yibo-zhang"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


def evaluate(model, dataloader, device_id, action_tokenizer):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_l1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, model.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            
            num_samples = len(batch["input_ids"])

            # Accumulate
            total_loss += loss * num_samples
            total_accuracy += action_accuracy * num_samples
            total_l1 += action_l1_loss * num_samples
            total_samples += num_samples

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_accuracy / total_samples,
        "l1_loss": total_l1 / total_samples,
    }

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+shf{cfg.shuffle_buffer_size}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    print(distributed_state)
    # if distributed_state:
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    '''

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    '''

    # Create train and validation datasets
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True
    )

    val_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=100,  # No shuffling for validation
        image_aug=False,  # Typically no augmentation for validation
        train=False
    )

    # Save dataset statistics (only need to do this once, using train stats)
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator (shared between train and val)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, 
        processor.tokenizer.pad_token_id, 
        padding_side="right"
    )

    # Train DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        # shuffle=True  # Shuffle at DataLoader level if needed
    )

    # Validation DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,  # Can use different batch size if needed
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        shuffle=False  # Never shuffle validation
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    def action_eval(output):
        # Compute Accuracy and L1 Loss for Logging
        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
        )
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
        return action_accuracy, action_l1_loss
    
    def save_model():
        if distributed_state.is_main_process:
            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

            # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
            save_dir = adapter_dir if cfg.use_lora else run_dir

            # Save Processor & Weights
            processor.save_pretrained(run_dir)
            vla.module.save_pretrained(save_dir)

        # Wait for processor and adapter weights to be saved by main process
        dist.barrier()

        # Merge LoRA weights into model backbone for faster inference
        #   =>> Note that merging is slow and can be done post-hoc to speed up training
        if cfg.use_lora:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
            merged_vla = merged_vla.merge_and_unload()
            if distributed_state.is_main_process:
                if cfg.save_latest_checkpoint_only:
                    # Overwrite latest checkpoint
                    merged_vla.save_pretrained(run_dir)

                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                else:
                    # Prepare to save checkpoint in new directory
                    checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt--loss-{val_loss_value:.3f}")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # Save dataset statistics to new directory
                    save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)

                    # Save processor and model weights to new directory
                    processor.save_pretrained(checkpoint_dir)
                    merged_vla.save_pretrained(checkpoint_dir)


                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

        # Block on Main Process Checkpointing
        dist.barrier()
    
    best_val_loss = float('inf')

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_accuracy, action_l1_loss = action_eval(output)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)  
            # smoothen, recent losses is defined as deque, no need to manually deque.  

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )


            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

                # ----- Validation & early stopping ----- 

                if (gradient_step_idx + 1) % cfg.validation_interval == 0:

                    # val_loss, val_accuracy, val_l1 = evaluate(vla, val_dataloader, device_id, action_tokenizer)
                    vla.eval()
                    total_loss = 0.0
                    total_accuracy = 0.0
                    total_l1 = 0.0
                    total_samples = 0

                    with torch.no_grad():
                        for batch in val_dataloader:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output: CausalLMOutputWithPast = vla(
                                    input_ids=batch["input_ids"].to(device_id),
                                    attention_mask=batch["attention_mask"].to(device_id),
                                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                                    labels=batch["labels"],
                                )
                                loss = output.loss


                            # Compute Accuracy and L1 Loss for Logging
                            action_accuracy, action_l1_loss = action_eval(output)
                            
                            num_samples = len(batch["input_ids"])

                            # Accumulate
                            total_loss += loss * num_samples
                            total_accuracy += action_accuracy * num_samples
                            total_l1 += action_l1_loss * num_samples
                            total_samples += num_samples

                    
                    val_loss = total_loss / total_samples,
                    val_accuracy = total_accuracy / total_samples,
                    val_l1 = total_l1 / total_samples,
                    
                    # Log validation metrics
                    if distributed_state.is_main_process:
                        wandb.log({
                            "val_loss": val_loss,
                            "val_action_accuracy": val_accuracy,
                            "val_l1_loss": val_l1,
                        }, step=gradient_step_idx)

                    # Early Stopping Check
                    val_loss_value = val_loss[0].item()
                    print('val_loss:', val_loss_value, 'best_val_loss:', best_val_loss)
                    if val_loss_value < best_val_loss:
                        best_val_loss = val_loss_value
                        patience_counter = 0
                        
                        # Save best model checkpoint
                        if distributed_state.is_main_process:
                            print(f"New best validation loss: {best_val_loss:.4f}")
                            processor.save_pretrained(run_dir)
                            vla.module.save_pretrained(run_dir)
                    else:
                        patience_counter += 1
                        cfg.learning_rate *= 0.5
                        validation_interval = int(validation_interval * 0.7 // 1)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = cfg.learning_rate
                        print(f"Updated learning rate to {cfg.learning_rate:.6f}")
                        wandb.log({"learning_rate": cfg.learning_rate}, step=gradient_step_idx)
                        if distributed_state.is_main_process:   
                            print(f"Current best validation loss: {best_val_loss:.4f}, patience counter: {patience_counter}/{cfg.patience}")

                        if patience_counter < cfg.patience:
                            print(f"Validation loss did not improve ({patience_counter}/{cfg.patience}), continuing training...")
                        else:
                            if distributed_state.is_main_process:
                                print(f"Validation loss did not improve ({patience_counter}/{cfg.patience}), stopping training...")
                            # Save the model checkpoint before stopping
                            save_model()
                            break
                        

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                save_model()


if __name__ == "__main__":
    finetune()
