# SwiftKV Checkpoint Loading Guide

This guide explains how to load and resume training from SwiftKV checkpoints.

## Overview

The SwiftKV training script now supports loading models from previously saved checkpoints. This allows you to:

1. **Resume interrupted training**: Continue training from where you left off
2. **Fine-tune trained models**: Load a trained SwiftKV model and continue training with different data or hyperparameters
3. **Transfer learning**: Use a checkpoint from one task as initialization for another

## How It Works

### Initial Training (From Teacher Model)

When training SwiftKV from scratch, you provide a path to a teacher model (e.g., a base Qwen3-8B model):

```yaml
model:
  name_or_path: Qwen/Qwen3-8B  # Teacher model
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
```

The training script will:
1. Load the teacher model
2. Freeze all teacher parameters
3. Initialize SwiftKV student parameters from the corresponding teacher layers
4. Train only the SwiftKV parameters

### Resuming from Checkpoint

When resuming from a SwiftKV checkpoint, you provide a path to the saved checkpoint:

```yaml
model:
  name_or_path: /path/to/checkpoint/qwen3-swiftkv-8b-6-consumers/global_step_17205
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
```

The training script will:
1. **Auto-detect** that this is a SwiftKV checkpoint (based on config.json and model files)
2. Load the model architecture
3. Skip parameter initialization (to avoid overwriting trained parameters)
4. Load the trained SwiftKV parameters from the checkpoint
5. Set the correct `requires_grad` flags
6. Continue training

## Configuration Options

### Automatic Detection

By default, the system auto-detects whether `name_or_path` points to a SwiftKV checkpoint by checking:
- Presence of model weight files (model.safetensors, pytorch_model.bin, etc.)
- SwiftKV-specific attributes in config.json (kv_sharing_map, model_type containing "swiftkv")

### Explicit Configuration

You can explicitly indicate checkpoint loading:

```yaml
model:
  name_or_path: /path/to/checkpoint
  resume_from_checkpoint: true  # Explicitly indicate this is a checkpoint
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
```

### Auto-Resume with DeepSpeed

For automatic checkpoint resumption (e.g., after interruptions), configure the checkpoint engine:

```yaml
checkpoint:
  - type: huggingface  # or deepspeed
    save_every_n_epochs: 1
    output_dir: ./checkpoint/my-model
    auto_resume: true  # Automatically resume from latest checkpoint
```

## Complete Examples

### Example 1: Initial Training

```yaml
type: swiftkv
code: ../train.py
model:
  name_or_path: Qwen/Qwen3-8B
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
checkpoint:
  - type: huggingface
    output_dir: ./checkpoint/qwen3-swiftkv-initial
    save_every_n_epochs: 1
```

### Example 2: Resume from Specific Checkpoint

```yaml
type: swiftkv
code: ../train.py
model:
  name_or_path: ./checkpoint/qwen3-swiftkv-initial/global_step_5000
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
checkpoint:
  - type: huggingface
    output_dir: ./checkpoint/qwen3-swiftkv-continued
    save_every_n_epochs: 1
```

### Example 3: Auto-Resume Training

```yaml
type: swiftkv
code: ../train.py
model:
  name_or_path: Qwen/Qwen3-8B
  num_key_value_layers: 30
  kv_sharing_map: {12: 11, 16: 15, 20: 19, 24: 23, 28: 27, 32: 31}
checkpoint:
  - type: deepspeed
    output_dir: ./checkpoint/qwen3-swiftkv-auto
    save_every_n_steps: 500
    auto_resume: true  # Will automatically load latest checkpoint if available
```

## Important Notes

1. **Checkpoint Compatibility**: Ensure the `kv_sharing_map` in your config matches the one used to train the checkpoint. Mismatched configurations will cause errors.

2. **Teacher vs Checkpoint**: The system distinguishes between:
   - **Teacher model** (HuggingFace model hub or base model): Used for initial training
   - **SwiftKV checkpoint** (saved during training): Used for resumption

3. **Parameter Preservation**: When loading from a checkpoint:
   - Trained SwiftKV parameters are loaded from the checkpoint
   - Teacher parameters remain frozen
   - Only SwiftKV parameters have `requires_grad=True`

4. **DeepSpeed Checkpoints**: When using DeepSpeed checkpoint engine with ZeRO optimization, the checkpoint includes:
   - Model parameters
   - Optimizer states
   - Learning rate scheduler state
   - Training step counter
   - Random number generator states

5. **HuggingFace Checkpoints**: When using HuggingFace checkpoint engine:
   - Only model parameters are saved
   - Optimizer and scheduler states are not preserved
   - Good for model distribution but not for exact training resumption

## Troubleshooting

### "No trainable parameters found"

If you see this error when loading from checkpoint:
- Check that `kv_sharing_map` matches the checkpoint's configuration
- Verify the checkpoint directory contains valid model files
- Try setting `resume_from_checkpoint: true` explicitly

### Parameters are reinitialized instead of loaded

If parameters seem to reset when loading a checkpoint:
- Ensure `name_or_path` points to the correct checkpoint directory
- Check that the directory contains model.safetensors or pytorch_model.bin
- Verify config.json contains SwiftKV-specific attributes

### Different results after resuming

If training behavior differs after resuming:
- Use DeepSpeed checkpoint engine with `auto_resume: true` for exact resumption
- HuggingFace checkpoints don't save optimizer state, so momentum etc. will reset
- Verify you're using the same hyperparameters (learning rate, batch size, etc.)

## Checkpoint Structure

A valid SwiftKV checkpoint directory should contain:

```
checkpoint_dir/
├── config.json              # Model configuration with kv_sharing_map
├── model.safetensors       # Model weights (preferred format)
├── pytorch_model.bin       # Alternative: PyTorch weights
└── training_args.bin       # Optional: Training arguments
```

For sharded models:
```
checkpoint_dir/
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
└── model-00004-of-00004.safetensors
```

## See Also

- Example config: `configs/qwen3-swiftkv-8b-resume.yaml`
- Training script: `train.py`
- ArcticTraining checkpoint documentation: `../../docs/checkpoint.rst`


