
# 🤖 deepseek-math-finetune

This project fine-tunes the **`deepseek-coder-1.3b-instruct`** model on a mathematical dataset using a low-rank adaptation (**LoRA**) technique. The goal is to improve the model's ability to solve mathematical problems and demonstrate a complete, reproducible fine-tuning workflow on a consumer-grade GPU.

## ✨ Features

  * **Efficient Fine-tuning:** Utilizes **QLoRA (Quantized LoRA)** to train a 1.3 billion parameter model on a single GPU with limited VRAM.
  * **Dataset:** Employs the **`davidheineman/deepmind-math-large`** dataset for a robust mathematical foundation.
  * **Automatic Resumption:** The training script automatically detects and resumes from the latest saved checkpoint, allowing for interrupted training runs.
  * **Reproducibility:** A self-contained Python script to handle data loading, tokenization, model setup, and training.
  * **Comprehensive Logging:** Integrates with **Weights & Biases** for detailed tracking of training progress, metrics, and model performance.

-----

## 🚀 How to Run

### 1\. Prerequisites

You'll need a system with a CUDA-enabled GPU and the necessary drivers installed. This script was tested on a GPU with at least 8GB of VRAM.

First, install the required Python libraries:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file is not included in the provided code, but you can create one by including the necessary libraries:

```
torch
transformers
peft
accelerate
datasets
wandb
bitsandbytes
```

-----

### 2\. Set Up Hugging Face and Weights & Biases Tokens

The script requires your Hugging Face and Weights & Biases API tokens to be set as environment variables.

You can set these tokens directly in the script for simplicity, or use `export` commands for a more secure approach:

```bash
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

> ⚠️ **Warning:** The provided code hardcodes the tokens, which is **not recommended** for production or public repositories. It's best to use environment variables to keep your tokens secure.

-----

### 3\. Run the Script

Simply execute the Python script:

```bash
python finetune_math.py
```

The script will handle the entire workflow, from downloading the dataset and the base model to starting the training process. It will automatically log its progress to your Weights & Biases dashboard.

-----

### 4\. Training Details

  * **Model:** `deepseek-coder-1.3b-instruct`
  * **Training Method:** QLoRA
  * **Batch Size:** 1 (with a gradient accumulation of 8)
  * **Epochs:** 1
  * **Learning Rate:** `1e-4`
  * **Dataset Split:** The script uses a small 0.5% test split for evaluation to ensure fast feedback.
  * **Checkpointing:** The model is saved every 245 steps, and the best model based on the `exact_match` metric is loaded at the end of training.

### 5\. Expected Output

The script will print progress to the console, including:

  * The number of training and test examples.
  * An example of the formatted training data.
  * A summary of trainable parameters.
  * Training and evaluation metrics reported by the Hugging Face `Trainer`.
  * A final evaluation of the best model.
  * An inference test with a sample question, showing the fine-tuned model's output.

-----

## 📊 Training Graphs

You can monitor the training progress and performance on Weights & Biases using the following links:

  * **Train vs. Loss:** [https://wandb.ai/hassanshaikh-using-id-iit-bombay/huggingface/runs/lexi48ea?nw=nwuserhassanshaikhusingid\&panelDisplayName=train%2Floss\&panelSectionName=train](https://wandb.ai/hassanshaikh-using-id-iit-bombay/huggingface/runs/lexi48ea?nw=nwuserhassanshaikhusingid&panelDisplayName=train%2Floss&panelSectionName=train)
  * **Train vs. Grad Norm:** [https://wandb.ai/hassanshaikh-using-id-iit-bombay/huggingface/runs/lexi48ea?nw=nwuserhassanshaikhusingid\&panelDisplayName=train%2Fgrad\_norm\&panelSectionName=train](https://wandb.ai/hassanshaikh-using-id-iit-bombay/huggingface/runs/lexi48ea?nw=nwuserhassanshaikhusingid&panelDisplayName=train%2Fgrad_norm&panelSectionName=train)

-----

## 6\. Fine-tuned Model

After the training is complete, the fine-tuned adapter weights will be saved in the `./deepseek_mathpatch` directory. You can then use these weights to load the model for inference without needing to retrain.

```python
# Example of loading the fine-tuned model for inference
from transformers import AutoTokenizer, AutoPeftModelForCausalLM

model_path = "./deepseek_mathpatch/checkpoint-XXXX" # Replace with the latest checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoPeftModelForCausalLM.from_pretrained(model_path)

# You can now use this 'model' for inference
```
