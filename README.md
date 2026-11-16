# Tenglish Intent Classifier – Fine-Tuning an LLM with LoRA

An LLM fine-tuning project focused on classifying **Tenglish (Telugu written using English characters)** messages into predefined intent categories using Parameter-Efficient Fine-Tuning (PEFT).

## Overview

People often communicate in Tenglish by mixing Telugu words with English characters. For example:

- "Naaku notes pampinchu"
- "Ela unnava?"
- "Ee problem ela solve cheyali?"
- "Naku login avvatledu"

Understanding informal, code-mixed text can be challenging because of spelling variations, sentence structures, and conversational expressions.

This project explores whether fine-tuning a pretrained Large Language Model can improve the consistency of intent classification for Tenglish messages.

## Project Objective

Fine-tune a small pretrained LLM to identify the intent behind a Tenglish message and return exactly one predefined label.

### Intent Categories

The model is trained to recognize five categories:

| Intent | Description | Example |
|---|---|---|
| `greeting` | Greetings and casual conversations | "Ela unnava?" |
| `query` | Asking questions or seeking information | "Exam eppudu undi?" |
| `complaint` | Expressing a problem or dissatisfaction | "App open avvatledu" |
| `request` | Asking someone to perform an action | "Naaku notes pampinchu" |
| `thanks` | Expressing gratitude | "Chala thanks" |

## Model and Fine-Tuning Approach

Instead of training an LLM from scratch, this project uses a pretrained model and adapts it to a specific classification task.

**Base Model:** Qwen2.5-1.5B-Instruct

**Fine-Tuning Technique:** LoRA (Low-Rank Adaptation)

**Quantization:** 4-bit quantization

### Why LoRA?

Full fine-tuning updates all model parameters, which requires significant GPU memory and computational resources.

LoRA introduces small trainable matrices into selected model layers while keeping the original model weights frozen.

This makes fine-tuning more memory-efficient and practical on limited GPU resources.

### Implementation Workflow

1. Prepare a custom Tenglish intent classification dataset in JSONL format.
2. Load and preprocess the dataset using Pandas.
3. Remove duplicate records and inspect class distribution.
4. Split the dataset into stratified training and testing sets.
5. Load Qwen2.5-1.5B-Instruct in 4-bit quantized format.
6. Apply LoRA adapters to selected attention and MLP projection layers.
7. Format examples using the model's chat template.
8. Fine-tune the model using Supervised Fine-Tuning (SFT).
9. Compare the pretrained model and fine-tuned model.
10. Evaluate predictions and inspect misclassified examples.

## Dataset Format

The dataset is stored in JSONL format.

Each record contains a text message and its corresponding intent label.

```json
{"text": "Naaku notes pampinchu", "label": "request"}
{"text": "Ela unnava?", "label": "greeting"}
{"text": "Exam eppudu undi?", "label": "query"}
{"text": "App open avvatledu", "label": "complaint"}
{"text": "Chala thanks", "label": "thanks"}
```

The quality and diversity of the dataset are important for improving classification performance.

## Tech Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Base Model:** Qwen2.5-1.5B-Instruct
- **Model Fine-Tuning:** LoRA / PEFT
- **Fine-Tuning Framework:** Unsloth, Hugging Face TRL
- **Model Libraries:** Transformers, PEFT
- **Dataset Processing:** Pandas
- **Evaluation:** Scikit-learn
- **Training Environment:** Google Colab

## Training Configuration

The notebook uses the following training configuration:

| Parameter | Value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit |
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 16 |
| Maximum Sequence Length | 256 |
| Training Batch Size | 4 |
| Gradient Accumulation Steps | 2 |
| Epochs | 3 |
| Learning Rate | 0.0002 |
| Optimizer | AdamW 8-bit |

These are the initial experimental settings and can be adjusted through further experimentation.

## Model Evaluation

The project compares the pretrained model against the fine-tuned model using the same test examples.

Evaluation metrics include:

- **Accuracy:** Overall percentage of correctly classified messages.
- **Precision:** How many predictions for a particular intent are correct.
- **Recall:** How many actual examples of an intent are correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Classification Report:** Detailed performance across all intent categories.
- **Error Analysis:** Inspection of incorrectly classified messages.

### Results

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Accuracy | To be measured | To be measured |
| Precision | To be measured | To be measured |
| Recall | To be measured | To be measured |
| F1-Score | To be measured | To be measured |

*Update this table with the actual evaluation results after running the notebook.*

## How to Run

The project is designed to run in Google Colab.

1. Clone or download this repository.
2. Open `tenglish_finetune.ipynb` in Google Colab.
3. Enable a GPU runtime if available.
4. Upload your `tenglish_intents.jsonl` dataset when prompted.
5. Execute the notebook cells sequentially.
6. Observe the baseline and fine-tuned model evaluations.
7. Save and download the trained LoRA adapter.

### Install Dependencies

The notebook installs the required libraries, including:

```bash
pip install unsloth
```

Other dependencies used in the notebook include:

```bash
pip install transformers datasets trl peft
pip install pandas scikit-learn
```

*Use the notebook's installation cell as the source of truth for compatible package versions.*

## Project Structure

```text
Tenglish-Intent-Classifier/
│
├── tenglish_finetune.ipynb
│
├── tenglish_intents.jsonl
│
├── tenglish_lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── README.md
│
└── requirements.txt
```

The `tenglish_lora/` directory is generated after saving the trained adapter. Include it in the repository only if you intend to distribute the model weights.

## Key Learnings

Through this project, I explored:

- Fine-tuning pretrained Large Language Models.
- Parameter-Efficient Fine-Tuning using LoRA.
- 4-bit quantization and memory-efficient model loading.
- Supervised Fine-Tuning using TRL.
- Preparing custom datasets for instruction tuning.
- Chat templates and response-only loss masking.
- Evaluating LLMs using classification metrics.
- Comparing pretrained and fine-tuned model behaviour.

## Future Improvements

- Expand the dataset with more diverse Tenglish expressions.
- Include different spelling variations and informal conversational patterns.
- Improve performance on ambiguous and mixed-intent messages.
- Experiment with different LoRA configurations.
- Compare multiple small open-source LLMs.
- Build a FastAPI endpoint for real-time intent prediction.
- Develop a simple web interface for testing Tenglish messages.

## Author

**K. Uday Kiran Achari**

Computer Science and Engineering Student | AI/ML & LLM Enthusiast

