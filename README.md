# Finetuning and Improving Prediction Results of LLMs Using Synthetic Data

Welcome to the repository for **"Finetuning and Improving Prediction Results of LLMs Using Synthetic Data"**. This repository primarily showcases the results of my bachelor’s thesis research, where I evaluated the impact of finetuning large language models (LLMs) for sustainability-focused conversational AI applications.

If you are interested in reading the entire thesis, you can access it [here (PDF)](https://www.theseus.fi/bitstream/handle/10024/860931/Macias_Melany.pdf?sequence=2&isAllowed=y).

## Overview

This project demonstrates the effectiveness of finetuning open-source LLMs using a synthetic dataset generated from sustainability-related documents. The focus is on presenting the results of this study, including performance metrics and comparative analyses. Additionally, the repository includes scripts for replicating the experiments or adapting the methods for personal use.

### Key Achievements

- Designed and implemented a **synthetic dataset** tailored for sustainability-related tasks.
- Finetuned and evaluated four models using state-of-the-art **performance metrics**.
- Delivered insights into resource-performance trade-offs in **LLM training**.
- Developed open-source scripts to enable easy **reproducibility** and **adaptation** for other researchers.

## Repository Structure

```plaintext
.
├── comparative_analysis
│   ├── Scripts and analyses for model comparison and error analysis
├── data
│   ├── Preprocessed datasets, sustainability PDFs, and Q&A pairs
├── evaluation
│   ├── Scripts for automated evaluation using BLEU, ROUGE, and METEOR
│   ├── Results and logs for evaluated models
├── results_tests
│   ├── Test cases and evaluation outputs
├── scripts
│   ├── batch
│   ├── training                # Scripts for finetuning models
│   ├── utilities               # Helper scripts for data preparation
├── .gitattributes
├── .gitignore
└── requirements.txt
```

## Main Results

This study evaluated the performance of four models—**Gemma-2B**, **Gemma-7B**, **Phi-2 (2.7B)**, and **Llama-3 (8B)**—on sustainability-related content. Below is a summary of the results:

### Highlights

- **Best Performing Model**: Gemma-7B achieved the highest METEOR score of **0.25**.
- **Training Efficiency**: Finetuning times ranged from ~3 hours for smaller models (Gemma-2B, Phi-2) to ~8.5 hours for larger ones (Llama-3).
- **Impact**: Finetuning consistently improved performance across all models, as measured by BLEU, ROUGE, and METEOR scores.

### Metrics Comparison

| Model         | BLEU  | ROUGE | METEOR | Training Time |
|---------------|-------|-------|--------|---------------|
| Phi-2 (2.7B)  | 0.06  | 0.18  | 0.23   | 03:37         |
| Gemma-2B      | 0.06  | 0.16  | 0.22   | 03:12         |
| Gemma-7B      | 0.09  | 0.19  | 0.25   | 07:17         |
| Llama-3 (8B)  | 0.05  | 0.15  | 0.20   | 08:25         |

For detailed evaluation results and logs, see the [evaluation directory](./evaluation).

## Using This Repository

This repository is structured to allow users to explore the results of this study and, optionally, replicate the finetuning and evaluation processes.

### Evaluation Scripts

The evaluation scripts are located in the [Finetuning-LLMs-for-Chatbot/evaluation](./evaluation) directory. These scripts use BLEU, ROUGE, and METEOR metrics to assess model performance. To evaluate a model:

1. Ensure you have Python 3.8+ installed and the required dependencies:
   
```bash
   pip install -r requirements.txt
 ```

2. Use the provided script to run evaluations.

```bash
python evaluation_script.py --model_path <path_to_model> --data_path <path_to_test_data>
```

### Training Scripts
For those interested in replicating the finetuning process or training their own models, the training scripts are available in the Finetuning-LLMs-for-Chatbot/scripts/training directory. The scripts are designed for compatibility with Hugging Face's transformers library.

Prepare your dataset in the required JSON format.

Run the training script:

```bash
python training_gemma_7b.py --data_path <path_to_training_data> --output_dir <output_directory>
```

## Data Preparation

The dataset used in this study, including synthetic Q&A pairs related to sustainability, is available on Hugging Face: [Sustainability Q&A Dataset](https://huggingface.co/datasets/nessa01macias/sustainability_qa) 
## Finetuned Models

The finetuned versions of the evaluated models are available on Hugging Face:

* [Gemma-2B](https://huggingface.co/nessa01macias/gemma-2b_sustainability-qa)
* [Gemma-7B](https://huggingface.co/nessa01macias/gemma-7b_sustainability-qa)
* [Phi-2 (2.7B)](https://huggingface.co/nessa01macias/phi-2_sustainability-qa)
* [Llama-3 (8B)](https://huggingface.co/nessa01macias/llama3-8b_sustainability-qa-ins)
 
## Key Findings and Discussion
The study revealed the following insights:
* Effectiveness of Finetuning: Even small models like Gemma-2B show significant improvements after finetuning.
* Resource-Performance Tradeoff: Larger models like Llama-3 offer slight performance improvements but demand higher computational resources.
For detailed insights, please refer to the comparative analysis results.

## Acknowledgments
This thesis was made possible through the support of Metropolia University of Applied Sciences, with guidance from supervisors Sakari Lukkarinen and Mika Hämäläinen.

