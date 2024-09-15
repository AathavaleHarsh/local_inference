# Local Model Inference Speed Measurement

## Project Overview

This Python project measures the local inference speed of machine learning models executed using the **Ollama** framework. The script runs a model via a subprocess, counts the number of tokens processed in real-time, and calculates the speed of inference in terms of tokens per second.

This tool is particularly useful for those who want to benchmark different models, analyze their performance locally, and log the results for further inspection.

## Features

- **Real-Time Token Counting**: The script counts tokens in real-time while the model processes the input.
- **Inference Speed Calculation**: It computes the speed of inference (tokens per second) to help benchmark different models.
- **Customizable**: Easily adaptable for different models and prompts.
- **Optional Resource Monitoring**: (Disabled by default) Tracks CPU and memory usage during inference.
- **Error Handling**: Graceful handling of unexpected encoding issues and other subprocess-related errors.
- **Logging**: Logs the inference summary (total tokens, inference time, and speed) to a file for future reference.

## Inference Summaries

### Example 1:
- **Model**: `gemma2:2b`
- **Total Tokens**: 14,082
- **Inference Time**: 28.14 seconds
- **Inference Speed**: 500.39 tokens/second

### Example 2:
- **Model**: `llama3.1:8b`
- **Total Tokens**: 62,129
- **Inference Time**: 564.83 seconds
- **Inference Speed**: 110.00 tokens/second

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
