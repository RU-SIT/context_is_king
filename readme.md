# Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs

## Overview
This project is the code for the paper Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs on wireless communication modulation classification using machine learning approaches. It explores how providing statistical context about signals improves classification performance on various modulation types.

## Key Features
- Advanced ML-based modulation classification
- Statistical context enhancement for improved accuracy
- Support for multiple signal conditions (noiseless and noisy)
- Evaluation of different language models (DeepSeek, Qwen)

## Project Structure
```
context_is_king/
├── exp/                  # Experiment results
│   └── ...               # CSV files with model evaluations
├── notebooks/            # Jupyter notebooks
│   ├── data_generation.ipynb    # Dataset generation scripts
│   └── qwen.ipynb               # Qwen model implementation
└── src/                  # Source code
    └── ...               # Data processing and model modules
```

### Experiments
- Contains result files from model evaluations
- Results for different models (DeepSeek, Qwen) under various signal conditions
- Includes both noiseless and noisy signal test results

### Notebooks
- `data_generation.ipynb`: Scripts for generating signal datasets
- `qwen.ipynb`: Implementation of modulation classification using Qwen models

### Source Code
- Various modules for data processing, model implementation, and evaluation

## Data Structure
The project uses signal data stored as `.npy` files with the following categories:

| Modulation Types | Signal Variants |
|------------------|----------------|
| 4ASK             | Noiseless      |
| 4PAM             | Noisy          |
| 8ASK             |                |
| 16PAM            |                |
| CPFSK            |                |
| DQPSK            |                |
| GFSK             |                |
| GMSK             |                |
| OQPSK            |                |
| OOK              |                |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd context_is_king

# Install required packages
pip -r requirements.txt
or
conda env create -f environment.yml

```

## Usage

### Data Generation
```bash
jupyter notebook notebooks/data_generation.ipynb
```

### Evaluation and examples
```bash
jupyter notebook notebooks/qwen.ipynb
```

### Scripts
The main code used for the paper is in `src/`

### Analyzing Results
Results are stored in CSV format in the `exp/` directory and can be analyzed using standard data analysis tools.

## Requirements
- Python 3.x
- NumPy
- PyTorch
- Transformers library (for Qwen models)
- Jupyter for running notebooks

For more details, please refer to the notebooks and source code documentation.