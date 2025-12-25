# Cognitive Balance in Large Language Model Collaboration: Quantification and Debate-Based Unlearning

This repository contains the implementation for the paper **"Cognitive Balance in Large Language Model Collaboration: Quantification and Debate-Based Unlearning"**. The research explores multi-agent debate frameworks to quantify cognitive orientations in LLMs and employs debate-based methods to calibrate and unlearn biases. The current version is a **demo release**. The complete codebase will be made publicly available upon official acceptance of the paper.

## 📂 Repository Structure

```
.
├── data                      # Four datasets, including MMLU, TruthfulQA, MedMCQA, Scalr
├── math_equivalence.py          # Utility for comparing mathematical expressions
├── main.py                      # Main debate simulation script
├── orientation_scoring_function.py  # BERT-based orientation scoring model
├── calibrate_orientation.py     # Probe-based calibration of model orientation
├── evaluate.py                  # Evaluation script for debate results
├── math_parsing.py             # Parsing utilities for math expressions
├── dataloader.py               # Dataset loaders for all supported tasks
├── commons.py                  # Common utilities (querying, parsing)
└── README.md                   # This file
```

## 🧠 Key Features

- **Multi-Agent Debate**: Simulates `a` agents debating over `t` rounds to reach consensus or calibrated answers.
- **Cognitive Orientation Scoring**: Uses a fine-tuned BERT-based model to score textual responses on a political/ideological spectrum.
- **Dataset Support**: Works with 4 datasets across reasoning, knowledge, and ethics domains.
- **Evaluation Suite**: Includes multiple evaluation metrics (accuracy, agreement, persuasiveness, majority voting, judge-based decisions).
- **Modular Design**: Clean separation of data loading, debate simulation, evaluation, and orientation calibration.

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mangoding71/Cognitive-Balance-in-Large-Language-Model-Collaboration-Quantification-and-Debate-Based-Unlearning.git
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- OpenAI Python SDK
- scikit-learn
- pandas
- tqdm

## 🚀 Usage

### Run Multi-Agent Debate

```bash
python main.py 
```

### Evaluate Debate Results

```bash
python evaluate.py
```

## 📈 Evaluation Metrics

- **Agent Accuracy**: Per-agent, per-turn correctness
- **Majority Voting**: Consensus-based decision accuracy
- **Judge Voting**: LLM-as-a-judge evaluation
- **Persuasiveness**: Influence of agents across rounds
- **RVS (Response Variance Score)**: Measures orientation consistency

## 📄 Citation

If you use this code in your research, please cite our paper (To be continued):
```bibtex

```

## 📬 Contact

For questions or collaborations, please open an issue or contact me with dinghongli@nudt.edu.cn.
