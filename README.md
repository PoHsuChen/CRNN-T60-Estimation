# CRNN-T60-Estimation

[cite_start]The model is based on the paper: **"Online Blind Reverberation Time Estimation Using CRNNs"** (Interspeech 2020)[cite: 26, 28].

![Result Analysis](ace_boxplot_filtered.png)
*(Result on ACE Eval Dataset: Prediction Error vs Ground Truth T60)*

## 📌 Features
* **End-to-End Learning**: Directly estimates $T_{60}$ from raw waveforms using Gammatone-based features.
* [cite_start]**Hybrid Architecture**: Combines CNN for feature extraction and LSTM for temporal context modeling[cite: 36].
* **Robustness**: Optimized training strategy with data augmentation (random cropping, noise injection) to handle variable input lengths.

## 🚀 Performance
[cite_start]Evaluation on the **ACE Challenge Eval Dataset** (Speech > 4s)[cite: 37]:

| Metric | Our Result | [cite_start]Paper Baseline [cite: 164] |
| :--- | :--- | :--- |
| **MSE** | **0.0355** | 0.0375 |
| **Bias** | 0.0417 | 0.1163 |
| **PCC** | 0.835 | 0.900 |

> Note: The model was trained purely on **synthetic RIRs** generated using the Image Method, demonstrating strong generalization to real-world recordings (ACE dataset).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PoHsuChen/CRNN-T60-Estimation.git
   cd CRNN-T60-Estimation
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## 📂 Dataset Preparation
This implementation uses the ACE Challenge Dataset for evaluation.
* Training: Synthetic RIRs convolved with LibriSpeech (simulated dynamically or pre-generated).
* Testing: ACE Eval Dataset
Please update the paths in utils.py:
    TRAIN_CSV_PATH = "./path/to/your/synthetic/metadata.csv"
    TEST_ROOT_DIR = "./path/to/ACE_Eval_dataset"

## ▶️ Usage
Training
```python
    python main.py
```


## 📜 Citation
If you find this code useful, please cite the original paper:
```bibtex
@inproceedings{deng2020online,
  title={Online Blind Reverberation Time Estimation Using CRNNs},
  author={Deng, Shuwen and Mack, Wolfgang and Habets, Emanuel AP},
  booktitle={Proc. Interspeech 2020},
  pages={5061--5065},
  year={2020}
}
```