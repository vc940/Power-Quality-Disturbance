# ⚡ Power Quality Disturbance Detection using Transformers

This project focuses on accurately classifying power quality disturbances in electrical signals using Transformer-based deep learning models. The model identifies 17 types of power quality disturbances by learning from waveform data and frequency-domain features.

---

## 🚀 Project Highlights

- 🧠 **Model**: Transformer-based neural network with attention mechanisms.
- 📊 **Accuracy**: Achieved **91% test accuracy** on a benchmark dataset.
- 📈 **Enhanced Representation**: Used **FFT-based feature engineering** to improve harmonic disturbance detection by 4–5%.
- 🧪 **Simulation**: Generated and validated power quality faults (e.g., sags, swells, harmonics) using **MATLAB**.
- 📡 **Applications**: Power quality monitoring, smart grid diagnostics, fault classification.

---

## 📁 Project Structure

```bash
├── README.md
├── results
└── src
    ├── data
    │   └── dataset
    │       ├── 5Kfs_1Cycle_50f_1000Sam_1A.mat
    │       ├── Details.txt
    │       ├── Flicker.csv
    │       ├── Flicker_with_Sag.csv
    │       ├── Flicker_with_Swell.csv
    │       ├── Harmonics.csv
    │       ├── Harmonics_with_Sag.csv
    │       ├── Harmonics_with_Swell.csv
    │       ├── Interruption.csv
    │       ├── Notch.csv
    │       ├── Oscillatory_Transient.csv
    │       ├── Pure_Sinusoidal.csv
    │       ├── Sag.csv
    │       ├── Sag_with_Harmonics.csv
    │       ├── Sag_with_Oscillatory_Transient.csv
    │       ├── Swell.csv
    │       ├── Swell_with_Harmonics.csv
    │       ├── Swell_with_Oscillatory_Transient.csv
    │       └── Transient.csv
    ├── model.py
    └── train.ipynb
```

---

## 🧠 Approach

1. **Signal Processing**:
   - Collected power signal waveforms from real/simulated scenarios.
   - Applied **Fast Fourier Transform (FFT)** to extract frequency-domain insights.

2. **Model Design**:
   - Used **Transformer Encoder** layers to model temporal dependencies.
   - Incorporated **attention mechanisms** to focus on critical signal features.

3. **Training**:
   - Loss Function: CrossEntropyLoss
   - Optimizer: Adam
   - Trained for 500 epochs on NVIDIA P100 Gpu

4. **Evaluation**:
   - Confusion matrix to analyze per-class accuracy
   - Special focus on **harmonic-rich classes**

---

## 📊 Results

| Metric         | Value     |
|----------------|-----------|
| Test Accuracy  | 91%       |
| Classes        | 17 types  |
| FFT Impact     | +4–5% acc |

---

## 🛠️ Technologies Used

- **Python**, **PyTorch**, **Matplotlib**
- **MATLAB** (for PQ signal simulation)
- **NumPy**
- **Jupyter Notebook**

---

## 📚 Power Quality Disturbances Covered
-Pure_Sinusoidal
-Sag
-Swell
-Interruption
-Transient
-Oscillatory_Transient
-Harmonics
-Harmonics_with_Sag
-Harmonics_with_Swell
-Flicker
-Flicker_with_Sag
-Flicker_with_Swell
-Sag_with_Oscillatory_Transient
-Swell_with_Oscillatory_Transient
-Sag_with_Harmonics
-Swell_with_Harmonics
-Notch

---

```bash
# Clone repository
git clone https://github.com/vc940/Power-Quality-Disturbance.git
cd Power-Quality-Disturbance

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Evaluate on test set
python evaluate.py
