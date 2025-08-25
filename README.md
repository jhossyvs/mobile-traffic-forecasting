# 📡 Mobile Traffic Forecasting  

Forecasting mobile network traffic in Paris using deep learning models.  
This project explores **multistep time series forecasting** of mobile users across different regions of Paris, based on the **Orange Telecom mobility dataset**.  

The goal is to predict the number of users in a given region for the **next 3 hours**, using both **univariate** and **multivariate** deep learning approaches.  

---

## 📂 Project Structure  

```
mobile-traffic-forecasting/
│── README.md              # Project overview
│── LICENSE                # Apache 2.0
│── requirements.txt       # Dependencies
│
├── data/                  # Datasets (raw and processed) 
│   ├── raw/               # Original Orange dataset
│   └── processed/         # Aggregated/cleaned series
│
├── notebooks/             # Experiments and exploratory work
│   ├── 01-eda.ipynb
│   └── 02-mlp-univariate.ipynb
│
├── src/                   # Reusable Python code
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   └── evaluation.py
│
├── results/               # Model outputs
│   ├── figures/           # RMSE curves, prediction plots
│   └── metrics/           # Saved evaluation metrics
│
└── docs/                  # Documentation
    └── report.md
```

---

## 📊 Dataset  

- **Name:** Multivariate-Mobility-Paris  
- **Provider:** Orange Telecom  
- **Source:** [Papers with Code](https://paperswithcode.com/dataset/multivariate-mobility-paris)  
- **Period:** 2020-08-24 to 2020-11-04 (72 days)  
- **Granularity:** 30-minute intervals (aggregated to hourly for this project)  
- **Features:** 6 regions of Paris (R1–R6), each with mobile user counts  

⚠️ *The dataset is not distributed in this repository due to licensing restrictions. Please download it from the official source.*  

---

## 🚀 Models Implemented  

- [x] **Univariate MLP** → predict next 3 hours of users in region R1  
- [ ] **Multivariate MLP** (coming soon)  
- [ ] **CNN for time series forecasting** (planned)  
- [ ] **LSTM for time series forecasting** (planned)  

---

## 📈 Results  

For each model, evaluation is performed with **walk-forward validation** on the last 10% of the time series.  
The main metric is **RMSE** (Root Mean Squared Error) across horizons (1h, 2h, 3h ahead).   

---

## 📈 Example Results (Univariate MLP)
![RMSE Curve](results/figures/univariate_mlp_rmse.png)

The MLP model shows increasing error across horizons, as expected (1h ahead is easier than 3h ahead).

🔧 How to Run Experiments
1. Preprocess the dataset:
   ```bash
   python src/preprocessing.py
   ```
2. Run the univariate MLP grid search (walk-forward evaluation):
   ```bash
   jupyter notebook notebooks/02-mlp-univariate.ipynb
   ```
3. Results will be stored under:
   - `results/models/univariate/mlp/` → Trained models
   - `results/metrics/univariate/mlp/` → CSV with evaluation metrics
   - `results/figures/` → RMSE plots

⚡ Note: Grid search is parallelized with joblib. Runtime may vary depending on CPU cores.

---

## ⚙️ Installation  

```bash
git clone https://github.com/jhossyvs/mobile-traffic-forecasting
cd mobile-traffic-forecasting
pip install -r requirements.txt
```

---

## 📝 License  

This project is licensed under the **Apache 2.0 License**.  
See the [LICENSE](./LICENSE) file for details.  

---

## 🙌 Acknowledgements  

Dataset courtesy of **Orange Telecom**.  
This project was initially developed as part of a **Time Series Forecasting course project**.  