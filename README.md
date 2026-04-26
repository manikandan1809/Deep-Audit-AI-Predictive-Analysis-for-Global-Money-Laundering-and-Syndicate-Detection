# 🔴 Deep Audit AI — v9.0
### AI/ML-Powered Financial Crime Detection System
**ED / CBI Financial Crimes Unit — Anti-Money Laundering Intelligence Platform**

---

## 📌 Overview

**Deep Audit AI** is an AI and Machine Learning-based financial fraud detection and anti-money laundering (AML) intelligence platform built for law enforcement and financial audit use cases.

It analyzes large volumes of financial transaction data to detect suspicious patterns, identify syndicate networks, and flag high-risk accounts — using a combination of **Isolation Forest**, **XGBoost**, **PageRank**, and **NLP-based fraud taxonomy classification**.

---

## 🎯 Key Features

- 🔍 **Anomaly Detection** — Isolation Forest + XGBoost dual-model pipeline
- 🕸️ **Syndicate Network Analysis** — NetworkX graph with PageRank-based boss detection
- 📊 **Real-Time Transaction Scoring** — Score any transaction instantly with risk level
- 🔴 **Live Dashboard** — Auto-refresh feed with real-time fraud monitoring (every 2 seconds)
- 🧠 **NLP Fraud Classification** — Detects Smurfing, Layering, Hawala, Integration, Terror Finance
- 👤 **KYC Account Investigator** — Full account profile with PAN, Aadhar, transaction history
- 📄 **PDF Report Generation** — PMLA 2002 compliant court-admissible forensic dossier
- 📧 **Email Alerts** — SMTP-based automatic alerts for HIGH risk transactions
- 🔐 **Role-Based Login** — Admin and Analyst access levels

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python |
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-learn (Isolation Forest), XGBoost |
| **Graph Analysis** | NetworkX, PageRank Algorithm |
| **Data Processing** | Pandas, NumPy, MinMaxScaler |
| **Visualization** | Plotly, Matplotlib |
| **PDF Generation** | ReportLab, fpdf2 |
| **Database** | SQLite |
| **NLP** | Rule-based Fraud Taxonomy Classifier |

---

## 📁 Project Structure

```
deep-audit-ai/
│
├── app.py                  # Main Streamlit application
├── transaction_generator.py  # Live transaction data simulator
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/manikandan1809/deep-audit-ai.git
cd deep-audit-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. (Optional) Start Live Transaction Generator
Open a second terminal and run:
```bash
python transaction_generator.py
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
networkx>=3.1
plotly>=5.18.0
fpdf2>=2.7.0
matplotlib>=3.8.0
pyarrow>=15.0.0
xgboost>=2.0.0
```

---

## 🔐 Default Login Credentials

| Username | Password | Role |
|---|---|---|
| admin | admin123 | ADMIN |
| analyst | analyst456 | ANALYST |

---

## 🧠 How It Works

### AI Pipeline (6 Steps)
1. **Feature Engineering** — 12 forensic features extracted (Amount, Time Gap, Rapid Fire, Velocity, Night Flag, etc.)
2. **Auto-tuned Contamination** — Dynamically estimated from IQR and night-time transaction rate (capped at 5%)
3. **Isolation Forest** — Flags anomalous transactions (n=100 estimators)
4. **XGBoost Boost** — Refines predictions using class-weighted classification
5. **PageRank Network** — Builds transaction graph, detects syndicate boss via composite score
6. **3D Network Visualization** — Interactive 200-node network rendered in browser

### Fraud Pattern Detection (NLP)
| Pattern | Description |
|---|---|
| SMURFING | Structured deposits below reporting threshold |
| LAYERING | Shell company / transit account routing |
| INTEGRATION | Offshore consolidation and wire transfers |
| HAWALA | Informal / undocumented value transfers |
| TERROR | Sanctioned or blacklisted account transactions |

### Dynamic Risk Thresholds (v9.0 Fix)
- **HIGH RISK** → Top 5% of dataset (95th percentile)
- **MEDIUM RISK** → Top 15% of dataset (85th percentile)
- **LOW / NORMAL** → Bottom 85%

---

## 📸 Application Pages

| Page | Description |
|---|---|
| 🏠 Command Center | System overview and fix changelog |
| 📁 Data Ingestion | Upload CSV/Parquet transaction files |
| 🔴 Live Dashboard | Real-time feed + AI scan on live data |
| ⚡ Real-Time Finder | Instantly score any transaction |
| 🕸️ Network Topology | Interactive 3D syndicate network |
| 📊 Intelligence Matrix | Charts, distributions, fraud breakdown |
| 📋 Threat Dossier | Syndicate role table + transaction chain |
| 🔬 Explainability Lab | Feature importance + model config |
| 📄 Export & Download | PDF report, CSV exports, FastAPI scorer |
| ⚙️ Settings | SMTP config, scan history, system status |

---

## 📄 CSV Input Format

Your transaction CSV should contain these columns:

| Column | Required | Description |
|---|---|---|
| Transaction_ID | Yes | Unique transaction identifier |
| Source_Acc_No | Yes | Sender account number |
| Dest_Acc_No | Yes | Receiver account number |
| Amount_INR | Yes | Transaction amount |
| Timestamp | Yes | Date and time |
| Transaction_Type | Yes | WIRE / NEFT / UPI / SWIFT / HAWALA |
| Is_International | Recommended | 0 or 1 |
| Txn_Description | Optional | Used for NLP fraud classification |

---

## 👨‍💻 Author

**Manikandan M**
- 📧 m.s.manikandan18.09.2004@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/manikandan-m-541781385)
- 🐙 [GitHub](https://github.com/manikandan1809)

---

## 📜 Disclaimer

This project is built for **academic and research purposes only**.
It simulates an AML intelligence platform and does not process real financial data.
All transaction data used is synthetically generated.

---

## ⭐ If you found this project useful, please give it a star!
# Deep-Audit-AI-Predictive-Analysis-for-Global-Money-Laundering-and-Syndicate-Detection
AI&amp;ML-powered Anti-Money Laundering platform — detects fraud patterns, syndicate networks, and high-risk transactions using Isolation Forest, NetworkX, XGBoost &amp; PageRank.
