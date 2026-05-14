# 📊 Telco Customer Churn — ML Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--Learn%20%7C%20Seaborn-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Abstract

Questo progetto è l'**evoluzione diretta** dell'analisi sviluppata nel corso di *Laboratorio di Big Data* ([repo precedente](https://github.com/diegoandruccioli/telco-churn-analysis)).

Partendo da una Logistic Regression baseline (~79% accuracy, ~57% Recall), l'obiettivo è ottimizzare sistematicamente i modelli tramite **K-Fold Cross-Validation** e **Grid Search**, e introdurre una **Rete Neurale Artificiale (MLP)**.  
La metrica di riferimento è il **Recall** — minimizzare i falsi negativi (churner non rilevati = costo di ri-acquisizione elevato).

---

## 🎯 Motivazione e Obiettivi

Il progetto precedente aveva identificato il 26.6% di clienti churner e costruito un modello baseline con Logistic Regression. Il limite principale era un Recall del 57%: quasi metà dei clienti a rischio non veniva rilevata.

**L'obiettivo di questo progetto è:**
1. **Ottimizzare** i modelli esistenti tramite GridSearchCV e K-Fold Cross-Validation
2. **Espandere** il set di modelli con l'introduzione di una rete neurale MLP
3. **Valutare** i risultati con un set completo di metriche (Recall, AUC-ROC, AUC-PR, MCC, Log-Loss)

---

## 📂 Struttura del Progetto

```
telco-churn-ml-optimization/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset IBM Telco (Kaggle)
├── notebooks/
│   └── churn_optimization.ipynb              # Notebook principale
├── requirements.txt                          # Dipendenze Python
└── README.md
```

---

## ⚙️ Struttura del Notebook

Il notebook segue la struttura richiesta per l'esame:

| Sezione | Contenuto |
|---|---|
| **0** | Punto di partenza — tabella baseline dal progetto precedente |
| **1** | Data Cleaning e EDA (analisi target, fedeltà, spesa, encoding, scaling) |
| **2** | Outlier Detection — IQR/Box-plot + Isolation Forest multivariato |
| **3** | Scelta dei modelli — ipotesi e motivazioni |
| **4** | Validazione e Ottimizzazione — baseline LR, K-Fold CV, GridSearch LR e Decision Tree |
| **5** | Rete Neurale MLP — baseline, loss curve, GridSearch |
| **6** | Confronto finale — tabella metriche, grafici comparativi, curve ROC e Precision-Recall |
| **7** | Conclusioni, raccomandazioni strategiche e sviluppi futuri |

---

## 🔬 Metodologia

### Outlier Detection (Sezione 2)
- **IQR / Box-plot** — rilevamento univariato sulle variabili numeriche continue
- **Isolation Forest** — rilevamento multivariato (anomalie contestuali); gli outlier rilevati vengono mantenuti perché rappresentano comportamenti reali

### Ottimizzazione Iperparametri (Sezione 4–5)
- **K-Fold Cross-Validation** (k=5) — stima robusta delle performance, riduce la varianza rispetto al singolo split
- **GridSearchCV** su tutti e tre i modelli, `scoring='recall'`

| Modello | Iperparametri ottimizzati |
|---|---|
| Logistic Regression | `C`, `solver`, `max_iter` |
| Decision Tree | `max_depth`, `min_samples_split`, `criterion` |
| MLP | `hidden_layer_sizes`, `activation`, `learning_rate_init` |

---

## 🔑 Risultati

| Modello | Accuracy | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| LR Baseline (vecchio progetto) | 80.7% | 56.7% | 60.9% | 0.842 |
| LR Ottimizzata (Grid Search) | 80.1% | 56.4% | 60.1% | 0.840 |
| Decision Tree (Grid Search) | 75.1% | 59.4% | 55.8% | 0.789 |
| MLP Baseline | 76.2% | 51.1% | 53.3% | 0.808 |
| **MLP Ottimizzato (Grid Search)** | **86.7%** | **71.7%** | **74.0%** | **0.934** |

**Miglioramento Recall:** +15 punti percentuali rispetto alla baseline (56.7% → 71.7%)

---

## 🚀 Installazione e Utilizzo

1. **Clona la repository:**
   ```bash
   git clone https://github.com/diegoandruccioli/telco-churn-ml-optimization.git
   cd telco-churn-ml-optimization
   ```

2. **Crea un ambiente virtuale (consigliato):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Installa le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Avvia Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Apri `notebooks/churn_optimization.ipynb` ed esegui le celle sequenzialmente.

---

## 🛠️ Tecnologie Utilizzate

- **Python 3.8+**
- **Pandas & NumPy** — manipolazione dati
- **Matplotlib & Seaborn** — visualizzazione
- **Scikit-Learn** — preprocessing, modelli ML, GridSearchCV, K-Fold CV, metriche

---

**Autori:** Diego Andruccioli, Rei Mici  
**Corso:** Laboratorio di Ottimizzazione, Intelligenza Artificiale e Machine Learning  
**Anno Accademico:** 2025/2026
