 # 📊 Telco Customer Churn — ML Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4c72b0)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-1.0%2B-F37626?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Abstract

Questo progetto è l'**evoluzione diretta** dell'analisi sviluppata nel corso di *Laboratorio di Big Data* ([repo precedente](https://github.com/diegoandruccioli/telco-churn-analysis)).

Partendo da una Logistic Regression baseline (~79% accuracy, ~57% Recall), l'obiettivo è ottimizzare sistematicamente i modelli tramite **K-Fold Cross-Validation** e **Grid Search**, e introdurre una **Rete Neurale Artificiale (MLP)**.  
La metrica di riferimento è il **Recall** — minimizzare i falsi negativi (churner non rilevati = costo di ri-acquisizione elevato).

---

## 📦 Dataset

**IBM Telco Customer Churn** — disponibile su [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

| Proprietà | Valore |
|---|---|
| Righe (clienti) | 7.043 |
| Colonne (feature) | 21 |
| Variabile target | `Churn` (Yes / No) |
| Sbilanciamento classi | ~26.6% Churn, ~73.4% No Churn |

Le feature coprono: dati demografici (genere, anzianità), servizi attivati (telefono, internet, streaming), tipo di contratto, metodo di pagamento, addebiti mensili e totali.

---

## 🔄 Progetto Precedente — Baseline

Il corso di *Laboratorio di Big Data* aveva prodotto una prima analisi end-to-end sullo stesso dataset:

| Fase | Tecnica | Risultato |
|---|---|---|
| Data Cleaning | Gestione valori mancanti, conversione `TotalCharges` | 0 valori nulli |
| EDA | Distribuzione Churn, correlazioni contratto/tenure/spesa | Contratto Month-to-Month = principale fattore di rischio |
| Clustering | K-Means (k=3), Elbow Method | Cluster critico: "Nuovi Alto-Spendenti" con churn >50% |
| Outlier Detection | IQR / Box-plot (univariato) | Outlier rilevati ma non rimossi |
| Classificazione | Logistic Regression (train/test split semplice) | Accuracy ~79%, **Recall ~57%** |

**Limite identificato:** con Recall del 57%, quasi metà dei clienti a rischio non veniva rilevata — il costo di acquisizione di un cliente perso supera quello di un falso allarme.

---

## 🎯 Obiettivi di questo Progetto

1. **Ottimizzare** i modelli tramite `GridSearchCV` e K-Fold Cross-Validation
2. **Espandere** il set di modelli con Decision Tree e rete neurale MLP
3. **Migliorare il Recall** come priorità di business
4. **Valutare** con un set completo di metriche: Recall, AUC-ROC, AUC-PR, MCC, Log-Loss

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
| **Introduzione** | Problema, tipo di task, obiettivo, interesse pratico |
| **Descrizione Dataset** | Origine, dimensione, tabella variabili per categoria, note sbilanciamento |
| **0** | Punto di partenza — tabella riepilogativa dei risultati del progetto precedente |
| **1** | Data Cleaning e EDA — analisi target, fedeltà, spesa; §1.5 encoding/scaling; §1.6 correlazioni |
| **2** | Outlier Detection — IQR/Box-plot (univariato) + Isolation Forest (multivariato) |
| **3** | Scelta dei modelli — ipotesi e motivazioni per ciascun algoritmo |
| **4** | Validazione e Ottimizzazione — baseline LR, K-Fold CV, GridSearch su LR, Decision Tree, KNN |
| **5** | Rete Neurale MLP — baseline, loss curve, GridSearch |
| **6** | Confronto finale — tabella metriche completa, grafici comparativi, curve ROC e Precision-Recall |
| **7** | Conclusioni, raccomandazioni strategiche e sviluppi futuri |

---

## 🔬 Metodologia

### Outlier Detection (Sezione 2)
- **IQR / Box-plot** — rilevamento univariato su `tenure`, `MonthlyCharges`, `TotalCharges`
- **Isolation Forest** — rilevamento multivariato (anomalie contestuali); gli outlier individuati vengono mantenuti perché rappresentano comportamenti reali di clienti, non errori di misurazione

### Preprocessing (Sezione 1.5)
- **Label Encoding** per variabili binarie (Yes/No, Male/Female)
- **One-Hot Encoding** per variabili multi-categoria (Contract, PaymentMethod, ecc.)
- **StandardScaler** — obbligatorio per Logistic Regression e MLP, entrambi sensibili alla scala perché ottimizzati tramite gradiente

### Ottimizzazione Iperparametri (Sezioni 4–5)
- **K-Fold Cross-Validation** (k=5, shuffle=True) — stima robusta; riduce la varianza rispetto al singolo split
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
**Miglioramento AUC-ROC:** da 0.842 a 0.934 (+0.092)

---

## 🚀 Installazione e Utilizzo

### Prerequisiti

- **Python 3.8+** — verificare con `python --version`
- **pip** — incluso nelle versioni moderne di Python

### 1. Clona la repository

```bash
git clone https://github.com/diegoandruccioli/telco-churn-ml-optimization.git
cd telco-churn-ml-optimization
```

### 2. Scarica il dataset

Il dataset non è incluso nella repository per ragioni di dimensione.

1. Vai su [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Scarica `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Posizionalo in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Crea un ambiente virtuale (consigliato)

```bash
python -m venv venv
```

Attiva l'ambiente:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 5. Avvia il notebook

**Opzione A — Jupyter Notebook (browser):**
```bash
jupyter notebook
```
Si apre il browser. Naviga in `notebooks/` e clicca su `churn_optimization.ipynb`.

**Opzione B — JupyterLab:**
```bash
pip install jupyterlab
jupyter lab
```

**Opzione C — VS Code:**
Installa l'estensione *Jupyter* da Microsoft, poi apri direttamente `notebooks/churn_optimization.ipynb`.

### 6. Esegui il notebook

Una volta aperto il notebook, esegui tutte le celle in sequenza:
- **Kernel → Restart & Run All** (Jupyter Notebook)
- **Run All** (JupyterLab / VS Code)

> **Nota:** la sezione §5.3 Grid Search MLP richiede ~5–10 minuti su hardware standard (80 combinazioni × 5-fold CV). Le `ConvergenceWarning` di sklearn durante questa sezione sono attese e già documentate nel notebook — non indicano errori.

---

## 🛠️ Tecnologie Utilizzate

| Libreria | Versione minima | Utilizzo |
|---|---|---|
| Python | 3.8+ | Linguaggio principale |
| Pandas | 2.0+ | Manipolazione e analisi dati |
| NumPy | 1.24+ | Operazioni numeriche |
| Matplotlib | 3.7+ | Visualizzazione dati |
| Seaborn | 0.12+ | Visualizzazione statistica |
| Scikit-Learn | 1.3+ | ML, GridSearchCV, K-Fold, metriche |
| Jupyter / Notebook | 1.0+ / 7.0+ | Ambiente di sviluppo interattivo |

---

**Autori:** Diego Andruccioli, Rei Mici  
**Corso:** Laboratorio di Ottimizzazione, Intelligenza Artificiale e Machine Learning  
**Anno Accademico:** 2025/2026
