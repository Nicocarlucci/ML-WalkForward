# 📈 FinML-WalkForward: Financial Machine Learning & Backtesting Framework

Un'infrastruttura in Python guidata dal Machine Learning per il **Walk-Forward Backtesting** su serie temporali finanziarie. Il framework integra la costruzione e pulizia automatizzata di feature tecniche avanzate (RSI, Bande di Bollinger %B) per la simulazione e validazione realistica di strategie di trading quantitative.

---

## 🎯 Obiettivi del Progetto

Nei mercati finanziari, il classici metodi di cross-validation rischiano di introdurre **look-ahead bias** e **data leakage**. Questo progetto implementa un approccio a finestra mobile (**Walk-Forward Testing**) per addestrare e testare i modelli di ML esclusivamente su dati storici antecedenti al periodo di simulazione.

---

## 🛠️ Caratteristiche Principali

- **Feature Engineering Ottimizzato:**
  - **Relative Strength Index (RSI):** Misurazione vettoriale del momentum a $N$ periodi esente da problemi di precedenza degli operatori o indicizzazione errata.
  - **Bollinger Bands %B:** Indicatore di volatilità normalizzato ($0 - 1$) per tracciare la posizione relativa del prezzo rispetto alle bande di Bollinger.
- **Data Cleaning & Pipeline Integration:**
  - Isolamento corretto delle serie temporali singole per evitare inquinamento da colonne ridondanti.
  - Rimozione automatica delle finestre iniziali incompletali (`NaN`) mantenendo un allineamento temporale stringente.
- **Walk-Forward Validation Engine:** Simulazione realistica che aggiorna periodicamente il modello su finestre temporali scorrevoli.

---

## 📐 Indicatori Implementati

### 1. RSI (Relative Strength Index)
L'RSI misura la velocità e l'entità dei movimenti direzionali dei prezzi:
$$RS = \frac{\text{EMA}(\text{Guadagni})}{\text{EMA}(\text{Perdite})}$$
$$\text{RSI} = 100 - \left( \frac{100}{1 + RS} \right)$$

### 2. Bollinger Bands %B
Misura la posizione del prezzo di chiusura rispetto alle bande di Bollinger:
$$\%B = \frac{\text{Prezzo} - \text{Banda Inferiore}}{\text{Banda Superiore} - \text{Banda Inferiore}}$$

---

## 💻 Installazione e Requisiti

Assicurati di avere Python 3.9+ installato. Clona il repository e installa le dipendenze:

```bash
# Clona il repository
git clone [https://github.com/tuo-username/ML-WalkForward.git](https://github.com/tuo-username/ML-WalkForward.git)
cd ML-WalkForward

# Crea e attiva l'ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa i pacchetti richiesti
pip install pandas numpy scikit-learn xgboost