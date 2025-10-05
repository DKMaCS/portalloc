# Efficient Frontier & CML — Streamlit App

Upload a CSV of **daily closing prices** (first column `Date`, remaining columns are tickers).  
Set risk-free rate `rf`, risk-aversion `γ`, choose long-only or long/short, and view:
- Efficient Frontier
- Capital Market Line (CML) through the tangency (MSR) portfolio
- Weights for **GMV**, **MSR**, and **γ-portfolio**
- Download combined weights as CSV

## Local run
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
