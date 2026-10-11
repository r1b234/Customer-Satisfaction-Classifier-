# Customer Satisfaction Classifier CRM

An NLP-based CRM dashboard built with **Streamlit**, **SQLite**, **Pandas**, and **NLTK (VADER)** that automatically processes customer survey comments, classifies them into **Satisfied**, **Neutral**, or **Dissatisfied**, tracks customer profiles, and computes metrics.

---

## Project Structure
```text
customer-satisfaction-classifier/
├── app.py                 # Main Streamlit Dashboard & NLP classification pipeline
├── requirements.txt       # Dependencies list
└── README.md              # Documentation and Test Cases
```

---

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Test Cases & Verification

| Test ID | Input Survey Comment | Expected Classification | VADER Score Condition | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | "Absolute game changer! Outstanding support." | Satisfied | Score $\ge 0.05$ | Passed |
| **TC-02** | "Terrible customer service. Total waste of money." | Dissatisfied | Score $\le -0.05$ | Passed |
| **TC-03** | "It's okay, nothing special. Does the job." | Neutral | $-0.05 < \text{Score} < 0.05$ | Passed |
