# PLC Doctor

**An AI-powered industrial maintenance assistant that detects likely machine faults and provides diagnostic guidance.**

## Overview

PLC Doctor combines Machine Learning and Google's Gemini AI to help technicians perform preliminary industrial fault diagnosis.

Users enter machine operating measurements. A ML classifier predicts one of six conditions, and Gemini generates an explanation with possible causes, verification steps and safety precautions.

## Features

- Machine fault classification using ML.
- Six categories: Motor Overload, Sensor Failure, Power Supply Failure, VFD Fault, PLC Communication Failure and Normal Operation.
- Fault probability visualization.
- Gemini AI diagnostic reports.
- Diagnostic history.
- English and French AI reports.

## Technology Stack

- Python 3.13
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Google Gemini API
- Joblib

## Installation

Clone the repository:

\\\ash
git clone https://github.com/SamyDries/PLC-Doctor.git
cd PLC-Doctor
\\\

Create and activate a virtual environment:

\\\ash
python -m venv .venv
\\\

Windows PowerShell:

\\\powershell
.venv\Scripts\Activate.ps1
\\\

Install dependencies:

\\\ash
pip install -r requirements.txt
\\\

## Gemini API Configuration

Create .streamlit/secrets.toml and add:

 your own Google Gemini API key.

Never publish your API key ;)

## Run the Application

\\\ash
streamlit run app.py
\\\

## Train the Model Again

\\\ash
python train_model.py
\\\

## Dataset and Limitations

This prototype uses a synthetic dataset of 600 observations across six categories.

The reported model accuracy reflects performance on this synthetic dataset. It does not establish real-world industrial diagnostic accuracy.

PLC Doctor is an educational prototype, not a certified diagnostic or safety system. All recommendations require verification by qualified personnel following official machine safety procedures.

## Developer

**Samy Dries — Solo Developer**

Responsible for project design, synthetic dataset generation, Machine Learning model, Gemini integration, Streamlit interface and testing.
