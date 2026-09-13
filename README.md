# CASEFILE: AI-Powered Missing Person Investigation and Probable Location Prediction System

## Project Overview

CASEFILE is an Advanced Machine Learning project designed to analyze historical GPS movement patterns and predict probable geographical areas for a fictional missing-person case.

The system analyzes movement patterns, frequently visited locations, anomalies, probable locations, probable routes, search-priority areas, and prediction explanations.

This project is an academic simulation and is not intended for real-world missing-person investigations.

## Objectives

- Analyze GPS trajectory data
- Identify frequently visited geographical areas
- Understand normal movement patterns
- Detect unusual movement using anomaly detection
- Predict probable geographical areas
- Predict probable movement routes
- Calculate search-priority scores
- Explain prediction results
- Visualize results using an interactive map
- Develop an interactive Streamlit dashboard

## Machine Learning Techniques

### 1. Movement Clustering

K-Means clustering is used to identify geographical movement areas.

### 2. Anomaly Detection

Isolation Forest is used to identify unusual movement patterns.

### 3. Location Prediction

Random Forest is used to predict probable geographical areas.

### 4. Route Prediction

Markov Chain transition probabilities are used to predict probable movement routes.

## Search Priority Score

The search-priority score combines:

- ML prediction probability — 30%
- Historical visit frequency — 20%
- Route similarity — 15%
- Distance relevance — 15%
- Time relevance — 10%
- Anomaly evidence — 10%

### Priority Levels

| Score | Priority |
|---|---|
| 0–30 | Low |
| 31–60 | Medium |
| 61–80 | High |
| 81–100 | Very High |

## Project Structure

```text
CASEFILE_MISSING_PERSON/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── requirements.txt
└── README.md