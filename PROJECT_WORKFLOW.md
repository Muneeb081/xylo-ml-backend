# PRECON Smart Energy ML System

## 1. Project Overview

The **PRECON Smart Energy ML System** is an end-to-end machine learning pipeline and real-time inference API designed to analyze household energy consumption. The project takes raw energy usage data (down to 1-minute or 15-minute intervals) and applies advanced machine learning models to provide deep insights, anomaly detection, and energy-saving recommendations.

### What Does This Project Do?
It performs **5 core AI-driven tasks**:

1. **Energy Consumption Prediction (Task 1)**: Forecasts household energy consumption using a highly optimized **XGBoost** model for immediate predictions and an **LSTM (Long Short-Term Memory)** neural network for multi-step sequence forecasting.
2. **Appliance Energy Disaggregation (Task 2)**: Breaks down the total household power draw (kW) into individual appliance/room usage (e.g., AC, Kitchen, Living Room). It utilizes individual XGBoost models for each appliance type to accurately estimate their load footprint.
3. **Room-Wise Anomaly Detection (Task 3)**: Monitors real-time device usage to detect unusual behavior using **Isolation Forests** and **Z-score** statistical methods. It generates human-readable alerts (e.g., *"⚠ WARNING — Living Room is consuming unusually high energy..."*).
4. **Peak Load Forecasting (Task 4)**: Predicts the maximum expected power load (Peak kW) for the day, allowing for proactive load management to avoid demand charges.
5. **Optimization Recommendations (Task 5)**: A smart rule-engine that synthesizes the current consumption, appliance states, and household metadata to generate actionable, real-time energy-saving tips.

---

## 2. Core Architecture

The codebase is divided into two primary environments:

### A. The Training Pipeline (`train_pipeline.py`)
This is the offline data ingestion and model training phase.
* **Data Ingestion**: Loads raw CSV/JSON data for the households.
* **Preprocessing**: Handles missing values, performs feature engineering (time-of-day, day-of-week, moving averages), and standardizes features using MinMax Scaling.
* **Feature Store**: Builds a formalized feature dataset (`fs_meta`) that guarantees consistency between training and inference.
* **Training & Export**: Trains all the XGBoost, LSTM, and Isolation Forest models, and exports them (`.joblib` and `.keras`) into the `models/` directory along with a centralized metrics state (`_pipeline_metrics.json`).

### B. The Inference API (`api/main.py`)
A high-performance **FastAPI** web service tailored for real-time inference.
* It loads the pre-trained models into memory on startup.
* Provides REST endpoints to query individual tasks (e.g., `/api/v1/predict/energy`, `/api/v1/disaggregate`).
* Provides a unified endpoint (`/api/v1/predict/all`) to run all 5 tasks simultaneously.
* Specifically supports **Firebase Realtime Database JSON streaming** (`/api/v1/stream/firebase`) for seamless IoT app integration.

---

## 3. Real-Time Workflow (App Integration)

When integrating this ML API into a real-world client application (e.g., an iOS/Android mobile app or a Web Dashboard), the operational workflow is as follows:

### Step 1: Data Collection (IoT / Firebase)
Smart meters or smart plugs in the user's home collect power draw (kW) readings every few minutes. These readings are either pushed to a Firebase Realtime Database or sent directly to your backend.

### Step 2: The API Request
The client app (or your backend watcher) sends a single POST request to the API. If using Firebase, you can pipe the raw Firebase JSON node directly to the API:

**POST `http://<server-ip>:8000/api/v1/stream/firebase`**
```json
{
  "house_id": 16,
  "timestamp": "2024-05-09T14:30:00",
  "data": {
    "Usage_kW": 3.8,
    "kitchen": 1.5,
    "ac": 2.0,
    "n_acs": 2,
    "n_people": 4
  }
}
```

### Step 3: Real-Time Processing (The ML Engine)
Within milliseconds, the FastAPI server:
1. Translates the raw payload into a scaled feature row perfectly matching the training data structure.
2. Passes the data through the **XGBoost models** to estimate total energy and predict the day's peak load.
3. Passes the data through the **Disaggregation models** to calculate what percentage of power is going to the AC vs. Kitchen.
4. Passes the data through the **Anomaly models** to detect if the AC is drawing 40% more power than its historical average for this time of day.
5. Computes rule-based **Recommendations**.

### Step 4: The Response & UI Rendering
The API returns a consolidated JSON payload back to the app:
```json
{
  "energy_prediction": { "predicted_kw": 3.9 },
  "peak_forecast": { "forecast_peak_kw": 5.2 },
  "disaggregation": [
    { "appliance": "ac", "kw": 2.0, "percentage": 52.6 },
    { "appliance": "kitchen", "kw": 1.5, "percentage": 39.4 }
  ],
  "anomaly_detection": {
    "alerts": [
      "⚠ WARNING — Kitchen is consuming unusually high energy compared to normal behavior."
    ]
  },
  "recommendations": [
    "Turn off 1 AC unit to save 1.2 kW immediately.",
    "Your peak load is approaching 5.0 kW, consider delaying laundry."
  ]
}
```

### Step 5: End-User Experience
* **Dashboard Chart**: The app updates a beautiful donut chart showing 52% usage by the AC and 39% by the Kitchen.
* **Notification/Push Alert**: The app triggers a push notification with the exact text from the anomaly alert (`"⚠ WARNING — Kitchen is consuming..."`).
* **Insights Tab**: The recommendations are displayed as dismissible "Smart Tips" cards.

## 4. How to Run

1. **Train the System**: 
   `python train_pipeline.py` (Full run) or `python train_pipeline.py --quick` (Fast test).
2. **Start the API**:
   `python run_api.py` (Runs the uvicorn server on port 8000).
3. **Test Health**:
   Navigate to `http://localhost:8000/docs` to see the automated Swagger UI and test endpoints directly.
