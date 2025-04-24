# Email Classifier API

A FastAPI-based application for classifying support emails into predefined categories while ensuring PII (Personally Identifiable Information) is masked to protect sensitive content.

---

## Features

- Email Classification: Automatically classifies incoming emails into relevant categories (e.g., Billing, Technical Support).
- Data Masking: Detects and masks sensitive PII data (e.g., emails, phone numbers, credit card details) using regex/NLP.
- Entity Extraction: Identifies and returns positions and types of masked entities.

---

## API Endpoints

- `POST /classify-email`:  
  Accepts raw email text, masks sensitive information, classifies the content, and returns:
  - Masked email content  
  - Detected entities with positions  
  - Predicted category

- `GET /docs`:  
  Access the Swagger UI for interactive API documentation.

---

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

---

## Setup and Run

1. Clone the Repository
   ```bash
   git clone https://github.com/vishwaraj14/EmailClassifier.git
   cd EmailClassifier
2.Install Dependencies

bash

pip install -r requirements.txt
3.Run the Application

uvicorn api:app --reload

4.Access the API Visit: http://127.0.0.1:8000/docs to test endpoints via Swagger UI