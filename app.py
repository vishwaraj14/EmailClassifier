from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import masking
import joblib
import re

app = FastAPI()

# Load trained models
classifier = joblib.load('secure_email_classifier.joblib')
label_encoder = joblib.load('secure_label_encoder.joblib')

# Initialize masker
masker = masking.PIIMasker()

class EmailRequest(BaseModel):
    email_body: str

def classify_email(masked_text: str) -> str:
    """Classify email using the trained LinearSVC model"""
    prediction = classifier.predict([masked_text])
    return label_encoder.inverse_transform(prediction)[0]

def extract_positions(original_text: str, masked_text: str, mapping: Dict[str, str]) -> List[Dict]:
    """Extract positions of masked entities in original text"""
    entities = []
    
    # Sort tokens by length descending to prevent partial matches
    for token in sorted(mapping.keys(), key=len, reverse=True):
        original_value = mapping[token]
        pattern = re.escape(token)
        
        # Find all token positions in masked text
        for match in re.finditer(pattern, masked_text):
            start, end = match.span()
            
            # Find corresponding original value in original text
            original_start = original_text.find(original_value, start, end + len(original_value))
            if original_start != -1:
                entities.append({
                    "position": [original_start, original_start + len(original_value)],
                    "classification": token.split('_')[1],  # Extract entity type from token
                    "entity": original_value
                })
    
    return entities

@app.get("/")
async def root():
    return {"message": "Welcome to the Email Classifier API!"}

@app.post("/classify-email", response_model=Dict)
async def process_email(request: EmailRequest):
    # Mask sensitive information
    original_text = request.email_body
    masked_text, mapping = masker.mask(original_text)
    
    # Classify email
    category = classify_email(masked_text)
    
    # Unmask for final output while preserving mapping
    demasked_text = masker.demask(masked_text, mapping)
    
    # Extract entity positions
    masked_entities = extract_positions(original_text, masked_text, mapping)
    
    return {
        "input_email_body": demasked_text,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_text,
        "category_of_the_email": category
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

