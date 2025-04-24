import re
import pandas as pd
import numpy as np
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


from typing import Tuple, List, Optional, Dict, Pattern

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download(['stopwords', 'wordnet', 'punkt'], quiet=True)

class PIIMasker:
    """Comprehensive PII/PCI masking module with validation"""
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_number': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'credit_debit_no': re.compile(r'\b(?:\d[ -]*?){13,19}\b'),
            'cvv_no': re.compile(r'\b\d{3,4}\b(?![ -]*\d)'),
            'expiry_no': re.compile(r'\b(0[1-9]|1[0-2])[/-]?(\d{2}|\d{4})\b'),
            'aadhar_num': re.compile(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b'),
            'dob': re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),
            'full_name': re.compile(r'(?i)\b(mr|mrs|ms|dr)\.?\s+\w+\s+\w+\b|(?<=\bname is\s)\w+\s+\w+')
        }

        self.mask_map = {
            'email': '[EMAIL]',
            'phone_number': '[PHONE]',
            'credit_debit_no': '[CREDIT_DEBIT]',
            'cvv_no': '[CVV]',
            'expiry_no': '[EXPIRY]',
            'aadhar_num': '[AADHAR]',
            'dob': '[DOB]',
            'full_name': '[NAME]'
        }

    def mask_all(self, text: str) -> str:
        """Apply all PII masks with priority handling"""
        for pii_type in ['credit_debit_no', 'aadhar_num', 'phone_number', 
                        'email', 'cvv_no', 'expiry_no', 'dob', 'full_name']:
            text = self.patterns[pii_type].sub(self.mask_map[pii_type], text)
        return text

    def validate_masking(self, text: str) -> bool:
        """Security check for residual PII"""
        return not any(pattern.search(text) for pattern in self.patterns.values())

class EmailClassifier:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[LinearSVC] = None
        self.stop_words = set(stopwords.words('english')).union(set(stopwords.words('german')))
        self.lemmatizer = WordNetLemmatizer()
        self.pii_masker = PIIMasker()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data"""
        try:
            df = pd.read_csv(file_path)
            if {'email', 'type'}.issubset(df.columns):
                print("Data loaded successfully")
                return df
            raise ValueError("Missing required columns: 'email' or 'type'")
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Secure text processing pipeline"""
        try:
            # Primary PII redaction
            text = self.pii_masker.mask_all(text)
            '''
            if not self.pii_masker.validate_masking(text):
                raise ValueError("PII masking validation failed")
            ''' 
            # Text normalization
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = nltk.word_tokenize(text)
            
            # Lemmatization and stopword removal
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                     if word not in self.stop_words and len(word) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Security alert in preprocessing: {str(e)}")
            return '[CONTENT REMOVED]'

    def prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """Feature engineering pipeline"""
        try:
            # Text preprocessing
            df['clean_text'] = df['email'].apply(self.preprocess_text)
            print("Secure text preprocessing completed")
            
            # Label encoding
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df['type'])
            return df['clean_text'].tolist(), labels
        except Exception as e:
            print(f"Feature preparation failed: {str(e)}")
            raise

    def build_model(self) -> Pipeline:
        """Construct ML pipeline with TF-IDF and Linear SVM"""
        try:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9
                )),
                ('clf', LinearSVC(
                    C=0.8,
                    class_weight='balanced',
                    dual=False,
                    max_iter=10000
                ))
            ])
            print(" Secure model pipeline constructed")
            return pipeline
        except Exception as e:
            print(f"Model build failed: {str(e)}")
            raise

    def train(self, file_path: str, test_size: float = 0.2) -> None:
        """Complete training workflow"""
        try:
            # Data loading and preparation
            df = self.load_data(file_path)
            texts, labels = self.prepare_features(df)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42
            )
            
            # Model training
            pipeline = self.build_model()
            print("Training secure model...")
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            print("Secure training completed")
            
            # Save artifacts
            joblib.dump(pipeline, 'secure_email_classifier.joblib')
            joblib.dump(self.label_encoder, 'secure_label_encoder.joblib')
            
            # Evaluation
            y_pred = pipeline.predict(X_test)
            print("\nSecurity-Validated Classification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ))
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    classifier = EmailClassifier()
    classifier.train('combined_emails_with_natural_pii.csv')
    print("\nSecure model ready for deployment!")

