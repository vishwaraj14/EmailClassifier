from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import List
from models import EmailClassifier

class EmailClassifierValidator(EmailClassifier):
    """Enhanced classifier with cross-validation"""

    def __init__(self):
        super().__init__()

    def cross_validate(self, file_path: str, n_splits: int = 5) -> None:
        """Cross-validation focused only on model performance"""
        try:
            df = self.load_data(file_path)
            texts, labels = self.prepare_features(df)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            all_accuracies = []
            all_f1_scores = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
                X_train = [texts[i] for i in train_idx]
                X_test = [texts[i] for i in test_idx]
                y_train = [labels[i] for i in train_idx]
                y_test = [labels[i] for i in test_idx]

                model = self.build_model()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                all_accuracies.append(acc)
                all_f1_scores.append(f1)

                print(f"Fold {fold+1}: Accuracy={acc:.4f}, F1-score={f1:.4f}")

            print("\nCross-Validation Results:")
            print(f"Mean Accuracy: {np.mean(all_accuracies):.4f}")
            print(f"Mean F1-score: {np.mean(all_f1_scores):.4f}")
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            raise

if __name__ == "__main__":
    validator = EmailClassifierValidator()
    try:
        validator.cross_validate('combined_emails_with_natural_pii.csv', n_splits=5)
        print("\nValidation complete - Model meets performance requirements!")
    except RuntimeError:
        print("\nValidation halted - Address issues before proceeding!")