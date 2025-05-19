import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# === 1. Load pretrained artifacts ===
vectorizer: TfidfVectorizer = joblib.load("vectorizer.pkl")
clf: OneVsRestClassifier = joblib.load("classifier.pkl")
mlb: MultiLabelBinarizer = joblib.load("mlb.pkl")

# === 2. Read your CSV ===
# Replace "data.csv" with your actual filename/path
df = pd.read_csv("ntcir_datasets_label_2.csv", dtype={"subject": str})

# === 3. Preprocess text ===
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["text"] = df["title"] + " " + df["description"]


# === 4. Parse manual subjects ===
def parse_subject(s):
    """Try JSON decode; if that fails, split on commas."""
    try:
        subs = json.loads(s)
        if isinstance(subs, list):
            return subs
    except (json.JSONDecodeError, TypeError):
        pass
    return [tok.strip() for tok in s.split(",") if tok.strip()]


df["manual_subjects"] = df["subject"].apply(parse_subject)

# === 5. Vectorize & Predict ===
X_new = vectorizer.transform(df["text"])
Y_pred = clf.predict(X_new)
pred_lists = mlb.inverse_transform(Y_pred)
df["predicted_subjects"] = list(pred_lists)

# === 6. Evaluate performance ===
Y_true = mlb.transform(df["manual_subjects"])
print("=== Classification Report ===")
print(classification_report(Y_true, Y_pred, target_names=mlb.classes_))
print(f"Accuracy    : {accuracy_score(Y_true, Y_pred):.4f}")
print(f"Hamming Loss: {hamming_loss(Y_true, Y_pred):.4f}")
