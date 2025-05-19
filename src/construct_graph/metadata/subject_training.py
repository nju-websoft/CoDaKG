import pandas as pd
import glob
import json
import joblib

# Path pattern for JSON files (adjust the path as needed)
file_paths = glob.glob("eu_open_data/*.json")

data_frames = []
for file_path in file_paths:
    try:
        # Try reading as JSON lines (each line is a JSON object)
        df = pd.read_json(file_path, lines=True)
    except ValueError:
        # If not JSON lines, try loading the entire file content
        with open(file_path, "r") as f:
            content = json.load(f)
            # If the file contains a list of objects, convert to DataFrame directly
            if isinstance(content, list):
                df = pd.DataFrame(content)
            else:
                # If the file is a single JSON object, put it into a list
                df = pd.DataFrame([content])
    data_frames.append(df)

# Concatenate all DataFrames into one
df = pd.concat(data_frames, ignore_index=True)

print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

# Replace NaNs with empty string to avoid issues in concatenation
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")

# Concatenate title and description into a new column 'text'
df["text"] = df["title"] + " " + df["description"]

# Drop the original title and description columns if no longer needed
df = df.drop(columns=["title", "description"])

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Initialize the binarizer
mlb = MultiLabelBinarizer()
# Fit and transform the list of label lists into a binary matrix
Y = mlb.fit_transform(df["categories"])

# Store the label names for reference
label_names = mlb.classes_
print(f"Transformed labels into binary matrix of shape {Y.shape}")
print(f"Sample labels (first 5):\n{df['categories'].head()}")
print(f"Corresponding binary rows (first 5):\n{Y[:5]}")

# (Optional) Analyze label frequency to see imbalance
label_counts = Y.sum(axis=0)
for label, count in zip(label_names, label_counts):
    print(f"Label '{label}' appears in {count} samples")

# We will use class_weight='balanced' in the RandomForest to address imbalance.
# This automatically weights each label inversely to its frequency in the training data&#8203;:contentReference[oaicite:3]{index=3}.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (e.g., 80% train, 20% test)
X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    df["text"], Y, test_size=0.2, random_state=42
)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

# Fit the vectorizer on the training text and transform training and test text
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

print(f"TF-IDF matrix shape (train): {X_train_vec.shape}, (test): {X_test_vec.shape}")

from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

base_clf = LGBMClassifier(n_estimators=400, class_weight="balanced", random_state=42)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train_vec, Y_train)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Predict on the test set
Y_pred = clf.predict(X_test_vec)

report = classification_report(
    Y_test,
    Y_pred,
    target_names=label_names,
    zero_division=0,  # avoids errors if a label is never predicted
)
print("Classification Report (per label):\n")
print(report)

# Compute micro and macro averaged precision, recall, F1
precision_micro = precision_score(Y_test, Y_pred, average="micro")
precision_macro = precision_score(Y_test, Y_pred, average="macro")
recall_micro = recall_score(Y_test, Y_pred, average="micro")
recall_macro = recall_score(Y_test, Y_pred, average="macro")
f1_micro = f1_score(Y_test, Y_pred, average="micro")
f1_macro = f1_score(Y_test, Y_pred, average="macro")

print(f"Precision (Micro-average): {precision_micro:.3f}")
print(f"Precision (Macro-average): {precision_macro:.3f}")
print(f"Recall (Micro-average):    {recall_micro:.3f}")
print(f"Recall (Macro-average):    {recall_macro:.3f}")
print(f"F1-score (Micro-average):  {f1_micro:.3f}")
print(f"F1-score (Macro-average):  {f1_macro:.3f}")

joblib.dump(vectorizer, "vectorizer.pkl")

# clf is your OneVsRestClassifier wrapping an LGBMClassifier
joblib.dump(clf, "classifier.pkl")

# mlb is your MultiLabelBinarizer
joblib.dump(mlb, "mlb.pkl")
