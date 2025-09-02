# Aspect-Based-Opinion-Analysis-of-healthcare-reviews-
📌 Overview

This project implements Aspect-Based Opinion Mining (ABOM) to analyze healthcare-related drug reviews and classify sentiments on specific aspects (e.g., effectiveness, side effects, dosage, cost, treatment, doctor interaction).

It uses a combination of deep learning models (BERT, RNN, LSTM, CNN) and lexicon-based preprocessing (VADER + aspect extraction) to predict star ratings (0–4 mapped to 1–5 stars) for different aspects mentioned in patient reviews.

The system also compares model performances, generates detailed visualizations, and performs error analysis to evaluate robustness.

⚙️ Features

Data Preprocessing: Cleans text, extracts aspect-opinion pairs, and balances classes.

Lexicon-Based Initialization: Uses VADER sentiment analyzer as a fallback during data generation.

Models Implemented:

BERT (transformers, HuggingFace)

RNN (PyTorch)

LSTM (PyTorch)

CNN (PyTorch)

Custom Loss Function: Implements Focal Loss to handle class imbalance.

Performance Metrics: Accuracy, F1-score, Precision, Recall, MAE, per-class F1.

Visualizations:

Model comparison (Accuracy, F1, MAE)

Training progress (loss, accuracy, F1, MAE per epoch)

Confusion matrices

Per-class F1 scores

Aspect distribution plots

Error Analysis: Saves misclassified samples for inspection.

Best Model Selection: Automatically picks the best-performing model for final predictions.

📂 Project Structure
healthcare_sentiment_final.py   # Main project script
error_analysis.csv              # Sample output: misclassified reviews
metrics_results.csv             # Model performance metrics
best_model_results.csv          # Predictions from the best model
visualizations/                 # Folder with plots and comparison charts

🛠️ Requirements

Make sure you have the following installed:

python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
transformers
nltk
tqdm

Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers nltk tqdm


Download NLTK VADER lexicon (handled automatically in code):

import nltk
nltk.download('vader_lexicon')

📊 Dataset

The project expects a CSV file with drug reviews dataset (from Drugs.com reviews
).

Required columns:

drugName, condition, review, rating

Example path used in code:

D:\minor_project\healthcare_sentiment\drug_reviews\drugsComTrain_raw.csv

🧾 Pre-trained Embeddings

Uses GloVe embeddings (200d) for RNN, LSTM, CNN.

Download GloVe embeddings
 and place at:

D:\minor_project\glove.6B.200d.txt

🚀 Running the Project

Run the script with:

python healthcare_sentiment_final.py


Modify file paths inside main() if needed:

file_path = "path/to/drugsComTrain_raw.csv"
glove_path = "path/to/glove.6B.200d.txt"

📈 Outputs

After execution, the project generates:

Metrics & Results

metrics_results.csv: Model-wise performance comparison

best_model_results.csv: Best model’s predictions on aspects

Visualizations (saved in visualizations/)

model_comparison.png: Accuracy, F1, MAE comparison

model_metrics_table.png: Tabular summary

training_progress.png: Training vs validation curves

*_confusion_matrix.png: Confusion matrices for each model

*_per_class_f1.png: F1 scores per class

aspect_category_distribution.png: Aspect vs sentiment distribution

Error Analysis

error_analysis.csv: Misclassified samples for deeper inspection

🧪 Example Output

Sample review:

"This drug works well but has bad side effects."


Extracted aspects:

Effectiveness → Positive

Side effects → Negative

Predicted aspect ratings (from best model):

Aspect Category	Aspect	Opinion	Star Rating
effectiveness	works	well	5
side_effects	side effect	bad	1
📌 Future Improvements

Expand aspect lexicon with domain-specific ontologies.

Incorporate attention-based aspect extraction instead of rule-based.

Deploy as a web-based ABOM system for healthcare reviews.

Experiment with RoBERTa, XLNet, and domain-specific transformers.
