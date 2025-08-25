
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm
import transformers
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Define models list
MODELS = ['BERT', 'RNN', 'LSTM', 'CNN']

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

print(f"Using transformers version: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
torch.backends.cudnn.benchmark = True

# Load GloVe embeddings
def load_glove_embeddings(path, vocab, embedding_dim=200):
    print("Loading GloVe embeddings...")
    try:
        embeddings = np.zeros((len(vocab), embedding_dim))
        word2idx = vocab
        found = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe"):
                values = line.split()
                word = values[0]
                if word in word2idx:
                    embeddings[word2idx[word]] = np.array(values[1:], dtype=np.float32)
                    found += 1
        print(f"Found {found}/{len(vocab)} words in GloVe")
        return embeddings
    except FileNotFoundError:
        print(f"GloVe file {path} not found. Using random embeddings.")
        return None

# Load and preprocess data
def load_and_preprocess(file_path):
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Unnamed: 0': 'uniqueID', 'drugName': 'drugName', 'condition': 'condition',
                                'review': 'review', 'rating': 'rating', 'date': 'date', 'usefulCount': 'usefulCount'})
        df = df[['uniqueID', 'drugName', 'condition', 'review', 'rating']].dropna(subset=['review', 'rating'])
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('Int64')
        print(f"Dataset loaded with {len(df)} entries")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file {file_path} not found")
        return pd.DataFrame()

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract aspect-opinion pairs
def extract_aspect_opinions(text, common_aspects=None):
    if common_aspects is None:
        common_aspects = {
            'effectiveness': ['effective', 'efficacy', 'effect', 'work', 'working', 'works', 'helped', 'help', 'relief',
                              'improve', 'success'],
            'side_effects': ['side effect', 'side effects', 'adverse', 'reaction', 'symptom', 'nausea', 'headache',
                             'dizzy', 'pain', 'fatigue'],
            'dosage': ['dose', 'dosage', 'pill', 'tablet', 'capsule', 'mg', 'milligram', 'intake'],
            'cost': ['cost', 'price', 'expensive', 'affordable', 'cheap', 'costly'],
            'doctor': ['doctor', 'physician', 'specialist', 'prescription', 'prescribed', 'consult'],
            'treatment': ['treatment', 'therapy', 'medication', 'cure', 'relief', 'mood', 'sleep', 'recovery']
        }

    aspect_opinions = []
    words = text.split()
    for i, word in enumerate(words):
        aspect_category = None
        aspect_term = None
        for category, terms in common_aspects.items():
            for term in terms:
                if term in word or (i > 0 and term in f"{words[i - 1]} {word}") or word.startswith(term):
                    aspect_category = category
                    aspect_term = term
                    break
            if aspect_category:
                break
        if aspect_category:
            opinions = []
            window = words[max(0, i - 5):i] + words[i + 1:min(len(words), i + 6)]
            for w in window:
                if w in ['good', 'great', 'excellent', 'helpful', 'effective', 'satisfactory', 'okay', 'better',
                         'amazing', 'awesome', 'well']:
                    opinions.append(w)
                elif w in ['bad', 'poor', 'terrible', 'ineffective', 'worse', 'horrible', 'awful', 'dreadful']:
                    opinions.append(w)
            if opinions:
                aspect_opinions.append((aspect_category, aspect_term, opinions))
    return aspect_opinions

# Generate training data
def generate_training_data(df):
    print("Generating training data...")
    sid = SentimentIntensityAnalyzer()
    common_aspects = {
        'effectiveness': ['effective', 'efficacy', 'effect', 'work', 'working', 'works', 'helped', 'help', 'relief',
                          'improve', 'success'],
        'side_effects': ['side effect', 'side effects', 'adverse', 'reaction', 'symptom', 'nausea', 'headache', 'dizzy',
                         'pain', 'fatigue'],
        'dosage': ['dose', 'dosage', 'pill', 'tablet', 'capsule', 'mg', 'milligram', 'intake'],
        'cost': ['cost', 'price', 'expensive', 'affordable', 'cheap', 'costly'],
        'doctor': ['doctor', 'physician', 'specialist', 'prescription', 'prescribed', 'consult'],
        'treatment': ['treatment', 'therapy', 'medication', 'cure', 'relief', 'mood', 'sleep', 'recovery']
    }

    data = []
    positive_words = ['good', 'great', 'excellent', 'helpful', 'effective', 'satisfactory', 'okay', 'better', 'amazing',
                      'awesome', 'well']
    negative_words = ['bad', 'poor', 'terrible', 'ineffective', 'worse', 'horrible', 'awful', 'dreadful']

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting pairs"):
        clean_review = clean_text(row['review'])
        aspect_opinions = extract_aspect_opinions(clean_review, common_aspects)
        rating = row['rating']
        if pd.isna(rating):
            continue

        if rating <= 3:
            fallback_label = 0
        elif rating == 4:
            fallback_label = 1
        elif rating == 5:
            fallback_label = 2
        elif rating == 6:
            fallback_label = 3
        else:
            fallback_label = 4

        for aspect_category, aspect, opinions in aspect_opinions:
            for opinion in opinions:
                text = f"The {aspect} is {opinion}"
                sentiment = sid.polarity_scores(text)['compound']
                if opinion in positive_words:
                    label = 4 if sentiment > 0.2 else 3
                elif opinion in negative_words:
                    label = 0 if sentiment < -0.2 else 1
                else:
                    label = 2
                if abs(sentiment) < 0.15:
                    label = fallback_label
                data.append({
                    'text': text,
                    'aspect_category': aspect_category,
                    'aspect': aspect,
                    'opinion': opinion,
                    'label': label,
                    'sentiment_score': sentiment
                })

    data_df = pd.DataFrame(data)
    print(f"Generated {len(data_df)} training samples")
    print("Sample data (first 5):")
    print(data_df[['text', 'opinion', 'sentiment_score', 'label']].head())

    label_counts = data_df['label'].value_counts()
    print("Initial label distribution:")
    print(label_counts)

    print("Balancing classes...")
    balanced_data = []
    available_labels = label_counts[label_counts > 0].index
    target_count = min(max(label_counts.get(label, 0) for label in range(5)), 10000)
    if len(available_labels) < 5:
        print(f"Warning: Only {len(available_labels)} labels have samples: {available_labels.tolist()}")
    for label in range(5):
        if label in available_labels:
            label_data = data_df[data_df['label'] == label].sample(target_count, replace=True, random_state=42)
            balanced_data.append(label_data)
        else:
            print(f"Skipping label {label} (no samples)")
    if balanced_data:
        data_df = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        print("Warning: No valid data after balancing. Using original data.")
    print(f"Final balanced dataset: {len(data_df)} samples")
    print("Final label distribution:")
    final_counts = data_df['label'].value_counts()
    print(final_counts)
    return data_df, final_counts

# Dataset class
class ABOMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, vocab=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            tokens = text.lower().split()
            indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens[:self.max_length]]
            if len(indices) < self.max_length:
                indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
            return {
                'input_ids': torch.tensor(indices, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }

# Build vocabulary
def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_counts = defaultdict(int)
    for text in texts:
        for word in text.lower().split():
            word_counts[word] += 1
    for word, _ in sorted(word_counts.items()):
        vocab[word] = len(vocab)
    return vocab

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, glove_embeddings=None):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if glove_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(glove_embeddings))
            self.embedding.weight.requires_grad = False
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, glove_embeddings=None):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if glove_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(glove_embeddings))
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, glove_embeddings=None):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if glove_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(glove_embeddings))
            self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(n_filters) for _ in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).unsqueeze(1)
        conved = [torch.relu(self.bn[i](conv(embedded).squeeze(3))) for i, conv in enumerate(self.convs)]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Train custom model
def train_custom_model(model, train_loader, val_loader, epochs=7, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    model = model.to(device)

    best_val_acc = 0
    best_val_f1 = 0
    best_model_state = None
    patience = 3
    no_improve = 0
    metrics_history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_mae = mean_absolute_error(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='weighted')
        train_recall = recall_score(train_labels, train_preds, average='weighted')
        train_f1_per_class = f1_score(train_labels, train_preds, average=None)
        metrics_history['train'].append({
            'accuracy': train_accuracy,
            'f1': train_f1,
            'mae': train_mae,
            'loss': train_loss / len(train_loader),
            'precision': train_precision,
            'recall': train_recall,
            'f1_per_class': train_f1_per_class.tolist()
        })

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1_per_class = f1_score(val_labels, val_preds, average=None)
        metrics_history['val'].append({
            'accuracy': val_accuracy,
            'f1': val_f1,
            'mae': val_mae,
            'loss': val_loss / len(val_loader),
            'precision': val_precision,
            'recall': val_recall,
            'f1_per_class': val_f1_per_class.tolist()
        })

        print(f"Epoch {epoch + 1}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}, "
              f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, "
              f"Train Precision={train_precision:.4f}, Val Precision={val_precision:.4f}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model, metrics_history, train_preds, train_labels, val_preds, val_labels

# Train BERT
def train_bert_model(train_df, val_df, tokenizer):
    train_dataset = ABOMDataset(train_df['text'].values, train_df['label'].values, tokenizer)
    val_dataset = ABOMDataset(val_df['text'].values, val_df['label'].values, tokenizer)

    training_args_dict = {
        'output_dir': './bert_results',
        'num_train_epochs': 7,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'logging_dir': './bert_logs',
        'logging_steps': len(train_df) // 8 // 4,
        'save_strategy': 'epoch',
        'eval_strategy': 'epoch',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'accuracy',
        'report_to': 'none',
        'fp16': True,
        'learning_rate': 2e-5,
        'gradient_accumulation_steps': 4
    }

    training_args = TrainingArguments(**training_args_dict)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        mae = mean_absolute_error(labels, preds)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1_per_class = f1_score(labels, preds, average=None)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'mae': mae,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class.tolist()
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        hidden_dropout_prob=0.3
    ).to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    train_results = trainer.predict(train_dataset)
    val_results = trainer.predict(val_dataset)

    metrics_history = {'train': [], 'val': []}
    # Process log_history for training and evaluation metrics
    for log in trainer.state.log_history:
        if 'eval_accuracy' in log:
            metrics_history['val'].append({
                'accuracy': log['eval_accuracy'],
                'f1': log['eval_f1'],
                'mae': log['eval_mae'],
                'loss': log['eval_loss'],
                'precision': log['eval_precision'],
                'recall': log['eval_recall'],
                'f1_per_class': log.get('eval_f1_per_class', [0]*5)
            })
        if 'loss' in log and 'epoch' in log and 'eval_loss' not in log:
            metrics_history['train'].append({
                'accuracy': 0,
                'f1': 0,
                'mae': 0,
                'loss': log['loss'],
                'precision': 0,
                'recall': 0,
                'f1_per_class': [0]*5
            })

    # Add final train and val metrics from predict
    metrics_history['train'].append({
        'accuracy': train_results.metrics['test_accuracy'],
        'f1': train_results.metrics['test_f1'],
        'mae': train_results.metrics['test_mae'],
        'loss': train_results.metrics.get('test_loss', 0),
        'precision': train_results.metrics['test_precision'],
        'recall': train_results.metrics['test_recall'],
        'f1_per_class': train_results.metrics['test_f1_per_class']
    })
    metrics_history['val'].append({
        'accuracy': val_results.metrics['test_accuracy'],
        'f1': val_results.metrics['test_f1'],
        'mae': val_results.metrics['test_mae'],
        'loss': val_results.metrics['test_loss'],
        'precision': val_results.metrics['test_precision'],
        'recall': val_results.metrics['test_recall'],
        'f1_per_class': val_results.metrics['test_f1_per_class']
    })

    train_preds = train_results.predictions.argmax(-1)
    train_labels = train_results.label_ids
    val_preds = val_results.predictions.argmax(-1)
    val_labels = val_results.label_ids

    return model, tokenizer, metrics_history, train_results, val_results, train_preds, train_labels, val_preds, val_labels

# Predict with custom model
def predict_custom_model(model, texts, vocab, max_length=128):
    model.eval()
    dataset = ABOMDataset(texts, [0] * len(texts), vocab=vocab, max_length=max_length)
    loader = DataLoader(dataset, batch_size=16)
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return np.array(preds) + 1

# Predict with BERT
def predict_bert_model(model, tokenizer, texts):
    dataset = ABOMDataset(texts, [0] * len(texts), tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=16)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return np.array(preds) + 1

# Visualize comparison
def visualize_comparison(metrics_history, data_df, df, initial_label_counts, train_results, val_results,
                        rnn_train_preds, rnn_train_labels, rnn_val_preds, rnn_val_labels,
                        lstm_train_preds, lstm_train_labels, lstm_val_preds, lstm_val_labels,
                        cnn_train_preds, cnn_train_labels, cnn_val_preds, cnn_val_labels):
    os.makedirs('visualizations', exist_ok=True)

    accuracies = [metrics_history[m]['val'][-1]['accuracy'] for m in MODELS]
    f1_scores = [metrics_history[m]['val'][-1]['f1'] for m in MODELS]
    maes = [metrics_history[m]['val'][-1]['mae'] for m in MODELS]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(MODELS))
    width = 0.25
    plt.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x, f1_scores, width, label='F1 Score', color='lightgreen')
    plt.bar(x + width, maes, width, label='MAE', color='salmon')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison: Accuracy, F1 Score, MAE')
    plt.xticks(x, MODELS)
    plt.legend()
    for i, (acc, f1, mae) in enumerate(zip(accuracies, f1_scores, maes)):
        plt.text(i - width, acc + 0.01, f'{acc:.3f}', ha='center')
        plt.text(i, f1 + 0.01, f'{f1:.3f}', ha='center')
        plt.text(i + width, mae + 0.01, f'{mae:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.axis('off')
    table_data = [[m, f"{acc:.3f}", f"{f1:.3f}", f"{mae:.3f}"] for m, acc, f1, mae in
                  zip(MODELS, accuracies, f1_scores, maes)]
    plt.table(cellText=table_data, colLabels=['Model', 'Accuracy', 'F1 Score', 'MAE'], loc='center', cellLoc='center')
    plt.title('Model Performance Summary')
    plt.savefig('visualizations/model_metrics_table.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pie(initial_label_counts, labels=initial_label_counts.index, autopct='%1.1f%%')
    ax1.set_title('Initial Label Distribution')
    ax2.pie(data_df['label'].value_counts(), labels=range(5), autopct='%1.1f%%')
    ax2.set_title('Balanced Label Distribution')
    plt.savefig('visualizations/label_proportion.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for model_name in MODELS:
        history = metrics_history[model_name]
        train_acc = [m['accuracy'] for m in history['train'] if m['accuracy'] > 0]
        val_acc = [m['accuracy'] for m in history['val']]
        train_loss = [m['loss'] for m in history['train']]
        val_loss = [m['loss'] for m in history['val']]
        train_f1 = [m['f1'] for m in history['train'] if m['f1'] > 0]
        val_f1 = [m['f1'] for m in history['val']]
        train_mae = [m['mae'] for m in history['train'] if m['mae'] > 0]
        val_mae = [m['mae'] for m in history['val']]

        if train_acc:
            axes[0, 0].plot(range(1, len(train_acc)+1), train_acc, label=f'{model_name} Train')
        axes[0, 0].plot(range(1, len(val_acc)+1), val_acc, '--', label=f'{model_name} Val')
        axes[0, 1].plot(range(1, len(train_loss)+1), train_loss, label=f'{model_name} Train')
        axes[0, 1].plot(range(1, len(val_loss)+1), val_loss, '--', label=f'{model_name} Val')
        if train_f1:
            axes[1, 0].plot(range(1, len(train_f1)+1), train_f1, label=f'{model_name} Train')
        axes[1, 0].plot(range(1, len(val_f1)+1), val_f1, '--', label=f'{model_name} Val')
        if train_mae:
            axes[1, 1].plot(range(1, len(train_mae)+1), train_mae, label=f'{model_name} Train')
        axes[1, 1].plot(range(1, len(val_mae)+1), val_mae, '--', label=f'{model_name} Val')

    axes[0, 0].set_title('Accuracy')
    axes[0, 1].set_title('Loss')
    axes[1, 0].set_title('F1 Score')
    axes[1, 1].set_title('MAE')
    for ax in axes.flatten():
        ax.legend()
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('visualizations/training_progress.png')
    plt.close()

    # Confusion Matrices
    preds_labels = {
        'BERT': (train_results.predictions.argmax(-1), train_results.label_ids, val_results.predictions.argmax(-1), val_results.label_ids),
        'RNN': (rnn_train_preds, rnn_train_labels, rnn_val_preds, rnn_val_labels),
        'LSTM': (lstm_train_preds, lstm_train_labels, lstm_val_preds, lstm_val_labels),
        'CNN': (cnn_train_preds, cnn_train_labels, cnn_val_preds, cnn_val_labels)
    }
    for model in MODELS:
        train_preds, train_labels, val_preds, val_labels = preds_labels[model]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        train_cm = confusion_matrix(train_labels, train_preds, labels=range(5))
        val_cm = confusion_matrix(val_labels, val_preds, labels=range(5))
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=range(5), yticklabels=range(5))
        ax1.set_title(f'{model} Train Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=range(5), yticklabels=range(5))
        ax2.set_title(f'{model} Validation Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        plt.tight_layout()
        plt.savefig(f'visualizations/{model}_confusion_matrix.png')
        plt.close()

    # Per-class F1 Scores
    for model in MODELS:
        train_f1 = metrics_history[model]['train'][-1]['f1_per_class']
        val_f1 = metrics_history[model]['val'][-1]['f1_per_class']
        plt.figure(figsize=(10, 6))
        x = np.arange(5)
        width = 0.35
        plt.bar(x - width/2, train_f1, width, label='Train', color='skyblue')
        plt.bar(x + width/2, val_f1, width, label='Validation', color='lightgreen')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title(f'{model} Per-class F1 Scores')
        plt.xticks(x, range(5))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'visualizations/{model}_per_class_f1.png')
        plt.close()

    print("Visualizations saved in 'visualizations/'")

# Main function
def main(file_path, glove_path):
    print("Starting Aspect-Based Opinion Mining analysis...")
    df = load_and_preprocess(file_path)
    if df.empty:
        return

    data_df, initial_label_counts = generate_training_data(df)
    if data_df.empty:
        print("No training data generated. Exiting.")
        return

    label_counts = data_df['label'].value_counts(normalize=True)
    print("Label distribution:")
    print(label_counts)

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])

    print("Training BERT...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model, bert_tokenizer, bert_metrics_history, bert_train_results, bert_val_results, bert_train_preds, bert_train_labels, bert_val_preds, bert_val_labels = train_bert_model(train_df, val_df, tokenizer)

    vocab = build_vocab(train_df['text'].values)
    glove_embeddings = load_glove_embeddings(glove_path, vocab)
    vocab_size = len(vocab)

    train_dataset = ABOMDataset(train_df['text'].values, train_df['label'].values, vocab=vocab)
    val_dataset = ABOMDataset(val_df['text'].values, val_df['label'].values, vocab=vocab)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)

    print("Training RNN...")
    rnn_model = RNNModel(vocab_size, embedding_dim=200, hidden_dim=768, output_dim=5, glove_embeddings=glove_embeddings)
    rnn_model, rnn_metrics_history, rnn_train_preds, rnn_train_labels, rnn_val_preds, rnn_val_labels = train_custom_model(rnn_model, train_loader, val_loader, epochs=7, lr=0.0001)

    print("Training LSTM...")
    lstm_model = LSTMModel(vocab_size, embedding_dim=200, hidden_dim=768, output_dim=5, glove_embeddings=glove_embeddings)
    lstm_model, lstm_metrics_history, lstm_train_preds, lstm_train_labels, lstm_val_preds, lstm_val_labels = train_custom_model(lstm_model, train_loader, val_loader)

    print("Training CNN...")
    cnn_model = CNNModel(vocab_size, embedding_dim=200, n_filters=100, filter_sizes=[3, 4, 5], output_dim=5, glove_embeddings=glove_embeddings)
    cnn_model, cnn_metrics_history, cnn_train_preds, cnn_train_labels, cnn_val_preds, cnn_val_labels = train_custom_model(cnn_model, train_loader, val_loader)

    metrics_history = {
        'BERT': bert_metrics_history,
        'RNN': rnn_metrics_history,
        'LSTM': lstm_metrics_history,
        'CNN': cnn_metrics_history
    }
    train_results = {
        'BERT': bert_train_results,
        'RNN': (rnn_train_preds, rnn_train_labels),
        'LSTM': (lstm_train_preds, lstm_train_labels),
        'CNN': (cnn_train_preds, cnn_train_labels)
    }
    val_results = {
        'BERT': bert_val_results,
        'RNN': (rnn_val_preds, rnn_val_labels),
        'LSTM': (lstm_val_preds, lstm_val_labels),
        'CNN': (cnn_val_preds, cnn_val_labels)
    }

    best_acc = -1
    best_model_name = None
    for model_name in MODELS:
        acc = metrics_history[model_name]['val'][-1]['accuracy']
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
    if best_model_name is None:
        best_model_name = 'BERT'
        best_acc = metrics_history['BERT']['val'][-1]['accuracy']
    print(f"\nBest model: {best_model_name} (Accuracy: {best_acc:.4f})")

    visualize_comparison(
        metrics_history, data_df, df, initial_label_counts, bert_train_results, bert_val_results,
        rnn_train_preds, rnn_train_labels, rnn_val_preds, rnn_val_labels,
        lstm_train_preds, lstm_train_labels, lstm_val_preds, lstm_val_labels,
        cnn_train_preds, cnn_train_labels, cnn_val_preds, cnn_val_labels
    )

    # Error Analysis
    error_analysis = []
    for model in MODELS:
        val_preds = locals()[f'{model.lower()}_val_preds']
        val_labels = locals()[f'{model.lower()}_val_labels']
        errors = np.where(val_preds != val_labels)[0]
        for idx in errors[:5]:
            error_analysis.append({
                'Model': model,
                'Index': idx,
                'Text': val_df.iloc[idx]['text'],
                'Predicted': val_preds[idx],
                'True': val_labels[idx]
            })
    error_df = pd.DataFrame(error_analysis)
    error_df.to_csv('error_analysis.csv', index=False)
    print("Saved error_analysis.csv")

    # Aspect Category Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data_df, x='aspect_category', hue='label')
    plt.title('Label Distribution per Aspect Category')
    plt.xlabel('Aspect Category')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig('visualizations/aspect_category_distribution.png')
    plt.close()

    sample_review = "This drug works well but has bad side effects."
    common_aspects = {
        'effectiveness': ['effective', 'efficacy', 'effect', 'work', 'working', 'works', 'helped', 'help', 'relief',
                          'improve', 'success'],
        'side_effects': ['side effect', 'side effects', 'adverse', 'reaction', 'symptom', 'nausea', 'headache', 'dizzy',
                         'pain', 'fatigue'],
        'dosage': ['dose', 'dosage', 'pill', 'tablet', 'capsule', 'mg', 'milligram', 'intake'],
        'cost': ['cost', 'price', 'expensive', 'affordable', 'cheap', 'costly'],
        'doctor': ['doctor', 'physician', 'specialist', 'prescription', 'prescribed', 'consult'],
        'treatment': ['treatment', 'therapy', 'medication', 'cure', 'relief', 'mood', 'sleep', 'recovery']
    }

    print(f"\nResults of best model ({best_model_name}) on sample review: '{sample_review}'")
    clean_review = clean_text(sample_review)
    aspect_opinions = extract_aspect_opinions(clean_review, common_aspects)
    print("Extracted aspect-opinions:", aspect_opinions)

    results = []
    if aspect_opinions:
        texts = [f"The {aspect} is {opinion}" for _, aspect, opinions in aspect_opinions for opinion in opinions]
        if best_model_name == 'BERT':
            star_ratings = predict_bert_model(bert_model, bert_tokenizer, texts)
        else:
            model = {'RNN': rnn_model, 'LSTM': lstm_model, 'CNN': cnn_model}[best_model_name]
            star_ratings = predict_custom_model(model, texts, vocab)

        i = 0
        for aspect_category, aspect, opinions in aspect_opinions:
            for opinion in opinions:
                results.append({
                    'aspect_category': aspect_category,
                    'aspect': aspect,
                    'opinion': opinion,
                    'star_rating': star_ratings[i]
                })
                i += 1

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No aspects found in the review.")
    else:
        print("Predicted aspect ratings:")
        print(results_df)
        results_df.to_csv('best_model_results.csv', index=False)
        print("Saved best_model_results.csv")

    metrics_df = []
    for model in MODELS:
        train_m = metrics_history[model]['train'][-1]
        val_m = metrics_history[model]['val'][-1]
        metrics_df.append({
            'Model': model,
            'Dataset': 'Train',
            'Accuracy': train_m['accuracy'],
            'F1': train_m['f1'],
            'MAE': train_m['mae'],
            'Loss': train_m['loss'],
            'Precision': train_m['precision'],
            'Recall': train_m['recall'],
            'F1_per_class_0': train_m['f1_per_class'][0],
            'F1_per_class_1': train_m['f1_per_class'][1],
            'F1_per_class_2': train_m['f1_per_class'][2],
            'F1_per_class_3': train_m['f1_per_class'][3],
            'F1_per_class_4': train_m['f1_per_class'][4]
        })
        metrics_df.append({
            'Model': model,
            'Dataset': 'Validation',
            'Accuracy': val_m['accuracy'],
            'F1': val_m['f1'],
            'MAE': val_m['mae'],
            'Loss': val_m['loss'],
            'Precision': val_m['precision'],
            'Recall': val_m['recall'],
            'F1_per_class_0': val_m['f1_per_class'][0],
            'F1_per_class_1': val_m['f1_per_class'][1],
            'F1_per_class_2': val_m['f1_per_class'][2],
            'F1_per_class_3': val_m['f1_per_class'][3],
            'F1_per_class_4': val_m['f1_per_class'][4]
        })
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv('metrics_results.csv', index=False)
    print("Saved metrics_results.csv")

    print("\nAnalysis completed!")

if __name__ == "__main__":
    file_path = "D:\\minor_project\\healthcare_sentiment\\drug_reviews\\drugsComTrain_raw.csv"
    glove_path = "D:\\minor_project\\glove.6B.200d.txt"
    main(file_path, glove_path)
