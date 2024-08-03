import psutil
import os
import re
import io
import string
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from collections import Counter
from typing import List, Tuple
import unicodedata

import numpy as np
from PIL import Image, UnidentifiedImageError
import pytesseract
import cv2

from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from transformers import BertTokenizer, BertModel

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from logging.handlers import RotatingFileHandler

# Change this to match the number of available CPU cores
NUM_WORKERS = 6
CATEGORIES = [
  'plan of care', 'discharge summary', 'prescription request',
  'progress note', 'prescription authorization', 'lab results',
  'result notification', 'formal records request', 'patient chart note',
  'return to work', 'answering service', 'spam', 'other'
]

# Setup logging
def setupLogger(log_file='document_processor.log'):
  logger = logging.getLogger('DocumentProcessor')
  logger.setLevel(logging.DEBUG)

  # Create handlers
  file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
  console_handler = logging.StreamHandler()

  # Create formatters
  file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(file_format)
  console_handler.setFormatter(console_format)

  # Add handlers to the logger
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  return logger
_logger = setupLogger()
def getLogger():
  global _logger
  return _logger
logger = getLogger()

class TextCleaner:
  def __init__(self):
    self.common_errors = {
      'l': 'i',
      '0': 'o',
      '1': 'i',
      '5': 's',
      '8': 'b',
      '9': 'g',
      '|': 'i',
      '—': '-',
    }

  def clean(self, text: str) -> str:
    """
    Main method to clean the input text using a series of cleaning steps.
    """
    #text = self.removeNonPrintableChars(text)
    text = self.removeSpecialChars(text)
    text = self.removeHeaderFooter(text)
    text = self.fixLineBreaks(text)
    text = self.removeExtraWhitespace(text)
    #text = self.correctCommonErrors(text)
    #text = self.normalizeUnicode(text)
    #text = self.correctWordSplits(text)
    return text
    
  def removeNonPrintableChars(self, text: str) -> str:
    """
    Remove non-printable characters from the input text.
    """
    return ''.join(ch for ch in text if ch.isprintable())
  
  def fixLineBreaks(self, text: str) -> str:
    """
    Fix inconsistent line breaks and remove unnecessary ones.
    """
    lines = text.splitlines()
    fixed_lines = []
    for line in lines:
      if line.strip():
        fixed_lines.append(line.strip())
    return ' '.join(fixed_lines)
  
  def removeExtraWhitespace(self, text: str) -> str:
    """
    Remove extra whitespace characters from the input text.
    """
    return ' '.join(text.split())
  
  def correctCommonErrors(self, text: str) -> str:
    """
    Correct common OCR misrecognition errors.
    """
    for error, correction in self.common_errors.items():
      text = text.replace(error, correction)
    return text
  
  def removeSpecialChars(self, text: str) -> str:
    """
    Remove special characters from the input text while keeping basic punctuation.
    """
    return re.sub(r'[^a-zA-Z0-9\s.,!?/-]', '', text)
  
  def normalizeUnicode(self, text: str) -> str:
    """
    Normalize Unicode characters to their closest ASCII equivalents.
    """
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
  
  def spellCheck(self, text: str) -> str:
    """
    [UNIMPLEMENTED] Perform spell checking on the input text.
    """
    return text
  
  def correctWordSplits(self, text: str) -> str:
    """
    Correct words that have been split incorrectly by OCR.
    """
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
  
  def isHeaderFooter(self, line: str) -> bool:
    """
    Helper method to identify potential header and footer text.
    """
    return bool(re.match(r'(page \d+/\d+|[ivxlcdm]+)', line.strip().lower()))
  
  def removeHeaderFooter(self, text: str) -> str:
    """
    Remove header and footer text from the input text.
    """
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not self.isHeaderFooter(line)]
    return '\n'.join(cleaned_lines)
  
class FewShotDocumentProcessor:
  def __init__(self, model_name='all-MiniLM-L6-v2', cache=None):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = SentenceTransformer(model_name).to(self.device)
    self.cache = cache if cache else FewShotCache()
    self.text_weight = 0.7
    self.layout_weight = 1.0 - self.text_weight
    self.threshold = 0.1
    self.scaler = None

  def extractFeatures(self, image_path):
    try:
      # extract text using OCR
      text, pages = readTifFile(image_path)

      # extract layout features
      layout_features_list = []
      for page in pages:
        gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
          horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 5)
          vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 5)
        else:
          horizontal_lines, vertical_lines = 0, 0

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = len(contours)
        white_space_ratio = 1 - (cv2.countNonZero(thresh) / (page.shape[0] * page.shape[1]))

        layout_features = np.array([
          horizontal_lines,
          vertical_lines,
          text_regions,
          white_space_ratio,
          cv2.countNonZero(edges),
          len(pytesseract.image_to_boxes(Image.fromarray(page)).splitlines())
        ])
        layout_features_list.append(layout_features)

      combined_layout_features = np.mean(layout_features_list, axis=0)
      return text, combined_layout_features.reshape(1, -1)
    
    except Exception as e:
      logger.error(f"Error extracting features: {type(e).__name__} - {str(e)}")
      return None, None

  def loadFewShotExamples(self, csv_path, force_reload=False):
    if not force_reload and self.cache.load():
      logger.info("Few-shot examples loaded from cache")
      self.few_shot_examples = self.cache.few_shot_examples
      self.few_shot_embeddings = self.cache.few_shot_embeddings
      self.few_shot_layouts = self.cache.few_shot_layouts
      self.scaler = self.cache.scaler
    else:
      logger.info("Few-shot examples not found in cache. Loading from CSV file")
      try:
        df = pd.read_csv(csv_path)
        self.few_shot_examples = df.to_dict('records')
        texts = []
        layouts = []

        for example in self.few_shot_examples:
          text, layout = self.extractFeatures(example['file_path'])
          texts.append(text)
          layouts.append(layout)

        self.few_shot_embeddings = self.model.encode(texts, convert_to_tensor=True)
        self.few_shot_layouts = np.vstack(layouts)
        self.scaler = StandardScaler()
        self.scaler.fit(self.few_shot_layouts)

        self.cache.few_shot_examples = self.few_shot_examples
        self.cache.few_shot_embeddings = self.few_shot_embeddings
        self.cache.few_shot_layouts = self.few_shot_layouts
        self.cache.scaler = self.scaler
        self.cache.save()
        logger.info("Saved few-shot examples to cache")

      except Exception as e:
        logger.error(f"Error loading few-shot examples: {type(e).__name__} - {str(e)}")

  def predict(self, image_path):
    text, layout = self.extractFeatures(image_path)
    if text is None or layout is None:
      return 'Error', 0.0
    query_embedding = self.model.encode(text, convert_to_tensor=True)
    
    if self.scaler is None:
      logger.error("Scaler is not initialized")
      return 'Error', 0.0
    layout = self.scaler.transform(layout)

    text_scores = util.cos_sim(query_embedding, self.few_shot_embeddings)[0]
    layout_distances = np.linalg.norm(self.few_shot_layouts - layout, axis=1)
    layout_scores = 1 / (1 + layout_distances)

    combined_scores = self.text_weight * text_scores + self.layout_weight * torch.tensor(layout_scores, device=self.device)

    top_k = 3
    top_indices = torch.topk(combined_scores, k=top_k).indices

    categories = [self.few_shot_examples[i]['category'] for i in top_indices]
    scores = combined_scores[top_indices].tolist()

    total_score = sum(scores)
    normalized_scores = [score / total_score for score in scores]

    if normalized_scores[0] > self.threshold:
      return categories[0], normalized_scores[0]
    else:
      return 'Uncertain', normalized_scores[0]
    #top_result = torch.argmax(combined_scores)
    #return self.few_shot_examples[top_result]['category']
  
  def predictProba(self, image_path):
    text, layout = self.extractFeatures(image_path)
    if text is None or layout is None:
      return 'Error', 0.0
    query_embedding = self.model.encode(text, convert_to_tensor=True)
    layout = self.scaler.transform(layout)

    text_scores = util.cos_sim(query_embedding, self.few_shot_embeddings)[0]
    layout_distances = np.linalg.norm(self.few_shot_layouts - layout, axis=1)
    layout_scores = 1 / (1 + layout_distances)

    combined_scores = self.text_weight * text_scores + self.layout_weight * torch.tensor(layout_scores, device=self.device)
    probabilities = torch.softmax(combined_scores, dim=0)

    category_probs = {}
    for i, example in enumerate(self.few_shot_examples):
      category = example['category']
      if category not in category_probs:
        category_probs[category] = probabilities[i].item()
      else:
        category_probs[category] = max(category_probs[category], probabilities[i].item())

    return category_probs

class FewShotCache:
  def __init__(self, cache_file='data/examples/cache/few_shot_cache.pkl'):
    self.cache_file = cache_file
    self.few_shot_examples = None
    self.few_shot_embeddings = None
    self.few_shot_layouts = None
    self.scaler = None

  def save(self):
    with open(self.cache_file, 'wb') as f:
      pickle.dump({
        'few_shot_examples': self.few_shot_examples,
        'few_shot_embeddings': self.few_shot_embeddings,
        'few_shot_layouts': self.few_shot_layouts,
        'scaler': self.scaler
      }, f)

  def load(self):
    if os.path.exists(self.cache_file):
      with open(self.cache_file, 'rb') as f:
        data = pickle.load(f)
        self.few_shot_examples = data.get('few_shot_examples')
        self.few_shot_embeddings = data.get('few_shot_embeddings')
        self.few_shot_layouts = data.get('few_shot_layouts')
        self.scaler = data.get('scaler')
      return True
    return False
  
  def clear(self):
    if os.path.exists(self.cache_file):
      os.remove(self.cache_file)
      logger.info(f"Cache file {self.cache_file} has been removed")
    self.init()
    logger.info("Cache has been cleared")

class PostClassificationValidator:
  def __init__(self):
    self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    self.features = None
    self.labels = None

  def prepareData(self, df):
    self.features = df[['confidence', 'text_length', 'num_pages']]
    self.labels = df['category']
  
  def train(self):
    X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
    self.classifier.fit(X_train, y_train)
    y_pred = self.classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

  def validate(self, new_results):
    features = new_results[['confidence', 'text_length', 'num_pages']]
    validated_categories = self.classifier.predict(features)
    validated_probas = self.classifier.predict_proba(features)

    new_results['validated_category'] = validated_categories
    new_results['validated_confidence'] = validated_probas.max(axis=1)
    return new_results
  
  def saveModel(self, model_file='data/models/post_classifier.pkl'):
    with open(model_file, 'wb') as f:
      pickle.dump(self.classifier, f)

  def loadModel(self, model_file='data/models/post_classifier.pkl'):
    with open(model_file, 'rb') as f:
      self.classifier = pickle.load(f)

class ActiveLearner:
  def __init__(self, initial_data, initial_labels):
    self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    self.learner = ActiveLearner(
      estimator=self.classifier,
      X_training=initial_data,
      y_training=initial_labels,
      query_strategy=uncertainty_sampling      
    )

  def query(self, unlabeled_data, n_instances=1):
    query_idx, query_instance = self.learner.query(unlabeled_data, n_instances=n_instances)
    return query_idx, query_instance
  
  def teach(self, X, y):
    self.learner.teach(X, y)

  def predict(self, X):
    return self.learner.predict(X)
  
  def predictProba(self, X):
    return self.learner.predict_proba(X)
  
def prepareDataForActiveLearning(df):
  features = df[['confidence', 'text_length', 'num_pages']].values
  labels = df['category'].values
  return features, labels

def activeLearningLoop(active_learner, unlabeled_data, batch_size=10):
  while len(unlabeled_data) > 0:
    query_idx, query_instance = active_learner.query(unlabeled_data, n_instances=batch_size)
    
    # simulate labeling
    human_labels = unlabeled_data['category'].iloc[query_idx].values

    active_learner.teach(query_instance, human_labels)
    unlabeled_data = unlabeled_data.drop(unlabeled_data.index[query_idx])
    logger.info(f"Remaining unlabeled data: {len(unlabeled_data)}")

  return active_learner

def getBertEmbeddings(text, max_length=512):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
  with torch.no_grad():
    outputs = model(**inputs)
  return outputs.last_hidden_state[:, 0, :].numpy()

def extractLayoutFeatures(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
  horizontal_lines = 0
  vertical_lines = 0
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      if abs(y2 - y1) < 5:
        horizontal_lines += 1
      elif abs(x2 - x1) < 5:
        vertical_lines += 1

  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  dilated = cv2.dilate(thresh, kernel, iterations=1)
  contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  text_regions = len(contours)
  white_space_ratio = 1 - (cv2.countNonZero(thresh) / (image.shape[0] * image.shape[1]))
  return np.array([horizontal_lines, vertical_lines, text_regions, white_space_ratio])

class CNNLSTM(nn.Module):
  def __init__(self, num_layout_features, text_embedding_dim, hidden_dim, num_classes):
    super(CNNLSTM, self).__init__()
    self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool1d(kernel_size=2)
    self.fc1 = nn.Linear(64 * (num_layout_features // 2), hidden_dim)

    self.lstm = nn.LSTM(text_embedding_dim, hidden_dim, batch_first=True)
    self.fc2 = nn.Linear(hidden_dim * 2, num_classes)

  def forward(self, layout_features, text_embeddings):
    x_layout = layout_features.unsqueeze(1)
    x_layout = self.pool(F.relu(self.conv1(x_layout)))
    x_layout = self.pool(F.relu(self.conv2(x_layout)))
    x_layout = x_layout.view(x_layout.size(0), -1)
    x_layout = F.relu(self.fc1(x_layout))

    _, (h_n, _) = self.lstm(text_embeddings)
    x_text = h_n.squeeze(0)

    x_combined = torch.cat((x_layout, x_text), dim=1)
    output = self.fc2(x_combined)
    return output
  
def trainCNNLSTM(model, train_loader, val_loader, num_epochs, learning_rate, model_file='data/models/cnn_lstm_model.pth'):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  best_accuracy = 0.0

  for epoch in range(num_epochs):
    model.train()
    for layout_features, text_embeddings, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(layout_features, text_embeddings)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for layout_features, text_embeddings, labels in val_loader:
        outputs = model(layout_features, text_embeddings)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%')

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      torch.save(model.state_dict(), model_file)
      logger.info(f'Model saved with accuracy: {best_accuracy:.2f}%')

  return model


class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  
class NeuralNetValidator:
  def __init__(self, input_size, hidden_size, num_classes):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = SimpleNN(input_size, hidden_size, num_classes).to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    self.scaler = StandardScaler()

    def prepareData(self, features, labels):
      X_scaled = self.scaler.fit_transform(features)
      X_train, X_val, y_train, y_val = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

      X_train_tensor = torch.FloatTensor(X_train).to(self.device)
      y_train_tensor = torch.LongTensor(y_train).to(self.device)
      X_val_tensor = torch.FloatTensor(X_val).to(self.device)
      y_val_tensor = torch.LongTensor(y_val).to(self.device)

      train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
      val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
      return train_dataset, val_dataset
    
  def train(self, train_dataset, val_dataset, num_epochs=50, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
      self.model.train()
      for batch_features, batch_labels in train_loader:
        self.optimizer.zero_grad()
        outputs = self.model(batch_features)
        loss = self.criterion(outputs, batch_labels)
        loss.backward()
        self.optimizer.step()

      # validation
      self.model.eval()
      val_loss = 0
      correct = 0
      total = 0
      with torch.no_grad():
        for batch_features, batch_labels in val_loader:
          outputs = self.model(batch_features)
          loss = self.criterion(outputs, batch_labels)
          val_loss += loss.item()
          _, predicted = torch.max(outputs.data, 1)
          total += batch_labels.size(0)
          correct += (predicted == batch_labels).sum().item()

      logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

  def validate(self, features):
    X_scaled = self.scaler.transform(features)
    X_tensor = torch.FloatTensor(X_scaled).to(self.device)
    with torch.no_grad():
      outputs = self.model(X_tensor)
      probabilities = torch.softmax(outputs, dim=1)
      predicted = torch.argmax(probabilities, dim=1)
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

class VotingClassifier:
  def __init__(self, threshold=0.6):
    self.threshold = threshold

  def combinePredictions(self, few_shot_pred, al_pred, nn_pred, cnn_lstm_pred,
                         few_shot_conf, al_conf, nn_conf, cnn_lstm_conf):
    predictions = [few_shot_pred, al_pred, nn_pred, cnn_lstm_pred]
    confidences = [few_shot_conf, al_conf, nn_conf, cnn_lstm_conf]

    if len(set(predictions)) == 1:
      return predictions[0], max(confidences)
    
    for pred in predictions:
      if predictions.count(pred) >= 3:
        conf = max([conf for p, conf in zip(predictions, confidences) if p == pred])
        if conf >= self.threshold:
          return pred, conf
        
    max_conf_index = confidences.index(max(confidences))
    return predictions[max_conf_index], confidences[max_conf_index]
  
def applyVoting(row, voter):
  return voter.combinePredictions(
    row['category'], row['al_category'], row['nn_category'], row['cnn_lstm_category'],
    row['confidence'], row['al_confidence'], row['nn_confidence'], row['cnn_lstm_confidence']
  )

def prepareDataForNeuralNet(df):
  features = df[['confidence', 'text_length', 'num_pages']].values
  labels = pd.factorize(df['category'])[0]
  return features, labels, len(set(labels))

def isFaxCover(text: str) -> Tuple[bool, List[str]]:
  keywords = ['fax', 'cover', 'sheet', 'attached']
  text = text.lower()
  matched_keywords = []
  keyword_count = 0
  for keyword in keywords:
    count = text.count(keyword)
    if count > 0:
      matched_keywords.append(keyword)
      keyword_count += count
    
    return keyword_count >= 2, matched_keywords
      
# Read the TIF file and extract text using OCR  
def readTifFile(file_path: str) -> Tuple[str, List[np.ndarray]]:
  """
  Reads a TIF file and extracts text from it using OCR.
  Iterates by page and returns cleaned text and images for each page if a fax cover page is not detected.

  returns:
  Tuple[str, List[np.ndarray]]: A tuple containing the cleaned text and a list of images.
  """
  cleaner = TextCleaner()
  try:
    with Image.open(file_path) as img:
      pages = []
      texts = []
      for i in range(img.n_frames):
        img.seek(i)
        rgb_image = img.convert('RGB')
        np_image = np.array(rgb_image)
        text = pytesseract.image_to_string(np_image)
        text = cleaner.clean(text)
        if i == 0:
          is_fax_cover, matched_keywords = isFaxCover(text)
          if is_fax_cover:
            logger.info(f"File '{file_path}': Detected a fax cover page. Keywords: {matched_keywords}")
            continue
        texts.append(text)
        pages.append(np_image)
    return " ".join(texts), pages
  except UnidentifiedImageError:
    logger.error(f"File '{file_path}': Not a valid image file or is corrupted")
    raise
  except OSError as e:
    logger.error(f"File '{file_path}': OS Error occurred when reading file: {str(e)}")
    raise
  except Exception as e:
    logger.error(f"File '{file_path}': Unexpected error reading file: {str(e)}")
    raise

# Process the document
def processDocument(args: Tuple[str, FewShotDocumentProcessor]) -> dict:
  file_path, analyzer = args

  result = {
    'file_path': file_path,
    'category': None,
    'confidence': None,
    'status': 'error',
    'error_message': None,
    'text': None,
    'text_length': 0
  }

  try:
    logger.info(f"Processing document {file_path}")
    document_text, _ = analyzer.extractFeatures(file_path)
    result['text'] = document_text
    result['text_length'] = len(document_text)

    if result['text_length'] == 0:
      logger.warning(f"File '{file_path}': Extracted text is empty for file: {file_path}")
      raise ValueError("Extracted text is empty")
    
    category, confidence = analyzer.predict(file_path)

    if category not in CATEGORIES:
      category = 'other'

    result.update({
      'category': category,
      'confidence': confidence,
      'status': 'success' if category and confidence else 'partial success'
    })
    logger.info(f"File '{file_path}': Processing complete (Status: {result["status"]})")

  except Exception as e:
    logger.error(f"File '{file_path}': Error processing document: {type(e).__name__} - {str(e)}")
    result['error_message'] = f'{type(e).__name__} - {str(e)}'

  return result

# Process the batches
def processBatch(batch: List[str], analyzer: FewShotDocumentProcessor) -> List[dict]:
  results = []
  with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_file = {executor.submit(processDocument, (file_path, analyzer)): file_path for file_path in batch}
    for future in as_completed(future_to_file):
      result = future.result()
      results.append(result)
      logger.debug(f'Processed document: {result["file_path"]} (Status: {result["status"]})')     
  return results

# Generate batches of files
def batchGenerator(file_paths, batch_size):
  for i in range(0, len(file_paths), batch_size):
    yield file_paths[i:i+batch_size]

# Process all the documents in the corpus
def processAllDocuments(file_paths: List[str], examples: str, batch_size: int = 30, force_reload: bool = False) -> pd.DataFrame:
  cache = FewShotCache()
  analyzer = FewShotDocumentProcessor(cache=cache)
  analyzer.loadFewShotExamples(examples, force_reload=force_reload)

  all_results = []
  total_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size else 0)

  logger.info(f'Starting to process {len(file_paths)} documents in {total_batches} batches')
  for i, batch in enumerate(batchGenerator(file_paths, batch_size), 1):
    logger.info(f'Processing batch {i}/{total_batches}')
    batch_results = processBatch(batch, analyzer)
    all_results.extend(batch_results)
    logger.info(f'Completed batch {i}/{total_batches}')

  df = pd.DataFrame(all_results)

  all_texts = []
  all_layout_features = []
  for file_path in df['file_path']:
    text, layout = analyzer.extractFeatures(file_path)
    all_texts.append(text)
    all_layout_features.append(layout.flatten())

  df['extracted_text'] = all_texts
  layout_features = np.array(all_layout_features)

  # Active learning
  features, labels = prepareDataForActiveLearning(df)
  sample_size = min(100, len(features))
  if sample_size == 0:
    logger.error("No features found for active learning")
    raise ValueError("No features found for active learning")
  initial_idx = np.random.choice(len(features), size=sample_size, replace=False)
  active_learner = ActiveLearner(features[initial_idx], labels[initial_idx])

  unlabeled_data = df.drop(df.index[initial_idx])
  active_learner = activeLearningLoop(active_learner, unlabeled_data)

  al_predictions = active_learner.predict(features)
  al_probas = active_learner.predictProba(features)
  df['al_category'] = al_predictions
  df['al_confidence'] = np.max(al_probas, axis=1)

  # Neural network (SimpleNN)
  nn_features, nn_labels, num_classes = prepareDataForNeuralNet(df)
  nn_validator = NeuralNetValidator(input_size=nn_features.shape[1], hidden_size=64, num_classes=num_classes)
  train_dataset, val_dataset = nn_validator.prepareData(nn_features, nn_labels)
  nn_validator.train(train_dataset, val_dataset)

  nn_predictions, nn_probas = nn_validator.validate(nn_features)
  df['nn_category'] = nn_predictions
  df['nn_confidence'] = np.max(nn_probas, axis=1)

  # CNN-LSTM
  text_embeddings = np.array([getBertEmbeddings(text) for text in df['extracted_text']])

  few_shot_layout_features = np.array([example['layout_features'].flatten() for example in cache.few_shot_examples])
  few_shot_text_embeddings = np.array([getBertEmbeddings(example['text']) for example in cache.few_shot_examples])
  few_shot_labels = LabelEncoder().fit_transform([example['category'] for example in cache.few_shot_examples])

  combined_layout_features = np.vstack([few_shot_layout_features, layout_features])
  combined_text_embeddings = np.vstack([few_shot_text_embeddings, text_embeddings])
  combined_labels = np.concatenate([few_shot_labels, nn_labels])

  dataset = TensorDataset(
    torch.FloatTensor(combined_layout_features),
    torch.FloatTensor(combined_text_embeddings),
    torch.LongTensor(combined_labels)
  )
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32)

  cnn_lstm_model = CNNLSTM(
    num_layout_features=combined_layout_features.shape[1],
    text_embedding_dim=combined_text_embeddings.shape[1],
    hidden_dim=64,
    num_classes=len(set(combined_labels))
  )

  model_file = 'data/models/cnn_lstm_model.pth'
  if os.path.exists(model_file):
    cnn_lstm_model.load_state_dict(torch.load(model_file))
    logger.info(f'Model loaded from file: {model_file}')
  else:
    cnn_lstm_model = trainCNNLSTM(cnn_lstm_model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, model_file=model_file)

  cnn_lstm_model.eval()
  with torch.no_grad():
    cnn_lstm_outputs = cnn_lstm_model(
      torch.FloatTensor(layout_features),
      torch.FloatTensor(text_embeddings)
    )
    cnn_lstm_probas = F.softmax(cnn_lstm_outputs, dim=1)
    cnn_lstm_predictions = torch.argmax(cnn_lstm_probas, dim=1)

  df['cnn_lstm_category'] = cnn_lstm_predictions.cpu().numpy()
  df['cnn_lstm_confidence'] = torch.max(cnn_lstm_probas, dim=1)[0].cpu().numpy()

  voter = VotingClassifier(threshold=0.6)
  df['voted_category'], df['voted_confidence'] = zip(*df.apply(lambda row: applyVoting(row, voter), axis=1))
      
  """ logger.info(f'Starting to validate {len(df)} documents')
  validator = PostClassificationValidator()
  validator.prepareData(df)
  validator.train()
  df = validator.validate(df) """

  logger.info(f'All documents processed and validated successfully. Success: {df['status'].value_counts().get("success", 0)}, Errors: {df['status'].value_counts().get("error", 0)}')
  return df

def fetchFiles(directory):
  files = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith('.tif')]
  logger.info(f'Found {len(files)} TIF files in directory: {directory}')
  return files