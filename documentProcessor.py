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
from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
import pytesseract
import cv2

from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from logging.handlers import RotatingFileHandler

# Change this to match the number of available CPU cores
NUM_WORKERS = 6
CATEGORIES = [
  'plan of care', 'discharge summary', 'prescription request',
  'progress note', 'prior authorization', 'lab results',
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

class DocumentPreviewer:
  def __init__(self, file_path, default_width=800, default_height=600):
    self.file_path = os.path.abspath(file_path)
    self.image = Image.open(self.file_path)
    self.page_index = 0
    self.num_pages = self.image.n_frames

    self.root = tk.Tk()
    self.root.title('Document Previewer - ' + os.path.basename(self.file_path))

    # Set default window dimensions and minimum size
    self.root.geometry(f"{default_width}x{default_height}")
    self.root.minsize(200, 200)  # Set a minimum window size

    self.label = tk.Label(self.root)
    self.label.pack(expand=True, fill=tk.BOTH)

    self.prev_button = tk.Button(self.root, text="Previous", command=self.show_prev_page)
    self.prev_button.pack(side=tk.LEFT)

    self.next_button = tk.Button(self.root, text="Next", command=self.show_next_page)
    self.next_button.pack(side=tk.RIGHT)

    self.root.bind('<Configure>', self.on_resize)

    self.show_page(self.page_index)
    self.root.mainloop()

  def on_resize(self, event):
      self.show_page(self.page_index)

  def show_page(self, index):
    self.image.seek(index)
    window_width = self.root.winfo_width()
    window_height = self.root.winfo_height()

    # Ensure dimensions are greater than zero
    if window_width <= 0 or window_height <= 0:
      return

    # Resize the image to fit the window while maintaining aspect ratio
    aspect_ratio = self.image.width / self.image.height
    if window_width / window_height > aspect_ratio:
      new_height = window_height
      new_width = int(window_height * aspect_ratio)
    else:
      new_width = window_width
      new_height = int(window_width / aspect_ratio)

    # Ensure new dimensions are greater than zero
    if new_width <= 0 or new_height <= 0:
      return

    resized_image = self.image.resize((new_width, new_height-50), Image.LANCZOS)
    photo = ImageTk.PhotoImage(resized_image)
    self.label.config(image=photo)
    self.label.image = photo

  def show_prev_page(self):
    if self.page_index > 0:
      self.page_index -= 1
      self.show_page(self.page_index)

  def show_next_page(self):
    if self.page_index < self.num_pages - 1:
      self.page_index += 1
      self.show_page(self.page_index)

  def close_window(self):
    self.root.destroy()

def display_image(file_path, default_width=720, default_height=960):
  viewer = DocumentPreviewer(file_path, default_width, default_height)
  return viewer

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
        try:
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

          text_boxes = pytesseract.image_to_boxes(Image.fromarray(page))
          num_text_boxes = len(text_boxes.splitlines()) if text_boxes else 0

          layout_features = np.array([
            horizontal_lines,
            vertical_lines,
            text_regions,
            white_space_ratio,
            cv2.countNonZero(edges),
            num_text_boxes
          ])
          layout_features_list.append(layout_features)
        except Exception as e:
          logger.error(f"Error extracting layout features for page: {type(e).__name__} - {str(e)}")
          layout_features_list.append(np.zeros((6,)))

      if not layout_features_list:
        logger.error(f"No layout features extracted for {image_path}. Using default features")
        combined_layout_features = np.zeros((1, 6))
      else:                     
        combined_layout_features = np.mean(layout_features_list, axis=0).reshape(1, -1)

      #logger.debug(f"Layout shape for {image_path}: {combined_layout_features.shape}")
      return text, combined_layout_features
    
    except Exception as e:
      logger.error(f"Error extracting features: {type(e).__name__} - {str(e)}")
      return None, np.zeros((1, 6))

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
        logger.debug(f"Loaded {len(df)} few-shot examples from CSV file")
        self.few_shot_examples = df.to_dict('records')
        texts = []
        layouts = []
        categories = []

        logger.debug("Extracting features from few-shot examples")
        for _, row in df.iterrows():
          text, layout = self.extractFeatures(row['file_path'])
          texts.append(text)
          layouts.append(layout)
          categories.append(row['category'])
          self.few_shot_examples.append({
            'file_path': row['file_path'],
            'text': text,
            'layout': layout,
            'category': row['category']
          })
          #logger.debug(f"Layout shape for {row['file_path']}: {layout.shape}")

        # Check for consistent dimensions
        layout_shapes = [layout.shape[1] for layout in layouts]
        if len(set(layout_shapes)) != 1:
          raise ValueError(f"Inconsistent layout dimensions: {layout_shapes}")

        logger.debug("Encoding texts")
        self.few_shot_embeddings = self.model.encode(texts, convert_to_tensor=True)
        logger.debug(f"Few-shot embeddings shape: {self.few_shot_embeddings.shape}")

        logger.debug("Stacking layouts")
        self.few_shot_layouts = np.vstack(layouts)
        logger.debug(f"Few-shot layouts shape: {self.few_shot_layouts.shape}")

        logger.debug("Fitting scaler")
        self.scaler = StandardScaler()
        self.scaler.fit(self.few_shot_layouts)

        logger.debug("Saving few-shot examples to cache")
        self.cache.few_shot_examples = self.few_shot_examples
        self.cache.few_shot_embeddings = self.few_shot_embeddings
        self.cache.few_shot_layouts = self.few_shot_layouts
        self.cache.scaler = self.scaler
        self.cache.save()
        logger.info("Saved few-shot examples to cache")

      except Exception as e:
        logger.error(f"Error loading few-shot examples: {type(e).__name__} - {str(e)}")

  def updateFewShotExamples(self, image_path, verified_category):
    text, layout = self.extractFeatures(image_path)
    new_example = {
      'file_path': image_path,
      'text': text,
      'layout': layout,
      'category': verified_category
    }
    self.cache.update(new_example)

  def predict(self, image_path):
    text, layout = self.extractFeatures(image_path)
    if text is None or layout is None:
      return 'Error', 0.0

    # Few-shot classification
    query_embedding = self.model.encode(text, convert_to_tensor=True)
    layout = self.scaler.transform(layout)

    text_scores = util.cos_sim(query_embedding, self.few_shot_embeddings)[0]
    layout_distances = np.linalg.norm(self.few_shot_layouts - layout, axis=1)
    layout_scores = 1 / (1 + layout_distances)

    combined_scores = self.text_weight * text_scores + self.layout_weight * torch.tensor(layout_scores, device=self.device)

    top_k = 3
    top_indices = torch.topk(combined_scores, k=top_k).indices.cpu().numpy()

    categories = [self.few_shot_examples[i]['category'] for i in top_indices]
    scores = combined_scores[top_indices].cpu().numpy()

    total_score = np.sum(scores)
    normalized_scores = scores / total_score if total_score != 0 else np.zeros_like(scores)

    if np.any(normalized_scores > self.threshold):
      return categories[0], float(normalized_scores[0])
    else:
      return 'other', float(normalized_scores[0])
  
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

class FewShotClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, few_shot_processor):
    self.few_shot_processor = few_shot_processor

  def fit(self, X, y):
    return self
  
  def predict(self, X):
    return [self.few_shot_processor.predict(image_path)[0] for image_path in X]
  
  def predict_proba(self, X):
    return [list(self.few_shot_processor.predictProba(image_path).values()) for image_path in X]

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
  
  def update(self, new_example):
    self.few_shot_examples.append(new_example)
    text, layout = new_example['text'], new_example['layout']
    self.few_shot_embeddings = torch.cat([self.few_shot_embeddings, self.model.encode([text], convert_to_tensor=True)])
    self.few_shot_layouts = np.vstack([self.few_shot_layouts, layout])
    self.scaler.partial_fit(layout.reshape(1, -1))
    self.save()
  
  def clear(self):
    if os.path.exists(self.cache_file):
      os.remove(self.cache_file)
      logger.info(f"Cache file {self.cache_file} has been removed")
    self.init()
    logger.info("Cache has been cleared")

class ActiveLearningIDP:
  def __init__(self, few_shot_processor, threshold=0.7):
    self.few_shot_processor = few_shot_processor
    self.threshold = threshold
    self.classifier = FewShotClassifier(few_shot_processor)
    self.learner = ActiveLearner(
      estimator=self.classifier,
      query_strategy=uncertainty_sampling
    )
    self.queued_samples = []
    self.available_categories = set()

  def processDocument(self, file_path):
    category, confidence = self.few_shot_processor.predict(file_path)
    proba = self.few_shot_processor.predictProba(file_path)

    self.available_categories.add(category)

    if confidence < self.threshold:
      self.queued_samples.append((file_path, category, confidence, proba))

    return {
      'file_path': file_path,
      'category': category,
      'confidence': confidence,
      'needs_review': confidence < self.threshold
    }

  def fetchSamples(self, n_samples=10):
    if len(self.queued_samples) < n_samples:
      return self.queued_samples
    else:
      return self.queued_samples[:n_samples]
    
  def updateModel(self, reviewed_samples):
    for file_path, verified_category in reviewed_samples:
      self.few_shot_processor.updateFewShotExamples(file_path, verified_category)
      self.available_categories.add(verified_category)

    self.queued_samples = [sample for sample in self.queued_samples if sample[0] not in [rs[0] for rs in reviewed_samples]]

    X = [example['file_path'] for example in self.few_shot_processor.few_shot_examples]
    y = [example['category'] for example in self.few_shot_processor.few_shot_examples]
    self.learner.fit(X, y)

def userReview(sample, available_categories):
  file_path, predicted_category, confidence, proba = sample
  print(f"\nReviewing document: {file_path}")
  print(f"\nInitial prediction: {predicted_category} (Confidence: {confidence:.2f})")
  
  print("\nAvailable categories:")
  for i, category in enumerate(available_categories):
    print(f"{i+1}. {category}")
  print(f"{len(available_categories)+1}. Other (Write-in)")

  while True:
    choice = input("\nAccept the prediction? (y) or enter the number of the correct category: ")
    if choice.lower() == 'y':
      return file_path, predicted_category
    try:
      choice = int(choice)
      if 1 <= choice <= len(available_categories):
        return file_path, list(available_categories)[choice-1]
      elif choice == len(available_categories) + 1:
        new_category = input("Enter the new category: ")
        return file_path, new_category
    except ValueError:
      pass
    print("Invalid input. Please enter a valid category number or 'y' to accept the prediction")

def isFaxCover(text: str) -> Tuple[bool, List[Tuple[str, int]]]:
  keywords = ['fax', 'cover', 'sheet', 'attached']
  text = text.lower()
  matched_keywords = []
  keyword_count = 0
  for keyword in keywords:
    count = text.count(keyword)
    if count > 0:
      matched_keywords.append((keyword, count))
      keyword_count += count
    
  return keyword_count >= 2, matched_keywords

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
        if i == 0 and img.n_frames > 1:
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

def processDocument(args: Tuple[str, FewShotDocumentProcessor]) -> dict:
  file_path, analyzer = args

  result = {
    'file_path': file_path,
    'category': None,
    'confidence': None,
    'status': 'error',
    'error_message': None,
    'text': None,
    'text_length': 0,
    'num_pages': 0
  }

  try:
    logger.info(f"Processing document {file_path}")
    document_text, pages = analyzer.extractFeatures(file_path)
    result['text'] = document_text
    result['text_length'] = len(document_text)
    result['num_pages'] = len(pages)

    if result['text_length'] == 0:
      logger.warning(f"File '{file_path}': Extracted text is empty for file: {file_path}")
      raise ValueError("Extracted text is empty")
    
    category, confidence = analyzer.predict(file_path)

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

def processBatch(batch: List[str], analyzer: FewShotDocumentProcessor) -> List[dict]:
  results = []
  with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_file = {executor.submit(processDocument, (file_path, analyzer)): file_path for file_path in batch}
    for future in as_completed(future_to_file):
      result = future.result()
      results.append(result)
      logger.debug(f'Processed document: {result["file_path"]} (Status: {result["status"]})')     
  return results

def batchGenerator(file_paths, batch_size):
  for i in range(0, len(file_paths), batch_size):
    yield file_paths[i:i+batch_size]

def processAllDocuments(file_paths: List[str], examples: str, batch_size: int = 30, force_reload: bool = False) -> pd.DataFrame:
  cache = FewShotCache()
  analyzer = FewShotDocumentProcessor(cache=cache)
  analyzer.loadFewShotExamples(examples, force_reload=force_reload)
  alIDP = ActiveLearningIDP(analyzer)

  all_results = []
  total_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size else 0)

  logger.info(f'Starting to process {len(file_paths)} documents in {total_batches} batches')
  for i, batch in enumerate(batchGenerator(file_paths, batch_size), 1):
    logger.info(f'Processing batch {i}/{total_batches}')
    batch_results = [alIDP.processDocument(file_path) for file_path in batch] # processBatch(batch, analyzer)
    all_results.extend(batch_results)
    logger.info(f'Completed batch {i}/{total_batches}')

    samples_to_review = alIDP.fetchSamples()
    if samples_to_review:
      logger.info(f'Found {len(samples_to_review)} samples to review')
      reviewed_samples = []
      for sample in samples_to_review:
        reviewed_sample = userReview(sample, alIDP.available_categories)
        reviewed_samples.append(reviewed_sample)
      alIDP.updateModel(reviewed_samples)
      logger.info(f'Updated model with {len(reviewed_samples)} reviewed samples')

  df = pd.DataFrame(all_results)

  logger.info(f'All documents processed and validated successfully. Success: {df['status'].value_counts().get("success", 0)}, Errors: {df['status'].value_counts().get("error", 0)}')
  return df

def fetchFiles(directory):
  files = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith('.tif')]
  logger.info(f'Found {len(files)} TIF files in directory: {directory}')
  return files