import torch
import psutil
import os
import re
import io
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from collections import Counter

import numpy as np
from PIL import Image, UnidentifiedImageError
import pytesseract
import cv2

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

from tqdm.notebook import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from logging.handlers import RotatingFileHandler

# Change this to match the number of available CPU cores
NUM_WORKERS = 6

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

def registerExtensions():
  if not Doc.has_extension("key_phrases"):
    Doc.set_extension("key_phrases", default=[])
  if not Span.has_extension("importance_score"):
    Span.set_extension("importance_score", default=0.0)

registerExtensions()

class DocumentLayoutAnalyzer:
  def __init__(self, n_clusters=15):
    self.n_clusters = n_clusters
    self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

  def extractLayoutFeatures(self, image):
    try:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      edges = cv2.Canny(gray, 50, 150, apertureSize=3)
      contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      features = []
      for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        features.append([area, aspect_ratio])
      return np.array(features)
    except Exception as e:
      logger.error(f"Error extracting layout features: {type(e).__name__} - {str(e)}")
      return []
    
  def fit(self, images):
    try:
      all_features = []
      for image in images:
        features = self.extractLayoutFeatures(image)
        all_features.extend(features)
      self.kmeans.fit(all_features)
    except Exception as e:
      logger.error(f"Error fitting KMeans model: {type(e).__name__} - {str(e)}")

  def predictLayout(self, image):
    try:
      features = self.extractLayoutFeatures(image)
      labels = self.kmeans.predict(features)
      return np.bincount(labels).argmax()
    except Exception as e:
      logger.error(f"Error predicting layout: {type(e).__name__} - {str(e)}")
      return None

class FaxCoverDetector:
  def __init__(self):
    self.keywords = ['fax', 'cover', 'sheet']

  def isFaxCover(self, image):
    try:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      text = pytesseract.image_to_string(gray).lower()
      keyword_count = sum(1 for keyword in self.keywords if keyword in text)
      return keyword_count > len(self.keywords) / 2
    except Exception as e:
      logger.error(f"Error detecting fax cover: {type(e).__name__} - {str(e)}")
      return False

class SupervisedDocumentAnalyzer:
  def __init__(self):
    self.text_vectorizer = TfidfVectorizer(max_features=1000)
    self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    self.label_encoder = LabelEncoder()

  def extractFeatures(self, image, text):
    try:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
      hist = hist.flatten() / hist.sum()

      text_features = self.text_vectorizer.transform([text]).toarray().flatten()
      return np.concatenate([hist, text_features])
    except Exception as e:
      logger.error(f"Error extracting features: {type(e).__name__} - {str(e)}")
      return None
    
  def fit(self, file_paths, labels):
    try:
      features = []
      texts = []
      for file_path in file_paths:
        with Image.open(file_path) as img:
          np_image = np.array(img.convert('RGB'))
          text = pytesseract.image_to_string(np_image)
          texts.append(text)
          features.append(self.extractFeatures(np_image, text))
      self.text_vectorizer.fit(texts)
      features = [self.extractFeatures(np.array(Image.open(fp).convert('RGB')), texts[i]) for i, fp in enumerate(file_paths)]
      encoded_labels = self.label_encoder.fit_transform(labels)
      self.classifier.fit(features, encoded_labels)

      # Evaluate
      X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
      self.classifier.fit(X_train, y_train)
      y_pred = self.classifier.predict(X_test)
      logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}")
    except Exception as e:
      logger.error(f"Error fitting classifier: {type(e).__name__} - {str(e)}")

  def predict(self, image, text):
    try:
      features = self.extractFeatures(image, text)
      pred = self.classifier.predict([features])
      return self.label_encoder.inverse_transform(pred)[0]
    except Exception as e:
      logger.error(f"Error predicting category: {type(e).__name__} - {str(e)}")
      return None
    
  def predictProba(self, image, text):
    try:
      features = self.extractFeatures(image, text)
      proba = self.classifier.predict_proba([features])
      return dict(zip(self.label_encoder.classes_, proba[0]))
    except Exception as e:
      logger.error(f"Error predicting category probabilities: {type(e).__name__} - {str(e)}")
      return None

class UnsupervisedDocumentAnalyzer:

  # Initialize the model
  def __init__(self, model_name='facebook/bart-large-mnli'):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
    self.classifier = pipeline('zero-shot-classification', model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
    self.summarizer = pipeline('summarization', model='google/pegasus-xsum', device=0 if self.device == 'cuda' else -1)
    
    self.nlp = spacy.load('en_core_web_sm')

    self.nlp.add_pipe('sentencizer')
    self.nlp.add_pipe("extractKeyPhrases", last=True)
    self.nlp.add_pipe("calculateSentenceImportance", last=True)
    
    self.vectorizer = TfidfVectorizer(max_features=1000)
    self.kmeans = KMeans(n_clusters=15, random_state=42)
    self.max_chunk_length = 1024

  def preprocessText(self, text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\d+', '', text) # Remove digits
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove non-alphanumeric characters except spaces
    tokens = nltk.word_tokenize(text)
    return tokens
  
  def removeStopwords(self, tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens
  
  def lemmatizeText(self, tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens
  
  def cleanText(self, text):
    tokens = self.preprocessText(text)
    filtered_tokens = self.removeStopwords(tokens)
    lemmatized_tokens = self.lemmatizeText(filtered_tokens)
    return " ".join(lemmatized_tokens)
    
  @staticmethod
  @spacy.Language.component("extractKeyPhrases")
  def extractKeyPhrases(doc):
    doc._.key_phrases = []
    for chunk in doc.noun_chunks:
      doc._.key_phrases.append(chunk)
    for token in doc:
      if token.pos == "VERB":
        verb_phrase = list(token.subtree)
        start = verb_phrase[0].i
        end = verb_phrase[-1].i + 1
        doc._.key_phrases.append(doc[start:end])
    return doc
  
  @staticmethod
  @spacy.Language.component("calculateSentenceImportance")
  def calculateSentenceImportance(doc):
    for sent in doc.sents:
      importance_score = sum([token.is_alpha and not token.is_stop for token in sent]) / len(sent)
      sent._.importance_score = importance_score
    return doc
  
  # Extract key information from the text
  def extractKeyInfo(self, text):    
    doc = self.nlp(text[:1000000])

    # Extract named entities
    entities = {ent.label_: ent.text for ent in doc.ents if ent.label_ in ['DATE', 'PERSON', 'ORG']}

    # Extract key phrases
    key_phrases = [chunk.text for chunk in doc._.key_phrases if len(chunk.text.split()) > 1][:5] # Extract only top 5 phrases with more than one word

    # Extract important sentences
    important_sentences = sorted(doc.sents, key=lambda s: s._.importance_score, reverse=True)[:3]
    important_sentences = [sent.text for sent in important_sentences]

    # Combine all key information
    key_info = {
      'Entities': entities,
      'Key_Phrases': key_phrases,
      'Important_Sentences': important_sentences
    }

    return key_info
  
  def chunkText(self, text):
    """
    Split the input text into chunks of specified maximum length.

    Args:
    text (str): The input text to split into chunks.
    max_chunk_length (int): The maximum length of each chunk.

    Returns:
    List[str]: A list of text chunks.
    """
    doc = self.nlp(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for sent in doc.sents:
      if current_chunk_length + len(sent) <= self.max_chunk_length:
        current_chunk.append(sent.text)
        current_chunk_length += len(sent)
      else:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sent.text]
        current_chunk_length = len(sent)

    if current_chunk:
      chunks.append(" ".join(current_chunk))
    
    return chunks
  
  # Extract the main content from the text
  def extractMainContent(self, text):
    try:
      doc = self.nlp(text[:1000000])
      content_sentences = [sent.text for sent in doc.sents if len(sent) > 5 and not sent.text.isupper()]
      return " ".join(content_sentences) if content_sentences else ""
    except Exception as e:
      logger.error(f"Error extracting main content: {type(e).__name__} - {str(e)}")
      return ""
  
  # Summarize the text
  def summarize(self, text, file_name):
    try:
      logger.debug(f"File '{file_name}': Attempting to summarize text of length {len(text)} characters")

      cleaned_text = self.cleanText(text)
      if not cleaned_text.strip():
        logger.warning(f"File '{file_name}': Cleaned text is empty")
        return "Key Info: {}\nSummary: Empty document"
      
      key_info = self.extractKeyInfo(cleaned_text)
      chunks = self.chunkText(cleaned_text)

      summaries = []
      for i, chunk in enumerate(chunks):
        try:
          logger.debug(f"File '{file_name}': Summarizing chunk {i+1} of {len(chunks)}")
          summary_result = self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)

          logger.debug(f"File '{file_name}': Raw summary result for chunk {i+1}: {summary_result}")

          if summary_result and isinstance(summary_result, list) and len(summary_result) > 0:
            summary = summary_result[0].get('summary_text', '').strip()
            if summary:
              summaries.append(summary)
              logger.debug(f"File '{file_name}': Successfully summarized chunk {i+1}. Length: {len(summary)} characters")
          else:
            logger.warning(f"File '{file_name}': Summarizer returned unexpected result for chunk {i+1}")
        except IndexError as e:
          logger.error(f"File '{file_name}': IndexError summarizing chunk {i+1}: {str(e)}")
        except Exception as e:
          logger.error(f"File '{file_name}': Error summarizing chunk {i+1}: {type(e).__name__} - {str(e)}")
      
      if not summaries:
        logger.warning(f"File '{file_name}': No valid summaries generated")
        return f"Key Info: {key_info}\nSummary: Unable to generate summary. Original text (truncated): {cleaned_text[:300]}..."
      
      combined_summary = " ".join(summaries)

      # Generate final summary
      try:
        logger.debug(f"File '{file_name}': Generating final summary")
        final_summary_result = self.summarizer(combined_summary, max_length=200, min_length=50, do_sample=False)

        logger.debug(f"File '{file_name}': Raw final summary result: {final_summary_result}")

        if final_summary_result and isinstance(final_summary_result, list) and len(final_summary_result) > 0:
          final_summary = final_summary_result[0].get('summary_text', '').strip()
          if not final_summary:
            logger.warning(f"File '{file_name}': Final summary is empty")
            final_summary = combined_summary
        else:
          logger.warning(f"File '{file_name}': Unexpected final summary result structure")
          final_summary = combined_summary
      except IndexError as e:
        logger.error(f"File '{file_name}': IndexError generating final summary: {str(e)}")
        final_summary = combined_summary
      except Exception as e:
        logger.error(f"File '{file_name}': Error generating final summary: {type(e).__name__} - {str(e)}")
        final_summary = combined_summary

      result = f"Key Info: {key_info}\nSummary: {final_summary}"
      logger.debug(f"File '{file_name}': Summarization successful. Result length: {len(final_summary)} characters")
      return result
    
    except Exception as e:
      logger.error(f"File '{file_name}': Error in summarization: {type(e).__name__} - {str(e)}")
      return f"Key Info: {{}}\nSummary: Error in summarization: {type(e).__name__} - {str(e)}"
  
  # Classify the text
  def classify(self, text, labels, file_name):
    try:
      logger.debug(f"File '{file_name}': Attempting to classify text of length {len(text)} with {len(labels)} labels")
      result = self.classifier(text, labels)
      logger.debug(f"File '{file_name}': Classification successful. Top label: {result['labels'][0]}, Confidence: {result['scores'][0]}")
      return result['labels'][0], result['scores'][0]
    except Exception as e:
      logger.error(f"File '{file_name}': Error in classification: {str(e)}")
      return None, None
  
  # Cluster the documents
  def cluster(self, documents):
    vectors = self.vectorizer.fit_transform(documents)
    clusters = self.kmeans.fit_predict(vectors)
    return clusters
  
  # Analyze the document =
  def analyzeDocument(self, text, labels):
    summary = self.summarize(text)
    if summary is None:
      return None, None, None
    category, confidence = self.classify(summary, labels)
    if category is None or confidence is None:
      return summary, None, None
    return summary, category, confidence
  
# Read the TIF file and extract text using OCR  
def readTifFile(file_path):
  try:
    with Image.open(file_path) as img:
      pages = []
      for i in range(img.n_frames):
        img.seek(i)
        page = np.array(img)
        text = pytesseract.image_to_string(page)
        pages.append(text)
    return " ".join(pages)
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
def processDocument(args):
  file_path, classifier, fax_detector = args

  result = {
    'file_path': file_path,
    'category': None,
    'confidence': None,
    'status': 'error',
    'error_message': None,
    'text_length': 0,
    'is_fax_cover': False,
    'num_pages': 0
  }

  try:
    logger.info(f"Processing document {file_path}")
    with Image.open(file_path) as img:
      result['num_pages'] = img.n_frames
      full_text = ""
      for i in range(img.n_frames):
        img.seek(i)
        np_image = np.array(img)
        if i == 0:
          result['is_fax_cover'] = fax_detector.isFaxCover(np_image)
          if result['is_fax_cover']:
            logger.info(f"File '{file_path}': Detected a fax cover page.")
            result['status'] = 'skipped'
            return result
          page_text = pytesseract.image_to_string(np_image, config='--psm 6')
          full_text += page_text + '\n|\n'
    result['text_length'] = len(full_text)

    if result['text_length'] == 0:
      logger.warning(f"File '{file_path}': Extracted text is empty for file: {file_path}")
      raise ValueError("Extracted text is empty")
    
    result['category'] = classifier.predict(np.image, full_text)
    probabilities = classifier.predictProba(np_image, full_text)
    result['confidence'] = probabilities.get(result['category'], 0.0)

    result['status'] = 'success'
    logger.info(f"File '{file_path}': Processing complete (Status: {result["status"]})")

  except Exception as e:
    logger.error(f"File '{file_path}': Error processing document: {type(e).__name__} - {str(e)}")
    result['error_message'] = f'{type(e).__name__} - {str(e)}'

  return result

# Process the batches
def processBatch(batch, classifier, fax_detector):
  results = []
  with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_file = {executor.submit(processDocument, (file_path, classifier, fax_detector)): file_path for file_path in batch}
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
def processAllDocuments(file_paths, analyzer, labels, batch_size=30):
  all_results = []
  total_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size else 0)

  logger.info(f'Starting to process {len(file_paths)} documents in {total_batches} batches')
  for i, batch in enumerate(batchGenerator(file_paths, batch_size), 1):
    logger.info(f'Processing batch {i}/{total_batches}')
    batch_results = processBatch(batch, analyzer, labels)
    all_results.extend(batch_results)
    logger.info(f'Completed batch {i}/{total_batches}')

  df = pd.DataFrame(all_results)
  logger.info(f'All documents processed successfully. Success: {df['status'].value_counts().get("success", 0)}, Errors: {df['status'].value_counts().get("error", 0)}')
  return df

def fetchFiles(directory):
  files = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith('.tif')]
  logger.info(f'Found {len(files)} TIF files in directory: {directory}')
  return files