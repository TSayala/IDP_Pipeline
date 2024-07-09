import torch
import psutil
import os
import re
import spacy
import numpy as np
from PIL import Image, UnidentifiedImageError
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
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

class UnsupervisedDocumentAnalyzer:

  # Initialize the model
  def __init__(self, model_name='facebook/bart-large-mnli'):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
    self.classifier = pipeline('zero-shot-classification', model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
    self.summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if self.device == 'cuda' else -1)
    self.nlp = spacy.load('en_core_web_sm')
    self.vectorizer = TfidfVectorizer(max_features=1000)
    self.kmeans = KMeans(n_clusters=15, random_state=42)
    self.max_chunk_length = 1024

  def cleanText(self, text):
    cleaned = re.sub(r'©.*?AM', '', text)
    cleaned = re.sub(r'~.*?of \d', '', cleaned) # Remove page numbers
    cleaned = re.sub(r'\n+', '\n', cleaned) # Replace multiple newlines with single newline
    cleaned = re.sub(r'\s+', ' ', cleaned) # Replace multiple spaces with single space
    cleaned = re.sub(r'In(?=\w)', '\n', cleaned) # Add newline after 'In' at the start of a sentence
    cleaned = re.sub(r'(?<=\w)In', '\n', cleaned) # Add newline before 'In' at the end of a sentence

    cleaned = re.sub(r'[^\w\s\.\,\:\-\(\)]', '', cleaned) # Remove special characters except .,:-()

    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return '\n'.join(lines)
    

  # Extract key information from the text
  def extractKeyInfo(self, text):

    patterns = {
      'Date': r'\d{2}-\d{2}-\d{4}', # DD-MM-YYYY
      'Patient': r'Patient:\s*([\w\s]+)',
      'DOB': r'Date of Birth:\s*([\d\-]+)',
    }
    
    """ date_pattern = r'\d{2}-\d{2}-\d{4}' # DD-MM-YYYY
    phone_pattern = r'\d{3}-\d{3}-\d{4}' # 123-456-7890
    name_pattern = r'(?:From:|Comment:)\s*([\w\s]+)'

    date = re.search(date_pattern, text)
    phone = re.search(phone_pattern, text)
    name = re.search(name_pattern, text) """

    key_info = []
    for key, pattern in patterns.items():
      match = re.search(pattern, text, re.DOTALL)
      if match:
        key_info[key] = match.group(1).strip()

    return key_info
  
  # Extract the main content from the text
  def extractMainContent(self, text):
    doc = self.nlp(text)
    content_sentences = [sent.text for sent in doc.sents if len(sent) > 5 and not sent.text.isupper()]
    return " ".join(content_sentences)
  
  # Summarize the text
  def summarize(self, text, file_name):
    try:
      logger.debug(f"File '{file_name}': Attempting to clean and summarize text of length {len(text)} characters")

      cleaned_text = self.cleanText(text)

      key_info = self.extractKeyInfo(cleaned_text)
      main_content = self.extractMainContent(cleaned_text)

      if len(main_content.split()) < 30:
        logger.warning(f"File '{file_name}': Text too short for summarization: {len(main_content.split())} words")
        return f"{key_info}\nContent: {main_content}"
      
      summary = self.summarizer(main_content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
      final_summary = f"Key Info: {key_info}\nSummary: {summary}"   

      logger.debug(f"File '{file_name}': Summary successful. Final summary length: {len(final_summary)} characters")
      return final_summary
    
    except Exception as e:
      logger.error(f"File '{file_name}': Error in summarization: {str(e)}")
      return text[:512] + '... (truncated)' if len(text) > 512 else text
  
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
  file_path, analyzer, labels = args

  result = {
    'file_path': file_path,
    'summary': None,
    'category': None,
    'confidence': None,
    'status': 'error',
    'error_message': None,
    'text_length': 0
  }

  try:
    logger.info(f"Processing document {file_path}")
    document_text = readTifFile(file_path)
    result['text_length'] = len(document_text)

    if result['text_length'] == 0:
      logger.warning(f"File '{file_path}': Extracted text is empty for file: {file_path}")
      raise ValueError("Extracted text is empty")
    
    summary = analyzer.summarize(document_text, file_path)
    category, confidence = analyzer.classify(summary, labels, file_path)

    """ summary, category, confidence = analyzer.analyzeDocument(document_text, labels)
    if summary is None or category is None or confidence is None:
      logger.warning(f'Error processing document {file_path}: Summary: {summary}, Category: {category}, Confidence: {confidence}')
      raise ValueError('Analysis failed') """

    result.update({
      'summary': summary,
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
def processBatch(batch, analyzer, labels):
  results = []
  with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    future_to_file = {executor.submit(processDocument, (file_path, analyzer, labels)): file_path for file_path in batch}
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