import torch
import psutil
import os
import re
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
    self.summarizer = pipeline('summarization', model='google/pegasus-xsum', device=0 if self.device == 'cuda' else -1)
    
    self.nlp = spacy.load('en_core_web_sm')
    self.nlp.add_pipe('sentencizer')
    if not Doc.has_extension("key_phrases"):
      Doc.set_extension("key_phrases", default=[])
    if not Span.has_extension("importance_score"):
      Span.set_extension("importance_score", default=0.0)
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
  @Language.component("extractKeyPhrases")
  def extractKeyPhrases(doc):
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
  @Language.component("calculateSentenceImportance")
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
          summary_result = self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)
          if summary_result and isinstance(summary_result, list) and len(summary_result) > 0:
            summary = summary_result[0].get('summary_text', '').strip()
            if summary:
              summaries.append(summary)
          else:
            logger.warning(f"File '{file_name}': Summarizer returned unexpected result for chunk {i+1}")
        except Exception as e:
          logger.error(f"File '{file_name}': Error summarizing chunk {i+1}: {type(e).__name__} - {str(e)}")
      
      if not summaries:
        logger.warning(f"File '{file_name}': No valid summaries generated")
        return f"Key Info: {key_info}\nSummary: {cleaned_text[:300]}... (truncated)"
      
      combined_summary = " ".join(summaries)

      # Generate final summary
      try:
        final_summary_result = self.summarizer(combined_summary, max_length=200, min_length=50, do_sample=False)
        if final_summary_result and isinstance(final_summary_result, list) and len(final_summary_result) > 0:
          final_summary = final_summary_result[0].get('summary_text', '').strip()
        else:
          logger.warning(f"File '{file_name}': Summarizer returned unexpected result for final summary")
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
  file_path, analyzer, labels = args

  result = {
    'file_path': file_path,
    'summary': None,
    'key_info': None,
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
    
    summary_result = analyzer.summarize(document_text, file_path)
    if summary_result.startswith("Key Info: {}"):
      result['error_message'] = summary_result.split('\nSummary: ', 1)[1]
      result['status'] = 'error'
      return result
    
    key_info_str, abstractive_summary = summary_result.split('\nSummary: ', 1)
    key_info = eval(key_info_str.replace("Key Info: ", ""))

    result['summary'] = abstractive_summary
    result['key_info'] = key_info

    if abstractive_summary:
      category, confidence = analyzer.classify(abstractive_summary, labels, file_path)
      result.update({
        'category': category,
        'confidence': confidence,
        'status': 'success' if category and confidence else 'partial success'
      })
    else:
      result['status'] = 'partial success'
      logger.warning(f"File '{file_path}': Empty abstractive summary, classification skipped")

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