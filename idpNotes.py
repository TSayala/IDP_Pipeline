import torch
import os
import re
import string

import nltk
from nltk.corpus import stopwords
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

import numpy as np
from PIL import Image, UnidentifiedImageError
import pytesseract
import cv2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd

import logging
from logging.handlers import RotatingFileHandler

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
def setupLogger(log_file='document_processor.log'):
    logger = logging.getLogger('DocumentProcessor')
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    console_handler = logging.StreamHandler()

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

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
            all_features = np.array(all_features)
            self.kmeans.fit(all_features)
        except Exception as e:
            logger.error(f"Error fitting KMeans: {type(e).__name__} - {str(e)}")

    def predict(self, images):
        try:
            predictions = []
            for image in images:
                features = self.extractLayoutFeatures(image)
                if features.size > 0:
                    pred = self.kmeans.predict(features)
                    predictions.append(pred)
                else:
                    predictions.append([])
            return predictions
        except Exception as e:
            logger.error(f"Error predicting layout features: {type(e).__name__} - {str(e)}")
            return []

class SupervisedDocumentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        self.model = RandomForestClassifier(random_state=42)
        self.label_encoder = LabelEncoder()

    def train(self, documents, labels):
        try:
            vectors = self.vectorizer.fit_transform(documents)
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.model.fit(vectors, encoded_labels)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {type(e).__name__} - {str(e)}")

    def classify(self, documents):
        try:
            vectors = self.vectorizer.transform(documents)
            predictions = self.model.predict(vectors)
            decoded_predictions = self.label_encoder.inverse_transform(predictions)
            return decoded_predictions
        except Exception as e:
            logger.error(f"Error during classification: {type(e).__name__} - {str(e)}")
            return []

def train_supervised_analyzer(documents, labels):
    analyzer = SupervisedDocumentAnalyzer()
    analyzer.train(documents, labels)
    return analyzer

def classify_documents(analyzer, document_images):
    texts = []
    for image in document_images:
        try:
            text = pytesseract.image_to_string(image)
            texts.append(text)
        except UnidentifiedImageError as e:
            logger.error(f"Error reading image: {type(e).__name__} - {str(e)}")
            texts.append("")
    return analyzer.classify(texts)

def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(directory_path, filename)
            try:
                image = Image.open(image_path)
                images.append(image)
            except UnidentifiedImageError as e:
                logger.error(f"Error loading image: {type(e).__name__} - {str(e)}")
    return images

def main(training_data, labels, test_directory):
    # Training the model
    analyzer = train_supervised_analyzer(training_data, labels)

    # Loading test images
    test_images = load_images_from_directory(test_directory)

    # Classifying test documents
    results = classify_documents(analyzer, test_images)

    # Creating a DataFrame with results
    filenames = [os.path.basename(image.filename) for image in test_images]
    df = pd.DataFrame({'Filename': filenames, 'Classification': results})
    return df

if __name__ == "__main__":
    # Example usage
    training_data = ["Sample document text 1", "Sample document text 2"]  # Replace with actual training data
    labels = ["label1", "label2"]  # Replace with actual labels
    test_directory = "/path/to/tif/files"  # Replace with actual path to TIF files

    result_df = main(training_data, labels, test_directory)
    print(result_df)
