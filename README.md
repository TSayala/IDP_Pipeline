# Intelligent Document Processing (IDP) Classification Pipeline
*By: Tanay Sayala*

The IDP classification pipeline I have developed is a robust system designed to efficiently process and classify large volumes of documents. The pipeline is highly configurable, allowing for verbose logging, CPU affinity settings, and batch processing. It leverages few-shot learning to handle limited labeled data and active learning to iteratively improve the model.

Jump to:
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Key Components](#key-components)
- [Pipeline Workflow](#pipeline-workflow)
- [Wiki](#wiki)

## Pipeline Overview

1. **Document Extraction**: The pipeline starts by reading TIF files and extracting text using Optical Character Recognition (OCR).

2. **Feature Extraction**: Both textual and layout features are extracted from each document and used in predictions.

3. **Few-Shot Learning**: A pre-trained model is used to generate initial classifications based on a small set of labeled examples.

4. **Active Learning**: The system identifies uncertain classifications and requests user feedback to improve its model.

5. **Batch Processing**: Documents are processed in batches for efficiency, as well as to allow for user review between batches.

6. **Continuous Improvement**: The model is updated based on user feedback, improving its accuracy over time.

## Usage

### Setup
1. Prepare a CSV file with initial few-shot examples. The CSV must have at least the following structure:
```csv
file_path,category
example1.tif,category1
example2.tif,category2
...
```
2. In ```pipeline.py```, set the paths for the example CSV file and the directory containing documents to be processed.

3. Ensure the required virtual environment is set up. If not, create a new virtual environment using included ```setup.bat``` file.

### Executing the pipeline
1. Open a Powershell terminal window with administrator privileges.

2. Navigate to the project directory.
```bash
cd <path_to_project_directory>
```
3. Activate the virtual environment.
```bash
./documentProcessor/scripts/activate
```
4. Run the pipeline with the desired arguments. The terminal will display information about the processing status. Detailed execution information is updated live in the log file.
```bash
./pipeline.py
```
5. After each batch is processed, the pipeline will prompt the user to review uncertain classifications. The user can accept the prediction, choose from existing categories, or add a new category. These messages will appear in the terminal. A preview of the document needing to be reviewed will also be displayed in a separate preview window.

6. The pipeline will continue processing batches until all documents are classified. A sample of the final results will be displayed in the terminal. The user will be prompted to export the results to a CSV file, to which they can specify the file name of the resulting CSV file. The CSV can then be loaded into Excel, pandas, or any other tool for further analysis.

## Key Components

### FewShotDocumentProcessor

This class is the core of the document processing pipeline. It uses a pre-trained sentence transformer model to encode text and extract features from documents. Key functionalities include:

- Extracting text and layout features from images
- Encoding text into embeddings
- Predicting document categories based on similarity to few-shot examples

### ActiveLearningIDP

This class implements the active learning component of the pipeline. It:

- Maintains a queue of uncertain samples for review
- Updates the model based on user feedback
- Uses a Random Forest classifier as the base model that is iteratively improved with the user's feedback

### FewShotCache

This class provides caching functionality to store and retrieve few-shot examples and their features, improving efficiency for repeated runs.

### TextCleaner

A utility class that cleans and normalizes extracted text, handling common OCR errors and formatting issues.

### DocumentPreviewer

A GUI component that allows users to view documents during the review process.

## Pipeline Workflow

1. **Initialization**: 
  - The pipeline loads few-shot examples from a CSV file or cache.
  - It initializes the FewShotDocumentProcessor and ActiveLearningIDP components.

2. **Batch Processing**:
  - Documents are processed in batches to balance efficiency and responsiveness.
  - For each document:
    - Text and layout features are extracted.
    - An initial classification is made using the few-shot learning model.
    - If the classification confidence is below a threshold, the document is queued for review.

3. **User Review**:
  - After each batch, documents with low confidence predictions are presented for user review.
  - The user can accept the prediction, choose from existing categories, or add a new category.
  - The DocumentPreviewer allows users to view the document during review.

4. **Model Update**:
  - After user reviews, the active learning model is updated with the new labeled examples.
  - The few-shot cache is also updated to include these new examples.

5. **Continuous Processing**:
  - The pipeline continues processing batches, learning and improving with each iteration.

## Wiki

### 1. Initialization and Setup
##### 1.1 Argument Parsing
The pipeline begins by parsing command-line arguments using argparse. The primary argument is --verbose, which enables verbose logging if specified.

##### 1.2 Logger and Timer Initialization
A logger is initialized using the getLogger function from documentProcessor.py, which sets up logging configurations based on the verbosity level.
A timer (TicToc) is initialized to measure the time taken for various operations.

##### 1.3 CPU Affinity
The pipeline sets CPU affinity using psutil to restrict the process to specific CPU cores, optimizing performance.

### 2. File Fetching and Preparation
##### 2.1 Define File Paths
The paths for example files and the directory containing documents to be processed are defined.

##### 2.2 Fetch Files
The fetchFiles function is called to retrieve all TIF files from the specified directory. The function logs the number of files found.

### 3. Document Processing
##### 3.1 Start Timer
The timer starts to measure the document processing time.

##### 3.2 Process Documents
The processAllDocuments function is called with the list of TIF files, example file path, batch size, and force reload flag.
This function orchestrates the entire document processing workflow.

### 4. Detailed Workflow in processAllDocuments
##### 4.1 Few-Shot Learning Setup
A FewShotCache is initialized to manage cached examples.
A FewShotDocumentProcessor is created, which loads few-shot examples from a CSV file. If force_reload is true, examples are reloaded from the file.
An ActiveLearningIDP instance is created to manage active learning for document processing.

##### 4.2 Initial Model Fitting
The initial model is fitted using the few-shot examples.

##### 4.3 Batch Processing
Documents are processed in batches using a batch generator.
For each batch, the processBatch function processes documents in parallel using a ProcessPoolExecutor.

##### 4.4 Result Compilation
Results from all batches are compiled into a DataFrame.
The DataFrame contains the processing results for all documents, including success and error counts.

### 5. Post-Processing and Export
##### 5.1 Display Results
A sample of the results is displayed using print and logged.
The value counts for the 'category' column are also printed.

##### 5.2 Export Results
The user is prompted to export the results to a CSV file. If the user agrees, the results are saved to the specified file.

### 6. Logging and Completion
The pipeline logs the completion of the document processing, including the total time taken.
Supporting Components in documentProcessor.py

##### 6.1 Logging Setup
The setupLogger and getLogger functions manage logging configurations.

##### 6.2 Text Cleaning
The TextCleaner class provides methods to clean and preprocess text extracted from documents.

##### 6.3 Document Previewing
The DocumentPreviewer class allows for visual inspection of documents during processing.

##### 6.4 Few-Shot Learning
The FewShotDocumentProcessor class handles feature extraction, example management, and prediction using few-shot learning.
The FewShotClassifier class wraps the few-shot processor for compatibility with scikit-learn.

##### 6.5 Active Learning
The ActiveLearningIDP class manages active learning, allowing the model to improve iteratively based on user feedback.

##### 6.6 Utility Functions
Functions like fetchFiles, readTifFile, and processBatch provide essential utilities for file handling and batch processing.
