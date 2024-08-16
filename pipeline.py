# Document Classification Pipeline
import argparse
import os
import psutil
from pytictoc import TicToc
import pandas as pd
from documentProcessor import processAllDocuments, getLogger, fetchFiles

# Parse arguments
parser = argparse.ArgumentParser(description='Run the document processing pipeline.')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Initialize logger and timer
logger = getLogger(verbose=args.verbose)
t = TicToc()

# Set CPU affinity
core_constraint = [2, 3, 4, 5, 6, 7]
psutil.Process(os.getpid()).cpu_affinity(core_constraint)
logger.info("INITIALIZING DOCUMENT PROCESSING PIPELINE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
logger.info(f"CPU affinity set to cores {core_constraint}")

# Define file paths and fetch files
fsExamples = 'utils/examples2.csv'
folder = 'examples2'  # Set the folder to process
directory = 'data/' + folder + '/'
tif_files = fetchFiles(directory)

# Start processing documents
t.tic()
logger.info(f'Starting document processing for {len(tif_files)} files in {directory}')
# ----------------------------------------------------------------
results_df = processAllDocuments(tif_files, fsExamples, batch_size=20, force_reload=False)
# ----------------------------------------------------------------
elapsed_time = t.tocvalue()
logger.info(f'Finished document processing for {len(tif_files)} files in {directory}. Time taken: {elapsed_time:.2f} seconds.')

# Display results
logger.info(f'\nSample of results for {len(results_df)} documents: ')
print(results_df.head(15))
print(results_df.value_counts('category'))

# Save results to CSV
export = input(f'\nExport all results to CSV? (y/n): ')
if export.lower() == 'y':
  file_name = input("Enter the name for the CSV file (without extension): ")
  results_df.to_csv(f'results/{file_name}.csv', index=False)
  logger.info(f'Results exported to {file_name}.csv')
else:
  logger.info(f'Skipping export')
logger.info("DOCUMENT PROCESSING PIPELINE COMPLETED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")