import numpy as np
import pandas as pd
import os
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging


# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_comment(comment):
    """Apply preprocess transformation to comment"""
    try:
        # convert to lowercase
        comment.lower()

        # removing trailing and leading spaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        raise 

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data : pd.DataFrame, data_path: str)-> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        processed_data_path = os.path.join(data_path,'processed')
        logger.debug(f"Creating directory {processed_data_path}")

        #create the data/raw directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)
        logger.debug(f"Directory {processed_data_path} created or already exists")

        # save the rain and test data
        train_data.to_csv(os.path.join(processed_data_path,"train_processed.csv"),index=False)
        test_data.to_csv(os.path.join(processed_data_path,"test_processed.csv"),index=False)
        
        logger.debug('Train and test data saved to %s', processed_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()