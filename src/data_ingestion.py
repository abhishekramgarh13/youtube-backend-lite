import numpy as np
import pandas as pd 
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path : str) -> dict:
    """Load Parameter from a YAML File."""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrived from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s",params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
        raise 
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise 

def load_data(data_url: str) -> pd.DataFrame:
    """load data from url and return as DataFrame."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv File: %s",e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occured while loading the data : %s",e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        #drop missing value
        df.dropna(inplace=True)
        # removing duplicates
        df.drop_duplicates(inplace=True)
        # remove empty string
        df = df[df['clean_comment'].str.strip() != ""]

        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data : pd.DataFrame, data_path: str)-> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path,'raw')

        #create the data/raw directory if it doesn't exist
        os.makedirs(raw_data_path, exist_ok=True)

        # save the rain and test data
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # load parameters from the params.ymal in the root directory
        base_dir = "."
        params = load_params(params_path=os.path.join(base_dir,"params.yaml"))
        test_size = params['data_ingestion']['test_size']

        #load data from url
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        #preprocess data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(base_dir, 'data'))
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()