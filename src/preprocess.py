# -*- coding: utf-8 -*-
"""
Data loading and preprocessing script.

This script handles:
- Loading the raw data.
- Applying data type conversions.
- Performing feature engineering and cleaning.
- Saving the processed data for the next steps.
"""

import pandas as pd
import numpy as np
from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    """Loads data from a CSV file."""
    logger.info(f"Loading data from {file_path}...")
    try:
        return pd.read_csv(file_path, sep=";", na_values=["9", "99", "999"])
    except FileNotFoundError:
        logger.error(f"Raw data file not found at: {file_path}")
        raise

def apply_data_types(df):
    """Applies correct data types to columns."""
    logger.info("Applying data types...")
    for col in config.COLS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in config.COLS_DATE:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    
    cols_restantes = [col for col in df.columns if col not in config.COLS_NUMERIC + config.COLS_DATE]
    for col in cols_restantes:
        df[col] = df[col].astype(str).replace('nan', np.nan)
    return df

def remove_low_variance_features(df):
    """Removes features with low variance."""
    logger.info(f"Removing features with variance below {config.LOW_VARIANCE_THRESHOLD} threshold...")
    cols_to_drop = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True, dropna=False)
        if len(counts) == 0 or counts.values[0] > config.LOW_VARIANCE_THRESHOLD:
            cols_to_drop.append(col)
            logger.info(f"  -> Dropping '{col}' (most frequent value covers >{config.LOW_VARIANCE_THRESHOLD*100}%)")
    return df.drop(columns=list(set(cols_to_drop)))

def feature_engineering(df):
    """Creates new features and cleans existing ones."""
    logger.info("Performing feature engineering...")
    # Target
    df[config.TARGET_COL] = df[config.STATUS_COL].map(config.STATUS_MAP)
    df = df.dropna(subset=[config.TARGET_COL, 'TEMPO_DIAS'])
    df = df[~df['TEMPO_DIAS'].isin([0])]
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

    # Binarizations
    df['RACACOR_BIN'] = df['RACACOR'].apply(lambda x: 'Branco' if x == '1.0' else 'Não Branco')
    df['ESCOLARIDADE_BIN'] = df['INSTRUC'].apply(lambda x: 'Médio/Superior' if x in ['4.0', '5.0'] else 'Outros')
    df['ESTCONJ_BIN'] = df['ESTCONJ'].apply(lambda x: 'Casado' if x in ['2.0', '5.0'] else 'Não Casado')
    df['ENCAMINHAMENTO_BIN'] = df['ORIGENCA'].apply(lambda x: 'SUS' if x == '1.0' else 'Não SUS')
    df['TRATAMENTO_BIN'] = df['PRITRATH'].apply(lambda x: 'Não' if x == '0.0' else 'Sim')

    # Drop redundant columns
    df = df.drop(columns=[c for c in config.COLS_LEAKAGE_AND_REDUNDANT if c in df.columns], errors='ignore')
    return df

def main():
    """Main preprocessing pipeline."""
    df = load_data(config.RAW_DATA_FILE)
    df = apply_data_types(df)
    df = remove_low_variance_features(df)
    df = feature_engineering(df)
    
    # Final cleaning
    df_proc = df.copy()
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    df_proc[cat_cols] = df_proc[cat_cols].fillna('Ignorado')
    df_proc = pd.get_dummies(df_proc, drop_first=True)
    df_proc = df_proc.fillna(df_proc.median())

    logger.info(f"Processed data shape: {df_proc.shape}")
    df_proc.to_parquet(config.PROCESSED_DATA_FILE, index=False)
    logger.info(f"Processed data saved to {config.PROCESSED_DATA_FILE}")

if __name__ == '__main__':
    main()
