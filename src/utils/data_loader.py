"""Data Loading Utilities"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple

def load_publications_data(file) -> pd.DataFrame:
    """Load publications data from various formats"""
    file_ext = Path(file.name).suffix.lower()

    if file_ext == '.csv':
        df = pd.read_csv(file)
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        df = pd.DataFrame(json.load(file))
    else:
        raise ValueError(f"Unsupported format: {file_ext}")

    df.columns = df.columns.str.lower()
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

    return df

def load_patents_data(file) -> pd.DataFrame:
    """Load patents data from various formats"""
    file_ext = Path(file.name).suffix.lower()

    if file_ext == '.csv':
        df = pd.read_csv(file)
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        df = pd.DataFrame(json.load(file))
    else:
        raise ValueError(f"Unsupported format: {file_ext}")

    df.columns = df.columns.str.lower()

    date_cols = ['application_date', 'grant_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

def validate_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """Validate dataframe"""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    if len(df) == 0:
        return False, "File is empty"

    return True, "Valid"

def get_data_summary(df: pd.DataFrame, data_type: str) -> dict:
    """Get data summary"""
    summary = {
        'total_records': len(df),
        'columns': list(df.columns)
    }

    if data_type == 'publications' and 'year' in df.columns:
        summary['year_range'] = f"{df['year'].min():.0f} - {df['year'].max():.0f}"

    return summary
