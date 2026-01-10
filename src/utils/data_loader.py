"""Data Loading Utilities for Publications and Patents"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Any

def load_publications_data(file) -> pd.DataFrame:
    """
    Load publications data from various file formats
    Handles column mapping for different data sources
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        DataFrame with standardized column names
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file, encoding='utf-8-sig')  # Handle BOM
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        data = json.load(file)
        df = pd.DataFrame(data)
    elif file_ext == '.bib':
        content = file.read().decode('utf-8')
        df = pd.DataFrame({'title': ['BibTeX support coming soon'], 'year': [2024]})
    elif file_ext == '.ris':
        content = file.read().decode('utf-8')
        df = pd.DataFrame({'title': ['RIS support coming soon'], 'year': [2024]})
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    # Column mapping for common data sources
    column_mapping = {
        'article title': 'title',
        'authors': 'author',
        'cited by': 'citations',
        'source title': 'journal',
        'author keywords': 'keywords',
        'affiliations': 'affiliation',
        'publication year': 'year'
    }
    
    # Apply mapping only for columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Ensure 'title' column exists (try alternative names)
    if 'title' not in df.columns:
        for alt_name in ['article title', 'paper title', 'document title']:
            if alt_name in df.columns:
                df['title'] = df[alt_name]
                break
    
    # Convert year to numeric if present
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Convert citations to numeric if present
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce')
    
    return df

def load_patents_data(file) -> pd.DataFrame:
    """
    Load patents data from various file formats
    Handles lens.org export format
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        DataFrame with standardized column names
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file, encoding='utf-8-sig')  # Handle BOM
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        data = json.load(file)
        df = pd.DataFrame(data)
    elif file_ext == '.xml':
        df = pd.DataFrame({'title': ['XML support coming soon'], 'application_date': ['2024-01-01']})
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    # Column mapping for lens.org and other patent databases
    column_mapping = {
        'publication date': 'application_date',
        'applicants': 'assignee',
        'inventors': 'inventor',
        'simple family size': 'family_size',
        'ipc classifications': 'ipc_class',
        'cpc classifications': 'cpc_class',
        'patent title': 'title'
    }
    
    # Apply mapping only for columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Ensure 'title' column exists
    if 'title' not in df.columns:
        for alt_name in ['patent title', 'invention title']:
            if alt_name in df.columns:
                df['title'] = df[alt_name]
                break
    
    # Convert date columns
    date_columns = ['application_date', 'publication date', 'grant_date', 'priority_date', 'earliest priority date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract year from application_date
    if 'application_date' in df.columns:
        df['year'] = df['application_date'].dt.year
    
    # Convert numeric columns
    numeric_cols = ['family_size', 'forward_citations', 'backward_citations', 'cited by patent count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """
    Validate that dataframe contains required columns (flexible checking)
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, message)
    """
    if len(df) == 0:
        return False, "Data file is empty"
    
    # After loading and mapping, we should have standardized column names
    # So just check if these exist
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    missing_cols = []
    for req_col in required_columns:
        req_col_lower = req_col.lower().strip()
        
        # Check if column exists (exact match or alternative)
        found = False
        if req_col_lower in df_columns_lower:
            found = True
        # For 'title' check alternatives
        elif req_col_lower == 'title' and any(alt in df_columns_lower for alt in ['article title', 'patent title']):
            found = True
        # For 'year' check alternatives  
        elif req_col_lower == 'year' and any(alt in df_columns_lower for alt in ['publication year']):
            found = True
        # For 'application_date' check alternatives
        elif req_col_lower == 'application_date' and any(alt in df_columns_lower for alt in ['publication date']):
            found = True
        
        if not found:
            missing_cols.append(req_col)
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(list(df.columns)[:10])}"
    
    return True, "Data is valid"

def get_data_summary(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
    """
    Get summary statistics for the data
    
    Args:
        df: Input dataframe
        data_type: 'publications' or 'patents'
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'columns': list(df.columns)
    }
    
    if data_type == 'publications':
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
        
        if 'author' in df.columns:
            summary['unique_authors'] = df['author'].nunique()
        
        if 'citations' in df.columns:
            total = df['citations'].sum()
            avg = df['citations'].mean()
            if pd.notna(total):
                summary['total_citations'] = int(total)
            if pd.notna(avg):
                summary['avg_citations'] = round(float(avg), 2)
        
        if 'journal' in df.columns:
            summary['unique_journals'] = df['journal'].nunique()
    
    elif data_type == 'patents':
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
        elif 'application_date' in df.columns:
            years = df['application_date'].dt.year.dropna()
            if len(years) > 0:
                summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
        
        if 'inventor' in df.columns:
            summary['unique_inventors'] = df['inventor'].nunique()
        
        if 'assignee' in df.columns:
            summary['unique_assignees'] = df['assignee'].nunique()
        
        if 'family_size' in df.columns:
            avg_family = df['family_size'].mean()
            if pd.notna(avg_family):
                summary['avg_family_size'] = round(float(avg_family), 2)
        
        if 'jurisdiction' in df.columns:
            summary['jurisdictions'] = df['jurisdiction'].nunique()
    
    return summary
