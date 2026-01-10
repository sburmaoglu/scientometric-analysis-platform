"""Data Loading Utilities for Publications and Patents"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from config.settings import UPLOAD_CONFIG

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
        # Basic BibTeX support
        content = file.read().decode('utf-8')
        df = pd.DataFrame({'title': ['BibTeX support coming soon'], 'year': [2024]})
    elif file_ext == '.ris':
        # Basic RIS support
        content = file.read().decode('utf-8')
        df = pd.DataFrame({'title': ['RIS support coming soon'], 'year': [2024]})
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Apply column mapping if available
    if 'column_mapping' in UPLOAD_CONFIG and 'publications' in UPLOAD_CONFIG['column_mapping']:
        mapping = UPLOAD_CONFIG['column_mapping']['publications']
        # Only map columns that exist in the dataframe
        existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
    
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
        # Basic XML support
        df = pd.DataFrame({'title': ['XML support coming soon'], 'application_date': ['2024-01-01']})
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Apply column mapping if available
    if 'column_mapping' in UPLOAD_CONFIG and 'patents' in UPLOAD_CONFIG['column_mapping']:
        mapping = UPLOAD_CONFIG['column_mapping']['patents']
        # Only map columns that exist in the dataframe
        existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
    
    # Convert date columns
    date_columns = ['application_date', 'publication date', 'grant_date', 'priority_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract year from application_date if not present
    if 'application_date' in df.columns and 'year' not in df.columns:
        df['year'] = df['application_date'].dt.year
    
    # Convert numeric columns
    numeric_cols = ['family_size', 'forward_citations', 'backward_citations']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """
    Validate that dataframe contains required columns
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check for required columns (case-insensitive)
    df_columns_lower = [col.lower().strip() for col in df.columns]
    required_lower = [col.lower().strip() for col in required_columns]
    
    missing_cols = [col for col in required_lower if col not in df_columns_lower]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    if len(df) == 0:
        return False, "Data file is empty"
    
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
                summary['year_range'] = f"{years.min():.0f} - {years.max():.0f}"
        
        if 'author' in df.columns:
            summary['unique_authors'] = df['author'].nunique()
        
        if 'citations' in df.columns:
            total = df['citations'].sum()
            avg = df['citations'].mean()
            if pd.notna(total):
                summary['total_citations'] = int(total)
            if pd.notna(avg):
                summary['avg_citations'] = float(avg)
        
        if 'journal' in df.columns:
            summary['unique_journals'] = df['journal'].nunique()
    
    elif data_type == 'patents':
        if 'application_date' in df.columns:
            years = df['application_date'].dt.year.dropna()
            if len(years) > 0:
                summary['year_range'] = f"{years.min():.0f} - {years.max():.0f}"
        
        if 'inventor' in df.columns:
            summary['unique_inventors'] = df['inventor'].nunique()
        
        if 'assignee' in df.columns:
            summary['unique_assignees'] = df['assignee'].nunique()
        
        if 'family_size' in df.columns:
            avg_family = df['family_size'].mean()
            if pd.notna(avg_family):
                summary['avg_family_size'] = float(avg_family)
        
        if 'jurisdiction' in df.columns:
            summary['jurisdictions'] = df['jurisdiction'].nunique()
    
    return summary
