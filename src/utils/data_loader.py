"""Data Loading Utilities for Publications and Patents - Lens.org Format"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Any

def load_publications_data(file):
    """
    Load publications data from lens.org format
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        DataFrame with standardized column names
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file, encoding='utf-8-sig')
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        data = json.load(file)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Column mapping for lens.org publications
    column_mapping = {
        'Lens ID': 'lens_id',
        'Title': 'title',
        'Date Published': 'date_published',
        'Publication Year': 'year',
        'Publication Type': 'publication_type',
        'Source Title': 'journal',
        'ISSNs': 'issn',
        'Publisher': 'publisher',
        'Source Country': 'country',
        'Author/s': 'author',
        'Abstract': 'abstract',
        'Volume': 'volume',
        'Issue Number': 'issue',
        'Start Page': 'page_start',
        'End Page': 'page_end',
        'Fields of Study': 'fields',
        'Keywords': 'keywords',
        'MeSH Terms': 'mesh_terms',
        'Chemicals': 'chemicals',
        'Funding': 'funding',
        'Source URLs': 'source_urls',
        'External URL': 'external_url',
        'PMID': 'pmid',
        'DOI': 'doi',
        'Microsoft Academic ID': 'mag_id',
        'PMCID': 'pmcid',
        'Citing Patents Count': 'citing_patents',
        'References': 'references',
        'Citing Works Count': 'citations',
        'Is Open Access': 'is_open_access',
        'Open Access License': 'oa_license',
        'Open Access Colour': 'oa_color'
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Convert year to numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Convert citations to numeric
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce')
    
    # Convert date_published to datetime
    if 'date_published' in df.columns:
        df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
    
    return df

def load_patents_data(file):
    """
    Load patents data from lens.org format
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        DataFrame with standardized column names
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file, encoding='utf-8-sig')
    elif file_ext == '.xlsx':
        df = pd.read_excel(file)
    elif file_ext == '.json':
        data = json.load(file)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Column mapping for lens.org patents
    column_mapping = {
        '#': 'index',
        'Jurisdiction': 'jurisdiction',
        'Kind': 'kind',
        'Display Key': 'display_key',
        'Lens ID': 'lens_id',
        'Publication Date': 'publication_date',
        'Publication Year': 'year',
        'Application Number': 'application_number',
        'Application Date': 'application_date',
        'Priority Numbers': 'priority_numbers',
        'Earliest Priority Date': 'earliest_priority_date',
        'Title': 'title',
        'Abstract': 'abstract',
        'Applicants': 'assignee',
        'Inventors': 'inventor',
        'Owners': 'owners',
        'URL': 'url',
        'Document Type': 'document_type',
        'Has Full Text': 'has_full_text',
        'Cites Patent Count': 'backward_citations',
        'Cited by Patent Count': 'forward_citations',
        'Simple Family Size': 'family_size',
        'Simple Family Members': 'family_members',
        'Simple Family Member Jurisdictions': 'family_jurisdictions',
        'Extended Family Size': 'extended_family_size',
        'Extended Family Members': 'extended_family_members',
        'Extended Family Member Jurisdictions': 'extended_family_jurisdictions',
        'Sequence Count': 'sequence_count',
        'CPC Classifications': 'cpc_class',
        'IPCR Classifications': 'ipc_class',
        'US Classifications': 'us_class',
        'NPL Citation Count': 'npl_citation_count',
        'NPL Resolved Citation Count': 'npl_resolved_count',
        'NPL Resolved Lens ID(s)': 'npl_lens_ids',
        'NPL Resolved External ID(s)': 'npl_external_ids',
        'NPL Citations': 'npl_citations',
        'Legal Status': 'legal_status'
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Convert date columns to datetime
    date_columns = ['publication_date', 'application_date', 'earliest_priority_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract year from application_date
    if 'application_date' in df.columns:
        if 'year' not in df.columns or df['year'].isna().all():
            df['year'] = df['application_date'].dt.year
    
    # Convert numeric columns
    numeric_cols = ['year', 'family_size', 'forward_citations', 'backward_citations', 
                    'extended_family_size', 'sequence_count', 'npl_citation_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_data(df, required_columns):
    """
    Validate that dataframe contains required columns
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, message)
    """
    if len(df) == 0:
        return False, "Data file is empty"
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        available = ', '.join(list(df.columns)[:10])
        return False, f"Missing columns: {', '.join(missing_cols)}. Available: {available}..."
    
    return True, "Data is valid"

def get_data_summary(df, data_type):
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
        # Year range
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
        
        # Unique authors
        if 'author' in df.columns:
            non_null = df['author'].dropna()
            if len(non_null) > 0:
                all_authors = []
                for authors_str in non_null:
                    if pd.notna(authors_str):
                        authors = str(authors_str).split(';')
                        all_authors.extend([a.strip() for a in authors if a.strip()])
                summary['unique_authors'] = len(set(all_authors))
        
        # Citations
        if 'citations' in df.columns:
            total = df['citations'].sum()
            avg = df['citations'].mean()
            if pd.notna(total):
                summary['total_citations'] = int(total)
            if pd.notna(avg):
                summary['avg_citations'] = round(float(avg), 2)
        
        # Journals
        if 'journal' in df.columns:
            summary['unique_journals'] = df['journal'].nunique()
        
        # Open access
        if 'is_open_access' in df.columns:
            oa_count = (df['is_open_access'] == True).sum()
            summary['open_access_count'] = int(oa_count)
            summary['open_access_pct'] = round(oa_count / len(df) * 100, 1)
    
    elif data_type == 'patents':
        # Year range
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
        
        # Unique inventors
        if 'inventor' in df.columns:
            non_null = df['inventor'].dropna()
            if len(non_null) > 0:
                all_inventors = []
                for inv_str in non_null:
                    if pd.notna(inv_str):
                        inventors = str(inv_str).split(';')
                        all_inventors.extend([i.strip() for i in inventors if i.strip()])
                summary['unique_inventors'] = len(set(all_inventors))
        
        # Unique assignees
        if 'assignee' in df.columns:
            non_null = df['assignee'].dropna()
            if len(non_null) > 0:
                all_assignees = []
                for ass_str in non_null:
                    if pd.notna(ass_str):
                        assignees = str(ass_str).split(';')
                        all_assignees.extend([a.strip() for a in assignees if a.strip()])
                summary['unique_assignees'] = len(set(all_assignees))
        
        # Family size
        if 'family_size' in df.columns:
            avg_family = df['family_size'].mean()
            if pd.notna(avg_family):
                summary['avg_family_size'] = round(float(avg_family), 2)
        
        # Jurisdictions
        if 'jurisdiction' in df.columns:
            summary['unique_jurisdictions'] = df['jurisdiction'].nunique()
        
        # Citations
        if 'forward_citations' in df.columns:
            total = df['forward_citations'].sum()
            avg = df['forward_citations'].mean()
            if pd.notna(total):
                summary['total_forward_citations'] = int(total)
            if pd.notna(avg):
                summary['avg_forward_citations'] = round(float(avg), 2)
    
    return summary
