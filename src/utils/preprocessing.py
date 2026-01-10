"""Text Preprocessing"""

import pandas as pd
import re
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

def preprocess_pipeline(df: pd.DataFrame, data_type: str, options: Dict) -> pd.DataFrame:
    """Main preprocessing pipeline"""
    df_processed = df.copy()

    if options.get('text_cleaning', True):
        df_processed = clean_text(df_processed)

    if options.get('ner', True) and nlp:
        df_processed = extract_entities(df_processed)

    return df_processed

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields"""
    for col in ['title', 'abstract']:
        if col in df.columns:
            df[f'{col}_cleaned'] = df[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else ''
            )
    return df

def extract_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Extract named entities"""
    entities_list = []

    text_col = 'abstract' if 'abstract' in df.columns else 'title'

    for text in df[text_col].fillna(''):
        if len(str(text)) > 10:
            doc = nlp(str(text)[:5000])
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities_list.append(entities)
        else:
            entities_list.append([])

    df['entities'] = entities_list
    return df
