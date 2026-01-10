"""Configuration Settings"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"
MODULES_DIR = BASE_DIR / "src" / "modules"

for dir_path in [DATA_DIR, CACHE_DIR, MODELS_DIR, EXPORTS_DIR, MODULES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
(DATA_DIR / "samples").mkdir(exist_ok=True)

PAGE_CONFIG = {
    "page_title": "ScientoMetrics Platform",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

THEME_CONFIG = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
"""

CUSTOM_CSS = """
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    .dataframe {
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
"""

VIZ_CONFIG = {
    "plotly_template": "plotly_white",
    "color_palettes": {
        "professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
        "academic": ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
    },
    "network_colors": {
        "publications": "#3498db",
        "patents": "#e74c3c",
        "cross_citation": "#95a5a6"
    }
}

NLP_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "tfidf_max_features": 10000,
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.95,
    "tfidf_ngram_range": (1, 3)
}

STATS_CONFIG = {
    "significance_level": 0.05,
    "confidence_level": 0.95,
    "min_sample_size": 30,
    "decimal_places": 4
}

# ==================== UPLOAD CONFIGURATION ====================
UPLOAD_CONFIG = {
    "max_file_size_mb": 200,
    "allowed_extensions": {
        "publications": [".csv", ".xlsx", ".json", ".bib", ".ris"],
        "patents": [".csv", ".xlsx", ".json", ".xml"]
    },
    "required_columns": {
        "publications": ["article title", "year"],  # Updated for your data
        "patents": ["title", "publication date"]  # Updated for lens.org format
    },
    "column_mapping": {
        "publications": {
            "article title": "title",
            "authors": "author",
            "cited by": "citations",
            "source title": "journal",
            "author keywords": "keywords"
        },
        "patents": {
            "publication date": "application_date",
            "applicants": "assignee",
            "inventors": "inventor",
            "simple family size": "family_size"
        }
    }
}

EXPORT_CONFIG = {
    "report_formats": ["PDF", "HTML", "DOCX"],
    "default_dpi": 300,
    "citation_styles": ["APA", "MLA", "Chicago"]
}
