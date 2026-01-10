"""Statistical Testing Functions"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

def perform_normality_test(data: pd.Series) -> Dict:
    """Test for normality"""
    if len(data) < 3:
        return {'test': 'insufficient_data', 'p_value': None}

    statistic, p_value = stats.shapiro(data.dropna())

    return {
        'test': 'Shapiro-Wilk',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': p_value > 0.05,
        'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
    }

def calculate_correlation(x: pd.Series, y: pd.Series, method: str = 'pearson') -> Dict:
    """Calculate correlation"""
    valid_mask = ~(x.isna() | y.isna())
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 3:
        return {'error': 'Insufficient data'}

    if method == 'pearson':
        r, p = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        r, p = stats.spearmanr(x_clean, y_clean)
    else:
        r, p = stats.kendalltau(x_clean, y_clean)

    return {
        'method': method,
        'correlation': float(r),
        'p_value': float(p),
        'n': len(x_clean),
        'significant': p < 0.05
    }

def calculate_descriptive_stats(data: pd.Series) -> Dict:
    """Calculate descriptive statistics"""
    return {
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'q25': float(data.quantile(0.25)),
        'q75': float(data.quantile(0.75)),
        'n': int(data.count())
    }
