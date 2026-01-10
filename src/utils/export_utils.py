"""Export and Report Generation"""

import pandas as pd
from datetime import datetime

def generate_statistical_report(results: dict, title: str) -> str:
    """Generate statistical report in markdown"""

    report = f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Statistical Results

"""

    for key, value in results.items():
        report += f"**{key}:** {value}\n\n"

    report += """
---

## Methodology

All analyses performed using established statistical methods.
Significance level: α = 0.05

## References

Statistical methods implemented using:
- SciPy (scipy.stats)
- Statsmodels
- Pingouin

"""

    return report

def export_to_csv(df: pd.DataFrame, filename: str):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')
