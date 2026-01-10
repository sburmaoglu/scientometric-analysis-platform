# 🔬 Advanced Scientometric Analysis Platform

A comprehensive, modular platform for analyzing publications and patents with rigorous statistical methods suitable for academic research and peer-reviewed publications.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## ✨ Key Features

### 📊 **Publication-Ready Statistical Analysis**
- Rigorous hypothesis testing with multiple testing corrections
- Confidence intervals and effect sizes (Cohen's d, r², η²)
- Assumption validation (normality, homogeneity, independence)
- Power analysis and sample size calculations
- Complete methodology documentation for citations

### 🎨 **Professional Visualizations**
- Interactive Plotly charts optimized for publications
- Network visualizations (2D/3D force-directed graphs)
- Temporal trend analysis with forecasting
- Geospatial mapping with flow diagrams
- Publication-quality exports (300-1200 DPI)

### 🔧 **Advanced Text Preprocessing**
- Named Entity Recognition (NER) for technologies, organizations
- Multi-level text normalization and cleaning
- TF-IDF keyword extraction
- Patent claim parsing (independent/dependent)
- Domain-specific preprocessing pipelines

### 🧩 **Modular Architecture**
- Easy to add new analysis modules
- Each module is self-contained
- Automatic module discovery and loading
- Configurable via JSON files

### 📈 **Comprehensive Analysis Modules**
1. **Publications Analysis** - Citation networks, author collaboration, impact metrics
2. **Patents Analysis** - Patent families, IPC/CPC classification, inventor networks
3. **Comparative Analysis** - Knowledge transfer, cross-domain influence, time lag analysis
4. **Network Analysis** - Multi-layer networks, community detection, centrality metrics
5. **Temporal Analysis** - Trend forecasting, hype cycles, innovation waves
6. **Geospatial Analysis** - Global collaboration, regional innovation systems
7. **Topic Modeling** - LDA, NMF, BERTopic with evolution tracking
8. **AI Insights** - Predictive modeling, anomaly detection, recommendations
9. **Custom Reports** - Automated report generation with full methodology

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/scientometric-analysis-platform.git
cd scientometric-analysis-platform

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
streamlit run app.py
```

### First Analysis

1. **Upload Data** - Navigate to "Data Upload" and upload your CSV/Excel/BibTeX files
2. **Preprocess** - Select preprocessing options and click "Start Preprocessing"
3. **Analyze** - Choose an analysis module from the sidebar
4. **Export** - Generate publication-ready reports with complete methodology

---

## 📁 Project Structure
```
scientometric-analysis-platform/
├── app.py                          # Main application
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── src/
│   ├── config/
│   │   └── settings.py            # All configuration
│   │
│   ├── core/
│   │   ├── module_loader.py       # Dynamic module loading
│   │   └── base_module.py         # Base class for modules
│   │
│   ├── utils/
│   │   ├── session_state.py       # State management
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── preprocessing.py       # Text preprocessing
│   │   ├── statistical_tests.py   # Statistical functions
│   │   └── export_utils.py        # Report generation
│   │
│   ├── visualizations/
│   │   ├── network_viz.py         # Network graphs
│   │   ├── temporal_viz.py        # Time series plots
│   │   ├── geospatial_viz.py     # Maps
│   │   └── statistical_viz.py     # Statistical plots
│   │
│   ├── pages/
│   │   ├── home.py                # Dashboard
│   │   └── data_upload.py         # Upload interface
│   │
│   └── modules/                    # Analysis modules
│       ├── publications_analysis/
│       ├── patents_analysis/
│       ├── comparative_analysis/
│       ├── network_analysis/
│       ├── temporal_analysis/
│       ├── geospatial_analysis/
│       ├── topic_modeling/
│       ├── ai_insights/
│       └── custom_reports/
│
├── data/                           # Data storage
├── exports/                        # Generated reports
└── models/                         # Saved models
```

---

## 📊 Data Formats

### Publications
**Required Fields:**
- `title` - Publication title
- `year` - Publication year

**Optional Fields:**
- `author`, `abstract`, `keywords`, `citations`, `doi`, `journal`, `affiliation`, `country`

**Supported Formats:**
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)
- BibTeX (.bib)
- RIS (.ris)

### Patents
**Required Fields:**
- `title` - Patent title
- `application_date` - Application date

**Optional Fields:**
- `inventor`, `assignee`, `abstract`, `claims`, `ipc_class`, `cpc_class`, `forward_citations`, `backward_citations`, `patent_number`, `legal_status`

**Supported Formats:**
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)
- XML (.xml)

---

## 🎓 Academic Use

### Citation

If you use this platform in your research, please cite:
```bibtex
@software{scientometrics2024,
  author = {Your Name},
  title = {Advanced Scientometric Analysis Platform},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/yourusername/scientometric-analysis-platform}
}
```

### Statistical Methods

All statistical analyses follow established methodologies:

- **Correlation Analysis**: Pearson, Spearman, Kendall correlations with significance tests
- **Regression**: OLS with diagnostics, robust regression, quantile regression
- **Hypothesis Testing**: t-tests, ANOVA, Mann-Whitney U, Kruskal-Wallis
- **Effect Sizes**: Cohen's d, r², η², ω²
- **Multiple Testing**: Bonferroni, Holm-Bonferroni, FDR corrections

References for all methods are included in exported reports.

---

## 🔧 Adding New Modules

Create a new analysis module in 3 steps:

1. **Create module directory:**
```bash
mkdir src/modules/my_analysis
```

2. **Create `module.py`:**
```python
from core.base_module import BaseModule
import streamlit as st

class MyAnalysisModule(BaseModule):
    def render(self):
        st.title("My Analysis")
        # Your analysis code here
```

3. **Create `config.json`:**
```json
{
  "module_name": "my_analysis",
  "display_name": "My Analysis",
  "icon": "📊",
  "enabled": true,
  "description": "My custom analysis module",
  "requires_data": ["publications"],
  "version": "1.0.0"
}
```

The module will be automatically discovered and loaded!

---

## 📈 Statistical Rigor

### Included in All Analyses

- ✅ **Hypothesis Testing** - Proper null/alternative hypothesis formulation
- ✅ **Significance Levels** - Configurable α (default: 0.05)
- ✅ **Confidence Intervals** - 95% CI by default (configurable)
- ✅ **Effect Sizes** - Cohen's d, Pearson's r, η², ω²
- ✅ **Assumption Checks** - Normality, homogeneity, independence
- ✅ **Power Analysis** - Post-hoc and a priori power calculations
- ✅ **Multiple Testing** - Bonferroni, Holm, FDR corrections
- ✅ **Diagnostic Plots** - Q-Q plots, residual plots, influence plots

### Report Contents

Every exported report includes:

1. **Executive Summary** - Key findings
2. **Methodology** - Complete statistical procedures
3. **Data Description** - Descriptive statistics, distributions
4. **Analysis Results** - All test statistics, p-values, CIs
5. **Visualizations** - High-resolution charts (300+ DPI)
6. **Interpretation** - Effect sizes and practical significance
7. **Assumptions** - Validation results
8. **References** - Citations for all methods
9. **Appendices** - Additional tables, code (optional)

---

## 🌐 Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/sburmaoglu/scientometric-analysis-platform/issues)
- **Email**: serhat.burmaoglu@ikcu.edu.tr; serhatburmaoglu@gmail.com

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Statistical methods based on established research
- Inspired by the scientometrics community

---

**Made with ❤️ for researchers who demand rigorous analysis**
