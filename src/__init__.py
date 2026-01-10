"""Pages package - imports all page modules"""

from . import home
from . import data_upload
from . import publications_analysis
from . import patents_analysis
from . import comparative_analysis
from . import temporal_analysis
from . import geographic_analysis

__all__ = [
    'home',
    'data_upload',
    'publications_analysis',
    'patents_analysis',
    'comparative_analysis',
    'temporal_analysis',
    'geographic_analysis'
]
