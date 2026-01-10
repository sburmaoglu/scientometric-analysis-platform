"""Dynamic Module Loading System"""

import json
import importlib
from pathlib import Path
from typing import Dict, Any
import streamlit as st

def load_available_modules() -> Dict[str, Any]:
    """
    Load all available analysis modules from the modules directory
    Each module should have a config.json file
    """
    modules_dir = Path(__file__).parent.parent / "modules"
    available_modules = {}
    
    if not modules_dir.exists():
        st.error(f"Modules directory not found: {modules_dir}")
        return available_modules
    
    for module_path in modules_dir.iterdir():
        if module_path.is_dir() and (module_path / "config.json").exists():
            try:
                with open(module_path / "config.json", 'r') as f:
                    config = json.load(f)
                    module_name = config.get('module_name', module_path.name)
                    available_modules[module_name] = config
            except Exception as e:
                st.warning(f"Failed to load config for {module_path.name}: {str(e)}")
    
    return available_modules

def get_module_instance(module_name: str):
    """
    Get an instance of a specific analysis module
    
    Args:
        module_name: Name of the module to load
        
    Returns:
        Instance of the module class or None if failed
    """
    try:
        # Import the module dynamically
        module_path = f"modules.{module_name}.module"
        module = importlib.import_module(module_path)
        
        # Find the Module class (should end with 'Module')
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Module'):
                return attr()
        
        st.error(f"No class ending with 'Module' found in {module_name}")
        return None
        
    except Exception as e:
        st.error(f"Failed to load module {module_name}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
