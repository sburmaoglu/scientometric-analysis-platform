"""Dynamic Module Loading System"""

import json
import importlib
from pathlib import Path
from typing import Dict, Any
import streamlit as st

def load_available_modules() -> Dict[str, Any]:
    """Load all available analysis modules"""
    modules_dir = Path(__file__).parent.parent / "modules"
    available_modules = {}

    if not modules_dir.exists():
        return available_modules

    for module_path in modules_dir.iterdir():
        if module_path.is_dir() and (module_path / "config.json").exists():
            try:
                with open(module_path / "config.json", 'r') as f:
                    config = json.load(f)
                    module_name = config.get('module_name', module_path.name)
                    available_modules[module_name] = config
            except Exception as e:
                st.warning(f"Failed to load module {module_path.name}: {str(e)}")

    return available_modules

def get_module_instance(module_name: str):
    """Get instance of a specific module"""
    try:
        module_path = f"modules.{module_name}.module"
        module = importlib.import_module(module_path)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Module'):
                return attr()

        return None
    except Exception as e:
        st.error(f"Failed to load module {module_name}: {str(e)}")
        return None
