import os
import sys

# Ensure project root is on path so we can import the main app
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the FastAPI app from the root-level api.py
from api import app

# Vercel expects a module-level variable named 'app'
__all__ = ["app"]
