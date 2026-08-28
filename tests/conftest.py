import os
import sys

# Let `pytest` be run from anywhere and still import the project's
# top-level modules (config, pipeline.*) the same way app.py does.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
