#!/usr/bin/env python3
# load_models.py
# Separate script to load models in a subprocess (isolates segfaults)

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import rag_engine
    
    print("Loading models...")
    rag_engine._load_data()
    success = rag_engine._load_models()
    
    if success and rag_engine.MODEL_READY:
        print("SUCCESS")
        sys.exit(0)
    else:
        print(f"FAILED: {rag_engine.MODEL_ERROR}")
        sys.exit(1)
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

