#!/usr/bin/env bash
# Run the app on macOS / Linux
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
echo "Open http://127.0.0.1:5000 in your browser"
python3 app.py
