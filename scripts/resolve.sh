#!/bin/bash
cd /opt/data/postprophet
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python3 postprophet.py resolve 2>&1
