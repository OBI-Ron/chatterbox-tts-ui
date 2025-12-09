#!/bin/bash
#
# Chatterbox-TTS Lite – Launch Script (Template)
# --------------------------------------------------
# Edit the VENV_PATH below to point to your virtual
# environment before running this script.
#
# Usage:
#   bash run_lite_template.sh
#
# or, once made executable:
#   ./run_lite_template.sh
#

# --------- EDIT THIS LINE ---------
VENV_PATH="/path/to/your/venv/bin/activate"
# ----------------------------------

# Activate the virtual environment
if [[ -f "$VENV_PATH" ]]; then
    source "$VENV_PATH"
else
    echo "❌ Could not find virtual environment at:"
    echo "   $VENV_PATH"
    echo "Please edit run_lite_template.sh and set the correct path."
    exit 1
fi

# Run the Lite interface
python lite/chatterbox-tts-lite-ui.py
# --------- OPTIONAL if running from nautilus ---------
exec $SHELL;
# ----------------------------------
