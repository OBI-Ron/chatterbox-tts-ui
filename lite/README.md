# Chatterbox-TTS UI

Lightweight Gradio interfaces for **Chatterbox-TTS**, ranging from a simple “Lite” interface to more advanced narrator and studio workflows.  
This repository currently focuses on the **Lite edition**, designed to be clean, minimal, and easy for new users.

---

## 📌 Overview

This project provides a local Gradio user interface for **Chatterbox-TTS**.  
The Lite edition enhances the basic Chatterbox demo by adding:

- **Automatic text chunking** (handles long passages gracefully)
- **Audio concatenation** (chunks are stitched into a single waveform)
- **Optional dual-mono stereo export** (for easier use in DAWs like Audacity)
- A simple, user-friendly UI

This interface **does not modify** Chatterbox-TTS itself — it simply provides an improved front-end.

---

## Running with a helper script (optional)

This repository includes a template launch script you may use to start the Lite interface:

- `run_lite_template.sh` (Linux/macOS)

Edit the `VENV_PATH` inside the script to point to your Python virtual environment.

Example (Linux):

```bash
VENV_PATH="/home/fred/venvs/chatterbox-lite/bin/activate"

```bash
bash run_lite_template.sh



## 🚀 Installation

This UI requires **Chatterbox-TTS** to be installed separately.

### Install Chatterbox-TTS  
Please refer to the official repository for installation instructions, platform notes, and dependency details:

➡️ **https://github.com/resemble-ai/chatterbox**

*This UI does not provide installation support for Chatterbox itself.*

### Install Gradio

