# Chatterbox-TTS UI

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/OBI-Ron/chatterbox-tts-ui?label=Lite%20Edition%20version)](https://github.com/OBI-Ron/chatterbox-tts-ui/releases)

Lightweight Gradio interfaces for **Chatterbox-TTS**, ranging from a simple “Lite” interface to more advanced narrator and studio workflows.


---

## 📌 Overview

This project provides a local Gradio user interface for **Chatterbox-TTS**.  
The Lite edition enhances the basic Chatterbox demo by adding:

- **Automatic text chunking** (handles long passages gracefully)
- **Audio concatenation** (chunks are stitched into a single waveform)
- **Optional dual-mono stereo export** (for easier use in DAWs like Audacity)
- **Adjustable pseudo-stereo effect** (10-30ms left channel delay, default 13ms)
- **Character management system** (save voice presets for audiobook creation)
- **Automatic browser launch** (opens automatically on startup)
- **Auto-shutdown after inactivity** (frees GPU after 5 minutes idle)
- **Example characters included** (Ralph and Susan with sample audio)
- **A simple, user-friendly UI**

This interface **does not modify** Chatterbox-TTS itself — it simply provides an improved front-end.

---

## 🚀 Installation

This UI requires **Chatterbox-TTS** to be installed separately.

### Install Chatterbox-TTS  
Please refer to the official repository for installation instructions, platform notes, and dependency details:

➡️ **https://github.com/resemble-ai/chatterbox**

*This UI does not provide installation support for Chatterbox itself.*

### Install Gradio
