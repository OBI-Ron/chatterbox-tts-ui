# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.2] - 2026-02-14

### Added

- Character management - providing support for audio book creation
  - From the Generate Audio tab you can:
    - Load a character preset
    - Save current settings for character creation
    - Regenerate character list
  - From the Manage Character tab you can:
    - Create & save characters
    - Delete characters
    - Load character into audio generator
  - Included two example characters (Ralph and Susan) each with sample audio
    - Sample voices generated from open source voices from https://brycebeattie.com/files/tts/
- Quasi stereo effect
  - Adjustable left channel delay (10-30ms, default 13ms) for pseudo-stereo output
  - Creates spatial separation in dual-mono exports
- Automatic browser launch on startup
- Automatic TTS server shutdown after 5 minutes of inactivity
- GPU memory cleanup on shutdown
  

## [0.1.1] – 2025-01-11

### Fixed

- Prevented CUDA out-of-memory errors when refreshing the browser page by:
  - Introducing a global singleton model instance in `load_model()`.
  - Ensuring the Chatterbox-TTS model is only loaded onto the GPU once per Python process.
- Optionally calling `torch.cuda.empty_cache()` after generation to help reduce GPU memory fragmentation.

---

## [0.1.0] – 2025-01-10

### Added

- Initial **Lite Edition** release:
  - Gradio-based local UI for Chatterbox-TTS.
  - Automatic text chunking for long inputs.
  - Audio concatenation into a single waveform.
  - Optional dual-mono stereo output for DAW-friendly WAV files.
  - Optional reference audio input for voice cloning.
  - Controls for exaggeration, temperature, CFG/pace weight, min_p, top_p, repetition penalty, and seed.
