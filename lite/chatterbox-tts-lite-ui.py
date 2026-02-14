#!/usr/bin/env python3
"""
Chatterbox-TTS Lite Gradio UI

- Automatic text chunking for long inputs
- Seamless audio concatenation into a single waveform
- Optional dual-mono stereo output (DAW-friendly)
- Simple, beginner-friendly Gradio interface

This script assumes that:
    pip install chatterbox-tts gradio

and that the Chatterbox-TTS model can be loaded on your system.
"""

import warnings
from pathlib import Path
import json
import shutil
from datetime import datetime
import signal
import sys
import atexit
import threading
import time

import numpy as np
import gradio as gr

from transformers import set_seed

# Try the most common import path for Chatterbox-TTS.
# Adjust here if your local install uses a different name.
from chatterbox.tts import ChatterboxTTS  # type: ignore

# -----------------------------
# Global configuration
# -----------------------------

# Maximum number of characters per text chunk fed to the model.
# This is a heuristic: small enough to avoid the 1000-token acoustic limit,
# but large enough to keep chunks reasonably long.
MAX_CHARS_PER_CHUNK = 280

# Global singleton to avoid re-loading the model on every page refresh
_GLOBAL_MODEL = None

# Decide whether to use CUDA or CPU.
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# Activity tracking for auto-shutdown
_last_activity_time = time.time()
_activity_lock = threading.Lock()
_shutdown_timeout = 300  # 5 minutes of inactivity before shutdown

def update_activity():
    """Update the last activity timestamp."""
    global _last_activity_time
    with _activity_lock:
        _last_activity_time = time.time()

# Optional: quiet down some noisy but harmless warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------
# Character Management
# ---------------------

CHARACTERS_DIR = Path(__file__).parent / "characters"


def ensure_characters_dir():
    """Create characters directory if it doesn't exist."""
    CHARACTERS_DIR.mkdir(exist_ok=True)


def create_character(
    character_name: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    seed: float,
    description: str = "",
) -> tuple[bool, str]:
    """
    Create a new character profile with the given settings.
    
    Returns:
        (success: bool, message: str)
    """
    ensure_characters_dir()
    
    if not character_name or not character_name.strip():
        return False, "Character name cannot be empty."
    
    # Sanitize character name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in character_name)
    character_path = CHARACTERS_DIR / safe_name
    
    if character_path.exists():
        return False, f"Character '{character_name}' already exists."
    
    try:
        character_path.mkdir(parents=True, exist_ok=True)
        
        # Create settings JSON
        settings = {
            "name": character_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "settings": {
                "exaggeration": float(exaggeration),
                "cfg_weight": float(cfg_weight),
                "temperature": float(temperature),
                "min_p": float(min_p),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "seed": float(seed),
            },
        }
        
        settings_file = character_path / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        
        # Create samples directory
        samples_dir = character_path / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        return True, f"Character '{character_name}' created successfully."
    except Exception as e:
        return False, f"Error creating character: {str(e)}"


def load_character(character_name: str) -> dict | None:
    """
    Load a character's settings from JSON file.
    
    Returns:
        Settings dict or None if character not found.
    """
    ensure_characters_dir()
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in character_name)
    settings_file = CHARACTERS_DIR / safe_name / "settings.json"
    
    if not settings_file.exists():
        return None
    
    try:
        with open(settings_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def list_characters() -> list[str]:
    """Get list of all available character names."""
    ensure_characters_dir()
    if not CHARACTERS_DIR.exists():
        return []
    
    characters = []
    for char_dir in CHARACTERS_DIR.iterdir():
        if char_dir.is_dir() and (char_dir / "settings.json").exists():
            settings = load_character(char_dir.name)
            if settings and "name" in settings:
                characters.append(settings["name"])
    
    return sorted(characters)


def delete_character(character_name: str) -> tuple[bool, str]:
    """Delete a character profile and all associated files."""
    ensure_characters_dir()
    
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in character_name)
    character_path = CHARACTERS_DIR / safe_name
    
    if not character_path.exists():
        return False, f"Character '{character_name}' not found."
    
    try:
        shutil.rmtree(character_path)
        return True, f"Character '{character_name}' deleted successfully."
    except Exception as e:
        return False, f"Error deleting character: {str(e)}"


def save_sample_audio(character_name: str, audio_array: np.ndarray, sample_rate: int, filename: str = None) -> tuple[bool, str]:
    """
    Save a generated audio sample to a character's samples folder.
    
    Returns:
        (success: bool, message: str)
    """
    ensure_characters_dir()
    
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in character_name)
    character_path = CHARACTERS_DIR / safe_name
    
    if not character_path.exists():
        return False, f"Character '{character_name}' not found."
    
    try:
        import scipy.io.wavfile as wavfile
        
        samples_dir = character_path / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_{timestamp}.wav"
        
        # Ensure .wav extension
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        
        output_path = samples_dir / filename
        wavfile.write(str(output_path), sample_rate, (audio_array * 32767).astype(np.int16))
        
        return True, f"Sample audio saved: {filename}"
    except Exception as e:
        return False, f"Error saving sample audio: {str(e)}"


# -----------------------------
# Helper functions
# -----------------------------


def split_text_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    """
    Split a long input string into smaller, sentence-like chunks that are
    safe to feed into the TTS model.

    Strategy:
        1. Normalize line endings and strip outer whitespace.
        2. Split into paragraphs using double newlines.
        3. Within each paragraph, walk characters and group into sentences
           based on punctuation (. ? ! ; :).
        4. Accumulate sentences into a chunk until adding another would
           exceed `max_chars`, then start a new chunk.

    Returns:
        List[str]: non-empty text chunks.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: list[str] = []

    # Step 1: paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        current_chunk = ""
        sentence = ""

        # Step 2: walk through characters to form sentences
        for ch in para:
            sentence += ch
            if ch in ".?!;:":
                # Sentence boundary encountered
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Step 3: add sentence to current chunk or start new one
                if current_chunk and len(current_chunk) + 1 + len(sentence) > max_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                sentence = ""

        # Any trailing text without terminal punctuation
        sentence = sentence.strip()
        if sentence:
            if current_chunk and len(current_chunk) + 1 + len(sentence) > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    return chunks


def load_model():
    """
    Load the ChatterboxTTS model the first time this function is called,
    then reuse the same instance for all future calls (e.g. after a
    browser refresh).

    Returns:
        ChatterboxTTS instance.
    """
    global _GLOBAL_MODEL

    # If we've already loaded the model once in this Python process,
    # just reuse it instead of trying to allocate it on the GPU again.
    if _GLOBAL_MODEL is not None:
        print("Reusing already loaded ChatterboxTTS model.")
        return _GLOBAL_MODEL

    print(f"Loading ChatterboxTTS model on device: {DEVICE}")
    _GLOBAL_MODEL = ChatterboxTTS.from_pretrained(DEVICE)
    print("Model successfully loaded.")
    return _GLOBAL_MODEL


# -----------------------------
# Core generation function
# -----------------------------


def generate(
    model: ChatterboxTTS,
    text: str,
    make_stereo: bool,
    audio_prompt_path: str | None,
    exaggeration: float,
    temperature: float,
    seed_num: float,
    cfg_weight: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    stereo_delay_ms: float = 13.0,
):
    """
    Generate audio from input text using Chatterbox-TTS with chunking and concatenation.

    Parameters
    ----------
    model : ChatterboxTTS
        Cached model instance from Gradio `State`.
    text : str
        Full text to synthesize. Can be long; will be split into chunks.
    make_stereo : bool
        Whether to return dual-mono stereo (2-channel) instead of mono.
    audio_prompt_path : str | None
        Optional reference audio path used for voice cloning.
    exaggeration : float
        Controls expressiveness of the voice.
    temperature : float
        Sampling temperature (higher = more variation).
    seed_num : float
        Random seed (0 = no explicit seeding).
    cfg_weight : float
        Pace / classifier-free guidance weight.
    min_p : float
        Minimum probability filter for sampling.
    top_p : float
        Nucleus sampling top-p.
    repetition_penalty : float
        Penalty factor for repeated tokens.
    stereo_delay_ms : float
        Delay in milliseconds to apply to the left channel for pseudo-stereo effect.

    Returns
    -------
    (sample_rate, np.ndarray), str
        Tuple for Gradio's Audio component, and a chunk info message.
    """
    # Track activity
    update_activity()
    
    if model is None:
        # In normal usage this should not happen, because the model is loaded
        # via demo.load() into the model_state. If it does, we raise an error.
        raise RuntimeError("Model is not initialized. Please reload the app.")

    # Apply seed once per full generation to maintain consistent style across chunks.
    if int(seed_num) != 0:
        set_seed(int(seed_num))

    # Split text into manageable chunks
    chunks = split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK)
    if not chunks:
        # If splitting produced nothing (e.g., only whitespace), fall back to stripped text
        stripped = text.strip()
        if stripped:
            chunks = [stripped]
        else:
            # Nothing to speak; return silence
            empty = np.zeros(1, dtype=np.float32)
            return (model.sr, empty), "No text provided."

    # Info for the UI
    chunk_info = f"Input text processed in **{len(chunks)}** chunk(s)."

    all_segments: list[np.ndarray] = []

    for idx, chunk in enumerate(chunks):
        # Optional: log to console for debugging
        print(f"[Chunk {idx+1}/{len(chunks)}] Text: {repr(chunk[:80])}...")

        wav_tensor = model.generate(
            chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Model returns shape (1, num_samples); convert to 1D NumPy
        wav_np = wav_tensor.squeeze(0).detach().cpu().numpy()
        all_segments.append(wav_np)

    # Stitch all chunks together into a single waveform
    if len(all_segments) == 1:
        full_wav = all_segments[0]
    else:
        full_wav = np.concatenate(all_segments, axis=-1)

    # Optional stereo (dual-mono): (num_samples,) -> (num_samples, 2)
    if make_stereo:
        # Calculate delay in samples
        delay_samples = int((stereo_delay_ms / 1000.0) * model.sr)
        
        # Create delayed left channel by trimming from the start
        if delay_samples > 0 and delay_samples < len(full_wav):
            left_channel = full_wav[delay_samples:]
            right_channel = full_wav[:len(left_channel)]
        else:
            # If delay is 0 or invalid, use standard dual-mono
            left_channel = full_wav
            right_channel = full_wav
        
        # Stack into stereo
        stereo_wav = np.stack([left_channel, right_channel], axis=-1)
        return (model.sr, stereo_wav), chunk_info
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Mono output
    return (model.sr, full_wav), chunk_info


# -----------------------------
# Gradio UI
# -----------------------------


def build_interface():
    """
    Build and return the Gradio Blocks interface.
    """
    with gr.Blocks(title="Chatterbox-TTS Lite") as demo:
        # Shared state: cache the model instance
        model_state = gr.State(None)

        gr.Markdown(
            """
            # Chatterbox-TTS Lite 🎧
            A lightweight local interface for Chatterbox-TTS with character management.

            - Automatic text chunking for long inputs  
            - Seamless audio stitching  
            - Optional dual-mono stereo output
            - Create and manage voice characters with custom settings
            """
        )

        # Show which device we're using (CPU/GPU)
        gr.Markdown(f"**Running on device:** `{DEVICE}`")

        with gr.Tabs():
            # ==================== TAB 1: GENERATION ====================
            with gr.Tab("Generate Audio"):
                with gr.Row():
                    with gr.Column():
                        # Character selector
                        def get_character_list():
                            return list_characters()
                        
                        character_dropdown = gr.Dropdown(
                            label="Load Character (optional)",
                            choices=get_character_list(),
                            value=None,
                            interactive=True,
                        )
                        
                        def load_char_settings(char_name):
                            """Load character settings into the sliders."""
                            if not char_name:
                                return (
                                    0.5, 0.6, 0.8, 0.04, 1.0, 1.25, 0,
                                    gr.update(label="Character loaded")
                                )
                            
                            char_data = load_character(char_name)
                            if char_data and "settings" in char_data:
                                s = char_data["settings"]
                                return (
                                    s.get("exaggeration", 0.5),
                                    s.get("cfg_weight", 0.6),
                                    s.get("temperature", 0.8),
                                    s.get("min_p", 0.04),
                                    s.get("top_p", 1.0),
                                    s.get("repetition_penalty", 1.25),
                                    s.get("seed", 0),
                                    gr.update(label=f"✓ {char_name} loaded")
                                )
                            return (
                                0.5, 0.6, 0.8, 0.04, 1.0, 1.25, 0,
                                gr.update(label="Character not found")
                            )
                        
                        # Text input
                        text_input = gr.Textbox(
                            label="Text to synthesize (will be split into chunks automatically)",
                            value=(
                                "Now let's make my mum's favourite. "
                                "So three mars bars into the pan. Then we add the tuna "
                                "and just stir for a bit, just let the chocolate and fish infuse. "
                                "A sprinkle of olive oil and some tomato ketchup. "
                                "Now smell that. Oh boy this is going to be incredible."
                            ),
                            lines=6,
                            max_lines=10,
                        )

                        # Optional reference audio for voice cloning
                        ref_audio = gr.Audio(
                            label="Reference audio (optional, for voice cloning)",
                            sources=["upload", "microphone"],
                            type="filepath",
                            value=None,
                        )

                        # Stereo toggle
                        stereo_checkbox = gr.Checkbox(
                            label="Export as stereo (dual-mono)",
                            value=True,
                        )
                        
                        # Stereo delay control
                        stereo_delay = gr.Slider(
                            label="Stereo delay (ms) - Left channel delay for pseudo-stereo effect",
                            minimum=10,
                            maximum=30,
                            step=1,
                            value=13,
                            visible=True,
                        )

                        exaggeration = gr.Slider(
                            label="Exaggeration (expressiveness)",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.5,
                        )

                        cfg_weight = gr.Slider(
                            label="CFG / Pace weight",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.6,
                        )

                        with gr.Accordion("Advanced settings", open=False):
                            seed_num = gr.Number(
                                label="Seed (0 = random each time)",
                                value=0,
                                precision=0,
                            )

                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                step=0.05,
                                value=0.8,
                            )

                            min_p = gr.Slider(
                                label="min_p",
                                minimum=0.0,
                                maximum=0.2,
                                step=0.005,
                                value=0.04,
                            )

                            top_p = gr.Slider(
                                label="top_p",
                                minimum=0.5,
                                maximum=1.0,
                                step=0.05,
                                value=1.0,
                            )

                            repetition_penalty = gr.Slider(
                                label="Repetition penalty",
                                minimum=1.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.25,
                            )

                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate Audio",
                                variant="primary",
                            )
                            save_sample_btn = gr.Button(
                                "Save as Sample for Character",
                                variant="secondary",
                            )

                    with gr.Column():
                        audio_output = gr.Audio(
                            label="Generated audio",
                            type="numpy",
                            interactive=False,
                        )

                        # Display how many chunks were used for this generation
                        chunk_info = gr.Markdown("Chunk information will appear here after generation.")
                        
                        char_save_status = gr.Textbox(
                            label="Character Save Status",
                            interactive=False,
                            value="Ready to save audio samples"
                        )

                # Wire the character loader
                character_dropdown.change(
                    fn=load_char_settings,
                    inputs=character_dropdown,
                    outputs=[exaggeration, cfg_weight, temperature, min_p, top_p, repetition_penalty, seed_num, char_save_status],
                )

                # Wire the button to the generate() function
                generate_btn.click(
                    fn=generate,
                    inputs=[
                        model_state,
                        text_input,
                        stereo_checkbox,
                        ref_audio,
                        exaggeration,
                        temperature,
                        seed_num,
                        cfg_weight,
                        min_p,
                        top_p,
                        repetition_penalty,
                        stereo_delay,
                    ],
                    outputs=[audio_output, chunk_info],
                )
                
                # Wire save sample button
                def save_sample_handler(audio_data, char_name):
                    if not char_name:
                        return "Please select a character first."
                    if audio_data is None:
                        return "Please generate audio first."
                    
                    try:
                        sample_rate, audio_array = audio_data
                        # Normalize if stereo
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.mean(axis=1)
                        success, msg = save_sample_audio(char_name, audio_array, sample_rate)
                        return msg
                    except Exception as e:
                        return f"Error saving sample: {str(e)}"
                
                save_sample_btn.click(
                    fn=save_sample_handler,
                    inputs=[audio_output, character_dropdown],
                    outputs=char_save_status,
                )

            # ==================== TAB 2: CHARACTER MANAGEMENT ====================
            with gr.Tab("Manage Characters"):
                gr.Markdown(
                    """
                    ## Character Management
                    Create, load, and manage voice characters with custom settings.
                    Each character is stored with their settings and sample audio files.
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Manage Existing Characters")
                        
                        existing_chars = gr.Dropdown(
                            label="Select Character",
                            choices=list_characters(),
                            interactive=True,
                        )
                        
                        char_details = gr.JSON(
                            label="Character Details",
                        )
                        
                        def show_char_details(char_name):
                            if not char_name:
                                return {}
                            char_data = load_character(char_name)
                            if char_data:
                                # Remove created_at for cleaner display
                                display = {
                                    "name": char_data.get("name"),
                                    "description": char_data.get("description", ""),
                                    "settings": char_data.get("settings", {}),
                                    "sample_count": len(list((CHARACTERS_DIR / "".join(c if c.isalnum() or c in "-_" else "_" for c in char_name) / "samples").glob("*.wav"))) if (CHARACTERS_DIR / "".join(c if c.isalnum() or c in "-_" else "_" for c in char_name) / "samples").exists() else 0,
                                }
                                return display
                            return {}
                        
                        existing_chars.change(
                            fn=show_char_details,
                            inputs=existing_chars,
                            outputs=char_details,
                        )
                        
                        load_to_generator_btn = gr.Button("Load into Generator", variant="secondary")
                        
                        def load_to_generator(char_name):
                            if not char_name:
                                return [gr.update(value=None), 0.5, 0.6, 0.8, 0.04, 1.0, 1.25, 0, gr.update(label="Character loaded")]
                            
                            # Load character settings
                            char_data = load_character(char_name)
                            if char_data and "settings" in char_data:
                                s = char_data["settings"]
                                return [
                                    gr.update(value=char_name, choices=list_characters()),
                                    s.get("exaggeration", 0.5),
                                    s.get("cfg_weight", 0.6),
                                    s.get("temperature", 0.8),
                                    s.get("min_p", 0.04),
                                    s.get("top_p", 1.0),
                                    s.get("repetition_penalty", 1.25),
                                    s.get("seed", 0),
                                    gr.update(label=f"✓ {char_name} loaded")
                                ]
                            return [gr.update(value=char_name, choices=list_characters()), 0.5, 0.6, 0.8, 0.04, 1.0, 1.25, 0, gr.update(label="Character not found")]
                        
                        load_to_generator_btn.click(
                            fn=load_to_generator,
                            inputs=existing_chars,
                            outputs=[character_dropdown, exaggeration, cfg_weight, temperature, min_p, top_p, repetition_penalty, seed_num, char_save_status],
                        )
                        
                        delete_btn = gr.Button("Delete Character", variant="stop")
                        delete_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready to delete"
                        )
                        
                        def delete_char_handler(char_name):
                            if not char_name:
                                return "Please select a character first."
                            success, msg = delete_character(char_name)
                            return msg
                        
                        delete_btn.click(
                            fn=delete_char_handler,
                            inputs=existing_chars,
                            outputs=delete_status,
                        )
                        
                        # Add button to refresh character list
                        refresh_btn = gr.Button("Refresh Character List")
                        
                        def refresh_lists():
                            new_choices = list_characters()
                            return [
                                gr.update(choices=new_choices),
                                gr.update(choices=new_choices),
                            ]
                        
                        refresh_btn.click(
                            fn=refresh_lists,
                            outputs=[character_dropdown, existing_chars],
                        )

                    with gr.Column():
                        gr.Markdown("### Create New Character")
                        
                        new_char_name = gr.Textbox(
                            label="Character Name",
                            placeholder="e.g., Wise Narrator, Excited Sally",
                        )
                        
                        new_char_desc = gr.Textbox(
                            label="Description (optional)",
                            placeholder="e.g., Deep voice, dramatic delivery",
                            lines=2,
                        )
                        
                        gr.Markdown("#### Voice Settings for New Character")
                        
                        new_exaggeration = gr.Slider(
                            label="Exaggeration",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.5,
                        )
                        
                        new_cfg_weight = gr.Slider(
                            label="CFG / Pace weight",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.6,
                        )
                        
                        new_temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            step=0.05,
                            value=0.8,
                        )
                        
                        new_min_p = gr.Slider(
                            label="min_p",
                            minimum=0.0,
                            maximum=0.2,
                            step=0.005,
                            value=0.04,
                        )
                        
                        new_top_p = gr.Slider(
                            label="top_p",
                            minimum=0.5,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                        )
                        
                        new_repetition_penalty = gr.Slider(
                            label="Repetition penalty",
                            minimum=1.0,
                            maximum=2.0,
                            step=0.05,
                            value=1.25,
                        )
                        
                        new_seed = gr.Number(
                            label="Seed (0 = random)",
                            value=42,
                            precision=0,
                        )
                        
                        create_btn = gr.Button("Create Character", variant="primary")
                        create_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready to create character"
                        )
                        
                        def create_char_handler(name, desc, exag, cfg, temp, min_p_val, top_p_val, rep_pen, seed_val):
                            success, msg = create_character(
                                name, exag, cfg, temp, min_p_val, top_p_val, rep_pen, seed_val, desc
                            )
                            new_choices = list_characters() if success else []
                            return msg, gr.update(choices=new_choices, value=name if success else None), gr.update(choices=new_choices)
                        
                        create_btn.click(
                            fn=create_char_handler,
                            inputs=[
                                new_char_name, new_char_desc,
                                new_exaggeration, new_cfg_weight, new_temperature,
                                new_min_p, new_top_p, new_repetition_penalty, new_seed
                            ],
                            outputs=[create_status, character_dropdown, existing_chars],
                        )

        # Pre-load the model when the app starts
        demo.load(
            fn=load_model,
            inputs=None,
            outputs=model_state,
        )

    return demo


def cleanup_on_exit():
    """Clean up GPU memory on exit."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is not None:
        print("\n[Cleanup] Releasing GPU memory...")
        _GLOBAL_MODEL = None
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass
    print("[Cleanup] Server shutdown complete.")


def monitor_inactivity():
    """Monitor for inactivity and shutdown server if idle too long."""
    global _last_activity_time
    while True:
        time.sleep(30)  # Check every 30 seconds
        with _activity_lock:
            idle_time = time.time() - _last_activity_time
        
        if idle_time > _shutdown_timeout:
            print(f"\n[Auto-Shutdown] No activity for {_shutdown_timeout/60:.1f} minutes. Shutting down...")
            cleanup_on_exit()
            import os
            os._exit(0)


if __name__ == "__main__":
    # Register cleanup handler
    atexit.register(cleanup_on_exit)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[Signal] Received interrupt signal, shutting down...")
        cleanup_on_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start inactivity monitor in background
    monitor_thread = threading.Thread(target=monitor_inactivity, daemon=True)
    monitor_thread.start()
    
    app = build_interface()
    
    # Launch with inbrowser=True to auto-open
    print("\n[Server] Starting Chatterbox-TTS Lite...")
    print("[Server] Browser will open automatically.")
    print(f"[Server] Server will auto-shutdown after {_shutdown_timeout/60:.0f} minutes of inactivity.")
    print("[Server] Press Ctrl+C to shutdown manually.\n")
    
    app.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        quiet=False,
    )
