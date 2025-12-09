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

# Decide whether to use CUDA or CPU.
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# Optional: quiet down some noisy but harmless warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    Load the ChatterboxTTS model once when the Gradio app starts.

    Returns:
        ChatterboxTTS instance.
    """
    print(f"Loading ChatterboxTTS model on device: {DEVICE}")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    print("Model successfully loaded.")
    return model


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

    Returns
    -------
    (sample_rate, np.ndarray), str
        Tuple for Gradio's Audio component, and a chunk info message.
    """
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
        stereo_wav = np.stack([full_wav, full_wav], axis=-1)
        return (model.sr, stereo_wav), chunk_info

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
            A lightweight local interface for Chatterbox-TTS.

            - Automatic text chunking for long inputs  
            - Seamless audio stitching  
            - Optional dual-mono stereo output  
            """
        )

        # Show which device we're using (CPU/GPU)
        gr.Markdown(f"**Running on device:** `{DEVICE}`")

        with gr.Row():
            with gr.Column():
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

                generate_btn = gr.Button(
                    "Generate Audio",
                    variant="primary",
                )

            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated audio",
                    type="numpy",
                )

                # Display how many chunks were used for this generation
                chunk_info = gr.Markdown("Chunk information will appear here after generation.")

        # Pre-load the model when the app starts
        demo.load(
            fn=load_model,
            inputs=None,
            outputs=model_state,
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
            ],
            outputs=[audio_output, chunk_info],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
