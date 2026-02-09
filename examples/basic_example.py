import os
import soundfile as sf
import torch
import time
import random
import logging
from neuttsair.neutts import NeuTTSAir

# -----------------------------------------------------------------------------
# Logging setup (timestamps in every line)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [neutts-air] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("neutts-air")

# -----------------------------------------------------------------------------
# Simulated "other process" input – 10 candidate texts
# Replace these with your real phrases
# -----------------------------------------------------------------------------
SAMPLE_TEXTS = [
    "Any circle that is too far from any point line is discarded. This removes many floating in empty space detections.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Real time speech generation is now running on the Mytelligent device.",
    "Please verify the output audio quality using the onboard speaker.",
    "This is a test sentence to measure end to end latency of the TTS engine.",
    "Edge AI allows language models and speech synthesis to run completely offline.",
    "Multiple sentences can be generated in sequence without reloading the model.",
    "Welcome to the Mytelligent operating system with embedded neural text to speech.",
    "This sample demonstrates random sentence selection for continuous testing.",
    "If you can hear this voice clearly, the NeuTTS Air codec is working correctly.",
]

def get_next_text():
    """
    Placeholder for 'text from another process'.

    Right now we just pick one of the SAMPLE_TEXTS at random.
    Later, you can replace this with:
      - a queue,
      - a FIFO pipe,
      - a socket,
      - or any IPC mechanism.
    """
    return random.choice(SAMPLE_TEXTS)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main(ref_codes_path, ref_text, backbone):
    log.info("Starting NeuTTS-Air engine...")
    engine = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu",
    )
    log.info("NeuTTS-Air engine initialized (backbone + codec loaded).")

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    # Continuous loop: simulate real-time incoming text
    # Adjust or add a break condition as needed for your service.
    try:
        while True:
            text = get_next_text()

            # Mark start time
            t_start = time.monotonic()
            log.info(f"Received text to synthesize: {text!r}")

            # Generate audio
            wav = engine.infer(text, ref_codes, ref_text)

            t_end = time.monotonic()
            gen_time = t_end - t_start
            audio_duration = len(wav) / 24000.0  # seconds, since sr=24000

            # Overwrite a single file (or add timestamped names if you prefer)
            output_path = f"output-{t_start}.wav"
            sf.write(output_path, wav, 24000)

            log.info(
                "Generated speech -> %s | gen_time = %.3f s | audio_duration = %.2f s",
                output_path,
                gen_time,
                audio_duration,
            )

            # Simulate waiting for next request.
            # If you wire this to real IPC, you can remove or adjust this sleep.
            time.sleep(2.0)

    finally:
        # Clean shutdown to avoid destructor noise
        log.info("Shutting down NeuTTS-Air engine...")
        try:
            engine.close()
        except AttributeError:
            pass
        log.info("NeuTTS-Air engine closed.")


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example")
    parser.add_argument(
        "--ref_codes", 
        type=str, 
        default="./samples/dave.pt", 
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt", 
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="neuphonic/neutts-air", 
        help="Huggingface repo containing the backbone checkpoint"
    )
    args = parser.parse_args()
    main(
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
    )
