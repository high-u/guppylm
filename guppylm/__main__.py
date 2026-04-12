"""Entry point for: python -m guppylm"""

import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"


def main():
    if len(sys.argv) < 2:
        print("GuppyLM — A tiny fish brain")
        print()
        print("Usage:")
        print("  python -m guppylm train        Train the model")
        print("  python -m guppylm prepare      Generate data & train tokenizer")
        print("  python -m guppylm chat         Chat with Guppy")
        print("    --safetensors                Use safetensors checkpoint")
        print("  python -m guppylm export       Export to ONNX")
        print("    --no-quantize                Skip uint8 quantization")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "chat":
        sf_path = CHECKPOINT_PATH.replace(".pt", ".safetensors")
        if "--safetensors" in sys.argv:
            if not os.path.exists(sf_path):
                print("safetensors not found. Train first:\n")
                print("  python -m guppylm train")
                return
            sys.argv.remove("--safetensors")
            sys.argv[0] = sf_path
        elif not os.path.exists(CHECKPOINT_PATH):
            print("Model not found. Train first:\n")
            print("  python -m guppylm prepare")
            print("  python -m guppylm train")
            return

        from .inference import main as inference_main
        inference_main()

    elif cmd == "export":
        from .export_onnx import main as export_main
        export_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m guppylm' for usage.")


main()
