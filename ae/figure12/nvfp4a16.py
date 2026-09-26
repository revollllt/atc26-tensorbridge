"""Make the NVFP4A16 (Marlin) checkpoint from an NVFP4A8 one.

The weights stay the same; only the activation quantization is removed from the
config, so vLLM picks its weight-only NVFP4 scheme (Marlin on Hopper) instead of
failing to find a scheme for NVFP4 weights with FP8 activations. The weight files
are symlinked, not copied. Usage: python nvfp4a16.py NVFP4A8_DIR OUTPUT_DIR
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name != "config.json" and not (output / path.name).exists():
            (output / path.name).symlink_to(path)
    config = json.loads((source / "config.json").read_text())
    for group in config["quantization_config"]["config_groups"].values():
        group["input_activations"] = None
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
