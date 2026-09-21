#!/usr/bin/env bash
# Run after activating a fresh Python 3.12 environment.
set -euo pipefail
odysseus_verl_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the reference environment"'
python -m pip install 'setuptools==78.1.1' 'wheel==0.45.1' 'ninja==1.13.0'
python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$odysseus_verl_root/requirements-odysseus.txt"
# Reference wheel for Linux x86_64, CPython 3.12, Torch 2.8 / CUDA 12, CXX11 ABI.
# Set ODYSSEUS_FLASH_ATTN_WHEEL to a matching local wheel or use source explicitly.
odysseus_flash_wheel=${ODYSSEUS_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}
python -m pip install --no-deps "$odysseus_flash_wheel"
python -m pip install --no-deps -e "$odysseus_verl_root"
python -m pip check
