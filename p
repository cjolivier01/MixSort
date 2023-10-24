#!/bin/bash
LD_LIBRARY_PATH="${HOME}/cuda/lib64:$LD_LIBRARY_PATH" \
  OMP_NUM_THREADS=16 \
  PYTHONPATH="$(pwd):$(pwd)/../../src" \
  python3 $@
