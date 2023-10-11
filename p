#!/bin/bash
LD_LIBRARY_PATH="${HOME}/cuda/lib64:$LD_LIBRARY_PATH" \
  PYTHONPATH=$(pwd) \
  python3 $@
