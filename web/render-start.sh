#!/bin/sh
set -eu
# Set allocator limits before Python and native model libraries initialize.
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VSL_LOG_MEMORY=1
exec python -m uvicorn test:app --app-dir web --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
