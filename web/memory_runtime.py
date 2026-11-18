"""Return freed native image/tracker buffers to memory-constrained Linux hosts."""
import ctypes
import gc
import logging
import os
import sys


def _load_trim():
    if not sys.platform.startswith("linux"):
        return None
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return trim
    except (AttributeError, OSError):
        return None


_trim = _load_trim()


def release_native_memory():
    # MediaPipe's native allocations can outlive Python reference cycles, and
    # glibc can retain already-freed pages in thread arenas until explicitly
    # trimmed. Do this at capture/request boundaries, never for every frame.
    gc.collect()
    if _trim is not None:
        _trim(0)
    if sys.platform.startswith("linux") and os.environ.get("VSL_LOG_MEMORY") == "1":
        try:
            with open("/proc/self/statm", encoding="ascii") as usage:
                resident = int(usage.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            logging.getLogger("uvicorn.error").info("Native buffers released; RSS %.1f MiB", resident / 1024**2)
        except (OSError, ValueError, IndexError):
            pass
