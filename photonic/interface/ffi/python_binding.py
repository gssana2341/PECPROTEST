# python_binding.py — Python binding for Photonic library
#
# เรียกใช้ photonic core ผ่าน ctypes/cffi
# ไม่ต้องติดตั้ง PyTorch

import ctypes
import os

# Path to the compiled shared library
_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'libphotonic.so')

class PhotonicLib:
    """Python wrapper for the Photonic C library."""

    def __init__(self, lib_path=None):
        self._lib = ctypes.CDLL(lib_path or _LIB_PATH)
        self._setup_functions()

    def _setup_functions(self):
        """Define C function signatures."""
        # pho_init
        self._lib.pho_init.restype = ctypes.c_int
        self._lib.pho_init.argtypes = []

        # pho_shutdown
        self._lib.pho_shutdown.restype = None
        self._lib.pho_shutdown.argtypes = []

        # pho_version
        self._lib.pho_version.restype = ctypes.c_char_p
        self._lib.pho_version.argtypes = []

    def init(self):
        return self._lib.pho_init()

    def shutdown(self):
        self._lib.pho_shutdown()

    def version(self):
        return self._lib.pho_version().decode('utf-8')


# TODO: Add higher-level Pythonic API wrappers
# e.g., PhotonicLayer class, numpy array conversion, etc.
