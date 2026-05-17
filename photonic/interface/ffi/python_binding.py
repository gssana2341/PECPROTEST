# python_binding.py — High-Level Python Wrapper for the Photonic C API
#
# Interacts with the shared library (libphotonic.so) via ctypes.
# Zero external dependencies (like PyTorch or NumPy) required!

import ctypes
import os
import sys

# Define standard ctypes structures
class PhoTrainConfig(ctypes.Structure):
    _fields_ = [
        ("epochs", ctypes.c_int),
        ("lr", ctypes.c_double),
        ("batch_size", ctypes.c_int)
    ]

class PhoResult(ctypes.Structure):
    _fields_ = [
        ("final_loss", ctypes.c_double),
        ("final_accuracy", ctypes.c_double)
    ]

class PhotonicNetwork:
    """Pythonic wrapper class for the Photonic Neuromorphic Computing Engine."""

    _lib = None

    @classmethod
    def _get_library(cls):
        """Loads and caches the compiled shared library (libphotonic.so)."""
        if cls._lib is not None:
            return cls._lib

        # Walk up to find the libphotonic.so in the root folder or lib folders
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, '..', '..', '..', 'libphotonic.so'),
            os.path.join(current_dir, '..', '..', 'libphotonic.so'),
            os.path.join(current_dir, '..', '..', '..', 'libphotonic.dylib'),
            os.path.join(current_dir, '..', '..', '..', 'libphotonic.dll'),
            'libphotonic.so'
        ]

        lib_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                lib_path = abs_path
                break

        try:
            cls._lib = ctypes.CDLL(lib_path or 'libphotonic.so')
            cls._setup_signatures(cls._lib)
        except Exception as e:
            print(f"[FFI Error] Failed to load libphotonic shared library: {e}", file=sys.stderr)
            print("Make sure to run 'make' or compile a shared library first.", file=sys.stderr)
            raise e

        return cls._lib

    @classmethod
    def _setup_signatures(cls, lib):
        """Setup function argument and return type signatures for the CDLL."""
        # pho_network_load
        lib.pho_network_load.argtypes = [ctypes.c_char_p]
        lib.pho_network_load.restype = ctypes.c_void_p

        # pho_network_train
        lib.pho_network_train.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PhoTrainConfig]
        lib.pho_network_train.restype = PhoResult

        # pho_network_predict
        lib.pho_network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.pho_network_predict.restype = ctypes.c_int

        # pho_network_save
        lib.pho_network_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.pho_network_save.restype = ctypes.c_int

        # pho_network_load_weights
        lib.pho_network_load_weights.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.pho_network_load_weights.restype = ctypes.c_void_p

        # pho_network_compress
        lib.pho_network_compress.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.pho_network_compress.restype = ctypes.c_double

        # pho_network_free
        lib.pho_network_free.argtypes = [ctypes.c_void_p]
        lib.pho_network_free.restype = None

        # pho_last_error
        lib.pho_last_error.argtypes = []
        lib.pho_last_error.restype = ctypes.c_char_p

    def __init__(self, pho_file: str = None, weights_file: str = None, _handle=None):
        """Initializes a new Photonic Network from .pho file, or loads trained weights."""
        self._lib = self._get_library()
        self._handle = None

        if _handle is not None:
            self._handle = _handle
        elif weights_file is not None and pho_file is not None:
            self._handle = self._lib.pho_network_load_weights(pho_file.encode('utf-8'), weights_file.encode('utf-8'))
            if not self._handle:
                err = self._lib.pho_last_error().decode('utf-8')
                raise RuntimeError(f"Failed to load weights: {err}")
        elif pho_file is not None:
            self._handle = self._lib.pho_network_load(pho_file.encode('utf-8'))
            if not self._handle:
                err = self._lib.pho_last_error().decode('utf-8')
                raise RuntimeError(f"Failed to load network definition: {err}")
        else:
            raise ValueError("Must provide either a .pho definition file path or an active handle.")

    def train(self, data_csv_path: str, epochs: int = -1, lr: float = -1.0, batch_size: int = -1) -> dict:
        """Trains the network on a CSV dataset. Returns the final evaluation loss and accuracy."""
        config = PhoTrainConfig(epochs=epochs, lr=lr, batch_size=batch_size)
        res = self._lib.pho_network_train(self._handle, data_csv_path.encode('utf-8'), config)
        return {
            "final_loss": res.final_loss,
            "final_accuracy": res.final_accuracy
        }

    def predict(self, input_pixels: list) -> int:
        """Predicts the digit category for a single image sample (784 float values)."""
        if len(input_pixels) != 784:
            raise ValueError(f"Input image must contain exactly 784 pixels, got {len(input_pixels)}")

        c_arr = (ctypes.c_double * 784)(*input_pixels)
        pred = self._lib.pho_network_predict(self._handle, c_arr, 784)
        if pred < 0:
            err = self._lib.pho_last_error().decode('utf-8')
            raise RuntimeError(f"Prediction failed: {err}")
        return pred

    def save_weights(self, weights_path: str):
        """Serializes the trained weights into a compact binary .phomodel file."""
        ret = self._lib.pho_network_save(self._handle, weights_path.encode('utf-8'))
        if ret < 0:
            err = self._lib.pho_last_error().decode('utf-8')
            raise RuntimeError(f"Failed to save weight file: {err}")

    def compress(self, dac_bits: int, pruning_threshold: float) -> float:
        """Applies DAC-aware pruning and quantization to weight matrices. Returns savings percentage."""
        return self._lib.pho_network_compress(self._handle, dac_bits, ctypes.c_double(pruning_threshold))

    def __del__(self):
        """Cleans up internal C network allocations on garbage collection."""
        if self._handle and self._lib:
            self._lib.pho_network_free(self._handle)
            self._handle = None
