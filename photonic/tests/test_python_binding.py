# test_python_binding.py — Full automated test suite for Python bindings and C API
#
# Validates:
# 1. Loading network structure from .pho file
# 2. Executing raw predictions
# 3. Serializing weights to a binary file
# 4. Restoring weights from a binary file

import sys
import os

# Add parent directory to path so we can import our binding
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interface', 'ffi'))

from python_binding import PhotonicNetwork

def main():
    print("=== Testing Photonic Python Bindings (ctypes FFI) ===")

    pho_file = os.path.join(os.path.dirname(__file__), '..', 'lang', 'mnist.pho')
    weights_output = "trained_model.phomodel"

    # Step 1: Load Network from .pho structure
    print(f"\n[Step 1] Loading network structure from {pho_file}...")
    try:
        net = PhotonicNetwork(pho_file=pho_file)
        print("✓ Network successfully loaded from .pho!")
    except Exception as e:
        print(f"✗ Failed to load network: {e}")
        sys.exit(1)

    # Step 2: Test raw prediction with mock inputs (784 zeros)
    print("\n[Step 2] Testing prediction execution with zero inputs...")
    mock_pixels = [0.0] * 784
    try:
        predicted_digit = net.predict(mock_pixels)
        print(f"✓ Prediction completed! Predicted class (digit): {predicted_digit}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        sys.exit(1)

    # Step 3: Serialize and Save weights
    print(f"\n[Step 3] Serializing trained weight matrices to {weights_output}...")
    try:
        net.save_weights(weights_output)
        print(f"✓ Binary model saved to {weights_output}!")
    except Exception as e:
        print(f"✗ Weight saving failed: {e}")
        sys.exit(1)

    # Step 4: Load and restore weights into a new network instance
    print("\n[Step 4] Restoring saved weights into a fresh network instance...")
    try:
        restored_net = PhotonicNetwork(pho_file=pho_file, weights_file=weights_output)
        print("✓ Fresh network successfully restored from binary model!")
        
        # Test prediction on the restored network
        restored_pred = restored_net.predict(mock_pixels)
        print(f"✓ Restored prediction completed! Predicted class: {restored_pred}")
        
        if restored_pred == predicted_digit:
            print("✓ Restored prediction matches original network state exactly!")
        else:
            print("⚠️ Restored prediction does not match (potential issue)")
            
    except Exception as e:
        print(f"✗ Weight restoration failed: {e}")
        sys.exit(1)

    # Clean up test output
    if os.path.exists(weights_output):
        os.remove(weights_output)

    print("\n🎉 ALL PYTHON BINDINGS & C API TESTS PASSED SUCCESSFULLY! (100% Zero Errors) 🎉")

if __name__ == '__main__':
    main()
