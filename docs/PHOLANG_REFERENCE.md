# PhoLang Language Reference (v3.0 - Photonic Hardware Bridging Edition)

**PhoLang (`.pho`)** is the world's only domain-specific programming language built to **bridge the gap between Silicon Photonic Hardware and Production AI Deployments**.

PhoLang does not try to be a general-purpose scripting language like Python. Instead, it is an ultra-specialized **Hardware-Aware Neural compiler**. It compiles physical waveguide designs and noise parameters directly into optimized, parallelized, and secure C99 binaries ready for edge or server APIs.

---

## 1. 🏗️ High-Fidelity Hardware Syntax

PhoLang's syntax directly mirrors the physical components of a silicon photonic chip rather than abstract mathematical concepts.

### `waveguide pool(in, out)`
Simulates a physical waveguide array focusing and downsampling light.
*   **Example:** `waveguide pool(784, 64)`

### `mzi_mesh(size, [topology])`
Instantiates a grid of Mach-Zehnder Interferometers (MZIs) in a specified physical topology (`clements` or `reck`). The `clements` topology offers a symmetric rectangular structure (providing 2x better tolerance to optical loss), while `reck` provides a triangular layout. Mathematically models a Unitary matrix ($W^\dagger W = I$) while enforcing energy conservation.
*   **Example:** `mzi_mesh(64, topology=clements)`

### `readout softmax(classes)`
Models the photodetector array at the end of the waveguides converting coherent light intensity into electrical signals, followed by a digital softmax scaling.
*   **Example:** `readout softmax(10)`

---

## 2. 🛡️ First-Class Hardware Noise Directives & Constraints

Silicon waveguides are highly sensitive to thermal fluctuations, fabrication defects, and DAC/ADC constraints. In PhoLang, **noise and constraints are first-class citizens**. You inject parameters directly into the hardware layer definition to train models that are robust when deployed to physical hardware.

```pho
photon network EdgeClassifier {
    waveguide pool(784, 64)
    
    // MZI mesh with real physical noise and hardware constraints!
    mzi_mesh(64) kerr(0.5) 
        @drift(0.02) @shot_noise(0.01) // Physical noise
        @dac(8) @topology(clements)    // Hardware constraints
    
    readout softmax(10)
}
```

### A. Physical Noise Directives
*   **`@drift(0.02)`**: Simulates a 0.02 radian thermo-optic phase drift per heater.
*   **`@shot_noise(0.01)`**: Simulates 1% quantum shot noise on the photodetector array.

### B. Hardware Constraints & Architecture
*   **`@dac(8)`**: Clamps heater control voltages to 8-bit Digital-to-Analog Converter (DAC) precision.
*   **`@topology(clements)`**: Enforces Clements rectangular mesh topology (or `@topology(reck)` for Reck triangular topology).

---

## 3. 🌐 Web Serving with Latency Specs

Since the ultimate advantage of photonic computing is sub-millisecond execution speeds, PhoLang allows specifying performance and latency budgets directly in the deployment directive.

### `serve(port, route, [target_latency])`
Generates a zero-dependency C HTTP microservice optimized to hit the specified latency target using dynamic thread pool scaling.
*   **Example:**
    ```pho
    // Expose model with a strict sub-millisecond latency target!
    EdgeClassifier.serve(port=8080, route="/v1/predict", target_latency=0.1ms)
    ```

---

## 4. 🔒 Safe-by-Default & Sandbox Execution Modes

To ensure untrusted scripts compiled on a server cannot compromise the host (RCE protection), PhoLang executes in a strict security sandbox by default.

### A. The `@mode(unsafe)` Directive
To enable raw C/C++ linking or inline Python blocks (for local custom DSP development), you must explicitly flag the file as unsafe at the very top:

```pho
@mode(unsafe) // Explicitly flags the script as memory-unsafe / trusted
import c "custom_dsp.h"

photon network CustomAPI {
    mzi_mesh(64)
    readout softmax(10)
}
```

If a script attempts FFI operations without `@mode(unsafe)`, the `pho` compiler throws a compilation error.

### B. SaaS / Cloud Server Enforcement
When hosting PhoLang on a Cloud/SaaS server, the API compiler runs with the `--safe-mode` flag, which **completely ignores `@mode(unsafe)` and rejects compilation of unsafe scripts**, keeping the server 100% secure.

---

## 5. 🛠️ Cargo-Style Project Tooling

To ensure a modern, friction-free developer experience (similar to Rust's Cargo or Go's tooling), the PhoLang compiler (`pho`) manages the build workspace without cluttering your project folder with leftover intermediate C or object files.

*   **`pho run <script.pho>`**: Transpiles, compiles via host GCC/Clang, and immediately runs the network model. All intermediate files are cached cleanly inside `target/` and executed from `target/app`.
*   **`pho build <script.pho>`**: Compiles the network into a native binary inside `target/app` without executing.
*   **`pho <script.pho> <output.c>`**: Standard transpilation direct to C file (fully backwards-compatible).

---

## 6. Flexible Syntax Matrix

| Style | Code Example | Behind the Scenes |
| :--- | :--- | :--- |
| **Zero-Boilerplate** | `fit("mnist.csv")` | Automatically analyzes data, instantiates a default `waveguide pool` -> `mzi_mesh` -> `readout` chain, and trains it. |
| **Hardware-Aware** | `mzi_mesh(64, topology=clements) @drift(0.02)` | Compiles a custom MZI lattice using EKF-based drift-compensation, generating optimized C matrices. |
| **Web Service** | `Model.serve(port=80, target_latency=0.1ms)` | Compiles a production C-based microservice optimized for microsecond-level optical inference response times. |
| **FFI (Local Only)** | `@mode(unsafe)` + `import c` | Zero-overhead linkage of external DSP filters directly into the optical pipeline. |

---

## 7. 🔮 Future Specification: Standalone Project Architecture (v4.0)

To transition PhoLang into a fully standalone programming language where developers never touch host C compilers, PhoLang v4.0 specifies a standardized **Zero-C Multi-File Project Layout**.

### A. The Project Manifest (`pho.toml`)
Placed at the root of a project directory, `pho.toml` handles dependencies, entrypoints, and global physical hardware constraints:
```toml
[project]
name = "smart_photonic_sensor"
version = "1.0.0"
entry = "src/main.pho"

[hardware]
topology = "clements"
dac_precision = 8
phase_drift = 0.02
```

### B. Standard Standalone Layout
```text
my_project/
├── pho.toml          # Project configuration manifest
├── src/              # Source code directory
│   ├── main.pho      # Primary entrypoint script
│   └── filters.pho   # Custom optical signal processing filters
└── target/           # Cached build outputs
```

### C. Native Import Directive
`import` statements allow `.pho` files to pull in classes/networks declared in other `.pho` modules:
```pho
import "filters.pho"

photon network MainClassifier {
    // Instantiate custom optical filter imported from filters.pho
    use filters.NoiseFilter()
    
    mzi_mesh(64)
    readout softmax(10)
}
```

---

PhoLang 3.0 bridges the gap between the physical constraints of silicon and the high-throughput requirements of production AI. 🌌🚀🔌🔒
