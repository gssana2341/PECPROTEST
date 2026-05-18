# PhoLang Quickstart Guide 🚀

Welcome! In this 5-minute guide, you will write your first **Photonic Neural Network** using our custom domain-specific language: **PhoLang** (`.pho`).

## 1. The Structure of a `.pho` File

PhoLang provides a very simple, declarative syntax to define your neural network architecture. Let's create a basic 3-layer network.

Create a new file called `my_network.pho`:

```pho
photon network MyFirstNetwork {
    // Downsample 28x28 image to 8x8 optical field using a physical waveguide pool
    waveguide pool(28, 8)
    
    // Photonic MZI meshes with Kerr nonlinearity and TIA gain
    mzi_mesh(64) kerr(0.5) gain(15.0)
    mzi_mesh(64) kerr(0.5) gain(15.0)
    
    // 10-class output
    readout softmax(10)
    
    // Riemannian optimizer
    train riemannian(lr=0.01, epochs=100, batch_size=32)
}
```

## 2. Compiling and Running the Network

Our compiler CLI (`pho`) acts as the project manager and compiler (similar to Rust's Cargo). It compiles and runs your `.pho` file directly, caching build artifacts cleanly inside the `target/` directory:

Run the network immediately:
```bash
./pho run my_network.pho
```

This will automatically transpile the code, build a native C executable in `target/app`, and execute it instantly with zero leftover files in your root workspace!

If you only want to build the production binary without running it, use:
```bash
./pho build my_network.pho
```

## 3. Running the MNIST Demo

Want to see it in action with real data? We've prepared an end-to-end MNIST demo that compiles a network and trains it on handwritten digits.

Run the demo target from the root directory:
```bash
make demo
```

**What happens?**
1. It compiles `photonic/lang/mnist.pho` into C code.
2. It links the generated C code with our `libphotonic` core engine.
3. It executes the simulation, reporting loss and accuracy!

## 4. Writing Your Own Application

To use your compiled network in your own C project:

1. Compile the generated code:
   ```bash
   gcc -c generated_network.c -o generated_network.o -I./photonic/core
   ```
2. Link it with your `main.c` and `libphotonic.a`:
   ```bash
   gcc main.c generated_network.o -L. -lphotonic -lm -fopenmp -o my_app
   ```

### Example `main.c`:
```c
#include <stdio.h>
// Include the generated header if you created one, or forward declare the init functions

int main() {
    printf("Starting Photonic Neural Network...\n");
    
    // 1. Initialize Network using generated initialization functions
    // 2. Load Weights / Data
    // 3. Run Forward Pass
    // 4. Print Results
    
    return 0;
}
```

---
**Happy coding!** 🌟 Let's accelerate AI with the speed of light.
