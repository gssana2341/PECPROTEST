# PhoLang Quickstart Guide 🚀

Welcome! In this 5-minute guide, you will write your first **Photonic Neural Network** using our custom domain-specific language: **PhoLang** (`.pho`).

## 1. The Structure of a `.pho` File

PhoLang provides a very simple, declarative syntax to define your neural network architecture. Let's create a basic 3-layer network.

Create a new file called `my_network.pho`:

```phoc
photon network MyFirstNetwork {
    // Downsample 28x28 image to 8x8 optical field
    lens pool(28, 8)
    
    // Photonic unitary layer with Kerr nonlinearity
    layer unitary(64) kerr(0.5) gain(15.0)
    layer unitary(64) kerr(0.5) gain(15.0)
    
    // 10-class output
    readout softmax(10)
    
    // Riemannian optimizer
    train riemannian(lr=0.01, epochs=100, batch_size=32)
}
```

## 2. Compiling the Network

Our compiler (`phoc`) translates your `.pho` file directly into optimized C code.

Run the compiler:
```bash
./phoc my_network.pho generated_network.c
```

This will generate a C file (`generated_network.c`) containing the network structures, memory allocation, and the forward pass functions tailored exactly to your architecture!

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
