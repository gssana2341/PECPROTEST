# Installation Guide

This guide will walk you through setting up the Silicon Photonics Neuromorphic Computing Simulation on Linux, macOS, and Windows (via WSL).

## Prerequisites

The project requires the following basic tools:
- **GCC (GNU Compiler Collection)**: For compiling the core engine and PhoLang compiler.
- **Make**: For building the project.
- **OpenMP**: For multi-threading support to speed up matrix operations.
- **Python 3.x** (Optional): If you plan to use Python bindings.

---

## 🐧 Linux (Ubuntu / Debian)

1. **Install Build Tools & OpenMP:**
   ```bash
   sudo apt update
   sudo apt install build-essential gcc make libomp-dev
   ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/gssana2341/PECPROTEST.git
   cd PECPROTEST
   ```

3. **Build the Project:**
   ```bash
   make       # Builds libphotonic.a and libphotonic.so
   make pho   # Builds the PhoLang Compiler
   ```

4. **Run the Test Suite:**
   ```bash
   make test
   ```

5. **(Optional) Install Compiler System-Wide:**
   ```bash
   sudo make install
   ```

---

## 🍏 macOS

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install GCC and OpenMP:**
   macOS uses Clang by default which might not have full OpenMP support enabled out of the box. Installing GCC is recommended.
   ```bash
   brew install gcc make libomp
   ```
   *Note: You may need to edit the `Makefile` to change `CC = gcc` to `CC = gcc-12` (or your installed GCC version) on macOS.*

3. **Clone & Build:**
   ```bash
   git clone https://github.com/gssana2341/PECPROTEST.git
   cd PECPROTEST
   make
   make pho
   ```

---

## 🪟 Windows (WSL 2)

We highly recommend using Windows Subsystem for Linux (WSL 2) for development.

1. **Install WSL 2 (Ubuntu):**
   Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
   Restart your PC if prompted, then open the "Ubuntu" app from the Start menu.

2. **Follow the Linux Instructions:**
   Once inside the WSL Ubuntu terminal, run the same commands as the Linux section:
   ```bash
   sudo apt update
   sudo apt install build-essential gcc make libomp-dev
   git clone https://github.com/gssana2341/PECPROTEST.git
   cd PECPROTEST
   make
   make pho
   make demo
   ```

---

## Next Steps
Now that everything is compiled, jump over to the [QUICKSTART.md](QUICKSTART.md) to learn how to write your first Photonic Neural Network using **PhoLang**!
