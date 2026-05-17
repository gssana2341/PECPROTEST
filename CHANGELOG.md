# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-05-17

### Added
- Initial project scaffold based on Photonic Neuromorphic Computing Blueprint v1.0.
- Core math library stubs (`complex.h`, `matrix.h`, `activation.h`, `loss.h`).
- Simulation and energy modeling stubs (`photonic_sim.c`, `noise_model.c`, `energy_model.c`).
- Training algorithm stubs (Multiplexed Gradient Descent, Feedback Alignment, Local Rules).
- PhoLang compiler stubs (Lexer, Parser, AST, Codegen, Runtime).
- Security primitives (`sandbox.h`, `input_validate.h`, `audit_log.c`).
- Clean C API header (`c_api.h`).
- Language binding stubs for Python and Rust.
- Hardware driver interface and IBM Quantum Photonic mock.
- Test harness and example project structures (XOR, MNIST Small, Bell State).
