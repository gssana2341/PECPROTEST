# Security Policy

## Supported Versions

Only the latest version of the Photonic Neuromorphic Computing library is currently supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

If you believe you have found a security vulnerability, please report it via email to the project maintainers. We will try to respond to the issue within 48 hours.

### Sandboxing & Resource Limits
This project provides a robust sandboxing mechanism (`security/sandbox.h`) designed for embedded, edge, and safety-critical environments. If you find a way to bypass these limits (e.g. memory allocation constraints, infinite loops escaping the iteration bounds), please classify this as a high-priority vulnerability.

### Input Validation
All API endpoints and mathematical operations strictly validate inputs (`security/input_validate.h`). Bypass of unitarity checks for photonic gates or missing bounds checking on matrix operations are considered security flaws.
