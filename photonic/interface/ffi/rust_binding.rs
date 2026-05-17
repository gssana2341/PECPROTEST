// rust_binding.rs — Rust binding for Photonic library
//
// เรียกใช้ photonic core ผ่าน FFI (bindgen)

#![allow(non_camel_case_types)]

// ─── C Types ────────────────────────────────────────────────────────

#[repr(C)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

#[repr(C)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: *mut f64,
}

// ─── FFI Bindings ───────────────────────────────────────────────────

extern "C" {
    pub fn pho_init() -> i32;
    pub fn pho_shutdown();
    pub fn pho_version() -> *const std::os::raw::c_char;

    // Complex number operations
    pub fn complex_new(real: f64, imag: f64) -> Complex;
    pub fn complex_add(a: Complex, b: Complex) -> Complex;
    pub fn complex_mul(a: Complex, b: Complex) -> Complex;
    pub fn complex_conj(a: Complex) -> Complex;
    pub fn complex_norm(a: Complex) -> f64;
}

// ─── Safe Rust Wrapper ──────────────────────────────────────────────

pub struct Photonic;

impl Photonic {
    pub fn init() -> Result<Self, &'static str> {
        let result = unsafe { pho_init() };
        if result == 0 {
            Ok(Photonic)
        } else {
            Err("Failed to initialize Photonic library")
        }
    }
}

impl Drop for Photonic {
    fn drop(&mut self) {
        unsafe { pho_shutdown(); }
    }
}

// TODO: Add higher-level Rust API wrappers
