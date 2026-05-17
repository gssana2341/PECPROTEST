# PHOTONIC — Master Blueprint
> Photonic Neuromorphic Computing · From Scratch · Solo Build  
> Version 1.0 · Start Date: 2026

---

## วิสัยทัศน์ (Vision)

สร้าง **software foundation** สำหรับ photonic neuromorphic computing  
ที่เบา เร็ว ปลอดภัย และ scale ได้ตั้งแต่คอมเครื่องเดียวจนถึง distributed cluster  
โดยไม่พึ่ง Python หรือ PyTorch — ทำจาก scratch ใน C และภาษาใหม่ชื่อ **PhoLang**

### ปัญหาที่แก้
ตอนนี้โลกทำ **inference** บน photonic hardware ได้แล้ว  
แต่ **on-chip training** ยังไม่มีใครแก้ได้แบบ scalable และเบาจริงๆ  
tools ที่มีอยู่ (Neurophox, LuxIA, ONNet) ทั้งหมดพึ่ง Python/PyTorch  
ซึ่ง scale ไม่ได้และหนักเกินไปสำหรับ embedded / edge / space applications

---

## โครงสร้างโปรเจกต์ (Project Structure)

```
photonic/
│
├── core/                        # หัวใจของระบบ — C ล้วน ไม่มี dependency
│   ├── complex.h                # complex number operations
│   ├── matrix.h                 # matrix multiply, transpose, inverse
│   ├── photonic.h               # photonic layer abstraction
│   ├── activation.h             # optical activation functions
│   ├── loss.h                   # loss functions
│   └── memory.h                 # memory management + bounds checking
│
├── sim/                         # simulator engine
│   ├── photonic_sim.c           # main simulator
│   ├── noise_model.c            # จำลอง hardware noise (realistic)
│   └── energy_model.c           # จำลอง energy consumption
│
├── training/                    # on-chip training algorithms
│   ├── mgd.c                    # Multiplexed Gradient Descent
│   ├── feedback_align.c         # Feedback Alignment (ไม่ใช่ backprop)
│   └── local_rules.c            # Hebbian / local learning rules
│
├── lang/                        # PhoLang compiler
│   ├── lexer.c                  # tokenizer
│   ├── parser.c                 # AST builder
│   ├── ast.h                    # AST node definitions
│   ├── codegen.c                # AST → C transpiler
│   └── runtime.h                # PhoLang runtime (inline C)
│
├── security/                    # ความปลอดภัย
│   ├── sandbox.h                # execution sandbox
│   ├── input_validate.h         # input validation + sanitization
│   └── audit_log.c              # audit logging สำหรับ production
│
├── interface/                   # เชื่อมต่อกับโลกภายนอก
│   ├── c_api.h                  # C API สำหรับ library ใช้
│   ├── ffi/                     # Foreign Function Interface
│   │   ├── python_binding.py    # เรียกจาก Python ได้
│   │   └── rust_binding.rs      # เรียกจาก Rust ได้
│   └── hardware/                # เชื่อมต่อ hardware จริง
│       ├── ibm_photonic.c       # IBM Quantum Photonic API
│       └── generic_driver.h     # generic hardware driver interface
│
├── tests/                       # test suite
│   ├── test_runner.c            # test harness หลัก
│   ├── unit/                    # unit tests แต่ละ function
│   ├── integration/             # integration tests
│   └── benchmark/               # performance benchmarks
│
├── examples/                    # ตัวอย่างการใช้งาน
│   ├── xor/                     # XOR problem (เริ่มต้น)
│   ├── mnist_small/             # MNIST subset
│   └── bell_state_sim/          # quantum-photonic hybrid demo
│
├── docs/                        # เอกสาร
│   ├── math/                    # อธิบาย math เบื้องหลัง
│   ├── api/                     # API reference
│   └── research/                # research notes + citations
│
├── Makefile                     # build system
├── SECURITY.md                  # security policy
├── CHANGELOG.md                 # version history
└── README.md                    # getting started
```

---

## Roadmap (แผนงาน)

### PHASE 1 — Foundation `[เดือน 1]`
> เป้าหมาย: เข้าใจ math และสร้าง simulator ที่รันได้จริง

#### 1.1 Math Prerequisites
- [ ] Complex number และ matrix multiplication
- [ ] Interference คืออะไร ทำไมแสงถึงคำนวณ matrix ได้
- [ ] Optical activation function vs ReLU
- [ ] Phase shift และ amplitude ในเชิง computation
- [ ] Unitary matrix คืออะไร และทำไม photonic ต้องเป็น unitary

#### 1.2 Core Library (`core/`)
- [ ] `complex.h` — add, mul, norm, conjugate
- [ ] `matrix.h` — multiply, transpose, unitary check
- [ ] `memory.h` — safe alloc/free + bounds checking
- [ ] `activation.h` — sigmoid_optical, softmax_optical
- [ ] `loss.h` — MSE, cross-entropy

#### 1.3 Simulator (`sim/`)
- [ ] `photonic_sim.c` — photonic layer forward pass
- [ ] ทดสอบ XOR problem ให้ accuracy > 99%
- [ ] `energy_model.c` — นับ operation count (ใช้ compare ทีหลัง)

#### 1.4 Test Harness (`tests/`)
- [ ] `test_runner.c` — run/pass/fail/report
- [ ] unit tests สำหรับทุก function ใน core/

**Checkpoint PHASE 1:**
```
✓ XOR forward pass accuracy > 99%
✓ ผลตรงกับ numpy reference ทุก decimal ที่ 6 ตำแหน่ง
✓ tests ทั้งหมดผ่าน
✓ zero memory leak (valgrind clean)
```

---

### PHASE 2 — Training `[เดือน 2]`
> เป้าหมาย: พิสูจน์ว่า on-chip training ทำได้ใน simulation

#### 2.1 Training Algorithms (`training/`)
- [ ] `feedback_align.c` — Feedback Alignment (ไม่ต้องใช้ backprop)
- [ ] `local_rules.c` — Hebbian learning rules
- [ ] `mgd.c` — Multiplexed Gradient Descent (จากงานวิจัย 2025)

#### 2.2 Noise Model (`sim/`)
- [ ] `noise_model.c` — จำลอง hardware noise realistic
- [ ] ทดสอบว่า training robust ต่อ noise ไหม

#### 2.3 Benchmarks
- [ ] เทียบ convergence speed กับ standard backprop
- [ ] เทียบ energy consumption (operation count)
- [ ] ทดสอบบน MNIST subset (1,000 samples)

**Checkpoint PHASE 2:**
```
✓ training converge ได้บน XOR
✓ MNIST accuracy > 85%
✓ energy model แสดงว่าใช้ operation น้อยกว่า backprop
```

---

### PHASE 3 — Language `[เดือน 3]`
> เป้าหมาย: สร้าง PhoLang ที่ express photonic concept ได้ตรงๆ

#### 3.1 Pain Point Collection (จาก Phase 1-2)
- [ ] จดทุกครั้งที่ C รู้สึกอึดอัดสำหรับ photonic concept
- [ ] รวบรวมเป็น spec document

#### 3.2 PhoLang Design
```
// ตัวอย่าง syntax เป้าหมาย
photon layer dense(input: signal[8], weights: matrix[8][8])

optical forward(x) {
    interfere x through weights
    activate as sigmoid_optical
    emit output
}

train with feedback_align {
    lr: 0.01
    epochs: 100
    noise_robust: true
}
```

#### 3.3 Compiler (`lang/`)
- [ ] `lexer.c` — tokenize PhoLang source
- [ ] `parser.c` — build AST
- [ ] `codegen.c` — AST → C transpiler
- [ ] ทดสอบโดยรัน XOR ผ่าน PhoLang

**Checkpoint PHASE 3:**
```
✓ PhoLang โค้ดสั้นกว่า C เทียบเท่า > 40%
✓ compile และรันได้โดยไม่มี error
✓ output ตรงกับ C version ทุก decimal
```

---

### PHASE 4 — Interface & Scale `[เดือน 4-5]`
> เป้าหมาย: ให้คนอื่นใช้ได้ และ scale ขึ้นได้

#### 4.1 C API (`interface/`)
- [ ] `c_api.h` — clean public API สำหรับ library
- [ ] Python binding — เรียกจาก Python ได้โดยไม่ต้อง PyTorch
- [ ] Rust binding — สำหรับ systems programming

#### 4.2 Hardware Interface
- [ ] `generic_driver.h` — abstract interface สำหรับ hardware จริง
- [ ] ทดสอบกับ IBM Quantum Photonic API (ถ้าเข้าถึงได้)

#### 4.3 Distributed (ถ้าพร้อม)
- [ ] หลาย node รัน photonic sim พร้อมกัน
- [ ] message passing ระหว่าง nodes

**Checkpoint PHASE 4:**
```
✓ Python สามารถ import และเรียกใช้ได้
✓ ทดสอบบน problem ใหญ่ขึ้น (CIFAR-10 subset)
✓ performance benchmark เทียบกับ PyTorch version
```

---

## ความปลอดภัย (Security Design)

### หลักการ
ระบบนี้ถูกออกแบบให้ embed ใน environment ที่ sensitive ได้  
เช่น medical device, satellite, autonomous vehicle  
ดังนั้น security ต้องคิดตั้งแต่ต้น ไม่ใช่ add-on ทีหลัง

### Input Validation (`security/input_validate.h`)
```c
// ทุก input ต้องผ่าน validate ก่อนเสมอ
typedef enum {
    VALIDATE_OK,
    VALIDATE_NULL_PTR,
    VALIDATE_OUT_OF_RANGE,
    VALIDATE_DIMENSION_MISMATCH,
    VALIDATE_NON_UNITARY,        // photonic gate ต้อง unitary
    VALIDATE_ENERGY_OVERFLOW     // ป้องกัน energy blow-up
} ValidationResult;

ValidationResult validate_signal(Signal* s, int expected_dim);
ValidationResult validate_weights(Matrix* w, int rows, int cols);
```

### Memory Safety (`core/memory.h`)
```c
// ห้าม raw malloc/free ใน codebase
// ทุกอย่างต้องผ่าน safe allocator
void* pho_alloc(size_t size, const char* tag);
void  pho_free(void* ptr, const char* tag);
void  pho_check_leaks(void);   // เรียกตอน shutdown
```

### Sandbox (`security/sandbox.h`)
```c
// จำกัด resource ที่ simulation ใช้ได้
typedef struct {
    size_t max_memory_mb;     // จำกัด RAM
    int    max_qubits;        // จำกัดขนาด simulation
    int    max_iterations;    // ป้องกัน infinite loop
    int    allow_file_io;     // ควบคุม file access
} SandboxConfig;
```

### Audit Logging (`security/audit_log.c`)
```c
// log ทุก operation สำคัญสำหรับ production
void audit_log(LogLevel level, const char* op, const char* detail);
// LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
```

---

## การทดสอบ (Test Strategy)

### ระดับที่ 1 — Correctness
```
XOR         → accuracy > 99.0%
Matrix mul  → ตรงกับ numpy ± 1e-6
Activation  → output range [0, 1] เสมอ
Unitary     → W†W = I ± 1e-8
```

### ระดับที่ 2 — Robustness
```
Noise test  → accuracy ไม่ตกเกิน 5% เมื่อมี 10% noise
Memory      → zero leak หลังรัน 10,000 iterations
Edge case   → handle zero input, NaN, Inf ได้ไม่ crash
```

### ระดับที่ 3 — Performance
```
Latency     → forward pass < 1ms สำหรับ 64-dim layer
Memory      → ใช้ RAM ไม่เกิน 2x ของ theoretical minimum
Energy      → operation count น้อยกว่า backprop อย่างน้อย 30%
```

---

## Extensibility (ต่อยอดได้ยังไง)

### เพิ่ม Gate ใหม่
```c
// ใน photonic.h เพิ่ม gate ใหม่ได้โดยไม่กระทบ existing code
typedef struct {
    const char* name;
    void (*apply)(QState* s, int* targets, double* params);
    int   n_targets;
    int   n_params;
} GateDefinition;

// register gate ใหม่
pho_register_gate(&(GateDefinition){
    .name     = "MY_GATE",
    .apply    = my_gate_fn,
    .n_targets = 1,
    .n_params  = 2
});
```

### เพิ่ม Hardware Backend
```c
// implement interface นี้ แล้ว plug เข้าระบบได้เลย
typedef struct {
    const char* name;
    int  (*init)(void* config);
    int  (*run_circuit)(Circuit* c, Result* out);
    void (*shutdown)(void);
} HardwareBackend;
```

### เพิ่มภาษา Binding ใหม่
```
C API → Python  (ctypes / cffi)
C API → Rust    (bindgen)
C API → Go      (cgo)
C API → Julia   (ccall)   ← ดีสำหรับ scientific computing
```

---

## Research References

| Paper | ความสำคัญ | Link |
|---|---|---|
| Multiplexed Gradient Descent (2025) | algorithm หลักที่จะใช้ | arxiv 2506.18041 |
| Photonic Neuromorphic On-Chip (2025) | state of the art review | mdpi chips4030034 |
| GHz Spiking PSNN (2025) | in-situ learning ล่าสุด | arxiv 2506.14272 |
| Neurophox Framework | open source reference | github solgaardlab |
| LuxIA Training Framework | scalable training reference | arxiv 2512.22264 |

---

## Milestones & Success Criteria

```
Month 1  → XOR ผ่าน, simulator รันได้, tests ผ่านทั้งหมด
Month 2  → on-chip training converge, MNIST > 85%
Month 3  → PhoLang compiler รันได้, โค้ดสั้นกว่า C 40%
Month 4  → Python binding ใช้ได้, benchmark เสร็จ
Month 5  → publish paper / release open source
```

---

## หมายเหตุสำคัญ

> ถ้า checkpoint ไหนไม่ผ่าน — **หยุดตรงนั้น แก้ก่อน ไม่ไปต่อ**  
> ถ้า pain point ที่เจอใน Phase 1-2 ไม่ตรงกับ PhoLang spec — **กลับไปออกแบบใหม่**  
> ถ้า training ไม่ converge — **ปัญหาอยู่ที่ math ไม่ใช่ code**

---

*Blueprint v1.0 — พร้อมสำหรับงานใหญ่*
