# third_party

Vendored sources the kernels compile against, shipped as package data. Files under each directory are copied
unmodified from upstream; update a dependency by replacing its directory and the entry below together (this file
is part of the JIT cache key).

## cutlass

- Upstream: https://github.com/NVIDIA/cutlass
- Commit: 2c85f497547eb441f0fe80bfb0fb2e6f9add1719 (tag v4.8.1). CUTLASS 4.2.1 does not compile under NVRTC
  against the CCCL 3.4 of CUDA 13: `cute/container/array.hpp` redeclares `std::tuple_size` with a different
  template signature.
- Contents: `include/` (CuTe and CUTLASS headers) and `LICENSE.txt` (BSD-3-Clause).
- Use: the JIT (`tensorbridge/jit/compiler.py`) adds `cutlass/include` to the NVRTC include path; the kernels
  under `tensorbridge/include/tensorbridge/` use the SM90 WGMMA, TMA, barrier and register-reconfiguration
  primitives from `cute/arch/` and `cutlass/arch/`.
