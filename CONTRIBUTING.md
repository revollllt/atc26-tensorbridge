# Contributing to TensorBridge

## Setup

```bash
pip install --no-build-isolation -e ".[cu13,dev]"     # or cu12; add vllm, experimental, bench as needed
ruff check .
```

Kernels compile on first use (NVRTC) into `TENSORBRIDGE_CACHE_DIR` (default `~/.tensorbridge/cache`).
C++ and CUDA follow `.clang-format`.

## Code style

1. Do not split code into small helpers used once or twice; extract a function only for real reuse or a complete
   concept. A linear 60-line function beats six 10-line ones.
2. No leading underscores except for functions nested in another function.
3. Handle exceptions only at real failure boundaries (I/O, external input, third-party calls, numerical
   degeneracy); an unreachable branch raises.
4. Define a constant where it is used; only shared constants go to a common module.
5. No `Any`: concrete types, `TypedDict`, `Protocol`, generics or unions, with every signature annotated.
6. Names carry their physical or mathematical meaning; note tensor shapes and dtypes where they help.
7. Branches are exhaustive without nesting: validate once at the function's entry and keep the body flat; no
   `else: pass`.
8. No dynamic attribute tricks (`getattr` / `hasattr` / `setattr` probing); express optional behavior with explicit
   fields, a `Protocol` or polymorphism.
9. Comments state what the code does, its correctness constraints and measured facts; put reasoning and
   unverified explanations in the commit message or a design note.
