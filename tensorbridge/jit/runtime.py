"""Host runtime: builds the C++ launcher once and keeps the compiled kernels."""

import functools
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.utils.cpp_extension
from filelock import FileLock

from tensorbridge.jit import codegen
from tensorbridge.jit.codegen import GemmConfig
from tensorbridge.jit.cuda_paths import filter_cuda_paths

CSRC_DIR = Path(__file__).resolve().parents[1] / "csrc"
TUNED_DIR = Path(__file__).resolve().parents[1] / "tuned"


@functools.cache
def load_launcher() -> None:
    """Build and load `csrc/launcher`, which registers `torch.ops.tensorbridge`."""
    from tensorbridge.jit import compiler

    sources = sorted(CSRC_DIR.glob("**/*.[ch]pp")) + sorted(CSRC_DIR.glob("**/*.h"))
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:16]
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = "torch" + ".".join(torch.__version__.split(".")[:2])
    build_dir = compiler.get_cache_dir() / "launcher" / f"{python_tag}_{torch_tag}_{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda = filter_cuda_paths(required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"])
    with FileLock(str(build_dir) + ".lock"):
        torch.utils.cpp_extension.load(
            name="tensorbridge_launcher",
            sources=[str(CSRC_DIR / "launcher" / "launcher.cpp")],
            extra_include_paths=list(cuda["include_paths"]),
            extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=["-O3"],
            build_directory=str(build_dir),
        )


def tuned_table_path() -> Path:
    return TUNED_DIR / (torch.cuda.get_device_name().replace(" ", "_") + ".json")


@functools.cache
def tuned_table() -> dict[str, GemmConfig]:
    """`MxNxK` -> config measured by `tensorbridge.autotune` on this GPU model."""
    path = tuned_table_path()
    if not path.exists():
        return {}
    return {shape: GemmConfig(**fields) for shape, fields in json.loads(path.read_text()).items()}


def select_config(
    m: int, n: int, k: int, use_stream_k: bool | None = None, tuned: bool = True
) -> GemmConfig:
    """The autotuned config for this exact shape if there is one, else the
    shape-only selection of `csrc/heuristics/sm90.hpp`."""
    if tuned and use_stream_k is None and (config := tuned_table().get(f"{m}x{n}x{k}")):
        return config
    load_launcher()
    num_sms = torch.ops.tensorbridge.get_num_sms(torch.cuda.current_device())
    override = -1 if use_stream_k is None else int(use_stream_k)
    return GemmConfig.from_list(torch.ops.tensorbridge.select_config(m, n, k, num_sms, override))


@functools.cache
def get_kernel(shape_n: int, shape_k: int, config: GemmConfig) -> int:
    """Compile (or fetch from the disk cache) and register one kernel; returns its id."""
    from tensorbridge.jit import compiler

    load_launcher()
    torch.cuda.set_device(torch.cuda.current_device())  # the driver API needs a current context
    source = codegen.generate(shape_n, shape_k, config)
    kernel = compiler.build(source, codegen.KERNEL_NAME)
    return torch.ops.tensorbridge.register_kernel(
        kernel.cubin_path, kernel.lowered_name, shape_n, shape_k, config.to_list(),
        kernel.smem_bytes,
    )
