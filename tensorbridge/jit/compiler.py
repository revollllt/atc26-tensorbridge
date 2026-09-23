"""NVRTC compilation with an on-disk cubin cache."""

import dataclasses
import glob
import hashlib
import json
import os
import struct
from pathlib import Path

from cuda.bindings import nvrtc
from elftools.elf.elffile import ELFFile
from filelock import FileLock

from tensorbridge.jit.cuda_paths import filter_cuda_paths

INCLUDE_DIR = Path(__file__).resolve().parents[1] / "include"

# NVRTC has no host standard library; map the few headers the kernel includes
# onto libcu++ and declare the one driver type it receives.
_STD_TYPE_TRAITS = """
    #include <cuda/std/type_traits>
    namespace std {
    using cuda::std::conditional_t;
    }
"""
_HEADER_SHIMS = {
    "cstdint": "#include <cuda/std/cstdint>\nusing namespace cuda::std;\n" + _STD_TYPE_TRAITS,
    "type_traits": _STD_TYPE_TRAITS,
    "cuda.h": """
        #pragma once
        #include <cuda/std/cstdint>
        using namespace cuda::std;
        typedef uint64_t cuuint64_t;
        #define CU_TENSOR_MAP_NUM_QWORDS 16
        typedef struct CUtensorMap_st {
            alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
        } CUtensorMap;
    """,
}


@dataclasses.dataclass(frozen=True)
class CompiledKernel:
    cubin_path: str
    lowered_name: str  # mangled name of the kernel instantiation
    smem_bytes: int


def get_cache_dir() -> Path:
    default = Path.home() / ".tensorbridge" / "cache"
    return Path(os.environ.get("TENSORBRIDGE_CACHE_DIR", default))


def get_flags() -> list[str]:
    cuda = filter_cuda_paths(required_headers=["cuda_runtime.h"])
    return [
        "--gpu-architecture=sm_90a",
        "-std=c++17",
        "--use_fast_math",
        "--dopt=on",
        "-extra-device-vectorization",
        "--ptxas-options=-O3",
        "--ptxas-options=--register-usage-level=10",
        "--ptxas-options=-v",  # register and spill statistics, in compile.log
        "--generate-line-info",  # source <-> PC mapping for ncu
        "--diag-suppress=39,161,174,177,940",
        "-default-device",
        f"-I{INCLUDE_DIR}",
        *(f"-I{path}" for path in cuda["include_paths"]),
    ]


def _cache_key(source: str, kernel_name: str, flags: list[str]) -> str:
    # The headers are part of the program: editing one must miss the cache.
    hasher = hashlib.sha256()
    _, major, minor = nvrtc.nvrtcVersion()
    hasher.update(json.dumps([major, minor, flags, kernel_name, source]).encode())
    for header in sorted(glob.glob(f"{INCLUDE_DIR}/**/*.cuh", recursive=True)):
        hasher.update(Path(header).read_bytes())
    return hasher.hexdigest()[:16]


def _read_uint32_symbol(cubin_path: Path, name: str) -> int:
    with cubin_path.open("rb") as file:
        elf = ELFFile(file)
        symbol = elf.get_section_by_name(".symtab").get_symbol_by_name(name)[0]
        data = elf.get_section(symbol["st_shndx"]).data()
        return struct.unpack_from("<I", data, symbol["st_value"])[0]


def _nvrtc_check(result: nvrtc.nvrtcResult, what: str) -> None:
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"{what} failed: {result}")


def _compile(source: str, kernel_name: str, flags: list[str], out_dir: Path) -> None:
    names = [name.encode() for name in _HEADER_SHIMS]
    contents = [content.encode() for content in _HEADER_SHIMS.values()]
    result, program = nvrtc.nvrtcCreateProgram(
        source.encode(), b"kernel.cu", len(names), contents, names
    )
    _nvrtc_check(result, "nvrtcCreateProgram")
    try:
        # Registering the instantiation forces it and lets us ask for its mangled name.
        result = nvrtc.nvrtcAddNameExpression(program, kernel_name.encode())[0]
        _nvrtc_check(result, "nvrtcAddNameExpression")
        options = [flag.encode() for flag in flags]
        result = nvrtc.nvrtcCompileProgram(program, len(options), options)[0]

        _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        log = b"\0" * log_size
        nvrtc.nvrtcGetProgramLog(program, log)
        log_text = log.decode(errors="replace").rstrip("\0")
        (out_dir / "compile.log").write_text(log_text)
        if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC failed, see {out_dir / 'compile.log'}:\n{log_text}")

        result, lowered_name = nvrtc.nvrtcGetLoweredName(program, kernel_name.encode())
        _nvrtc_check(result, "nvrtcGetLoweredName")
        _, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        cubin = b"\0" * cubin_size
        _nvrtc_check(nvrtc.nvrtcGetCUBIN(program, cubin)[0], "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)

    cubin_path = out_dir / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    meta = {
        "lowered_name": lowered_name.decode(),
        "smem_bytes": _read_uint32_symbol(cubin_path, "SMEM_BYTES"),
    }
    # Written last: its presence marks the directory as complete.
    (out_dir / "meta.json").write_text(json.dumps(meta))


def build(source: str, kernel_name: str) -> CompiledKernel:
    flags = get_flags()
    out_dir = get_cache_dir() / _cache_key(source, kernel_name, flags)
    out_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(out_dir / "lock"):
        if not (out_dir / "meta.json").exists():
            (out_dir / "kernel.cu").write_text(source)
            _compile(source, kernel_name, flags, out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    return CompiledKernel(str(out_dir / "kernel.cubin"), meta["lowered_name"], meta["smem_bytes"])
