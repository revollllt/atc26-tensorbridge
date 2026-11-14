"""NVRTC compilation with an on-disk cubin cache."""

import dataclasses
import glob
import hashlib
import json
import os
import re
import struct
from pathlib import Path

from cuda.bindings import nvrtc
from elftools.elf.elffile import ELFFile
from filelock import FileLock

from tensorbridge.jit.cuda_paths import filter_cuda_paths

INCLUDE_DIR = Path(__file__).resolve().parents[1] / "include"
# CuTe and CUTLASS headers, vendored at the commit `tensorbridge/third_party/README.md` records; package data of
# the wheel.
THIRD_PARTY_DIR = Path(__file__).resolve().parents[1] / "third_party"
CUTLASS_INCLUDE_DIR = THIRD_PARTY_DIR / "cutlass" / "include"
# The kernels' argument struct, which the launcher compiles too.
PARAMS_HEADER = INCLUDE_DIR / "tensorbridge" / "common" / "params.h"

# NVRTC has no host standard library; map the few headers the kernel includes
# onto libcu++ and declare the one driver type it receives. `type_traits` gives
# `std::conditional_t`.
STD_TYPE_TRAITS = """
    #include <cuda/std/type_traits>
    namespace std {
    using cuda::std::conditional_t;
    }
"""
HEADER_SHIMS = {
    "cstdint": "#include <cuda/std/cstdint>\nusing namespace cuda::std;\n" + STD_TYPE_TRAITS,
    "type_traits": STD_TYPE_TRAITS,
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
class PtxasStats:
    """What ptxas reports about the kernel in `compile.log` (`--ptxas-options=-v`). NVRTC keeps its own
    cache of compiled programs in the driver's compute cache, across processes and cache directories;
    on a hit it skips ptxas and the log is empty, so the statistics are not always there (`CUDA_CACHE_DISABLE=1`
    turns that cache off)."""

    registers: int  # per thread, as launched
    spill_bytes: int  # spill stores plus spill loads
    stack_bytes: int  # stack frame, e.g. local arrays that did not stay in registers
    # ptxas serialized the WGMMAs: C7511 and C7512 for lack of registers, C7513 for non-WGMMA
    # instructions that define a WGMMA's input registers within a pipeline stage
    wgmma_serialized: bool

    @classmethod
    def parse(cls, log: str) -> "PtxasStats | None":
        """The statistics in `log`, or None if ptxas did not run (see the class)."""
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
        registers = re.search(r"Used (\d+) registers", log)
        if frame is None or registers is None:
            return None
        return cls(
            registers=int(registers[1]), spill_bytes=int(frame[2]) + int(frame[3]), stack_bytes=int(frame[1]),
            wgmma_serialized="wgmma.mma_async instructions are serialized" in log,
        )


@dataclasses.dataclass(frozen=True)
class CompiledKernel:
    cubin_path: str
    lowered_name: str  # mangled name of the kernel instantiation
    smem_bytes: int
    ptxas: PtxasStats | None  # None where NVRTC's own cache spared the compile


def get_cache_dir() -> Path:
    default = Path.home() / ".tensorbridge" / "cache"
    return Path(os.environ.get("TENSORBRIDGE_CACHE_DIR", default))


def get_flags(include_root: Path = INCLUDE_DIR) -> list[str]:
    """NVRTC flags; `include_root` holds the `tensorbridge/` kernel headers (another worktree's
    for an out-of-tree A/B variant)."""
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
        f"-I{include_root}",
        f"-I{CUTLASS_INCLUDE_DIR}",
        *(f"-I{path}" for path in cuda["include_paths"]),
    ]


def cache_key(source: str, kernel_name: str, flags: list[str], include_root: Path) -> str:
    # The headers are part of the program: editing one must miss the cache. The vendored CUTLASS
    # changes only with its commit, which the README records.
    hasher = hashlib.sha256()
    _, major, minor = nvrtc.nvrtcVersion()
    hasher.update(json.dumps([major, minor, flags, kernel_name, source]).encode())
    hasher.update((THIRD_PARTY_DIR / "README.md").read_bytes())
    for header in sorted(glob.glob(f"{include_root}/**/*.cuh", recursive=True) +
                         glob.glob(f"{include_root}/**/*.h", recursive=True)):
        hasher.update(Path(header).read_bytes())
    return hasher.hexdigest()[:16]


def read_uint32_symbol(cubin_path: Path, name: str) -> int:
    with cubin_path.open("rb") as file:
        elf = ELFFile(file)
        symbol = elf.get_section_by_name(".symtab").get_symbol_by_name(name)[0]
        data = elf.get_section(symbol["st_shndx"]).data()
        return struct.unpack_from("<I", data, symbol["st_value"])[0]


def check_nvrtc(result: nvrtc.nvrtcResult, what: str) -> None:
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"{what} failed: {result}")


def compile_program(source: str, kernel_name: str, flags: list[str], out_dir: Path) -> None:
    names = [name.encode() for name in HEADER_SHIMS]
    contents = [content.encode() for content in HEADER_SHIMS.values()]
    result, program = nvrtc.nvrtcCreateProgram(
        source.encode(), b"kernel.cu", len(names), contents, names
    )
    check_nvrtc(result, "nvrtcCreateProgram")
    try:
        # Registering the instantiation forces it and lets us ask for its mangled name.
        result = nvrtc.nvrtcAddNameExpression(program, kernel_name.encode())[0]
        check_nvrtc(result, "nvrtcAddNameExpression")
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
        check_nvrtc(result, "nvrtcGetLoweredName")
        _, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        cubin = b"\0" * cubin_size
        check_nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin)[0], "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)

    cubin_path = out_dir / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    meta = {
        "lowered_name": lowered_name.decode(),
        "smem_bytes": read_uint32_symbol(cubin_path, "SMEM_BYTES"),
    }
    # Written last: its presence marks the directory as complete.
    (out_dir / "meta.json").write_text(json.dumps(meta))


def build(source: str, kernel_name: str, include_root: Path = INCLUDE_DIR) -> CompiledKernel:
    flags = get_flags(include_root)
    out_dir = get_cache_dir() / cache_key(source, kernel_name, flags, include_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(out_dir / "lock"):
        if not (out_dir / "meta.json").exists():
            (out_dir / "kernel.cu").write_text(source)
            compile_program(source, kernel_name, flags, out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    stats = PtxasStats.parse((out_dir / "compile.log").read_text())
    return CompiledKernel(str(out_dir / "kernel.cubin"), meta["lowered_name"], meta["smem_bytes"], stats)
