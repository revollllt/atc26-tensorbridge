"""Host runtime: builds the C++ launcher once, selects configs and keeps the compiled kernels."""

import dataclasses
import functools
import hashlib
import sys
import warnings
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.utils.cpp_extension
from filelock import FileLock

from tensorbridge.jit import codegen
from tensorbridge.jit.codegen import Bridge, GemmConfig
from tensorbridge.jit.cuda_paths import filter_cuda_paths

CSRC_DIR = Path(__file__).resolve().parents[1] / "csrc"


class Launcher(Protocol):
    def get_num_sms(self, device: int) -> int: ...


@functools.cache
def load_launcher() -> Launcher:
    """Build and load `csrc/launcher`, which registers `torch.ops.tensorbridge`."""
    from tensorbridge.jit import compiler

    # The launcher also compiles the kernels' argument struct (`include/tensorbridge/common/params.h`).
    sources = sorted(CSRC_DIR.glob("**/*.[ch]pp")) + sorted(CSRC_DIR.glob("**/*.h")) + [compiler.PARAMS_HEADER]
    # The checkout's path is part of the key: `torch.utils.cpp_extension` writes absolute source and include
    # paths into build.ninja, so checkouts sharing a build directory would rebuild it in turn.
    digest = hashlib.sha256(str(CSRC_DIR).encode() + b"".join(path.read_bytes() for path in sources)).hexdigest()[:16]
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = "torch" + ".".join(torch.__version__.split(".")[:2])
    build_dir = compiler.get_cache_dir() / "launcher" / f"{python_tag}_{torch_tag}_{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda = filter_cuda_paths(required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"])
    with FileLock(str(build_dir) + ".lock"):
        launcher = cast(Launcher, torch.utils.cpp_extension.load(
            name="tensorbridge_launcher",
            sources=[str(CSRC_DIR / "launcher" / "launcher.cpp")],
            extra_include_paths=list(cuda["include_paths"]) + [str(compiler.INCLUDE_DIR)],
            # No -lcuda: the launcher takes its driver functions from the CUDA runtime (`launcher/driver.h`).
            extra_ldflags=["-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=["-O3"],
            build_directory=str(build_dir),
        ))
        return launcher


def get_num_sms(device: int) -> int:
    """Available SMs of the current stream; Green Contexts on one device may differ."""
    return load_launcher().get_num_sms(device)


# The config of each shape by (device, available SMs, M, N, K, experts, layout, bridge, input top-k, Stream-K
# override, group scales), N and K the weight's: the selection rule's choice, computed on first use, or the one
# autotune measured (`override_config`). The two bridges compile different kernels and an indexed layout's gather
# depends on top-k, so measured configs are kept apart by both.
SelectionKey = tuple[int, int, int, int, int, int, codegen.GroupedLayout, Bridge, int, bool | None, bool]
selected_configs: dict[SelectionKey, GemmConfig] = {}


def select_config(
    m: int, n: int, k: int, num_experts: int = 0, layout: codegen.GroupedLayout = "dense", device: int | None = None,
    *, bridge: Bridge = "exact", top_k: int = 1, use_stream_k: bool | None = None, group_scales: bool = False,
) -> GemmConfig:
    """The config for this shape on `device` (default: the current device): the one autotune measured for it, else
    the choice of the selection rule of `csrc/heuristics/sm90_rule.hpp`, computed once per shape and SM count. `n` and
    `k` are the weight's; the rule sees them rounded up to multiples of 128, the shape the kernel runs on
    (`codegen.pad_to_unit`). `m` is the routed row count of a grouped layout. `use_stream_k` forces the Stream-K tail
    of a dense GEMM off or on, and skips autotune's choice; grouped GEMM never splits K. `group_scales` selects for
    the `[1, 128]` activation-scale kernel (`impls/sm90_w4a8_pgroup.cuh`)."""
    device = torch.cuda.current_device() if device is None else device
    # The hot path of every call: one native SM-count query and one lookup (`test_repeated_selection_is_a_lookup`).
    key = (device, load_launcher().get_num_sms(device), m, n, k, num_experts, layout, bridge, top_k, use_stream_k,
           group_scales)
    if (config := selected_configs.get(key)) is not None:
        return config
    num_sms = key[1]
    stream_k_override = 0 if layout != "dense" else -1 if use_stream_k is None else int(use_stream_k)
    with torch.cuda.device(device):  # the op reads the current device's properties
        try:
            selected = torch.ops.tensorbridge.select_config(
                max(m, 1), codegen.pad_to_unit(n), codegen.pad_to_unit(k), num_sms, stream_k_override, num_experts,
                codegen.LAYOUT_IDS[layout], group_scales)
        except RuntimeError as error:
            raise ValueError(f"no config runs (M, N, K) = ({m}, {n}, {k}) with layout {layout}") from error
    config = selected_configs[key] = GemmConfig.from_list(selected)
    return config


def override_config(
    m: int, n: int, k: int, config: GemmConfig, device: int,
    num_experts: int = 0, layout: codegen.GroupedLayout = "dense", bridge: Bridge = "exact", top_k: int = 1,
    group_scales: bool = False,
) -> None:
    """Serve this shape with `config` from now on (autotune), at the stream's current SM count."""
    key = (device, get_num_sms(device), m, n, k, num_experts, layout, bridge, top_k, None, group_scales)
    selected_configs[key] = config


def autotune_candidates(
    m: int, n: int, k: int, count: int, device: int | None = None,
    num_experts: int = 0, layout: codegen.GroupedLayout = "dense", group_scales: bool = False,
) -> list[GemmConfig]:
    """At most `count` configs autotune times for this shape on `device`: the rule's config, which
    `select_config` chooses, then its neighbours that selection may take (`autotune_candidates` of
    `csrc/heuristics/sm90_rule.hpp`), for the weight's `(n, k)` rounded up as `select_config` rounds it."""
    device = torch.cuda.current_device() if device is None else device
    with torch.cuda.device(device):  # the op reads the current device's properties
        num_sms = get_num_sms(device)
        flat = torch.ops.tensorbridge.autotune_candidates(max(m, 1), codegen.pad_to_unit(n), codegen.pad_to_unit(k),
                                                          num_sms, count, num_experts, codegen.LAYOUT_IDS[layout],
                                                          group_scales)
    width = len(dataclasses.fields(GemmConfig))
    return [GemmConfig.from_list(flat[i:i + width]) for i in range(0, len(flat), width)]


@functools.cache
def ctas_fit(ctas_per_sm: int, smem_bytes: int, optin_bytes: int) -> bool:
    """Whether `ctas_per_sm` CTAs with `smem_bytes` of shared memory each are resident on one SM at
    once. They must be, since a Stream-K slice may wait for a CTA that has not started; each one
    reserves 1 KB of shared memory besides its own."""
    return ctas_per_sm * (smem_bytes + 1024) <= optin_bytes + 1024


def get_kernel(
    n: int, k: int, config: GemmConfig, include_root: Path | None = None,
    bridge: codegen.Bridge = "exact", num_experts: int = 0, layout: codegen.GroupedLayout = "dense",
    source: codegen.KernelSource | None = None, device: int | None = None,
) -> int:
    """The id of one kernel on `device` (default: the current device), compiled (or fetched from
    the disk cache) and registered on the first request; later requests return the same id. `(n, k)`
    is the weight's shape, which `codegen.generate` pads.

    `include_root` is the directory of the kernel headers, by default this package's; another
    worktree's `tensorbridge/include` compiles that version's kernel, which must take the same
    arguments as this launcher; `source` is then that version's kernel for the config (its own
    codegen's), by default this tree's. `bridge` is the weights' mode (`NVFP4Weight.bridge`): the
    same source, another template instantiation.

    `fp8_nvfp4_gemm` asks for its kernel on every call; the cache keeps that a lookup, and keeps every call of a
    CUDA graph on one module.
    """
    from tensorbridge.jit import compiler

    device = torch.cuda.current_device() if device is None else device
    kernel = source or codegen.generate(n, k, config, num_experts, layout, bridge)
    return register_device_kernel(device, kernel, include_root or compiler.INCLUDE_DIR, n, k, config,
                                  num_experts, layout)


@functools.cache
def register_device_kernel(
    device: int, kernel: codegen.KernelSource, include_root: Path, n: int, k: int, config: GemmConfig,
    num_experts: int = 0, layout: codegen.GroupedLayout = "dense", activation_scale_groups: int = 1,
    scale_m_major: bool = False, block_scales: bool = False,
) -> int:
    """Compile `kernel` against the headers under `include_root` and register it on `device` for the weight shape
    `(n, k)`; returns its id. `activation_scale_groups` and `scale_m_major` describe the activation scales the kernel
    reads, `block_scales` the layout of its weight scales (`launcher.cpp`, `Kernel`)."""
    from tensorbridge.jit import compiler

    load_launcher()
    with torch.cuda.device(device):  # the driver API needs a current context
        compiled = compiler.build(kernel.source, kernel.name, include_root)
        if compiled.ptxas is not None and compiled.ptxas.spill_bytes:
            # The selector only proposes configs that compiled without spilling (the census); this one did not.
            warnings.warn(f"{config} spills {compiled.ptxas.spill_bytes} bytes for (N, K) = ({n}, {k})",
                          RuntimeWarning, stacklevel=3)
        if config.ctas_per_sm > 1:
            optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
            if not ctas_fit(config.ctas_per_sm, compiled.smem_bytes, optin):
                raise ValueError(f"{config} needs {compiled.smem_bytes} B of shared memory per CTA, too much for "
                                 f"{config.ctas_per_sm} CTAs per SM")
        return torch.ops.tensorbridge.register_kernel(
            compiled.cubin_path, compiled.lowered_name, n, k, config.to_list(),
            compiled.smem_bytes, num_experts, codegen.LAYOUT_IDS[layout], activation_scale_groups, scale_m_major,
            block_scales,
        )


def get_grouped_kernel(
    n: int, k: int, num_experts: int, layout: codegen.GroupedLayout, config: GemmConfig,
    include_root: Path | None = None, bridge: codegen.Bridge = "exact", device: int | None = None,
    source: codegen.KernelSource | None = None, *, group_scales: bool = False,
) -> int:
    """Compile a grouped specialization through the shared JIT and registration path."""
    if layout == "dense" or num_experts < 1:
        raise ValueError("a grouped kernel needs a grouped layout and at least one expert")
    if config.use_stream_k:
        raise ValueError("grouped GEMM does not support Stream-K")
    if group_scales:
        from tensorbridge.jit import compiler

        device = torch.cuda.current_device() if device is None else device
        kernel = source or codegen.generate_pgroup(n, k, config, False, bridge, num_experts, layout)
        return register_device_kernel(device, kernel, include_root or compiler.INCLUDE_DIR, n, k, config,
                                      num_experts, layout, k // 128, block_scales=True)
    else:
        return get_kernel(n, k, config, include_root, bridge, num_experts, layout, source, device)
