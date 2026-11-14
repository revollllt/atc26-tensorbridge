"""The CUDA installs the JIT (`compiler.py`) and the launcher (`runtime.py`) compile against: the nvidia-* wheels of
the Python environment, or a system install under /usr/local."""

import functools
import glob
import json
import os
import re
import sys
from typing import Literal, TypedDict


class CudaInstall(TypedDict):
    source: Literal["pypi", "system"]
    path: str
    major: int | None
    minor: int | None
    include_paths: list[str]


def parse_major_minor(text: str) -> tuple[int | None, int | None]:
    """The first `major.minor` (or bare `major`) version in `text`."""
    match = re.search(r"(\d+)\.(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1)), 0
    return None, None


def read_system_version(cuda_home: str) -> tuple[int | None, int | None]:
    """The version of a system install: its `version.json`, else `version.txt`, else a directory name `cuda-X.Y`.
    A file that cannot be read or parsed leaves the decision to the next source."""
    version_json = os.path.join(cuda_home, "version.json")
    if os.path.isfile(version_json):
        try:
            with open(version_json) as handle:
                version = json.load(handle).get("cuda", {}).get("version")
        except (OSError, ValueError):
            version = None
        if version:
            return parse_major_minor(version)
    version_txt = os.path.join(cuda_home, "version.txt")
    if os.path.isfile(version_txt):
        try:
            with open(version_txt) as handle:
                return parse_major_minor(handle.read())
        except OSError:
            pass  # unreadable: the directory name decides
    match = re.match(r"cuda-(\d+(?:\.\d+)?)$", os.path.basename(os.path.realpath(cuda_home)))
    if match:
        return parse_major_minor(match.group(1))
    return None, None


def include_paths(root: str, recurse_components: bool = False) -> list[str]:
    """`root/include` (and its `cccl`), and with `recurse_components` the same of every component directory of
    `root` but the `cuXX` ones, as the nvidia-* wheels lay out CUDA 12."""
    directories = [os.path.join(root, "include")]
    if recurse_components:
        directories += [os.path.join(root, name, "include") for name in sorted(os.listdir(root))
                        if not (name.startswith("cu") and name[2:].isdigit())]
    paths = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        paths.append(directory)
        cccl = os.path.join(directory, "cccl")
        if os.path.isdir(cccl):
            paths.append(cccl)
    return paths


def find_pypi_installs() -> list[CudaInstall]:
    """The nvidia-* wheels on `sys.path`: CUDA 12 as one component directory each, CUDA 13 under `nvidia/cu13`."""
    results: list[CudaInstall] = []
    seen = set()
    for entry in sys.path:
        nvidia_root = os.path.join(entry, "nvidia")
        if not os.path.isdir(nvidia_root) or nvidia_root in seen:
            continue
        seen.add(nvidia_root)
        if any(os.path.isdir(os.path.join(nvidia_root, sub)) for sub in ("cuda_runtime", "cuda_nvcc")):
            results.append(CudaInstall(source="pypi", path=nvidia_root, major=12, minor=None,
                                       include_paths=include_paths(nvidia_root, recurse_components=True)))
        cu13_root = os.path.join(nvidia_root, "cu13")
        if os.path.isdir(cu13_root) and os.path.isdir(os.path.join(cu13_root, "include")):
            results.append(CudaInstall(source="pypi", path=cu13_root, major=13, minor=None,
                                       include_paths=include_paths(cu13_root)))
    return results


def filter_cuda_paths(required_headers: list[str]) -> CudaInstall:
    """The CUDA install of torch's major version that has `required_headers`."""
    import torch

    target_major, _ = parse_major_minor(torch.version.cuda)

    def has_header(install: CudaInstall, header: str) -> bool:
        return any(os.path.exists(os.path.join(path, header)) for path in install["include_paths"])

    results = [install for install in find_all_cuda_paths()
               if install["major"] == target_major and all(has_header(install, h) for h in required_headers)]
    # The nvidia-* wheels come first: they match the NVRTC that cuda-bindings loads and are pinned
    # with the Python environment, so the kernels do not depend on the host's CUDA install.
    results.sort(key=lambda e: (e["source"] != "pypi", e["minor"] is None, -(e["minor"] or 0)))
    if results:
        return results[0]
    if target_major in (12, 13):
        raise RuntimeError(
            f"No suitable CUDA {target_major} environment found. "
            f"Try: pip install tensorbridge[cu{target_major}]"
        )
    raise RuntimeError(f"No suitable CUDA {target_major} environment found.")


@functools.lru_cache(maxsize=1)
def find_all_cuda_paths() -> list[CudaInstall]:
    """The system installs `/usr/local/cuda` and `/usr/local/cuda-*` (each real directory once), then the wheels."""
    results: list[CudaInstall] = []
    seen_real = set()
    for path in ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*")):
        if not os.path.isdir(path):
            continue
        real = os.path.realpath(path)
        if real in seen_real:
            continue
        seen_real.add(real)
        major, minor = read_system_version(path)
        results.append(CudaInstall(source="system", path=path, major=major, minor=minor,
                                   include_paths=include_paths(path)))
    results.extend(find_pypi_installs())
    return results
