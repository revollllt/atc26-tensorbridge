"""Model- and shape-specific piecewise GemmConfig profiles indexed by hardware, SM budget, and M.

A profile maps `(device_name, num_sms, n, k, num_experts, layout, group_scales, bridge)` to an
ordered list of `(max_m, GemmConfig)` segments. Because a model's `gate_up` and `down` weights have
fixed `(n, k, num_experts)` and only `m` (routed rows) varies at run time, offline full-space sweeps
(`benchmarks/sweep_moe_profiles.py`) can store optimal configs per model and per SM partition
(full GPU or Green Context compute partition) with zero steady-state selection overhead.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from tensorbridge.jit.codegen import Bridge, GemmConfig, GroupedLayout, pad_to_unit

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"

ProfileKey = tuple[str, int, int, int, int, GroupedLayout, bool, Bridge]


@dataclasses.dataclass(frozen=True)
class ProfileSegment:
    """One M-range segment: serves `m <= max_m` (`max_m == 0` means unbounded upper tail)."""

    max_m: int
    config: GemmConfig


_registry: dict[ProfileKey, tuple[ProfileSegment, ...]] = {}
_loaded_default_sources = False


def _normalize_segments(segments: Iterable[ProfileSegment]) -> tuple[ProfileSegment, ...]:
    bounded = sorted((s for s in segments if s.max_m > 0), key=lambda s: s.max_m)
    unbounded = [s for s in segments if s.max_m == 0]
    if len(unbounded) > 1:
        raise ValueError("a profile entry may have at most one unbounded (max_m=0) tail segment")
    return tuple(bounded + unbounded)


def register_profile(
    device_name: str,
    num_sms: int,
    n: int,
    k: int,
    num_experts: int,
    layout: GroupedLayout,
    group_scales: bool,
    bridge: Bridge,
    segments: Sequence[ProfileSegment],
) -> None:
    """Register (or replace) a piecewise config profile for one fixed shape and SM count."""
    key: ProfileKey = (
        device_name,
        int(num_sms),
        pad_to_unit(int(n)),
        pad_to_unit(int(k)),
        int(num_experts),
        layout,
        bool(group_scales),
        bridge,
    )
    _registry[key] = _normalize_segments(segments)


def clear_profiles(*, reload_defaults: bool = False) -> None:
    """Clear all loaded profiles; optionally keep default disk profiles ready to reload."""
    global _loaded_default_sources
    _registry.clear()
    _loaded_default_sources = not reload_defaults


def load_profile_file(path: Path | str) -> int:
    """Load profile entries from a JSON file and return the number of entries loaded."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["entries"] if isinstance(data, dict) else data
    count = 0
    for entry in entries:
        segments = [
            ProfileSegment(int(seg["max_m"]), GemmConfig.from_list(seg["config"]))
            for seg in entry["segments"]
        ]
        register_profile(
            device_name=str(entry["device_name"]),
            num_sms=int(entry["num_sms"]),
            n=int(entry["n"]),
            k=int(entry["k"]),
            num_experts=int(entry["num_experts"]),
            layout=entry["layout"],
            group_scales=bool(entry["group_scales"]),
            bridge=entry["bridge"],
            segments=segments,
        )
        count += 1
    return count


def save_profile_file(
    path: Path | str,
    entries: Sequence[tuple[ProfileKey, Sequence[ProfileSegment]]],
    metadata: dict[str, object] | None = None,
) -> None:
    """Write profile entries to a JSON file in the canonical schema."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "metadata": metadata or {},
        "entries": [
            {
                "device_name": key[0],
                "num_sms": key[1],
                "n": key[2],
                "k": key[3],
                "num_experts": key[4],
                "layout": key[5],
                "group_scales": key[6],
                "bridge": key[7],
                "segments": [
                    {"max_m": seg.max_m, "config": seg.config.to_list()}
                    for seg in _normalize_segments(segs)
                ],
            }
            for key, segs in entries
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_default_sources() -> None:
    global _loaded_default_sources
    if _loaded_default_sources:
        return
    _loaded_default_sources = True
    if PROFILES_DIR.is_dir():
        for json_file in sorted(PROFILES_DIR.glob("*.json")):
            load_profile_file(json_file)
    env_paths = os.environ.get("TENSORBRIDGE_PROFILE_PATH", "")
    for raw in env_paths.split(os.pathsep):
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            for json_file in sorted(p.glob("*.json")):
                load_profile_file(json_file)
        elif p.is_file():
            load_profile_file(p)


def lookup(
    device_name: str,
    num_sms: int,
    n: int,
    k: int,
    num_experts: int,
    layout: GroupedLayout,
    group_scales: bool,
    bridge: Bridge,
    m: int,
) -> GemmConfig | None:
    """Look up a registered profile config for `m` routed rows, or None to fall back to the rule."""
    _load_default_sources()
    if not _registry:
        return None
    key: ProfileKey = (
        device_name,
        int(num_sms),
        pad_to_unit(int(n)),
        pad_to_unit(int(k)),
        int(num_experts),
        layout,
        bool(group_scales),
        bridge,
    )
    segments = _registry.get(key)
    if not segments:
        return None
    for seg in segments:
        if seg.max_m == 0 or m <= seg.max_m:
            return seg.config
    return None


def compress_segments(
    points: Sequence[tuple[int, GemmConfig]],
    *,
    unbounded_tail: bool = False,
) -> list[ProfileSegment]:
    """Merge consecutive M buckets that select the identical GemmConfig into compact segments.

    `points` must contain `(m, config)` pairs. Each segment covers up to the geometric midpoint
    between adjacent M buckets where the winning config changes, or up to the last measured M
    (or unbounded if `unbounded_tail=True`).
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda item: item[0])
    segments: list[ProfileSegment] = []
    for idx, (m_val, cfg) in enumerate(ordered):
        is_last = idx == len(ordered) - 1
        if not is_last and ordered[idx + 1][1] == cfg:
            continue
        if is_last:
            bound = 0 if unbounded_tail else m_val
        else:
            next_m = ordered[idx + 1][0]
            # Split at the integer geometric mean between adjacent sampled M values.
            bound = max(m_val, int((m_val * next_m) ** 0.5))
        segments.append(ProfileSegment(bound, cfg))
    return segments
