"""RAM used/total must match ``free``'s Mem line, not MemFree subtraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.local_runtime.hardware import (
    _linux_ram_stats,
    _parse_proc_meminfo,
    _ram_from_meminfo,
    _ram_stats,
)

# KiB in one GiB — meminfo fields are KiB.
_GIB_KIB = 1024 * 1024


def test_ram_used_never_exceeds_total():
    total, used, _avail = _ram_from_meminfo({
        "MemTotal": 64 * _GIB_KIB,
        "MemFree": 2 * _GIB_KIB,
        "MemAvailable": 40 * _GIB_KIB,
        "Buffers": 1 * _GIB_KIB,
        "Cached": 30 * _GIB_KIB,
        "SReclaimable": 2 * _GIB_KIB,
    })
    assert used <= total


def test_large_cache_is_not_counted_as_used():
    """MemFree-as-available (getconf _AVPHYS) treats reclaimable cache as used."""
    fields = {
        "MemTotal": 64 * _GIB_KIB,
        "MemFree": 2 * _GIB_KIB,
        "MemAvailable": 40 * _GIB_KIB,
        "Buffers": 1 * _GIB_KIB,
        "Cached": 30 * _GIB_KIB,
        "SReclaimable": 2 * _GIB_KIB,
    }
    total, used, avail = _ram_from_meminfo(fields)
    memfree_used = (fields["MemTotal"] - fields["MemFree"]) * 1024
    assert used <= total
    assert used != memfree_used
    assert used < memfree_used
    # Never display MemAvailable as the used figure.
    assert used != avail
    # procps-ng 4.x: used = MemTotal − MemAvailable.
    assert used == total - avail


def test_without_memavailable_used_is_not_total_minus_available():
    """Older kernels: used is total − free − buffers − cache, not total − MemFree."""
    fields = {
        "MemTotal": 64 * _GIB_KIB,
        "MemFree": 2 * _GIB_KIB,
        "Buffers": 1 * _GIB_KIB,
        "Cached": 30 * _GIB_KIB,
        "SReclaimable": 2 * _GIB_KIB,
    }
    total, used, avail = _ram_from_meminfo(fields)
    classic = (
        fields["MemTotal"] - fields["MemFree"] - fields["Buffers"]
        - fields["Cached"] - fields["SReclaimable"]
    ) * 1024
    assert used == classic
    assert used <= total
    # available falls back to MemFree; cache is large, so used ≠ total − available.
    assert avail == fields["MemFree"] * 1024
    assert used != total - avail


@pytest.mark.linux_only
def test_live_meminfo_used_is_not_total_minus_memfree_when_cache_large():
    text = Path("/proc/meminfo").read_text()
    fields = _parse_proc_meminfo(text)
    parsed = _ram_from_meminfo(fields)
    assert parsed is not None
    total, used, avail = parsed
    assert used <= total
    cache = ((fields.get("Cached") or 0) + (fields.get("SReclaimable") or 0)) * 1024
    if cache < (4 << 30) or not fields.get("MemAvailable"):
        pytest.skip("need a large reclaimable cache to distinguish MemFree from used")
    memfree_used = (fields["MemTotal"] - fields["MemFree"]) * 1024
    assert used != memfree_used
    assert used == total - avail


@pytest.mark.linux_only
def test_ram_stats_agree_with_meminfo_parser():
    stats = _linux_ram_stats()
    assert stats is not None
    total, used, avail = stats
    assert used <= total
    again = _ram_stats()
    assert again[0] == total
    # Two sequential reads can drift; stay inside a small window, not a snapshot.
    assert abs(again[1] - used) < (256 << 20)
    assert abs(again[2] - avail) < (256 << 20)
