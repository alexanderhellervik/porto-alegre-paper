"""Shared plumbing for the reproduction pipeline (Python side).

Every stage script starts with `ctx = init(stage=N, name="...")`, which loads
config.yaml, fixes the seed and opens a run log; `stage` only labels the log.
Nothing in this module makes an analytical decision -- those all live in
config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config.yaml"
INPUT_MANIFEST = "input_manifest.json"
SUMS_FILE = REPO / "data" / "SHA256SUMS"


@dataclass
class Context:
    cfg: dict[str, Any]
    stage: int
    name: str
    log: logging.Logger
    started: datetime
    counts: list[dict[str, Any]] = field(default_factory=list)

    # -- path helpers ---------------------------------------------------------
    def path(self, *keys: str) -> Path:
        node: Any = self.cfg["paths"]
        for k in keys:
            node = node[k]
        return REPO / node

    def interim(self, filename: str) -> Path:
        p = self.path("interim") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def report(self, filename: str) -> Path:
        p = self.path("reports") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def table(self, filename: str) -> Path:
        p = self.path("outputs") / "tables" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- record counts in / out at every filter --------------------------------
    def count(self, step: str, n: int, note: str = "") -> int:
        prev = self.counts[-1]["n"] if self.counts else None
        delta = None if prev is None else n - prev
        self.counts.append({"step": step, "n": n, "delta": delta, "note": note})
        arrow = "" if delta is None else f"  ({delta:+d})"
        self.log.info("count  %-34s %8d%s %s", step, n, arrow, note)
        return n

    def write_counts(self, filename: str) -> Path:
        out = self.table(filename)
        import csv

        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", "n", "delta", "note"])
            w.writeheader()
            w.writerows(self.counts)
        self.log.info("wrote %s", out.relative_to(REPO))
        return out

    def finish(self) -> None:
        secs = (datetime.now(timezone.utc) - self.started).total_seconds()
        self.log.info("stage %d (%s) finished in %.1fs", self.stage, self.name, secs)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _logger(stage: int, name: str, cfg: dict) -> logging.Logger:
    log = logging.getLogger(f"stage{stage}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    logdir = REPO / cfg["paths"]["logs"]
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"stage{stage:02d}_{name}.log", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def init(stage: int, name: str) -> Context:
    """Load config, fix the seed and open the run log for one pipeline step."""
    cfg = load_config()
    log = _logger(stage, name, cfg)

    seed = int(cfg["repro"]["seed"])
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    log.info("stage %d — %s", stage, name)
    log.info("seed=%d  crs=EPSG:%s", seed, cfg["repro"]["crs_epsg"])
    return Context(
        cfg=cfg, stage=stage, name=name, log=log, started=datetime.now(timezone.utc)
    )


# -- container limits ---------------------------------------------------------
#
# Inside a container, `free` and `nproc` report the host's resources rather
# than the container's. The cgroup v2 files report the container's own limits,
# so every stage that records memory use reads them through here.

CGROUP_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_PEAK = Path("/sys/fs/cgroup/memory.peak")
CGROUP_MAX = Path("/sys/fs/cgroup/memory.max")


def read_cgroup_int(path: Path) -> int | None:
    """Read a cgroup v2 scalar; None when the file is absent or says `max`."""
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError):  # not a cgroup v2 host
        return None


class MemoryWatch:
    """Sample the cgroup's `memory.current` in the background; report the peak.

    `memory.peak` is a high-water mark that cannot be reset from inside a
    read-only cgroup mount, so a per-run peak has to be sampled rather than read
    off. Used as a context manager around anything large enough to be worth a
    number in a manifest.
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            v = read_cgroup_int(CGROUP_CURRENT)
            if v is not None and v > self.peak_bytes:
                self.peak_bytes = v

    def __enter__(self) -> Self:
        self.peak_bytes = read_cgroup_int(CGROUP_CURRENT) or 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / 2**30

    def mark(self) -> float:
        """Sample now, fold into the peak, and return the current GiB."""
        v = read_cgroup_int(CGROUP_CURRENT) or 0
        self.peak_bytes = max(self.peak_bytes, v)
        return v / 2**30


# -- integrity ----------------------------------------------------------------


def checksum(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def checksum_dir(path: Path, algo: str = "sha256") -> tuple[str, int, int]:
    """Order-stable digest over a directory tree.

    Hashes each file's repo-relative path and content, so a rename or an added
    file changes the digest. Returns (digest, n_files, total_bytes).
    """
    h = hashlib.new(algo)
    files = sorted(f for f in path.rglob("*") if f.is_file())
    total = 0
    for f in files:
        h.update(str(f.relative_to(path)).encode())
        h.update(checksum(f, algo).encode())
        total += f.stat().st_size
    return h.hexdigest(), len(files), total


def repo_relative(p: Path) -> str:
    """`p` relative to the repository root when it lies inside it, else as given."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def input_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Every declared input (`paths.raw.*`, `paths.frozen.*`), keyed `group.key`."""
    return {
        f"{group}.{key}": REPO / rel
        for group in ("raw", "frozen")
        for key, rel in cfg["paths"][group].items()
    }


def manifest(paths: dict[str, Path], algo: str = "sha256") -> dict[str, dict]:
    """Checksum and size for a set of inputs, with repo-relative paths.

    Directory inputs (e.g. a shapefile bundle) are digested recursively.
    """
    out = {}
    for key, p in paths.items():
        p = Path(p)
        if not p.exists():
            out[key] = {"status": "MISSING", "path": repo_relative(p)}
            continue
        real = p.resolve()
        rec = {"status": "ok", "path": repo_relative(real)}
        if real.is_dir():
            digest, nfiles, total = checksum_dir(real, algo)
            rec.update(
                {"kind": "directory", "n_files": nfiles, "bytes": total, algo: digest}
            )
        else:
            rec.update(
                {
                    "kind": "file",
                    "bytes": real.stat().st_size,
                    algo: checksum(real, algo),
                }
            )
        out[key] = rec
    return out


def write_manifest(ctx: Context, paths: dict[str, Path], filename: str) -> Path:
    algo = ctx.cfg["repro"]["checksum_algo"]
    m = manifest(paths, algo)
    out = ctx.interim(filename)
    out.write_text(json.dumps(m, indent=2, sort_keys=True))
    ctx.log.info("wrote manifest %s (%d entries)", out.relative_to(REPO), len(m))
    return out


def verify_manifest(ctx: Context, filename: str = INPUT_MANIFEST) -> list[str]:
    """Re-check a stored manifest; returns the list of drifted keys.

    The input manifest is written by `make inventory`. When it does not exist
    yet (a fresh clone), it is built now from the declared inputs, so the check
    starts from this run's files and detects drift in later runs.
    """
    path = ctx.interim(filename)
    if not path.exists():
        if filename != INPUT_MANIFEST:
            raise FileNotFoundError(path)
        ctx.log.info("%s not found; building it now", path.relative_to(REPO))
        write_manifest(ctx, input_paths(ctx.cfg), filename)
    stored = json.loads(path.read_text())
    algo = ctx.cfg["repro"]["checksum_algo"]
    drift = []
    for key, rec in stored.items():
        if rec.get("status") != "ok":
            continue
        p = REPO / rec["path"]
        if not p.exists():
            drift.append(key)
            continue
        now = checksum_dir(p, algo)[0] if p.is_dir() else checksum(p, algo)
        if now != rec[algo]:
            drift.append(key)
    if drift:
        ctx.log.warning("input drift in: %s", ", ".join(drift))
    else:
        ctx.log.info("all %d inputs match stored checksums", len(stored))
    return drift


def sums_targets() -> list[Path]:
    """The files `data/SHA256SUMS` covers: raw data, frozen layers, the wheel."""
    files: list[Path] = []
    for base in (
        REPO / "data" / "raw",
        REPO / "data" / "frozen",
        REPO / "env" / "wheels",
    ):
        files += [
            f
            for f in base.rglob("*")
            if f.is_file()
            and f.name != ".gitkeep"
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in f.relative_to(REPO).parts
            )
        ]
    return sorted(files)


def write_sums(path: Path = SUMS_FILE) -> int:
    """Write `<sha256>  <repo-relative path>` lines (the `sha256sum` format)."""
    lines = [f"{checksum(f)}  {f.relative_to(REPO)}" for f in sums_targets()]
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_sums(path: Path = SUMS_FILE) -> list[str]:
    """Check every entry of `data/SHA256SUMS`; returns the failing entries."""
    bad = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        f = REPO / rel.strip()
        if not f.exists():
            bad.append(f"{rel}: missing")
        elif checksum(f) != digest:
            bad.append(f"{rel}: checksum differs")
    return bad


# -- exclusions: the order of the steps changes the sample -------------------


def apply_exclusions(ctx: Context, df: Any, level: str) -> Any:
    """Python twin of `apply_exclusions()` in src/common.R — same config, same
    order, same count labels, so a funnel run in Python is comparable line for
    line with the R gate's.

    Column names are resolved through `measures.columns[<level>]` where a
    logical name exists (`land_value`, `plot_area`, ...) and taken literally
    otherwise, exactly as the R version does. Callers passing a rebuilt
    dependent variable should therefore name its columns the way the config's
    `exclusions.positivity` list does.

    One difference from the R version: R ends by re-ordering the frame with
    `order_for_weights()`, so that `knearneigh`'s tie-breaking depends on the
    rows' attributes rather than on the file's row order. Python does not build
    a neighbour graph among the observations (`sjoin_nearest` snaps address
    points to street segments, which the order does not affect), so the step is
    not mirrored here. The counts and the funnel are unaffected by the ordering.
    """
    ex = ctx.cfg["exclusions"]
    cmap = ctx.cfg["measures"]["columns"][level]

    def resolve(nm: str) -> str:
        return cmap.get(nm, nm)

    pos_cols = [resolve(c) for c in ex["positivity"][level]]
    lv_col = resolve(ex["iqr_outliers"]["variable"])

    ctx.count(f"{level}: loaded", len(df))
    order = ex["order"].get(level)
    if order is None:
        raise KeyError(f"no exclusion order declared for level '{level}' in config")
    ctx.log.info("exclusion order (%s): %s", level, " -> ".join(order))
    if (
        "iqr_outliers" in order
        and "positivity" in order
        and order.index("iqr_outliers") < order.index("positivity")
    ):
        raise ValueError(
            "exclusions.order: the IQR fence is computed after positivity "
            "in the Python pipeline; it cannot come first"
        )

    for step in order:
        if step == "positivity":
            keep = None
            for cc in pos_cols:
                if cc not in df.columns:
                    raise KeyError(f"positivity column not found: {cc}")
                v = df[cc]
                m = v.notna() & (v > 0)
                keep = m if keep is None else (keep & m)
            df = df[keep]
            ctx.count("after positivity", len(df), ",".join(pos_cols))
        elif step == "iqr_outliers":
            v = df[lv_col]
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            m = ex["iqr_outliers"]["multiplier"]
            lo, hi = q1 - m * (q3 - q1), q3 + m * (q3 - q1)
            df = df[v.notna() & (v >= lo) & (v <= hi)]
            ctx.count(
                "after IQR outliers",
                len(df),
                f"{lv_col} in [{lo:.2f}, {hi:.2f}], fence on the rows reaching this step",
            )
        else:
            raise ValueError(f"unknown exclusion step: {step}")
    return df


# -- environment capture ------------------------------------------------------


def capture_env(ctx: Context) -> Path:
    """Record this run's Python environment under `logs/env/`.

    The tracked files in `env/` record the environment of the runs of record;
    `make env` copies `logs/env/*` there when that record is to be replaced.
    """
    envdir = REPO / ctx.cfg["paths"]["logs"] / "env"
    envdir.mkdir(parents=True, exist_ok=True)
    out = envdir / "pip_freeze.txt"
    res = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    out.write_text(res.stdout)
    (envdir / "python_version.txt").write_text(sys.version + "\n")
    ctx.log.info("wrote %s", out.relative_to(REPO))
    return out
