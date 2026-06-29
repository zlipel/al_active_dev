from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Mapping, Any

from .config import ALConfig



_DEFAULT_FMT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "model=%(model)s iter=%(iter)s front=%(front)s tag=%(tag)s"
    "%(extra_kv)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ContextFilter(logging.Filter):
    """
    Injects config context (model/iter/front/tag) into every log record,
    plus optional arbitrary key/values (seq_id, cand_id, etc).
    """
    def __init__(self, base_ctx: Mapping[str, Any]) -> None:
        super().__init__()
        self.base_ctx = dict(base_ctx)

    def filter(self, record: logging.LogRecord) -> bool:
        # Required fields for formatter:
        record.model = self.base_ctx.get("model", "-")
        record.iter = self.base_ctx.get("iter", "-")
        record.front = self.base_ctx.get("front", "-")
        record.tag = self.base_ctx.get("tag", "-")

        # Optional: build a " extra_kv" suffix from anything else in base_ctx
        extras = {k: v for k, v in self.base_ctx.items()
                  if k not in ("model", "iter", "front", "tag")}
        if extras:
            record.extra_kv = " " + " ".join(f"{k}={v}" for k, v in extras.items())
        else:
            record.extra_kv = ""
        return True


def _make_file_handler(log_path: Path, level: int) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    return fh


def _make_stream_handler(level: int) -> logging.Handler:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    return sh


def _get_or_create_logger(
    name: str,
    *,
    log_path: Path,
    ctx: Mapping[str, Any],
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    """
    Create a logger that writes to log_path (and optionally stdout).
    Ensures handlers are not duplicated on repeated imports/calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log via root

    # If we've already configured this logger, don't add handlers again.
    # We detect by checking for a FileHandler pointing at the same file.
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename) == log_path:
                    return logger
            except Exception:
                pass

    # Fresh configuration
    logger.handlers.clear()

    f = _ContextFilter(ctx)
    logger.addFilter(f)

    logger.addHandler(_make_file_handler(log_path, level))
    if also_stdout:
        logger.addHandler(_make_stream_handler(level))

    return logger


#### public api
@dataclass(frozen=True)
class LogPaths:
    master_log: Path
    child_log_dir: Path

    def child_log(self, seq_id: int) -> Path:
        return self.child_log_dir / f"child_{seq_id:02d}.log"


def get_log_paths(cfg: ALConfig) -> LogPaths:
    """
    Central place to define where logs go.

    In the IDP problem, this is int he directory:
      
    logs_dir = /home/.../MODEL_COMPARISON/<MODEL>/logs/iteration_<front>_<iter> --> defined in paths
    Add additional for overall logging in the same directory, AL_master_*, so we have:
    
      - AL_master_<tag>.log
      - child_01.log ... child_24.log
    """
    p = cfg.paths
    master = p.logs_dir / f"AL_master_{p.tag}.log" # total logging path
    return LogPaths(master_log=master, child_log_dir=p.logs_dir)


def get_master_logger(
    cfg: ALConfig,
    *,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    p = cfg.paths
    lp = get_log_paths(cfg)
    ctx = {
        "model": cfg.model,
        "iter": cfg.iteration,
        "front": cfg.front,
        "tag": p.tag,
        "phase": "master",
    }
    return _get_or_create_logger(
        f"al_pipeline.master.{cfg.model}.{cfg.front}.{cfg.iteration}",
        log_path=lp.master_log,
        ctx=ctx,
        level=level,
        also_stdout=also_stdout,
    )


def get_child_logger(
    cfg: ALConfig,
    seq_id: int,
    *,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    p = cfg.paths
    lp = get_log_paths(cfg)
    ctx = {
        "model": cfg.model,
        "iter": cfg.iteration,
        "front": cfg.front,
        "tag": p.tag,
        "phase": "child",
        "seq_id": seq_id,
    }
    return _get_or_create_logger(
        f"al_pipeline.child.{cfg.model}.{cfg.front}.{cfg.iteration}.seq{seq_id}",
        log_path=lp.child_log(seq_id),
        ctx=ctx,
        level=level,
        also_stdout=also_stdout,
    )


def with_context(logger: logging.Logger, **kwargs: Any) -> logging.LoggerAdapter:
    """
    NOTE to self: this does NOT change the base formatter fields (those come from the filter),
    """
    return logging.LoggerAdapter(logger, extra=kwargs)
