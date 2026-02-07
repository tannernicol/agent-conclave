"""CLI package shim for `python -m conclave.cli`."""
from __future__ import annotations

from importlib import util as _import_util
from pathlib import Path as _Path

_CLI_PATH = _Path(__file__).resolve().parent.parent / "cli.py"
_spec = _import_util.spec_from_file_location("conclave._cli_main", _CLI_PATH)
if _spec and _spec.loader:
    _module = _import_util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    main = _module.main
    build_parser = _module.build_parser
    __all__ = ["main", "build_parser"]
else:
    __all__ = []
