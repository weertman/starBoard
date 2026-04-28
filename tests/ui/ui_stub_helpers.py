"""Shared helpers for lightweight UI module import scaffolds in tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any

MISSING = object()


def stub_class(name: str, **attrs: Any):
    return type(name, (), attrs)


def stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def install_src_stubs(
    stubbed_modules: dict[str, types.ModuleType],
    *,
    data_modules: Iterable[str] = (),
    ui_modules: Iterable[str] = (),
    utils_modules: Iterable[str] = (),
) -> dict[str, types.ModuleType | object]:
    """Install a stubbed src package tree and return modules to restore later.

    Some UI unit tests import large PySide modules via importlib while replacing
    heavy dependencies with lightweight stubs. This helper centralizes the
    fake src/src.data/src.ui/src.utils package wiring so each test only lists
    the module stubs it actually needs.
    """

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    data_pkg = types.ModuleType("src.data")
    data_pkg.__path__ = []
    ui_pkg = types.ModuleType("src.ui")
    ui_pkg.__path__ = []
    utils_pkg = types.ModuleType("src.utils")
    utils_pkg.__path__ = []

    src_pkg.data = data_pkg
    src_pkg.ui = ui_pkg
    src_pkg.utils = utils_pkg

    for attr in data_modules:
        setattr(data_pkg, attr, stubbed_modules[f"src.data.{attr}"])
    for attr in ui_modules:
        setattr(ui_pkg, attr, stubbed_modules[f"src.ui.{attr}"])
    for attr in utils_modules:
        setattr(utils_pkg, attr, stubbed_modules[f"src.utils.{attr}"])

    stubbed_modules.update({
        "src": src_pkg,
        "src.data": data_pkg,
        "src.ui": ui_pkg,
        "src.utils": utils_pkg,
    })
    previous_modules = {name: sys.modules.get(name, MISSING) for name in stubbed_modules}
    sys.modules.update(stubbed_modules)
    return previous_modules


def restore_modules(previous_modules: Mapping[str, types.ModuleType | object]) -> None:
    for name, previous in previous_modules.items():
        if previous is MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous  # type: ignore[assignment]
