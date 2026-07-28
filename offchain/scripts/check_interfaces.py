"""Introspect the interfaces the backtest engine depends on.

Prints the real shape of the schema, market view, and agent classes so
that engine assumptions can be checked against them without guessing.
Touches neither the network nor the frozen snapshot.
"""

from __future__ import annotations

import importlib
import inspect

MODULES = [
    "bounded_vault.schema",
    "bounded_vault.market.view",
    "bounded_vault.agents.base",
    "bounded_vault.agents.yield_proportional",
]


def report(module_path: str) -> None:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        print(f"  could not import: {exc}")
        return

    found = False
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_path:
            continue
        found = True
        bases = ", ".join(b.__name__ for b in obj.__bases__)
        print(f"  class {name}({bases})")

        for field, kind in getattr(obj, "__annotations__", {}).items():
            print(f"      {field}: {kind}")

        for method_name, method in inspect.getmembers(obj, inspect.isfunction):
            if method_name.startswith("_"):
                continue
            print(f"      def {method_name}{inspect.signature(method)}")

    if not found:
        print("  no classes defined in this module")


for path in MODULES:
    print(f"=== {path} ===")
    report(path)
    print()