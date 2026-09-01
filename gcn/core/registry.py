# -*- coding: utf-8 -*-
"""GCN 扩展注册表: 新增技术指标 / 指标配方的统一入口。

用法
----
    from gcn.core.registry import register_indicator, register_recipe

    @register_indicator("supertrend")
    def supertrend(df, period=10, mult=3):
        ...

注册后即可在配方 (recipes) 中按名引用, 便于发现与测试。
"""
from __future__ import annotations

import importlib
from typing import Callable

INDICATORS: dict[str, Callable] = {}
RECIPES: dict[str, Callable] = {}


def _ensure_builtin_indicators() -> None:
    """Load built-ins lazily so the registry works as a direct entry point."""
    if not INDICATORS:
        importlib.import_module("gcn.core.indicators")


def register_indicator(name: str):
    def deco(fn):
        INDICATORS[name] = fn
        fn.registry_name = name
        return fn
    return deco


def register_recipe(name: str):
    def deco(fn):
        RECIPES[name] = fn
        fn.registry_name = name
        return fn
    return deco


def get_indicator(name: str) -> Callable:
    _ensure_builtin_indicators()
    return INDICATORS[name]


def list_indicators() -> list[str]:
    _ensure_builtin_indicators()
    return sorted(INDICATORS)


# Direct imports expose the built-ins as well as the registration helpers.
_ensure_builtin_indicators()
