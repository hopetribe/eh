# -*- coding: utf-8 -*-
"""测试运行器: python3 tests/run_all.py (兼容 pytest: pytest tests/)。"""
import importlib
import sys
import traceback
from pathlib import Path

MODULES = ["tests.test_tdx", "tests.test_golden", "tests.test_indicators",
           "tests.test_recipe", "tests.test_backtest", "tests.test_cli",
           "tests.test_server", "tests.test_data_service",
           "tests.test_screener", "tests.test_radar", "tests.test_radar_email",
           "tests.test_webui", "tests.test_shadow_validation",
           "tests.test_shadow_revision", "tests.test_shadow_evaluation"]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    failed = 0
    total = 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for fn in sorted(n for n in dir(mod) if n.startswith("test_")):
            total += 1
            try:
                getattr(mod, fn)()
                print(f"  ✓ {name}.{fn}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  ✗ {name}.{fn}: {e}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} 通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
