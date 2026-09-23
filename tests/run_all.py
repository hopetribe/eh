# -*- coding: utf-8 -*-
"""测试运行器: python3 tests/run_all.py (兼容 pytest: pytest tests/)。"""
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import pytest
    return pytest.main([str(root / "tests"), "-q", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
