"""雷达信号契约：与主图使用相同版本、默认参数和指标列。"""
from gcn.recipes.gcn_main import VERSIONS

SIGNALS = {
    "bSignal": ("B买", "B_SIGNAL"),
    "sSignal": ("S卖", "S_SIGNAL"),
    "sCondition": ("S条件", "S_CONDITION"),
    "juefan": ("绝反", "ICON_JUEFAN"),
    "upNine": ("上九转9", "NINE2_UP_9"),
    "downNine": ("下九转9", "NINE2_DOWN_9"),
    "stageSetup": ("B买Setup", "B_SETUP", "B_STAGE_SETUP"),
    "stageEntry": ("B买确认", "B_ENTRY_SIGNAL", "B_STAGE_ENTRY_SIGNAL"),
    "stageExpired": ("Setup过期", "B_SETUP_EXPIRED", "B_STAGE_EXPIRED"),
}


def signal_options(version):
    return [{"id": key, "label": value[0]} for key, value in SIGNALS.items()
            if not key.startswith("stage") or version in {"v4-exp", "v5"}]


def normalize_config(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("雷达配置必须为对象")
    version = config.get("version", "v5")
    if not isinstance(version, str) or version not in VERSIONS:
        raise ValueError(f"未知 GCN 版本: {version}")
    selected = config.get("signals", ["bSignal", "juefan"])
    allowed = [item["id"] for item in signal_options(version)]
    if not isinstance(selected, list) or not selected or any(
            not isinstance(key, str) or key not in allowed for key in selected):
        raise ValueError("请选择当前 GCN 版本支持的信号（至少一种）")
    return {"version": version, "signals": [key for key in allowed if key in selected]}
