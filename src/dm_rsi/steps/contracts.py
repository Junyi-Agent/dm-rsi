"""文件契约的校验（ADR 0001）。在 sub-RSI 的 venv 里用 pandas 读，dm-rsi 自己不依赖 pandas。

分数文件：``<scores_dir>/<name>_<split>.parquet``，列至少有 ``id, y, p``，``y`` ∈ {0,1}，``p`` ∈ [0,1]，id 唯一。

交接时重算（ADR 0001，2026-11-24 第二次修订，借鉴 Borui 的 ``rsi register``）：sub-RSI 报告里的指标，dm-rsi 从同一份
分数文件自己再算一遍（:func:`recompute`），对不上就当成契约问题（:func:`check_reported`），不拿没核对过的数做选择。
"""

from __future__ import annotations

import json
import math
import subprocess
from typing import TYPE_CHECKING, cast

from dm_rsi import subenv

if TYPE_CHECKING:
    from pathlib import Path

    from xtra_rsi.payload import Payload

    from dm_rsi.config import Config


class ContractError(ValueError):
    """产物不符合契约。"""


_CHECK = r"""
import json, sys
import pandas as pd
scores_dir, name, splits, convert = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4] == "1"
from pathlib import Path
out = {}
for s in splits:
    pq = Path(scores_dir) / f"{name}_{s}.parquet"
    csv = Path(scores_dir) / f"{name}_{s}.csv"
    if not pq.exists() and csv.exists() and convert:
        pd.read_csv(csv).to_parquet(pq, index=False)
        csv.unlink()
    if not pq.exists():
        out[s] = {"error": f"{pq} 不存在"}
        continue
    df = pd.read_parquet(pq)
    errs = [f"缺列 {c}" for c in ("id", "y", "p") if c not in df.columns]
    if not errs:
        if not df["id"].is_unique: errs.append("id 不唯一")
        if not set(df["y"].unique()) <= {0, 1}: errs.append("y 不是 0/1")
        if df["p"].isna().any() or (df["p"] < 0).any() or (df["p"] > 1).any(): errs.append("p 不在 [0,1]")
    out[s] = {"rows": int(len(df)), "pos": int(df["y"].sum()) if "y" in df else None,
              "columns": list(map(str, df.columns)), "errors": errs}
print(json.dumps(out))
"""


def check_scores(
    cfg: Config, scores_dir: Path, name: str, splits: list[str], *, convert_csv: bool = False
) -> Payload:
    """校验分数文件，``convert_csv`` 时把同名 csv 转成 parquet（合成数据的打分器写 csv）。"""
    proc = subprocess.run(
        [
            str(subenv.python(cfg)),
            "-c",
            _CHECK,
            str(scores_dir),
            name,
            ",".join(splits),
            "1" if convert_csv else "0",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**subenv.base_env(scores_dir), "PATH": "/usr/bin:/bin"},
        timeout=600,
    )
    if proc.returncode != 0:
        msg = f"校验分数文件失败：{proc.stderr.strip()[-2000:]}"
        raise ContractError(msg)
    report = cast("Payload", json.loads(proc.stdout))
    bad = {
        s: r
        for s, r in report.items()
        if isinstance(r, dict) and (r.get("error") or r.get("errors"))
    }
    if bad:
        msg = f"{scores_dir}/{name}_*.parquet 不符合契约：{bad}"
        raise ContractError(msg)
    return report


_METRICS = r"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
scores_dir, split, names = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
out = {}
for key, name in names.items():
    df = pd.read_parquet(f"{scores_dir}/{name}_{split}.parquet")
    y = df["y"].to_numpy().astype(int)
    p = df["p"].to_numpy().astype(float)
    two = 0 < y.sum() < len(y)
    out[key] = {"rows": int(len(y)), "pos": int(y.sum()),
                "auprc": float(average_precision_score(y, p)) if two else None,
                "auroc": float(roc_auc_score(y, p)) if two else None,
                "brier": float(np.mean((p - y) ** 2))}
print(json.dumps(out))
"""

# 和阈值无关、可以只从 (y, p) 算出来的指标；f1 / precision / recall 依赖 sub-RSI 选的阈值，不核对
CHECKED = ("auprc", "auroc", "brier")
TOLERANCE = 1e-6


def recompute(cfg: Config, scores_dir: Path, names: dict[str, str], split: str) -> Payload:
    """从分数文件重算 ``{key: {rows, pos, auprc, auroc, brier}}``（``names``：key → 分数文件名前缀）。"""
    out = subenv.run_code(cfg, None, _METRICS, [str(scores_dir), split, json.dumps(names)])
    return cast("Payload", json.loads(out))


def _num(v: object) -> float | None:
    """数值；None、NaN、非数值都当成「没有」。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
        return None
    return float(v)


def check_reported(reported: dict[str, dict[str, object]], recomputed: Payload, where: str) -> None:
    """sub-RSI 报告的数（``{key: {指标: 值}}``）和重算的对不上时报 :class:`ContractError`，写明字段、报告值、重算值。"""
    bad: list[str] = []
    for key, mine in recomputed.items():
        theirs = reported.get(key)
        if theirs is None:
            bad.append(f"{key}：报告里没有这个模型")
            continue
        m = cast("dict[str, object]", mine)
        for field in ("rows", "pos", *CHECKED):
            if field not in theirs:
                continue
            a, b = _num(theirs[field]), _num(m.get(field))
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > TOLERANCE:
                bad.append(f"{key}.{field}：报告 {a}，dm-rsi 重算 {b}")
    if bad:
        msg = f"{where} 报告的指标和分数文件重算的对不上：" + "；".join(bad)
        raise ContractError(msg)
