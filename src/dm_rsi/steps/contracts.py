"""文件契约的校验（ADR 0001）。在 sub-RSI 的 venv 里用 pandas 读，dm-rsi 自己不依赖 pandas。

分数文件：``<scores_dir>/<name>_<split>.parquet``，列至少有 ``id, y, p``，``y`` ∈ {0,1}，``p`` ∈ [0,1]，id 唯一。
"""

from __future__ import annotations

import json
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
