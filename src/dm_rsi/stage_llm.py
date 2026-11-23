"""``DMRSI_META=llm`` 时 reasoning / decision 阶段的命令：把阶段 AGENTS.md 和 upstream 交给 ``claude -p``。

xtra-rsi ``CommandStage`` 协议：stdin 是 YAML（``stage``、``iteration``、``instructions``、``upstream``），
stdout 是阶段输出（YAML 映射），带 ``_usage`` 给节点记账。没有工具：只读输入、给出判断。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import cast

from xtra_rsi.command_stage import strip_fence
from xtra_rsi.yamlio import dump, load_mapping


def main() -> int:
    """读 stdin，调 claude，写 stdout。"""
    req = load_mapping(sys.stdin.read(), where="stdin")
    prompt = (
        f"{req.get('instructions')}\n\n## 本轮输入（upstream）\n\n```yaml\n{dump(req.get('upstream') or {})}```\n\n"
        "只输出一个 YAML 映射（可以用 ```yaml 代码块包住），不要别的文字。"
    )
    argv = [
        os.environ.get("DMRSI_CLAUDE", "claude"),
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",
        prompt,
    ]
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        env={**os.environ, "DMRSI_AGENT": "1"},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    res = cast("dict[str, object]", json.loads(proc.stdout))
    out = load_mapping(strip_fence(str(res.get("result") or "")), where="claude 输出")
    out["_usage"] = {"cost_usd": float(cast("float", res.get("total_cost_usd") or 0.0))}
    sys.stdout.write(dump(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
