"""读写 YAML 文件：解析走 xtra_rsi.yamlio（YAML 1.2 语义），写入用原子替换。"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from xtra_rsi.yamlio import dump, load_mapping

if TYPE_CHECKING:
    from xtra_rsi.payload import Payload


def now() -> str:
    """UTC 时间，秒精度，ISO 8601。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read(path: Path) -> Payload:
    """读一个顶层为映射的 YAML 文件。"""
    return load_mapping(path.read_text(encoding="utf-8"), where=str(path))


def write(path: Path, obj: Payload) -> None:
    """原子写入：先写临时文件再替换，读的人不会看到半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dump(obj))
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
