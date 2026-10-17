"""Write files by replacing only a completely flushed temporary file."""
import json
import os
from pathlib import Path
import tempfile


def write_text_atomic(path, text):
    """原子写入文本：先完整 flush 临时文件，再替换目标。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=target.parent, prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path, value):
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False))
