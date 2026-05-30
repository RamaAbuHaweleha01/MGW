from typing import Any, Optional, List, Dict, Tuple, Callable, Union

def get_mark_comment(slot: int) -> Any:
    ...

def get_marked_pos(slot: int) -> ida_idaapi.ea_t:
    ...

def mark_position(ea: ida_idaapi.ea_t, lnnum: int, x: short, y: short, slot: int, comment: str) -> None:
    ...

SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module