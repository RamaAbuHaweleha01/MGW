from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Middle-level search functions.

They all are controlled by Search flags 
    
"""

def find_code(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_data(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_defined(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_error(ea: ida_idaapi.ea_t, sflag: int) -> int:
    ...

def find_imm(ea: ida_idaapi.ea_t, sflag: int, search_value: int) -> int:
    ...

def find_not_func(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_notype(ea: ida_idaapi.ea_t, sflag: int) -> int:
    ...

def find_reg_access(out: reg_access_t, start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, regname: str, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_suspop(ea: ida_idaapi.ea_t, sflag: int) -> int:
    ...

def find_text(start_ea: ida_idaapi.ea_t, y: int, x: int, ustr: str, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_unknown(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def search_down(sflag: int) -> bool:
    r"""Is the SEARCH_DOWN bit set?
    
    """
    ...

SEARCH_BRK: int  # 256
SEARCH_CASE: int  # 4
SEARCH_DEF: int  # 1024
SEARCH_DOWN: int  # 1
SEARCH_IDENT: int  # 128
SEARCH_NEXT: int  # 2
SEARCH_NOBRK: int  # 16
SEARCH_NOSHOW: int  # 32
SEARCH_REGEX: int  # 8
SEARCH_UP: int  # 0
SEARCH_USE: int  # 512
SEARCH_USESEL: int  # 2048
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module