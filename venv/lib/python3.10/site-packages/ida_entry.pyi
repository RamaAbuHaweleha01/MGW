from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with entry points.

Exported functions are considered as entry points as well.
IDA maintains list of entry points to the program. Each entry point:
* has an address
* has a name
* may have an ordinal number 


    
"""

def add_entry(ord: int, ea: ida_idaapi.ea_t, name: str, makecode: bool, flags: int = 0) -> bool:
    r"""Add an entry point to the list of entry points. 
            
    :param ord: ordinal number if ordinal number is equal to 'ea' then ordinal is not used
    :param ea: linear address
    :param name: name of entry point. If the specified location already has a name, the old name will be appended to the regular comment.
    :param makecode: should the kernel convert bytes at the entry point to instruction(s)
    :param flags: See AEF_*
    :returns: success (currently always true)
    """
    ...

def get_entry(ord: int) -> ida_idaapi.ea_t:
    r"""Get entry point address by its ordinal 
            
    :param ord: ordinal number of entry point
    :returns: address or BADADDR
    """
    ...

def get_entry_forwarder(ord: int) -> str:
    r"""Get forwarder name for the entry point by its ordinal. 
            
    :param ord: ordinal number of entry point
    :returns: size of entry forwarder name or -1
    """
    ...

def get_entry_name(ord: int) -> str:
    r"""Get name of the entry point by its ordinal. 
            
    :param ord: ordinal number of entry point
    :returns: size of entry name or -1
    """
    ...

def get_entry_ordinal(idx: size_t) -> int:
    r"""Get ordinal number of an entry point. 
            
    :param idx: internal number of entry point. Should be in the range 0..get_entry_qty()-1
    :returns: ordinal number or 0.
    """
    ...

def get_entry_qty() -> int:
    r"""Get number of entry points.
    
    """
    ...

def rename_entry(ord: int, name: str, flags: int = 0) -> bool:
    r"""Rename entry point. 
            
    :param ord: ordinal number of the entry point
    :param name: name of entry point. If the specified location already has a name, the old name will be appended to a repeatable comment.
    :param flags: See AEF_*
    :returns: success
    """
    ...

def set_entry_forwarder(ord: int, name: str, flags: int = 0) -> bool:
    r"""Set forwarder name for ordinal. 
            
    :param ord: ordinal number of the entry point
    :param name: forwarder name for entry point.
    :param flags: See AEF_*
    :returns: success
    """
    ...

AEF_IDBENC: int  # 1
AEF_NODUMMY: int  # 2
AEF_NOFORCE: int  # 8
AEF_UTF8: int  # 0
AEF_WEAK: int  # 4
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module