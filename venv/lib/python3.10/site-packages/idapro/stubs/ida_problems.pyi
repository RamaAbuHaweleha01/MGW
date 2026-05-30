from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with the list of problems.

There are several problem lists. An address may be inserted to any list. The kernel simply maintains these lists, no additional processing is done.
The problem lists are accessible for the user from the View->Subviews->Problems menu item.
Addresses in the lists are kept sorted. In general IDA just maintains these lists without using them during analysis (except PR_ROLLED). 
    
"""

def forget_problem(type: problist_id_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Remove an address from a problem list 
            
    :param type: problem list type
    :param ea: linear address
    :returns: success
    """
    ...

def get_problem(type: problist_id_t, lowea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get an address from the specified problem list. The address is not removed from the list. 
            
    :param type: problem list type
    :param lowea: the returned address will be higher or equal than the specified address
    :returns: linear address or BADADDR
    """
    ...

def get_problem_desc(t: problist_id_t, ea: ida_idaapi.ea_t) -> str:
    r"""Get the human-friendly description of the problem, if one was provided to remember_problem. 
            
    :param t: problem list type.
    :param ea: linear address.
    :returns: the message length or -1 if none
    """
    ...

def get_problem_name(type: problist_id_t, longname: bool = True) -> str:
    r"""Get problem list description.
    
    """
    ...

def is_problem_present(t: problist_id_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Check if the specified address is present in the problem list.
    
    """
    ...

def remember_problem(type: problist_id_t, ea: ida_idaapi.ea_t, msg: str = None) -> None:
    r"""Insert an address to a list of problems. Display a message saying about the problem (except of PR_ATTN,PR_FINAL) PR_JUMP is temporarily ignored. 
            
    :param type: problem list type
    :param ea: linear address
    :param msg: a user-friendly message to be displayed instead of the default more generic one associated with the type of problem. Defaults to nullptr.
    """
    ...

def was_ida_decision(ea: ida_idaapi.ea_t) -> bool:
    ...

PR_ATTN: int  # 12
PR_BADSTACK: int  # 11
PR_COLLISION: int  # 15
PR_DECIMP: int  # 16
PR_DISASM: int  # 7
PR_END: int  # 17
PR_FINAL: int  # 13
PR_HEAD: int  # 8
PR_ILLADDR: int  # 9
PR_JUMP: int  # 6
PR_MANYLINES: int  # 10
PR_NOBASE: int  # 1
PR_NOCMT: int  # 4
PR_NOFOP: int  # 3
PR_NONAME: int  # 2
PR_NOXREFS: int  # 5
PR_ROLLED: int  # 14
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
cvar: swigvarlink
ida_idaapi: module
weakref: module