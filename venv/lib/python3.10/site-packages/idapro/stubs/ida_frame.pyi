from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Routines to manipulate function stack frames, stack variables, register variables and local labels.

The frame is represented as a structure: 
  +------------------------------------------------+
  | function arguments                             |
  +------------------------------------------------+
  | return address (isn't stored in func_t)        |
  +------------------------------------------------+
  | saved registers (SI, DI, etc - func_t::frregs) |
  +------------------------------------------------+ <- typical BP
  |                                                |  |
  |                                                |  | func_t::fpd
  |                                                |  |
  |                                                | <- real BP
  | local variables (func_t::frsize)               |
  |                                                |
  |                                                |
  +------------------------------------------------+ <- SP

To access the structure of a function frame and stack variables, use:
* tinfo_t::get_func_frame(const func_t *pfn) (the preferred way)
* get_func_frame(tinfo_t *out, const func_t *pfn)
* tinfo_t::get_udt_details() gives info about stack variables: their type, names, offset, etc 


    
"""

class regvar_t:
    @property
    def canon(self) -> Any: ...
    @property
    def cmt(self) -> Any: ...
    @property
    def end_ea(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def user(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: range_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: range_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: range_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: range_t) -> bool:
        ...
    def __lt__(self, r: range_t) -> bool:
        ...
    def __ne__(self, r: range_t) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> Any:
        r"""Return str(self)."""
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def clear(self) -> None:
        r"""Set start_ea, end_ea to 0.
        
        """
        ...
    def compare(self, r: range_t) -> int:
        ...
    def contains(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. contains(ea: ida_idaapi.ea_t) -> bool
            1. contains(r: const range_t &) -> bool
        
        # 0: contains(ea: ida_idaapi.ea_t) -> bool
        
        Compare two range_t instances, based on the start_ea.
        
        Is 'ea' in the address range? 
                
        
        # 1: contains(r: const range_t &) -> bool
        
        Is every ea in 'r' also in this range_t?
        
        
        """
        ...
    def empty(self) -> bool:
        r"""Is the size of the range_t <= 0?
        
        """
        ...
    def extend(self, ea: ida_idaapi.ea_t) -> None:
        r"""Ensure that the range_t includes 'ea'.
        
        """
        ...
    def intersect(self, r: range_t) -> None:
        r"""Assign the range_t to the intersection between the range_t and 'r'.
        
        """
        ...
    def overlaps(self, r: range_t) -> bool:
        r"""Is there an ea in 'r' that is also in this range_t?
        
        """
        ...
    def size(self) -> int:
        r"""Get end_ea - start_ea.
        
        """
        ...
    def swap(self, r: regvar_t) -> None:
        ...

class stkpnt_t:
    @property
    def ea(self) -> Any: ...
    @property
    def spd(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: stkpnt_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: stkpnt_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: stkpnt_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: stkpnt_t) -> bool:
        ...
    def __lt__(self, r: stkpnt_t) -> bool:
        ...
    def __ne__(self, r: stkpnt_t) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> Any:
        r"""Return str(self)."""
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def compare(self, r: stkpnt_t) -> int:
        ...

class stkpnts_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: stkpnts_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: stkpnts_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: stkpnts_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: stkpnts_t) -> bool:
        ...
    def __lt__(self, r: stkpnts_t) -> bool:
        ...
    def __ne__(self, r: stkpnts_t) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> Any:
        r"""Return str(self)."""
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def compare(self, r: stkpnts_t) -> int:
        ...

class xreflist_entry_t:
    @property
    def ea(self) -> Any: ...
    @property
    def opnum(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: xreflist_entry_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: xreflist_entry_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: xreflist_entry_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: xreflist_entry_t) -> bool:
        ...
    def __lt__(self, r: xreflist_entry_t) -> bool:
        ...
    def __ne__(self, r: xreflist_entry_t) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> Any:
        r"""Return str(self)."""
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def compare(self, r: xreflist_entry_t) -> int:
        ...

class xreflist_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: xreflist_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> xreflist_entry_t:
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __iter__(self) -> Any:
        r"""Helper function, to be set as __iter__ method for qvector-, or array-based classes."""
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: xreflist_t) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, i: size_t, v: xreflist_entry_t) -> None:
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> Any:
        r"""Return str(self)."""
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def add_unique(self, x: xreflist_entry_t) -> bool:
        ...
    def append(self, x: xreflist_entry_t) -> None:
        ...
    def at(self, _idx: size_t) -> xreflist_entry_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> qvector:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> qvector:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: xreflist_t) -> None:
        ...
    def extract(self) -> xreflist_entry_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: xreflist_entry_t) -> bool:
        ...
    def inject(self, s: xreflist_entry_t, len: size_t) -> None:
        ...
    def insert(self, it: xreflist_entry_t, x: xreflist_entry_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> xreflist_entry_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: xreflist_t) -> None:
        ...
    def truncate(self) -> None:
        ...

def add_auto_stkpnt(pfn: func_t, ea: ida_idaapi.ea_t, delta: int) -> bool:
    r"""Add automatic SP register change point. 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address where SP changes. usually this is the end of the instruction which modifies the stack pointer ( insn_t::ea+ insn_t::size)
    :param delta: difference between old and new values of SP
    :returns: success
    """
    ...

def add_frame(pfn: func_t, frsize: int, frregs: ushort, argsize: asize_t) -> bool:
    r"""Add function frame. 
            
    :param pfn: pointer to function structure
    :param frsize: size of function local variables
    :param frregs: size of saved registers
    :param argsize: size of function arguments range which will be purged upon return. this parameter is used for __stdcall and __pascal calling conventions. for other calling conventions please pass 0.
    :returns: 1: ok
    :returns: 0: failed (no function, frame already exists)
    """
    ...

def add_frame_member(pfn: func_t, name: str, offset: int, tif: tinfo_t, repr: value_repr_t = None, etf_flags: uint = 0) -> bool:
    r"""Add member to the frame type 
            
    :param pfn: pointer to function
    :param name: variable name, nullptr means autogenerate a name
    :param offset: member offset in the frame structure, in bytes
    :param tif: variable type
    :param repr: variable representation
    :returns: success
    """
    ...

def add_regvar(pfn: func_t, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, canon: str, user: str, cmt: str) -> int:
    r"""Define a register variable. 
            
    :param pfn: function in which the definition will be created
    :param ea1: range of addresses within the function where the definition will be used
    :param ea2: range of addresses within the function where the definition will be used
    :param canon: name of a general register
    :param user: user-defined name for the register
    :param cmt: comment for the definition
    :returns: Register variable error codes
    """
    ...

def add_user_stkpnt(ea: ida_idaapi.ea_t, delta: int) -> bool:
    r"""Add user-defined SP register change point. 
            
    :param ea: linear address where SP changes
    :param delta: difference between old and new values of SP
    :returns: success
    """
    ...

def build_stkvar_name(pfn: func_t, v: int) -> str:
    r"""Build automatic stack variable name. 
            
    :param pfn: pointer to function (can't be nullptr!)
    :param v: value of variable offset
    :returns: length of stack variable name or -1
    """
    ...

def build_stkvar_xrefs(out: xreflist_t, pfn: func_t, start_offset: int, end_offset: int) -> None:
    r"""Fill 'out' with a list of all the xrefs made from function 'pfn' to specified range of the pfn's stack frame. 
            
    :param out: the list of xrefs to fill.
    :param pfn: the function to scan.
    :param start_offset: start frame structure offset, in bytes
    :param end_offset: end frame structure offset, in bytes
    """
    ...

def calc_frame_offset(pfn: func_t, off: int, insn: insn_t = None, op: op_t = None) -> int:
    r"""Calculate the offset of stack variable in the frame. 
            
    :param pfn: pointer to function (cannot be nullptr)
    :param off: the offset relative to stack pointer or frame pointer
    :param insn: the instruction
    :param op: the operand
    :returns: the offset in the frame
    """
    ...

def calc_stkvar_struc_offset(pfn: func_t, insn: insn_t, n: int) -> ida_idaapi.ea_t:
    r"""Calculate offset of stack variable in the frame structure. 
            
    :param pfn: pointer to function (cannot be nullptr)
    :param insn: the instruction
    :param n: 0..UA_MAXOP-1 operand number -1 if error, return BADADDR
    :returns: BADADDR if some error (issue a warning if stack frame is bad)
    """
    ...

def define_stkvar(pfn: func_t, name: str, off: int, tif: tinfo_t, repr: value_repr_t = None) -> bool:
    r"""Define/redefine a stack variable. 
            
    :param pfn: pointer to function
    :param name: variable name, nullptr means autogenerate a name
    :param off: offset of the stack variable in the frame. negative values denote local variables, positive - function arguments.
    :param tif: variable type
    :param repr: variable representation
    :returns: success
    """
    ...

def del_frame(pfn: func_t) -> bool:
    r"""Delete a function frame. 
            
    :param pfn: pointer to function structure
    :returns: success
    """
    ...

def del_regvar(pfn: func_t, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, canon: str) -> int:
    r"""Delete a register variable definition. 
            
    :param pfn: function in question
    :param ea1: range of addresses within the function where the definition holds
    :param ea2: range of addresses within the function where the definition holds
    :param canon: name of a general register
    :returns: Register variable error codes
    """
    ...

def del_stkpnt(pfn: func_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Delete SP register change point. 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address
    :returns: success
    """
    ...

def delete_frame_members(pfn: func_t, start_offset: int, end_offset: int) -> bool:
    r"""Delete frame members 
            
    :param pfn: pointer to function
    :param start_offset: member offset to start deletion from, in bytes
    :param end_offset: member offset which not included in the deletion, in bytes
    :returns: success
    """
    ...

def find_regvar(args: Any) -> regvar_t:
    r"""This function has the following signatures:
    
        0. find_regvar(pfn: func_t *, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, canon: str, user: str) -> regvar_t *
        1. find_regvar(pfn: func_t *, ea: ida_idaapi.ea_t, canon: str) -> regvar_t *
    
    # 0: find_regvar(pfn: func_t *, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, canon: str, user: str) -> regvar_t *
    
    Find a register variable definition (powerful version). One of 'canon' and 'user' should be nullptr. If both 'canon' and 'user' are nullptr it returns the first regvar definition in the range. 
            
    :returns: nullptr-not found, otherwise ptr to regvar_t
    
    # 1: find_regvar(pfn: func_t *, ea: ida_idaapi.ea_t, canon: str) -> regvar_t *
    
    Find a register variable definition. 
            
    :returns: nullptr-not found, otherwise ptr to regvar_t
    
    """
    ...

def frame_off_args(pfn: func_t) -> ida_idaapi.ea_t:
    r"""Get starting address of arguments section.
    
    """
    ...

def frame_off_lvars(pfn: func_t) -> ida_idaapi.ea_t:
    r"""Get start address of local variables section.
    
    """
    ...

def frame_off_retaddr(pfn: func_t) -> ida_idaapi.ea_t:
    r"""Get starting address of return address section.
    
    """
    ...

def frame_off_savregs(pfn: func_t) -> ida_idaapi.ea_t:
    r"""Get starting address of saved registers section.
    
    """
    ...

def free_regvar(v: regvar_t) -> None:
    ...

def get_effective_spd(pfn: func_t, ea: ida_idaapi.ea_t) -> int:
    r"""Get effective difference between the initial and current values of ESP. This function returns the sp-diff used by the instruction. The difference between get_spd() and get_effective_spd() is present only for instructions like "pop [esp+N]": they modify sp and use the modified value. 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address
    :returns: 0 or the difference, usually a negative number
    """
    ...

def get_frame_part(range: range_t, pfn: func_t, part: frame_part_t) -> None:
    r"""Get offsets of the frame part in the frame. 
            
    :param range: pointer to the output buffer with the frame part start/end(exclusive) offsets, can't be nullptr
    :param pfn: pointer to function structure, can't be nullptr
    :param part: frame part
    """
    ...

def get_frame_retsize(pfn: func_t) -> int:
    r"""Get size of function return address. 
            
    :param pfn: pointer to function structure, can't be nullptr
    """
    ...

def get_frame_size(pfn: func_t) -> int:
    r"""Get full size of a function frame. This function takes into account size of local variables + size of saved registers + size of return address + number of purged bytes. The purged bytes correspond to the arguments of the functions with __stdcall and __fastcall calling conventions. 
            
    :param pfn: pointer to function structure, may be nullptr
    :returns: size of frame in bytes or zero
    """
    ...

def get_func_frame(out: tinfo_t, pfn: func_t) -> bool:
    r"""Get type of function frame 
            
    :param out: type info
    :param pfn: pointer to function structure
    :returns: success
    """
    ...

def get_sp_delta(pfn: func_t, ea: ida_idaapi.ea_t) -> int:
    r"""Get modification of SP made at the specified location 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address
    :returns: 0 if the specified location doesn't contain a SP change point. otherwise return delta of SP modification.
    """
    ...

def get_spd(pfn: func_t, ea: ida_idaapi.ea_t) -> int:
    r"""Get difference between the initial and current values of ESP. 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address of the instruction
    :returns: 0 or the difference, usually a negative number. returns the sp-diff before executing the instruction.
    """
    ...

def has_regvar(pfn: func_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Is there a register variable definition? 
            
    :param pfn: function in question
    :param ea: current address
    """
    ...

def is_anonymous_member_name(name: str) -> bool:
    r"""Is member name prefixed with "anonymous"?
    
    """
    ...

def is_dummy_member_name(name: str) -> bool:
    r"""Is member name an auto-generated name?
    
    """
    ...

def is_funcarg_off(pfn: func_t, frameoff: int) -> bool:
    ...

def is_special_frame_member(tid: tid_t) -> bool:
    r"""Is stkvar with TID the return address slot or the saved registers slot ? 
            
    :param tid: frame member type id return address or saved registers member?
    """
    ...

def lvar_off(pfn: func_t, frameoff: int) -> int:
    ...

def recalc_spd(cur_ea: ida_idaapi.ea_t) -> bool:
    r"""Recalculate SP delta for an instruction that stops execution. The next instruction is not reached from the current instruction. We need to recalculate SP for the next instruction.
    This function will create a new automatic SP register change point if necessary. It should be called from the emulator (emu.cpp) when auto_state == AU_USED if the current instruction doesn't pass the execution flow to the next instruction. 
            
    :param cur_ea: linear address of the current instruction
    :returns: 1: new stkpnt is added
    :returns: 0: nothing is changed
    """
    ...

def recalc_spd_for_basic_block(pfn: func_t, cur_ea: ida_idaapi.ea_t) -> bool:
    r"""Recalculate SP delta for the current instruction. The typical code snippet to calculate SP delta in a proc module is:
    
    if ( may_trace_sp() && pfn != nullptr )
      if ( !recalc_spd_for_basic_block(pfn, insn.ea) )
        trace_sp(pfn, insn);
    
    where trace_sp() is a typical name for a function that emulates the SP change of an instruction.
    
    :param pfn: pointer to the function
    :param cur_ea: linear address of the current instruction
    :returns: true: the cumulative SP delta is set
    :returns: false: the instruction at CUR_EA passes flow to the next instruction. SP delta must be set as a result of emulating the current instruction.
    """
    ...

def rename_regvar(pfn: func_t, v: regvar_t, user: str) -> int:
    r"""Rename a register variable. 
            
    :param pfn: function in question
    :param v: variable to rename
    :param user: new user-defined name for the register
    :returns: Register variable error codes
    """
    ...

def set_auto_spd(pfn: func_t, ea: ida_idaapi.ea_t, new_spd: int) -> bool:
    r"""Add such an automatic SP register change point so that at EA the new cumulative SP delta (that is, the difference between the initial and current values of SP) would be equal to NEW_SPD. 
            
    :param pfn: pointer to the function. may be nullptr.
    :param ea: linear address of the instruction
    :param new_spd: new value of the cumulative SP delta
    :returns: success
    """
    ...

def set_frame_member_type(pfn: func_t, offset: int, tif: tinfo_t, repr: value_repr_t = None, etf_flags: uint = 0) -> bool:
    r"""Change type of the frame member 
            
    :param pfn: pointer to function
    :param offset: member offset in the frame structure, in bytes
    :param tif: variable type
    :param repr: variable representation
    :returns: success
    """
    ...

def set_frame_size(pfn: func_t, frsize: asize_t, frregs: ushort, argsize: asize_t) -> bool:
    r"""Set size of function frame. Note: The returned size may not include all stack arguments. It does so only for __stdcall and __fastcall calling conventions. To get the entire frame size for all cases use frame.get_func_frame(pfn).get_size() 
            
    :param pfn: pointer to function structure
    :param frsize: size of function local variables
    :param frregs: size of saved registers
    :param argsize: size of function arguments that will be purged from the stack upon return
    :returns: success
    """
    ...

def set_purged(ea: ida_idaapi.ea_t, nbytes: int, override_old_value: bool) -> bool:
    r"""Set the number of purged bytes for a function or data item (funcptr). This function will update the database and plan to reanalyze items referencing the specified address. It works only for processors with PR_PURGING bit in 16 and 32 bit modes. 
            
    :param ea: address of the function of item
    :param nbytes: number of purged bytes
    :param override_old_value: may overwrite old information about purged bytes
    :returns: success
    """
    ...

def set_regvar_cmt(pfn: func_t, v: regvar_t, cmt: str) -> int:
    r"""Set comment for a register variable. 
            
    :param pfn: function in question
    :param v: variable to rename
    :param cmt: new comment
    :returns: Register variable error codes
    """
    ...

def soff_to_fpoff(pfn: func_t, soff: int) -> int:
    r"""Convert struct offsets into fp-relative offsets. This function converts the offsets inside the udt_type_data_t object into the frame pointer offsets (for example, EBP-relative). 
            
    """
    ...

def update_fpd(pfn: func_t, fpd: asize_t) -> bool:
    r"""Update frame pointer delta. 
            
    :param pfn: pointer to function structure
    :param fpd: new fpd value. cannot be bigger than the local variable range size.
    :returns: success
    """
    ...

FPC_ARGS: int  # 0
FPC_LVARS: int  # 3
FPC_RETADDR: int  # 1
FPC_SAVREGS: int  # 2
FRAME_UDM_NAME_R: str  # __return_address
FRAME_UDM_NAME_S: str  # __saved_registers
REGVAR_ERROR_ARG: int  # -1
REGVAR_ERROR_NAME: int  # -3
REGVAR_ERROR_OK: int  # 0
REGVAR_ERROR_RANGE: int  # -2
STKVAR_KEEP_EXISTING: int  # 2
STKVAR_VALID_SIZE: int  # 1
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
ida_range: module
weakref: module