from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with fixup information.

A loader should setup fixup information using set_fixup(). 
    
"""

class fixup_data_t:
    @property
    def displacement(self) -> Any: ...
    @property
    def off(self) -> Any: ...
    @property
    def sel(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, value: Any) -> Any:
        r"""Return self==value."""
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, value: Any) -> Any:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> Any:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> Any:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> Any:
        r"""Return self!=value."""
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
    def calc_size(self) -> int:
        r"""calc_fixup_size()
        
        """
        ...
    def clr_extdef(self) -> None:
        ...
    def clr_unused(self) -> None:
        ...
    def get(self, source: ida_idaapi.ea_t) -> bool:
        r"""get_fixup()
        
        """
        ...
    def get_base(self) -> ida_idaapi.ea_t:
        r"""Get base of fixup. 
                
        """
        ...
    def get_desc(self, source: ida_idaapi.ea_t) -> str:
        r"""get_fixup_desc()
        
        """
        ...
    def get_flags(self) -> int:
        r"""Fixup flags Fixup flags.
        
        """
        ...
    def get_handler(self) -> fixup_handler_t:
        r"""get_fixup_handler()
        
        """
        ...
    def get_type(self) -> fixup_type_t:
        r"""Fixup type Types of fixups.
        
        """
        ...
    def get_value(self, ea: ida_idaapi.ea_t) -> int:
        r"""get_fixup_value()
        
        """
        ...
    def has_base(self) -> bool:
        r"""Is fixup relative?
        
        """
        ...
    def is_custom(self) -> bool:
        r"""is_fixup_custom()
        
        """
        ...
    def is_extdef(self) -> bool:
        ...
    def is_unused(self) -> bool:
        ...
    def patch_value(self, ea: ida_idaapi.ea_t) -> bool:
        r"""patch_fixup_value()
        
        """
        ...
    def set(self, source: ida_idaapi.ea_t) -> None:
        r"""set_fixup()
        
        """
        ...
    def set_base(self, new_base: ida_idaapi.ea_t) -> None:
        r"""Set base of fixup. The target should be set before a call of this function. 
                
        """
        ...
    def set_extdef(self) -> None:
        ...
    def set_sel(self, seg: segment_t) -> None:
        ...
    def set_target_sel(self) -> None:
        r"""Set selector of fixup to the target. The target should be set before a call of this function. 
                
        """
        ...
    def set_type(self, type_: fixup_type_t) -> None:
        ...
    def set_type_and_flags(self, type_: fixup_type_t, flags_: int = 0) -> None:
        ...
    def set_unused(self) -> None:
        ...
    def was_created(self) -> bool:
        r"""Is fixup artificial?
        
        """
        ...

class fixup_info_t:
    @property
    def ea(self) -> Any: ...
    @property
    def fd(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, value: Any) -> Any:
        r"""Return self==value."""
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, value: Any) -> Any:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> Any:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> Any:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> Any:
        r"""Return self!=value."""
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

def calc_fixup_size(type: fixup_type_t) -> int:
    r"""Calculate size of fixup in bytes (the number of bytes the fixup patches) 
            
    :returns: -1: means error
    """
    ...

def contains_fixups(ea: ida_idaapi.ea_t, size: asize_t) -> bool:
    r"""Does the specified address range contain any fixup information?
    
    """
    ...

def del_fixup(source: ida_idaapi.ea_t) -> None:
    r"""Delete fixup information.
    
    """
    ...

def exists_fixup(source: ida_idaapi.ea_t) -> bool:
    r"""Check that a fixup exists at the given address.
    
    """
    ...

def find_custom_fixup(name: str) -> fixup_type_t:
    ...

def gen_fix_fixups(_from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, size: asize_t) -> None:
    r"""Relocate the bytes with fixup information once more (generic function). This function may be called from loader_t::move_segm() if it suits the goal. If loader_t::move_segm is not defined then this function will be called automatically when moving segments or rebasing the entire program. Special parameter values (from = BADADDR, size = 0, to = delta) are used when the function is called from rebase_program(delta). 
            
    """
    ...

def get_first_fixup_ea() -> ida_idaapi.ea_t:
    ...

def get_fixup(fd: fixup_data_t, source: ida_idaapi.ea_t) -> bool:
    r"""Get fixup information.
    
    """
    ...

def get_fixup_desc(source: ida_idaapi.ea_t, fd: fixup_data_t) -> str:
    r"""Get FIXUP description comment.
    
    """
    ...

def get_fixup_handler(type: fixup_type_t) -> fixup_handler_t:
    r"""Get handler of standard or custom fixup.
    
    """
    ...

def get_fixup_value(ea: ida_idaapi.ea_t, type: fixup_type_t) -> int:
    r"""Get the operand value. This function get fixup bytes from data or an instruction at `ea` and convert them to the operand value (maybe partially). It is opposite in meaning to the `patch_fixup_value()`. For example, FIXUP_HI8 read a byte at `ea` and shifts it left by 8 bits, or AArch64's custom fixup BRANCH26 get low 26 bits of the insn at `ea` and shifts it left by 2 bits. This function is mainly used to get a relocation addend. 
            
    :param ea: address to get fixup bytes from, the size of the fixup bytes depends on the fixup type.
    :param type: fixup type
    :returns: operand: value
    """
    ...

def get_fixups(out: fixups_t, ea: ida_idaapi.ea_t, size: asize_t) -> bool:
    ...

def get_next_fixup_ea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_prev_fixup_ea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def handle_fixups_in_macro(ri: refinfo_t, ea: ida_idaapi.ea_t, other: fixup_type_t, macro_reft_and_flags: int) -> bool:
    r"""Handle two fixups in a macro. We often combine two instruction that load parts of a value into one macro instruction. For example: 
           ADRP  X0, #var@PAGE
               ADD   X0, X0, #var@PAGEOFF  --> ADRL X0, var
          lui   $v0, %hi(var)
               addiu $v0, $v0, %lo(var)    --> la   $v0, var
    
    
            
    :returns: success ('false' means that RI was not changed)
    """
    ...

def is_fixup_custom(type: fixup_type_t) -> bool:
    r"""Is fixup processed by processor module?
    
    """
    ...

def patch_fixup_value(ea: ida_idaapi.ea_t, fd: fixup_data_t) -> bool:
    r"""Patch the fixup bytes. This function updates data or an instruction at `ea` to the fixup bytes. For example, FIXUP_HI8 updates a byte at `ea` to the high byte of `fd->off`, or AArch64's custom fixup BRANCH26 updates low 26 bits of the insn at `ea` to the value of `fd->off` shifted right by 2. 
            
    :param ea: address where data are changed, the size of the changed data depends on the fixup type.
    :param fd: fixup data
    :returns: false: the fixup bytes do not fit (e.g. `fd->off` is greater than 0xFFFFFFC for BRANCH26). The database is changed even in this case.
    """
    ...

def set_fixup(source: ida_idaapi.ea_t, fd: fixup_data_t) -> None:
    r"""Set fixup information. You should fill fixup_data_t and call this function and the kernel will remember information in the database. 
            
    :param source: the fixup source address, i.e. the address modified by the fixup
    :param fd: fixup data
    """
    ...

FIXUPF_CREATED: int  # 8
FIXUPF_EXTDEF: int  # 2
FIXUPF_LOADER_MASK: int  # -268435456
FIXUPF_REL: int  # 1
FIXUPF_UNUSED: int  # 4
FIXUP_CUSTOM: int  # 32768
FIXUP_HI16: int  # 7
FIXUP_HI8: int  # 6
FIXUP_LOW16: int  # 9
FIXUP_LOW8: int  # 8
FIXUP_OFF16: int  # 1
FIXUP_OFF16S: int  # 15
FIXUP_OFF32: int  # 4
FIXUP_OFF32S: int  # 16
FIXUP_OFF64: int  # 12
FIXUP_OFF8: int  # 13
FIXUP_OFF8S: int  # 14
FIXUP_PTR16: int  # 3
FIXUP_PTR32: int  # 5
FIXUP_SEG16: int  # 2
SWIG_PYTHON_LEGACY_BOOL: int  # 1
V695_FIXUP_VHIGH: int  # 10
V695_FIXUP_VLOW: int  # 11
annotations: _Feature
ida_idaapi: module
weakref: module