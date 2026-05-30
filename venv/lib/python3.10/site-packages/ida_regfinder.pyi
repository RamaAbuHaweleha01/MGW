from typing import Any, Optional, List, Dict, Tuple, Callable, Union

class reg_value_def_t:
    ABORTED: int  # 3
    NOVAL: int  # 0
    SPVAL: int  # 2
    UVAL: int  # 1
    @property
    def LIKE_GOT(self) -> Any: ...
    @property
    def PC_BASED(self) -> Any: ...
    @property
    def SHORT_INSN(self) -> Any: ...
    @property
    def def_ea(self) -> Any: ...
    @property
    def def_itype(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def val(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reg_value_def_t) -> bool:
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
    def __lt__(self, r: reg_value_def_t) -> bool:
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
    def __str__(self) -> str:
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
    def dstr(self, how: dstr_val_t, pm: procmod_t = None) -> str:
        r"""Return the string representation.
        
        """
        ...
    def is_like_got(self) -> bool:
        ...
    def is_pc_based(self) -> bool:
        ...
    def is_short_insn(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_short_insn() -> bool
            1. is_short_insn(insn: const insn_t &) -> bool
        
        # 0: is_short_insn() -> bool
        
        
        # 1: is_short_insn(insn: const insn_t &) -> bool
        
        
        """
        ...

class reg_value_info_t:
    ADD: int  # 0
    AND: int  # 3
    AND_NOT: int  # 5
    CONTAINED: int  # 2
    CONTAINS: int  # 1
    EQUAL: int  # 0
    MOVT: int  # 9
    NEG: int  # 10
    NOT: int  # 11
    NOT_COMPARABLE: int  # 3
    OR: int  # 2
    SAR: int  # 8
    SLL: int  # 6
    SLR: int  # 7
    SUB: int  # 1
    XOR: int  # 4
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, value: Any) -> bool:
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
    def __getitem__(self, i: size_t) -> reg_value_def_t:
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
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
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
    def __str__(self) -> str:
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
    def aborted(self) -> bool:
        r"""Return 'true' if the tracking process was aborted.
        
        """
        ...
    def add(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Add R to the value, save INSN as a defining instruction. 
                
        """
        ...
    def add_num(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. add_num(r: int, insn: const insn_t &) -> None
            1. add_num(r: int) -> None
        
        # 0: add_num(r: int, insn: const insn_t &) -> None
        
        Add R to the value, save INSN as a defining instruction. 
                
        
        # 1: add_num(r: int) -> None
        
        Add R to the value, do not change the defining instructions. 
                
        
        """
        ...
    def band(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Make bitwise AND of R to the value, save INSN as a defining instruction. 
                
        """
        ...
    def bandnot(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Make bitwise AND of the inverse of R to the value, save INSN as a defining instruction. 
                
        """
        ...
    def bnot(self, insn: insn_t) -> None:
        r"""Make bitwise inverse of the value, save INSN as a defining instruction. 
                
        """
        ...
    def bor(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Make bitwise OR of R to the value, save INSN as a defining instruction. 
                
        """
        ...
    def bxor(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Make bitwise eXclusive OR of R to the value, save INSN as a defining instruction. 
                
        """
        ...
    def clear(self) -> None:
        r"""Undefine the value.
        
        """
        ...
    def empty(self) -> bool:
        r"""Return 'true' if we know nothing about a value.
        
        """
        ...
    def extend(self, pm: procmod_t, width: int, is_signed: bool) -> None:
        r"""Sign-, or zero-extend the number or SP delta value to full size. The initial value is considered to be of size WIDTH. 
                
        """
        ...
    def get_aborting_depth(self) -> int:
        r"""Return the aborting depth if the value is ABORTED.
        
        """
        ...
    def get_def_ea(self) -> ida_idaapi.ea_t:
        r"""Return the defining address.
        
        """
        ...
    def get_def_itype(self) -> uint16:
        r"""Return the defining instruction code (processor specific).
        
        """
        ...
    def get_num(self) -> bool:
        r"""Return the number if the value is a constant. 
                
        """
        ...
    def get_spd(self) -> bool:
        r"""Return the SP delta if the value depends on the stack pointer. 
                
        """
        ...
    def has_any_vals_flag(self, val_flags: uint16) -> bool:
        ...
    def have_all_vals_flag(self, val_flags: uint16) -> bool:
        r"""Check the given flag for each value.
        
        """
        ...
    def is_all_vals_like_got(self) -> bool:
        ...
    def is_all_vals_pc_based(self) -> bool:
        ...
    def is_any_vals_like_got(self) -> bool:
        ...
    def is_any_vals_pc_based(self) -> bool:
        ...
    def is_badinsn(self) -> bool:
        r"""Return 'true' if the value is unknown because of a bad insn.
        
        """
        ...
    def is_dead_end(self) -> bool:
        r"""Return 'true' if the value is undefined because of a dead end.
        
        """
        ...
    def is_known(self) -> bool:
        r"""Return 'true' if the value is known (i.e. it is a number or SP delta).
        
        """
        ...
    def is_num(self) -> bool:
        r"""Return 'true' if the value is a constant.
        
        """
        ...
    def is_spd(self) -> bool:
        r"""Return 'true' if the value depends on the stack pointer.
        
        """
        ...
    def is_special(self) -> bool:
        r"""Return 'true' if the value requires special handling.
        
        """
        ...
    def is_unkfunc(self) -> bool:
        r"""Return 'true' if the value is unknown from the function start.
        
        """
        ...
    def is_unkinsn(self) -> bool:
        r"""Return 'true' if the value is unknown after executing the insn.
        
        """
        ...
    def is_unkloop(self) -> bool:
        r"""Return 'true' if the value is unknown because it changes in a loop.
        
        """
        ...
    def is_unkmult(self) -> bool:
        r"""Return 'true' if the value is unknown because the register has incompatible values (a number and SP delta). 
                
        """
        ...
    def is_unknown(self) -> bool:
        r"""Return 'true' if the value is unknown.
        
        """
        ...
    def is_unkvals(self) -> bool:
        r"""Return 'true' if the value is unknown because the register has too many values. 
                
        """
        ...
    def is_unkxref(self) -> bool:
        r"""Return 'true' if the value is unknown because there are too many xrefs.
        
        """
        ...
    def is_value_unique(self) -> bool:
        r"""Check that the value is unique.
        
        """
        ...
    def make_aborted(self, bblk_ea: ida_idaapi.ea_t, aborting_depth: int = -1) -> reg_value_info_t:
        r"""Return the value after aborting. 
                
        """
        ...
    def make_badinsn(self, insn_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value after a bad insn. 
                
        """
        ...
    def make_dead_end(self, dead_end_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the undefined value because of a dead end. 
                
        """
        ...
    def make_initial_sp(self, func_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the value that is the initial stack pointer. 
                
        """
        ...
    def make_num(self, args: Any) -> reg_value_info_t:
        r"""This function has the following signatures:
        
            0. make_num(rval: int, insn: const insn_t &, val_flags: uint16=0) -> reg_value_info_t
            1. make_num(rval: int, val_ea: ida_idaapi.ea_t, val_flags: uint16=0) -> reg_value_info_t
        
        # 0: make_num(rval: int, insn: const insn_t &, val_flags: uint16=0) -> reg_value_info_t
        
        Return the value that is the RVAL number. 
                
        
        # 1: make_num(rval: int, val_ea: ida_idaapi.ea_t, val_flags: uint16=0) -> reg_value_info_t
        
        Return the value that is the RVAL number. 
                
        
        """
        ...
    def make_unkfunc(self, func_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value from the function start. 
                
        """
        ...
    def make_unkinsn(self, insn: insn_t) -> reg_value_info_t:
        r"""Return the unknown value after executing the insn. 
                
        """
        ...
    def make_unkloop(self, bblk_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value if it changes in a loop. 
                
        """
        ...
    def make_unkmult(self, bblk_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value if the register has incompatible values. 
                
        """
        ...
    def make_unkvals(self, bblk_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value if the register has too many values. 
                
        """
        ...
    def make_unkxref(self, bblk_ea: ida_idaapi.ea_t) -> reg_value_info_t:
        r"""Return the unknown value if there are too many xrefs. 
                
        """
        ...
    def movt(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Replace the top 16 bits with bottom 16 bits of R, leaving the bottom 16 bits untouched, save INSN as a defining instruction. 
                
        """
        ...
    def neg(self, insn: insn_t) -> None:
        r"""Negate the value, save INSN as a defining instruction.
        
        """
        ...
    def sar(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Shift arithmetically the value right by R, save INSN as a defining instruction. 
                
        """
        ...
    def set_aborted(self, bblk_ea: ida_idaapi.ea_t, aborting_depth: int = -1) -> None:
        r"""Set the value after aborting. 
                
        """
        ...
    def set_all_vals_flag(self, val_flags: uint16) -> None:
        r"""Set the given flag for each value.
        
        """
        ...
    def set_all_vals_got_based(self) -> None:
        ...
    def set_all_vals_pc_based(self) -> None:
        ...
    def set_badinsn(self, insn_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown after a bad insn. 
                
        """
        ...
    def set_dead_end(self, dead_end_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be undefined because of a dead end. 
                
        """
        ...
    def set_num(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. set_num(rval: int, insn: const insn_t &, val_flags: uint16=0) -> None
            1. set_num(rvals: uvalvec_t *, insn: const insn_t &) -> None
            2. set_num(rval: int, val_ea: ida_idaapi.ea_t, val_flags: uint16=0) -> None
        
        # 0: set_num(rval: int, insn: const insn_t &, val_flags: uint16=0) -> None
        
        Set the value to be a number after executing an insn. 
                
        
        # 1: set_num(rvals: uvalvec_t *, insn: const insn_t &) -> None
        
        Set the value to be numbers after executing an insn. 
                
        
        # 2: set_num(rval: int, val_ea: ida_idaapi.ea_t, val_flags: uint16=0) -> None
        
        Set the value to be a number before an address. 
                
        
        """
        ...
    def set_unkfunc(self, func_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown from the function start. 
                
        """
        ...
    def set_unkinsn(self, insn: insn_t) -> None:
        r"""Set the value to be unknown after executing the insn. 
                
        """
        ...
    def set_unkloop(self, bblk_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown because it changes in a loop. 
                
        """
        ...
    def set_unkmult(self, bblk_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown because the register has incompatible values. 
                
        """
        ...
    def set_unkvals(self, bblk_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown because the register has too many values. 
                
        """
        ...
    def set_unkxref(self, bblk_ea: ida_idaapi.ea_t) -> None:
        r"""Set the value to be unknown because there are too many xrefs. 
                
        """
        ...
    def shift_left(self, r: int) -> None:
        r"""Shift the value left by R, do not change the defining instructions. 
                
        """
        ...
    def shift_right(self, r: int) -> None:
        r"""Shift the value right by R, do not change the defining instructions. 
                
        """
        ...
    def sll(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Shift the value left by R, save INSN as a defining instruction. 
                
        """
        ...
    def slr(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Shift logically the value right by R, save INSN as a defining instruction. 
                
        """
        ...
    def sub(self, r: reg_value_info_t, insn: insn_t) -> None:
        r"""Subtract R from the value, save INSN as a defining instruction. 
                
        """
        ...
    def swap(self, r: reg_value_info_t) -> None:
        ...
    def trunc_uval(self, pm: procmod_t) -> None:
        r"""Truncate the number to the application bitness. 
                
        """
        ...
    def vals_union(self, r: reg_value_info_t) -> reg_value_info_t:
        r"""Add values from R into THIS ignoring duplicates. 
                
        :returns: EQUAL: THIS is not changed
        :returns: CONTAINS: THIS is not changed
        :returns: CONTAINED: THIS is a copy of R
        :returns: NOT_COMPARABLE: values from R are added to THIS
        """
        ...

def find_nearest_rvi(rvi: reg_value_info_t, ea: ida_idaapi.ea_t, reg: Any) -> int:
    r"""Find the value of any of the two registers using the register tracker. First, this function tries to find the registers in the basic block of EA, and if it could not do this, then it tries to find in the entire function. 
            
    :param rvi: the found value with additional attributes
    :param ea: the address to find a value at
    :param reg: the registers to find
    :returns: the index of the found register or -1
    """
    ...

def find_reg_value(ea: ida_idaapi.ea_t, reg: int) -> uint64:
    r"""Find register value using the register tracker. 
            
    :param ea: the address to find a value at
    :param reg: the register to find
    :returns: 0: no value (the value is varying or the find depth is not enough to find a value)
    :returns: 1: the found value is in VAL
    :returns: -1: the processor module does not support a register tracker
    """
    ...

def find_reg_value_info(rvi: reg_value_info_t, ea: ida_idaapi.ea_t, reg: int, max_depth: int = 0) -> bool:
    r"""Find register value using the register tracker. 
            
    :param rvi: the found value with additional attributes
    :param ea: the address to find a value at
    :param reg: the register to find
    :param max_depth: the number of basic blocks to look before aborting the search and returning the unknown value. 0 means the value of REGTRACK_MAX_DEPTH from ida.cfg for ordinal registers or REGTRACK_FUNC_MAX_DEPTH for the function-wide registers, -1 means the value of REGTRACK_FUNC_MAX_DEPTH from ida.cfg.
    :returns: 'false': the processor module does not support a register tracker
    :returns: 'true': the found value is in RVI
    """
    ...

def find_sp_value(ea: ida_idaapi.ea_t, reg: int = -1) -> int64:
    r"""Find a value of the SP based register using the register tracker. 
            
    :param ea: the address to find a value at
    :param reg: the register to find. by default the SP register is used.
    :returns: 0: no value (the value is varying or the find depth is not enough to find a value)
    :returns: 1: the found value is in VAL
    :returns: -1: the processor module does not support a register tracker
    """
    ...

def invalidate_regfinder_cache(args: Any) -> None:
    r"""The control flow from FROM to TO has removed (CREF==fl_U) or added (CREF!=fl_U). Try to update the register tracker cache after this change. If TO == BADADDR then clear the entire cache. 
            
    """
    ...

def invalidate_regfinder_xrefs_cache(args: Any) -> None:
    r"""The data reference to TO has added (DREF!=dr_O) or removed (DREF==dr_O). Update the regtracker xrefs cache after this change. If TO == BADADDR then clear the entire xrefs cache. 
            
    """
    ...

SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
cvar: swigvarlink
ida_idaapi: module
weakref: module