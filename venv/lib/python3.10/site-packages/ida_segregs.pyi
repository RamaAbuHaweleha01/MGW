from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with the segment registers.

If your processor doesn't use segment registers, then these functions are of no use for you. However, you should define two virtual segment registers - CS and DS (for code segment and data segment) and specify their internal numbers in the LPH structure (processor_t::reg_code_sreg and processor_t::reg_data_sreg). 
    
"""

class sreg_range_t:
    @property
    def end_ea(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def tag(self) -> Any: ...
    @property
    def val(self) -> Any: ...
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
    def __init__(self) -> Any:
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

def copy_sreg_ranges(dst_rg: int, src_rg: int, map_selector: bool = False) -> None:
    r"""Duplicate segment register ranges. 
            
    :param dst_rg: number of destination segment register
    :param src_rg: copy ranges from
    :param map_selector: map selectors to linear addresses using sel2ea()
    """
    ...

def del_sreg_range(ea: ida_idaapi.ea_t, rg: int) -> bool:
    r"""Delete segment register range started at ea. When a segment register range is deleted, the previous range is extended to cover the empty space. The segment register range at the beginning of a segment cannot be deleted. 
            
    :param ea: start_ea of the deleted range
    :param rg: the segment register number
    :returns: success
    """
    ...

def get_prev_sreg_range(out: sreg_range_t, ea: ida_idaapi.ea_t, rg: int) -> bool:
    r"""Get segment register range previous to one with address. 
            
    :param out: segment register range
    :param ea: any linear address in the program
    :param rg: the segment register number
    :returns: success
    """
    ...

def get_sreg(ea: ida_idaapi.ea_t, rg: int) -> sel_t:
    r"""Get value of a segment register. This function uses segment register range and default segment register values stored in the segment structure. 
            
    :param ea: linear address in the program
    :param rg: number of the segment register
    :returns: value of the segment register, BADSEL if value is unknown or rg is not a segment register.
    """
    ...

def get_sreg_range(out: sreg_range_t, ea: ida_idaapi.ea_t, rg: int) -> bool:
    r"""Get segment register range by linear address. 
            
    :param out: segment register range
    :param ea: any linear address in the program
    :param rg: the segment register number
    :returns: success
    """
    ...

def get_sreg_range_num(ea: ida_idaapi.ea_t, rg: int) -> int:
    r"""Get number of segment register range by address. 
            
    :param ea: any address in the range
    :param rg: the segment register number
    :returns: -1 if no range occupies the specified address. otherwise returns number of the specified range (0..get_srranges_qty()-1)
    """
    ...

def get_sreg_ranges_qty(rg: int) -> int:
    r"""Get number of segment register ranges. 
            
    :param rg: the segment register number
    """
    ...

def getn_sreg_range(out: sreg_range_t, rg: int, n: int) -> bool:
    r"""Get segment register range by its number. 
            
    :param out: segment register range
    :param rg: the segment register number
    :param n: number of range (0..qty()-1)
    :returns: success
    """
    ...

def set_default_dataseg(ds_sel: sel_t) -> None:
    r"""Set default value of DS register for all segments.
    
    """
    ...

def set_default_sreg_value(sg: segment_t, rg: int, value: sel_t) -> bool:
    r"""Set default value of a segment register for a segment. 
            
    :param sg: pointer to segment structure if nullptr, then set the register for all segments
    :param rg: number of segment register
    :param value: its default value. this value will be used by get_sreg() if value of the register is unknown at the specified address.
    :returns: success
    """
    ...

def set_sreg_at_next_code(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, rg: int, value: sel_t) -> None:
    r"""Set the segment register value at the next instruction. This function is designed to be called from idb_event::sgr_changed handler in order to contain the effect of changing a segment register value only until the next instruction.
    It is useful, for example, in the ARM module: the modification of the T register does not affect existing instructions later in the code. 
            
    :param ea1: address to start to search for an instruction
    :param ea2: the maximal address
    :param rg: the segment register number
    :param value: the segment register value
    """
    ...

def split_sreg_range(ea: ida_idaapi.ea_t, rg: int, v: sel_t, tag: uchar, silent: bool = False) -> bool:
    r"""Create a new segment register range. This function is used when the IDP emulator detects that a segment register changes its value. 
            
    :param ea: linear address where the segment register will have a new value. if ea==BADADDR, nothing to do.
    :param rg: the number of the segment register
    :param v: the new value of the segment register. If the value is unknown, you should specify BADSEL.
    :param tag: the register info tag. see Segment register range tags
    :param silent: if false, display a warning() in the case of failure
    :returns: success
    """
    ...

R_cs: int  # 30
R_ds: int  # 32
R_es: int  # 29
R_fs: int  # 33
R_gs: int  # 34
R_ss: int  # 31
SR_auto: int  # 3
SR_autostart: int  # 4
SR_inherit: int  # 1
SR_user: int  # 2
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
ida_range: module
weakref: module