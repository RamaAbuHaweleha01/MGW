from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that work with the autoanalyzer queue.

The autoanalyzer works when IDA is not busy processing the user keystrokes. It has several queues, each queue having its own priority. The analyzer stops when all queues are empty.
A queue contains addresses or address ranges. The addresses are kept sorted by their values. The analyzer will process all addresses from the first queue, then switch to the second queue and so on. There are no limitations on the size of the queues.
This file also contains functions that deal with the IDA status indicator and the autoanalysis indicator. You may use these functions to change the indicator value. 
    
"""

class auto_display_t:
    @property
    def ea(self) -> Any: ...
    @property
    def state(self) -> Any: ...
    @property
    def type(self) -> Any: ...
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

def auto_apply_tail(tail_ea: ida_idaapi.ea_t, parent_ea: ida_idaapi.ea_t) -> None:
    r"""Plan to apply the tail_ea chunk to the parent 
            
    :param tail_ea: linear address of start of tail
    :param parent_ea: linear address within parent. If BADADDR, automatically try to find parent via xrefs.
    """
    ...

def auto_apply_type(caller: ida_idaapi.ea_t, callee: ida_idaapi.ea_t) -> None:
    r"""Plan to apply the callee's type to the calling point.
    
    """
    ...

def auto_cancel(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> None:
    r"""Remove an address range (ea1..ea2) from queues AU_CODE, AU_PROC, AU_USED. To remove an address range from other queues use auto_unmark() function. 'ea1' may be higher than 'ea2', the kernel will swap them in this case. 'ea2' doesn't belong to the range. 
            
    """
    ...

def auto_get(type: atype_t, lowEA: ida_idaapi.ea_t, highEA: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Retrieve an address from queues regarding their priority. Returns BADADDR if no addresses not lower than 'lowEA' and less than 'highEA' are found in the queues. Otherwise *type will have queue type. 
            
    """
    ...

def auto_is_ok() -> bool:
    r"""Are all queues empty? (i.e. has autoanalysis finished?). 
            
    """
    ...

def auto_make_code(ea: ida_idaapi.ea_t) -> None:
    r"""Plan to make code.
    
    """
    ...

def auto_make_proc(ea: ida_idaapi.ea_t) -> None:
    r"""Plan to make code&function.
    
    """
    ...

def auto_make_step(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> bool:
    r"""Analyze one address in the specified range and return true. 
            
    :returns: if processed anything. false means that there is nothing to process in the specified range.
    """
    ...

def auto_mark(ea: ida_idaapi.ea_t, type: atype_t) -> None:
    r"""Put single address into a queue. Queues keep addresses sorted.
    
    """
    ...

def auto_mark_range(start: ida_idaapi.ea_t, end: ida_idaapi.ea_t, type: atype_t) -> None:
    r"""Put range of addresses into a queue. 'start' may be higher than 'end', the kernel will swap them in this case. 'end' doesn't belong to the range. 
            
    """
    ...

def auto_postpone_analysis(ea: ida_idaapi.ea_t) -> bool:
    r"""Plan to reanalyze on the second pass The typical usage of this function in emu.cpp is: if ( !auto_postpone_analysis(ea) ) op_offset(ea, 0, ...); (we make an offset only on the second pass) 
            
    """
    ...

def auto_recreate_insn(ea: ida_idaapi.ea_t) -> int:
    r"""Try to create instruction 
            
    :param ea: linear address of callee
    :returns: the length of the instruction or 0
    """
    ...

def auto_unmark(start: ida_idaapi.ea_t, end: ida_idaapi.ea_t, type: atype_t) -> None:
    r"""Remove range of addresses from a queue. 'start' may be higher than 'end', the kernel will swap them in this case. 'end' doesn't belong to the range. 
            
    """
    ...

def auto_wait() -> bool:
    r"""Process everything in the queues and return true. 
            
    :returns: false if the user clicked cancel. (the wait box must be displayed by the caller if desired)
    """
    ...

def auto_wait_range(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> ssize_t:
    r"""Process everything in the specified range and return true. 
            
    :returns: number of autoanalysis steps made. -1 if the user clicked cancel. (the wait box must be displayed by the caller if desired)
    """
    ...

def enable_auto(enable: bool) -> bool:
    r"""Temporarily enable/disable autoanalyzer. Not user-facing, but rather because IDA sometimes need to turn AA on/off regardless of inf.s_genflags:INFFL_AUTO 
            
    :returns: old state
    """
    ...

def get_auto_display(auto_display: auto_display_t) -> bool:
    r"""Get structure which holds the autoanalysis indicator contents.
    
    """
    ...

def get_auto_state() -> atype_t:
    r"""Get current state of autoanalyzer. If auto_state == AU_NONE, IDA is currently not running the analysis (it could be temporarily interrupted to perform the user's requests, for example). 
            
    """
    ...

def is_auto_enabled() -> bool:
    r"""Get autoanalyzer state.
    
    """
    ...

def may_create_stkvars() -> bool:
    r"""Is it allowed to create stack variables automatically?. This function should be used by IDP modules before creating stack vars. 
            
    """
    ...

def may_trace_sp() -> bool:
    r"""Is it allowed to trace stack pointer automatically?. This function should be used by IDP modules before tracing sp. 
            
    """
    ...

def peek_auto_queue(low_ea: ida_idaapi.ea_t, type: atype_t) -> ida_idaapi.ea_t:
    r"""Peek into a queue 'type' for an address not lower than 'low_ea'. Do not remove address from the queue. 
            
    :returns: the address or BADADDR
    """
    ...

def plan_and_wait(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, final_pass: bool = True) -> int:
    r"""Analyze the specified range. Try to create instructions where possible. Make the final pass over the specified range if specified. This function doesn't return until the range is analyzed. 
            
    :returns: 1: ok
    :returns: 0: Ctrl-Break was pressed
    """
    ...

def plan_ea(ea: ida_idaapi.ea_t) -> None:
    r"""Plan to perform reanalysis.
    
    """
    ...

def plan_range(sEA: ida_idaapi.ea_t, eEA: ida_idaapi.ea_t) -> None:
    r"""Plan to perform reanalysis.
    
    """
    ...

def reanalyze_callers(ea: ida_idaapi.ea_t, noret: bool) -> None:
    r"""Plan to reanalyze callers of the specified address. This function will add to AU_USED queue all instructions that call (not jump to) the specified address. 
            
    :param ea: linear address of callee
    :param noret: !=0: the callee doesn't return, mark to undefine subsequent instructions in the caller. 0: do nothing.
    """
    ...

def revert_ida_decisions(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> None:
    r"""Delete all analysis info that IDA generated for for the given range.
    
    """
    ...

def set_auto_state(new_state: atype_t) -> atype_t:
    r"""Set current state of autoanalyzer. 
            
    :param new_state: new state of autoanalyzer
    :returns: previous state
    """
    ...

def set_ida_state(st: idastate_t) -> idastate_t:
    r"""Change IDA status indicator value 
            
    :param st: - new indicator status
    :returns: old indicator status
    """
    ...

def show_addr(ea: ida_idaapi.ea_t) -> None:
    r"""Show an address on the autoanalysis indicator. The address is displayed in the form " @:12345678". 
            
    :param ea: - linear address to display
    """
    ...

def show_auto(args: Any) -> None:
    r"""Change autoanalysis indicator value. 
            
    :param ea: linear address being analyzed
    :param type: autoanalysis type (see Autoanalysis queues)
    """
    ...

AU_CHLB: int  # 90
AU_CODE: int  # 20
AU_FCHUNK: int  # 38
AU_FINAL: int  # 200
AU_LBF2: int  # 70
AU_LBF3: int  # 80
AU_LIBF: int  # 60
AU_NONE: int  # 0
AU_PROC: int  # 30
AU_TAIL: int  # 35
AU_TYPE: int  # 50
AU_UNK: int  # 10
AU_USD2: int  # 45
AU_USED: int  # 40
AU_WEAK: int  # 25
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
cvar: swigvarlink
ida_idaapi: module
st_Ready: int  # 0
st_Think: int  # 1
st_Waiting: int  # 2
st_Work: int  # 3
weakref: module