from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""There are 2 representations of the binary code in the decompiler:

Hex-Rays Decompiler project Copyright (c) 1990-2025 Hex-Rays ALL RIGHTS RESERVED.

* microcode: processor instructions are translated into it and then the decompiler optimizes and transforms it
* ctree: ctree is built from the optimized microcode and represents AST-like tree with C statements and expressions. It can be printed as C code.


Microcode is represented by the following classes:
* mba_t keeps general info about the decompiled code and array of basic blocks. usually mba_t is named 'mba'
* mblock_t a basic block. includes list of instructions
* minsn_t an instruction. contains 3 operands: left, right, and destination
* mop_t an operand. depending on its type may hold various info like a number, register, stack variable, etc.
* mlist_t list of memory or register locations; can hold vast areas of memory and multiple registers. this class is used very extensively in the decompiler. it may represent list of locations accessed by an instruction or even an entire basic block. it is also used as argument of many functions. for example, there is a function that searches for an instruction that refers to a mlist_t.


See [https://hex-rays.com/blog/microcode-in-pictures](https://hex-rays.com/blog/microcode-in-pictures) for a few pictures.
Ctree is represented by:
* cfunc_t keeps general info about the decompiled code, including a pointer to mba_t. deleting cfunc_t will delete mba_t too (however, decompiler returns cfuncptr_t, which is a reference counting object and deletes the underlying function as soon as all references to it go out of scope). cfunc_t has 'body', which represents the decompiled function body as cinsn_t.
* cinsn_t a C statement. can be a compound statement or any other legal C statements (like if, for, while, return, expression-statement, etc). depending on the statement type has pointers to additional info. for example, the 'if' statement has poiner to cif_t, which holds the 'if' condition, 'then' branch, and optionally 'else' branch. Please note that despite of the name cinsn_t we say "statements", not "instructions". For us instructions are part of microcode, not ctree.
* cexpr_t a C expression. is used as part of a C statement, when necessary. cexpr_t has 'type' field, which keeps the expression type.
* citem_t a base class for cinsn_t and cexpr_t, holds common info like the address, label, and opcode.
* cnumber_t a constant 64-bit number. in addition to its value also holds information how to represent it: decimal, hex, or as a symbolic constant (enum member). please note that numbers are represented by another class (mnumber_t) in microcode.


See [https://hex-rays.com/blog/hex-rays-decompiler-primer](https://hex-rays.com/blog/hex-rays-decompiler-primer) for more pictures and more details.
Both microcode and ctree use the following class:
* lvar_t a local variable. may represent a stack or register variable. a variable has a name, type, location, etc. the list of variables is stored in mba->vars.
* lvar_locator_t holds a variable location (vdloc_t) and its definition address.
* vdloc_t describes a variable location, like a register number, a stack offset, or, in complex cases, can be a mix of register and stack locations. very similar to argloc_t, which is used in ida. the differences between argloc_t and vdloc_t are:
* vdloc_t never uses ARGLOC_REG2
* vdloc_t uses micro register numbers instead of processor register numbers
* the stack offsets are never negative in vdloc_t, while in argloc_t there can be negative offsets




The above are the most important classes in this header file. There are many auxiliary classes, please see their definitions in the header file.
See also the description of Virtual Machine used by Microcode. 
    
"""

class DecompilationFailure(Exception):
    args: getset_descriptor  # <attribute 'args' of 'BaseException' objects>
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        r"""Initialize self.  See help(type(self)) for accurate signature."""
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __reduce__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        r"""Return repr(self)."""
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setstate__(self, *args: Any, **kwargs: Any) -> Any:
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
    def with_traceback(self, *args: Any, **kwargs: Any) -> Any:
        r"""Exception.with_traceback(tb) --
            set self.__traceback__ to tb and return self.
        """
        ...

class Hexrays_Hooks:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _flags: int = 0, _hkcb_flags: int = 1) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def begin_inlining(self, cdg: codegen_t, decomp_flags: int) -> int:
        r"""Starting to inline outlined functions. 
                  
        :param cdg: (codegen_t *)
        :param decomp_flags: (int)
        :returns: Microcode error code This is an opportunity to inline other ranges.
        """
        ...
    def build_callinfo(self, blk: mblock_t, type: tinfo_t) -> Any:
        r"""Analyzing a call instruction. 
                  
        :param blk: (mblock_t *) blk->tail is the call.
        :param type: (tinfo_t *) buffer for the output type.
        """
        ...
    def callinfo_built(self, blk: mblock_t) -> bool:
        r"""A call instruction has been anallyzed. 
                  
        :param blk: (mblock_t *) blk->tail is the call.
        """
        ...
    def calls_done(self, mba: mba_t) -> int:
        r"""All calls have been analyzed. 
                  
        :param mba: (mba_t *) This event is generated immediately after analyzing all calls, before any optimizitions, call unmerging and block merging.
        """
        ...
    def close_pseudocode(self, vu: vdui_t) -> bool:
        r"""Pseudocode view is being closed. 
                  
        :param vu: (vdui_t *)
        """
        ...
    def cmt_changed(self, cfunc: cfunc_t, loc: treeloc_t, cmt: str) -> int:
        r"""Comment got changed. 
                  
        :param cfunc: (cfunc_t *)
        :param loc: (const treeloc_t *)
        :param cmt: (const char *)
        """
        ...
    def collect_warnings(self, cfunc: cfunc_t) -> int:
        r"""Collect warning messages from plugins. These warnings will be displayed at the function header, after the user-defined comments. 
                  
        :param cfunc: (cfunc_t *)
        """
        ...
    def combine(self, blk: mblock_t, insn: minsn_t) -> int:
        r"""Trying to combine instructions of basic block. 
                  
        :param blk: (mblock_t *)
        :param insn: (minsn_t *) Should return: 1 if combined the current instruction with a preceding one -1 if the instruction should not be combined 0 else
        """
        ...
    def create_hint(self, vu: vdui_t) -> Any:
        r"""Create a hint for the current item. 
                  
        :param vu: (vdui_t *)
        :returns: 0: continue collecting hints with other subscribers
        :returns: 1: stop collecting hints
        """
        ...
    def curpos(self, vu: vdui_t) -> int:
        r"""Current cursor position has been changed. (for example, by left-clicking or using keyboard)
        
                  
        :param vu: (vdui_t *)
        """
        ...
    def double_click(self, vu: vdui_t, shift_state: int) -> int:
        r"""Mouse double click. 
                  
        :param vu: (vdui_t *)
        :param shift_state: (int) Should return: 1 if the event has been handled
        """
        ...
    def flowchart(self, fc: qflow_chart_t, mba: mba_t, reachable_blocks: bitset_t, decomp_flags: int) -> int:
        r"""Flowchart has been generated. 
                  
        :param fc: (qflow_chart_t *)
        :param mba: (mba_t *)
        :param reachable_blocks: (bitset_t *)
        :param decomp_flags: (int)
        :returns: Microcode error code
        """
        ...
    def func_printed(self, cfunc: cfunc_t) -> int:
        r"""Function text has been generated. Plugins may modify the text in cfunc_t::sv. However, it is too late to modify the ctree or microcode. The text uses regular color codes (see lines.hpp) COLOR_ADDR is used to store pointers to ctree items. 
                  
        :param cfunc: (cfunc_t *)
        """
        ...
    def glbopt(self, mba: mba_t) -> int:
        r"""Global optimization has been finished. If microcode is modified, MERR_LOOP must be returned. It will cause a complete restart of the optimization. 
                  
        :param mba: (mba_t *)
        :returns: Microcode error code
        """
        ...
    def hook(self) -> bool:
        ...
    def inlined_func(self, cdg: codegen_t, blk: int, mbr: mba_ranges_t, i1: int, i2: int) -> int:
        r"""A set of ranges got inlined. 
                  
        :param cdg: (codegen_t *)
        :param blk: (int) the block containing call/jump to inline
        :param mbr: (mba_ranges_t *) the range to inline
        :param i1: (int) blknum of the first inlined block
        :param i2: (int) blknum of the last inlined block (excluded)
        """
        ...
    def inlining_func(self, cdg: codegen_t, blk: int, mbr: mba_ranges_t) -> int:
        r"""A set of ranges is going to be inlined. 
                  
        :param cdg: (codegen_t *)
        :param blk: (int) the block containing call/jump to inline
        :param mbr: (mba_ranges_t *) the range to inline
        """
        ...
    def interr(self, errcode: int) -> int:
        r"""Internal error has occurred. 
                  
        :param errcode: (int )
        """
        ...
    def keyboard(self, vu: vdui_t, key_code: int, shift_state: int) -> int:
        r"""Keyboard has been hit. 
                  
        :param vu: (vdui_t *)
        :param key_code: (int) VK_...
        :param shift_state: (int) Should return: 1 if the event has been handled
        """
        ...
    def locopt(self, mba: mba_t) -> int:
        r"""Basic block level optimization has been finished. 
                  
        :param mba: (mba_t *)
        :returns: Microcode error code
        """
        ...
    def lvar_cmt_changed(self, vu: vdui_t, v: lvar_t, cmt: str) -> int:
        r"""Local variable comment got changed. 
                  
        :param vu: (vdui_t *)
        :param v: (lvar_t *)
        :param cmt: (const char *) Please note that it is possible to read/write user settings for lvars directly from the idb.
        """
        ...
    def lvar_mapping_changed(self, vu: vdui_t, frm: lvar_t, to: lvar_t) -> int:
        r"""Local variable mapping got changed. 
                  
        :param vu: (vdui_t *)
        :param to: (lvar_t *) Please note that it is possible to read/write user settings for lvars directly from the idb.
        """
        ...
    def lvar_name_changed(self, vu: vdui_t, v: lvar_t, name: str, is_user_name: bool) -> int:
        r"""Local variable got renamed. 
                  
        :param vu: (vdui_t *)
        :param v: (lvar_t *)
        :param name: (const char *)
        :param is_user_name: (bool) Please note that it is possible to read/write user settings for lvars directly from the idb.
        """
        ...
    def lvar_type_changed(self, vu: vdui_t, v: lvar_t, tinfo: tinfo_t) -> int:
        r"""Local variable type got changed. 
                  
        :param vu: (vdui_t *)
        :param v: (lvar_t *)
        :param tinfo: (const tinfo_t *) Please note that it is possible to read/write user settings for lvars directly from the idb.
        """
        ...
    def maturity(self, cfunc: cfunc_t, new_maturity: ctree_maturity_t) -> int:
        r"""Ctree maturity level is being changed. 
                  
        :param cfunc: (cfunc_t *)
        :param new_maturity: (ctree_maturity_t)
        """
        ...
    def mba_maturity(self, mba: mba_t, reqmat: mba_maturity_t) -> int:
        r"""Maturity level of an MBA was changed. 
                  
        :param mba: (mba_t *)
        :param reqmat: (mba_maturity_t) requested maturity level
        :returns: Microcode error code
        """
        ...
    def microcode(self, mba: mba_t) -> int:
        r"""Microcode has been generated. 
                  
        :param mba: (mba_t *)
        :returns: Microcode error code
        """
        ...
    def open_pseudocode(self, vu: vdui_t) -> vdui_t:
        r"""New pseudocode view has been opened. 
                  
        :param vu: (vdui_t *)
        """
        ...
    def populating_popup(self, widget: TWidget, popup_handle: TPopupMenu, vu: vdui_t) -> int:
        r"""Populating popup menu. We can add menu items now. 
                  
        :param widget: (TWidget *)
        :param popup_handle: (TPopupMenu *)
        :param vu: (vdui_t *)
        """
        ...
    def pre_structural(self, ct: control_graph_t, cfunc: cfunc_t, g: simple_graph_t) -> int:
        r"""Structure analysis is starting. 
                  
        :param ct: (control_graph_t *) in/out: control graph
        :param cfunc: (cfunc_t *) in: the current function
        :param g: (const simple_graph_t *) in: control flow graph
        :returns: Microcode error code; MERR_BLOCK means that the analysis has been performed by a plugin
        """
        ...
    def prealloc(self, mba: mba_t) -> int:
        r"""Local variables: preallocation step begins. 
                  
        :param mba: (mba_t *) This event may occur several times. Should return: 1 if modified microcode Negative values are Microcode error code
        """
        ...
    def preoptimized(self, mba: mba_t) -> int:
        r"""Microcode has been preoptimized. 
                  
        :param mba: (mba_t *)
        :returns: Microcode error code
        """
        ...
    def print_func(self, cfunc: cfunc_t, vp: vc_printer_t) -> None:
        r"""Printing ctree and generating text. 
                  
        :param cfunc: (cfunc_t *)
        :param vp: (vc_printer_t *) Returns: 1 if text has been generated by the plugin It is forbidden to modify ctree at this event.
        """
        ...
    def prolog(self, mba: mba_t, fc: qflow_chart_t, reachable_blocks: bitset_t, decomp_flags: int) -> int:
        r"""Prolog analysis has been finished. 
                  
        :param mba: (mba_t *)
        :param fc: (qflow_chart_t *)
        :param reachable_blocks: (const bitset_t *)
        :param decomp_flags: (int)
        :returns: Microcode error code This event is generated for each inlined range as well.
        """
        ...
    def refresh_pseudocode(self, vu: vdui_t) -> int:
        r"""Existing pseudocode text has been refreshed. Adding/removing pseudocode lines is forbidden in this event. 
                  
        :param vu: (vdui_t *) See also hxe_text_ready, which happens earlier
        """
        ...
    def resolve_stkaddrs(self, mba: mba_t) -> int:
        r"""The optimizer is about to resolve stack addresses. 
                  
        :param mba: (mba_t *)
        """
        ...
    def right_click(self, vu: vdui_t) -> int:
        r"""Mouse right click. Use hxe_populating_popup instead, in case you want to add items in the popup menu. 
                  
        :param vu: (vdui_t *)
        """
        ...
    def stkpnts(self, mba: mba_t, _sps: stkpnts_t) -> int:
        r"""SP change points have been calculated. 
                  
        :param mba: (mba_t *)
        :returns: Microcode error code This event is generated for each inlined range as well.
        """
        ...
    def structural(self, ct: control_graph_t) -> int:
        r"""Structural analysis has been finished. 
                  
        :param ct: (control_graph_t *)
        """
        ...
    def switch_pseudocode(self, vu: vdui_t) -> int:
        r"""Existing pseudocode view has been reloaded with a new function. Its text has not been refreshed yet, only cfunc and mba pointers are ready. 
                  
        :param vu: (vdui_t *)
        """
        ...
    def text_ready(self, vu: vdui_t) -> int:
        r"""Decompiled text is ready. 
                  
        :param vu: (vdui_t *) This event can be used to modify the output text (sv). Obsolete. Please use hxe_func_printed instead.
        """
        ...
    def unhook(self) -> bool:
        ...

class array_of_bitsets:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: array_of_bitsets) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: array_of_bitsets) -> bool:
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
    def __setitem__(self, i: size_t, v: bitset_t) -> None:
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
    def add_unique(self, x: bitset_t) -> bool:
        ...
    def append(self, x: bitset_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: array_of_bitsets) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: bitset_t) -> bool:
        ...
    def inject(self, s: bitset_t, len: size_t) -> None:
        ...
    def insert(self, it: bitset_t, x: bitset_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: array_of_bitsets) -> None:
        ...
    def truncate(self) -> None:
        ...

class array_of_ivlsets:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: array_of_ivlsets) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: array_of_ivlsets) -> bool:
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
    def __setitem__(self, i: size_t, v: ivlset_t) -> None:
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
    def add_unique(self, x: ivlset_t) -> bool:
        ...
    def append(self, x: ivlset_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: array_of_ivlsets) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: ivlset_t) -> bool:
        ...
    def inject(self, s: ivlset_t, len: size_t) -> None:
        ...
    def insert(self, it: ivlset_t, x: ivlset_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: array_of_ivlsets) -> None:
        ...
    def truncate(self) -> None:
        ...

class array_of_node_bitset_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class bit_bound_t:
    @property
    def nbits(self) -> Any: ...
    @property
    def sbits(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, n: int = 0, s: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class bitset_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: bitset_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: bitset_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: bitset_t) -> bool:
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
        ...
    def __le__(self, r: bitset_t) -> bool:
        ...
    def __len__(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def __lt__(self, r: bitset_t) -> bool:
        ...
    def __ne__(self, r: bitset_t) -> bool:
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
    def add(self, args: Any) -> voff_t:
        r"""This function has the following signatures:
        
            0. add(bit: int) -> bool
            1. add(bit: int, width: int) -> bool
            2. add(ml: const bitset_t &) -> bool
        
        # 0: add(bit: int) -> bool
        
        
        # 1: add(bit: int, width: int) -> bool
        
        
        # 2: add(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: bitset_t) -> int:
        ...
    def copy(self, m: bitset_t) -> bitset_t:
        ...
    def count(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def cut_at(self, maxbit: int) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def fill_with_ones(self, maxbit: int) -> None:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, bit: int) -> bool:
        ...
    def has_all(self, bit: int, width: int) -> bool:
        ...
    def has_any(self, bit: int, width: int) -> bool:
        ...
    def has_common(self, ml: bitset_t) -> bool:
        ...
    def inc(self, p: iterator, n: int = 1) -> None:
        ...
    def includes(self, ml: bitset_t) -> bool:
        ...
    def intersect(self, ml: bitset_t) -> int:
        ...
    def is_subset_of(self, ml: bitset_t) -> bool:
        ...
    def itat(self, n: int) -> bitset_t:
        ...
    def itv(self, it: iterator) -> int:
        ...
    def last(self) -> int:
        ...
    def shift_down(self, shift: int) -> None:
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(bit: int) -> bool
            1. sub(bit: int, width: int) -> bool
            2. sub(ml: const bitset_t &) -> bool
        
        # 0: sub(bit: int) -> bool
        
        
        # 1: sub(bit: int, width: int) -> bool
        
        
        # 2: sub(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def swap(self, r: bitset_t) -> None:
        ...

class block_chains_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: block_chains_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: block_chains_iterator_t) -> bool:
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

class block_chains_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def dstr(self) -> str:
        ...
    def get_chain(self, args: Any) -> chain_t:
        r"""This function has the following signatures:
        
            0. get_chain(k: const voff_t &, width: int=1) -> const chain_t *
            1. get_chain(k: const voff_t &, width: int=1) -> chain_t *
            2. get_chain(ch: const chain_t &) -> const chain_t *
            3. get_chain(ch: const chain_t &) -> chain_t *
        
        # 0: get_chain(k: const voff_t &, width: int=1) -> const chain_t *
        
        Get chain for the specified value offset. 
                
        
        # 1: get_chain(k: const voff_t &, width: int=1) -> chain_t *
        
        
        # 2: get_chain(ch: const chain_t &) -> const chain_t *
        
        Get chain similar to the specified chain 
                
        
        # 3: get_chain(ch: const chain_t &) -> chain_t *
        
        
        """
        ...
    def get_reg_chain(self, reg: mreg_t, width: int = 1) -> chain_t:
        r"""Get chain for the specified register 
                
        :param reg: register number
        :param width: size of register in bytes
        """
        ...
    def get_stk_chain(self, off: int, width: int = 1) -> chain_t:
        r"""Get chain for the specified stack offset 
                
        :param off: stack offset
        :param width: size of stack value in bytes
        """
        ...

class block_chains_vec_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __ne__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: block_chains_t) -> None:
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
    def append(self, x: block_chains_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: block_chains_vec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: block_chains_t, len: size_t) -> None:
        ...
    def insert(self, it: block_chains_t, x: block_chains_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: block_chains_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class boundaries_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: boundaries_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: boundaries_iterator_t) -> bool:
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

class boundaries_t:
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: cinsn_t) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def keytype(self, args: Any) -> Any:
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, args: Any) -> Any:
        ...

class carg_t(cexpr_t, citem_t):
    op_to_typename: dict
    @property
    def a(self) -> Any: ...
    @property
    def cexpr(self) -> Any: ...
    @property
    def cinsn(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def exflags(self) -> Any: ...
    @property
    def formal_type(self) -> Any: ...
    @property
    def fpc(self) -> Any: ...
    @property
    def helper(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    @property
    def insn(self) -> Any: ...
    @property
    def is_vararg(self) -> Any: ...
    @property
    def label_num(self) -> Any: ...
    @property
    def m(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def n(self) -> Any: ...
    @property
    def obj_ea(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    @property
    def operands(self) -> Any: ...
    @property
    def opname(self) -> Any: ...
    @property
    def ptrsize(self) -> Any: ...
    @property
    def refwidth(self) -> Any: ...
    @property
    def string(self) -> Any: ...
    @property
    def to_specific_type(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def v(self) -> Any: ...
    @property
    def x(self) -> Any: ...
    @property
    def y(self) -> Any: ...
    @property
    def z(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: carg_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: carg_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: carg_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: carg_t) -> bool:
        ...
    def __lt__(self, r: carg_t) -> bool:
        ...
    def __ne__(self, r: carg_t) -> bool:
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
    def assign(self, r: cexpr_t) -> cinsn_t:
        ...
    def calc_type(self, recursive: bool) -> None:
        r"""Calculate the type of the expression. Use this function to calculate the expression type when a new expression is built 
                
        :param recursive: if true, types of all children expression will be calculated before calculating our type
        """
        ...
    def cleanup(self) -> None:
        r"""Cleanup the expression. This function properly deletes all children and sets the item type to cot_empty. 
                
        """
        ...
    def compare(self, r: carg_t) -> int:
        ...
    def consume_cexpr(self, e: cexpr_t) -> None:
        ...
    def contains_comma(self, times: int = 1) -> bool:
        r"""Does the expression contain a comma operator?
        
        """
        ...
    def contains_comma_or_insn_or_label(self, maxcommas: int = 1) -> bool:
        r"""Does the expression contain a comma operator or an embedded statement operator or a label?
        
        """
        ...
    def contains_expr(self, e: cexpr_t) -> bool:
        r"""Does the item contain an expression?
        
        """
        ...
    def contains_insn(self, times: int = 1) -> bool:
        r"""Does the expression contain an embedded statement operator?
        
        """
        ...
    def contains_insn_or_label(self) -> bool:
        r"""Does the expression contain an embedded statement operator or a label?
        
        """
        ...
    def contains_label(self) -> bool:
        r"""Does the item contain a label?
        
        """
        ...
    def contains_operator(self, needed_op: ctype_t, times: int = 1) -> bool:
        r"""Check if the expression contains the specified operator. 
                
        :param needed_op: operator code to search for
        :param times: how many times the operator code should be present
        :returns: true if the expression has at least TIMES children with NEEDED_OP
        """
        ...
    def cpadone(self) -> bool:
        r"""Pointer arithmetic correction done for this expression?
        
        """
        ...
    def dstr(self) -> str:
        ...
    def equal_effect(self, r: cexpr_t) -> bool:
        r"""Compare two expressions. This function tries to compare two expressions in an 'intelligent' manner. For example, it knows about commutitive operators and can ignore useless casts. 
                
        :param r: the expression to compare against the current expression
        :returns: true expressions can be considered equal
        """
        ...
    def find_closest_addr(self, _ea: ida_idaapi.ea_t) -> citem_t:
        ...
    def find_num_op(self) -> cexpr_t:
        r"""Find the operand with a numeric value.
        
        """
        ...
    def find_op(self, _op: ctype_t) -> cexpr_t:
        r"""Find the child with the specified operator.
        
        """
        ...
    def find_parent_of(self, item: citem_t) -> citem_t:
        r"""Find parent of the specified item. 
                
        :param item: Item to find the parent of. The search will be performed among the children of the item pointed by `this`.
        :returns: nullptr if not found
        """
        ...
    def get_1num_op(self, o1: cexpr_t, o2: cexpr_t) -> bool:
        r"""Get pointers to operands. at last one operand should be a number o1 will be pointer to the number 
                
        """
        ...
    def get_const_value(self) -> bool:
        r"""Get expression value. 
                
        :returns: true if the expression is a number.
        """
        ...
    def get_high_nbit_bound(self) -> bit_bound_t:
        r"""Get max number of bits that can really be used by the expression. For example, x % 16 can yield only 4 non-zero bits, higher bits are zero 
                
        """
        ...
    def get_low_nbit_bound(self) -> int:
        r"""Get min number of bits that are certainly required to represent the expression. For example, constant 16 always uses 5 bits: 10000. 
                
        """
        ...
    def get_ptr_or_array(self) -> cexpr_t:
        r"""Find pointer or array child.
        
        """
        ...
    def get_type_sign(self) -> type_sign_t:
        r"""Get expression sign.
        
        """
        ...
    def get_v(self) -> var_ref_t:
        ...
    def has_side_effects(self) -> bool:
        r"""Check if the expression has side effects. Calls, pre/post inc/dec, and assignments have side effects. 
                
        """
        ...
    def is_call_arg_of(self, parent: citem_t) -> bool:
        r"""Is call argument? 
                
        :returns: true if our expression is a call argument of the specified parent expression.
        """
        ...
    def is_call_object_of(self, parent: citem_t) -> bool:
        r"""Is call object? 
                
        :returns: true if our expression is the call object of the specified parent expression.
        """
        ...
    def is_child_of(self, parent: citem_t) -> bool:
        r"""Verify if the specified item is our parent. 
                
        :param parent: possible parent item
        :returns: true if the specified item is our parent
        """
        ...
    def is_const_value(self, _v: uint64) -> bool:
        r"""Check if the expression is a number with the specified value.
        
        """
        ...
    def is_cstr(self) -> bool:
        ...
    def is_expr(self) -> bool:
        r"""Is an expression?
        
        """
        ...
    def is_fpop(self) -> bool:
        ...
    def is_jumpout(self) -> bool:
        ...
    def is_negative_const(self) -> bool:
        r"""Check if the expression is a negative number.
        
        """
        ...
    def is_nice_cond(self) -> bool:
        r"""Is nice condition?. Nice condition is a nice expression of the boolean type. 
                
        """
        ...
    def is_nice_expr(self) -> bool:
        r"""Is nice expression? Nice expressions do not contain comma operators, embedded statements, or labels. 
                
        """
        ...
    def is_non_negative_const(self) -> bool:
        r"""Check if the expression is a non-negative number.
        
        """
        ...
    def is_non_zero_const(self) -> bool:
        r"""Check if the expression is a non-zero number.
        
        """
        ...
    def is_odd_lvalue(self) -> bool:
        ...
    def is_type_signed(self) -> bool:
        r"""Is expression signed?
        
        """
        ...
    def is_type_unsigned(self) -> bool:
        r"""Is expression unsigned?
        
        """
        ...
    def is_undef_val(self) -> bool:
        ...
    def is_vftable(self) -> bool:
        ...
    def is_zero_const(self) -> bool:
        r"""Check if the expression is a zero.
        
        """
        ...
    def maybe_ptr(self) -> bool:
        r"""May the expression be a pointer?
        
        """
        ...
    def numval(self) -> uint64:
        r"""Get numeric value of the expression. This function can be called only on cot_num expressions! 
                
        """
        ...
    def print1(self, func: cfunc_t) -> None:
        r"""Print expression into one line. 
                
        :param func: parent function. This argument is used to find out the referenced variable names.
        """
        ...
    def put_number(self, args: Any) -> None:
        r"""Assign a number to the expression. 
                
        :param func: current function
        :param value: number value
        :param nbytes: size of the number in bytes
        :param sign: number sign
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def requires_lvalue(self, child: cexpr_t) -> bool:
        r"""Check if the expression requires an lvalue. 
                
        :param child: The function will check if this child of our expression must be an lvalue.
        :returns: true if child must be an lvalue.
        """
        ...
    def set_cpadone(self) -> None:
        ...
    def set_v(self, v: var_ref_t) -> None:
        ...
    def set_vftable(self) -> None:
        ...
    def swap(self, r: cexpr_t) -> None:
        ...
    def theother(self, what: cexpr_t) -> cexpr_t:
        r"""Get the other operand. This function returns the other operand (not the specified one) for binary expressions. 
                
        """
        ...

class carglist_t(qvector_carg_t):
    @property
    def flags(self) -> Any: ...
    @property
    def functype(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: carglist_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: carglist_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: carglist_t) -> bool:
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
    def __le__(self, r: carglist_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: carglist_t) -> bool:
        ...
    def __ne__(self, r: carglist_t) -> bool:
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
    def __setitem__(self, i: size_t, v: carg_t) -> None:
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
    def add_unique(self, x: carg_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: carglist_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_carg_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: carg_t) -> bool:
        ...
    def inject(self, s: carg_t, len: size_t) -> None:
        ...
    def insert(self, it: carg_t, x: carg_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_carg_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class casm_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: casm_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: casm_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: casm_t) -> bool:
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
    def __le__(self, r: casm_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: casm_t) -> bool:
        ...
    def __ne__(self, r: casm_t) -> bool:
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
    def __setitem__(self, i: size_t, v: int) -> None:
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
    def add_unique(self, x: int) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: casm_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: uint64vec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, x: int) -> bool:
        ...
    def inject(self, s: int, len: size_t) -> None:
        ...
    def insert(self, it: iterator, x: int) -> qvector:
        ...
    def one_insn(self) -> bool:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: uint64vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class catchexpr_t:
    @property
    def fake_type(self) -> Any: ...
    @property
    def obj(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: catchexpr_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: catchexpr_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: catchexpr_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: catchexpr_t) -> bool:
        ...
    def __lt__(self, r: catchexpr_t) -> bool:
        ...
    def __ne__(self, r: catchexpr_t) -> bool:
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
    def compare(self, r: catchexpr_t) -> int:
        ...
    def is_catch_all(self) -> bool:
        ...
    def swap(self, r: catchexpr_t) -> None:
        ...

class cblock_pos_t:
    @property
    def blk(self) -> Any: ...
    @property
    def p(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def insn(self) -> cinsn_t:
        ...
    def is_first_insn(self) -> bool:
        ...
    def prev_insn(self) -> cinsn_t:
        ...

class cblock_posvec_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __ne__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: cblock_pos_t) -> None:
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
    def append(self, x: cblock_pos_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: cblock_posvec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: cblock_pos_t, len: size_t) -> None:
        ...
    def insert(self, it: cblock_pos_t, x: cblock_pos_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: cblock_posvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class cblock_t(cinsn_list_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cblock_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cblock_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: cblock_t) -> bool:
        ...
    def __init__(self) -> Any:
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
    def __le__(self, r: cblock_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: cblock_t) -> bool:
        ...
    def __ne__(self, r: cblock_t) -> bool:
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
    def __setitem__(self, i: size_t, v: cinsn_t) -> None:
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
    def at(self, index: Any) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: cblock_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def erase(self, p: cinsn_list_t_iterator) -> None:
        ...
    def find(self, item: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def index(self, item: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def pop_front(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def push_front(self, x: cinsn_t) -> None:
        ...
    def rbegin(self, args: Any) -> qlist:
        ...
    def remove(self, v: cinsn_t) -> bool:
        ...
    def rend(self, args: Any) -> qlist:
        ...
    def size(self) -> int:
        ...
    def splice(self, pos: iterator, other: cinsn_list_t, first: iterator, last: iterator) -> None:
        ...
    def swap(self, x: cinsn_list_t) -> None:
        ...

class ccase_t(cinsn_t, citem_t):
    op_to_typename: dict
    @property
    def casm(self) -> Any: ...
    @property
    def cblock(self) -> Any: ...
    @property
    def cdo(self) -> Any: ...
    @property
    def cexpr(self) -> Any: ...
    @property
    def cfor(self) -> Any: ...
    @property
    def cgoto(self) -> Any: ...
    @property
    def cif(self) -> Any: ...
    @property
    def cinsn(self) -> Any: ...
    @property
    def creturn(self) -> Any: ...
    @property
    def cswitch(self) -> Any: ...
    @property
    def cthrow(self) -> Any: ...
    @property
    def ctry(self) -> Any: ...
    @property
    def cwhile(self) -> Any: ...
    @property
    def details(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    @property
    def label_num(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    @property
    def opname(self) -> Any: ...
    @property
    def to_specific_type(self) -> Any: ...
    @property
    def values(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ccase_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ccase_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: ccase_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: ccase_t) -> bool:
        ...
    def __lt__(self, r: ccase_t) -> bool:
        ...
    def __ne__(self, r: ccase_t) -> bool:
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
    def assign(self, r: cinsn_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        r"""Cleanup the statement. This function properly deletes all children and sets the item type to cit_empty. 
                
        """
        ...
    def collect_free_breaks(self, breaks: cinsnptrvec_t) -> bool:
        r"""Collect free `break` statements. This function finds all free `break` statements within the current statement. A `break` statement is free if it does not have a loop or switch parent that that is also within the current statement. 
                
        :param breaks: pointer to the variable where the vector of all found free `break` statements is returned. This argument can be nullptr.
        :returns: true if some free `break` statements have been found
        """
        ...
    def collect_free_continues(self, continues: cinsnptrvec_t) -> bool:
        r"""Collect free `continue` statements. This function finds all free `continue` statements within the current statement. A `continue` statement is free if it does not have a loop parent that that is also within the current statement. 
                
        :param continues: pointer to the variable where the vector of all found free `continue` statements is returned. This argument can be nullptr.
        :returns: true if some free `continue` statements have been found
        """
        ...
    def compare(self, r: ccase_t) -> int:
        ...
    def contains_expr(self, e: cexpr_t) -> bool:
        r"""Does the item contain an expression?
        
        """
        ...
    def contains_free_break(self) -> bool:
        r"""Check if the statement has free `break` statements.
        
        """
        ...
    def contains_free_continue(self) -> bool:
        r"""Check if the statement has free `continue` statements.
        
        """
        ...
    def contains_insn(self, type: ctype_t, times: int = 1) -> bool:
        r"""Check if the statement contains a statement of the specified type. 
                
        :param type: statement opcode to look for
        :param times: how many times TYPE should be present
        :returns: true if the statement has at least TIMES children with opcode == TYPE
        """
        ...
    def contains_label(self) -> bool:
        r"""Does the item contain a label?
        
        """
        ...
    def create_if(self, cnd: cexpr_t) -> cif_t:
        r"""Create a new if-statement. The current statement must be a block. The new statement will be appended to it. 
                
        :param cnd: if condition. It will be deleted after being copied.
        """
        ...
    def dstr(self) -> str:
        ...
    def find_closest_addr(self, _ea: ida_idaapi.ea_t) -> citem_t:
        ...
    def find_parent_of(self, item: citem_t) -> citem_t:
        r"""Find parent of the specified item. 
                
        :param item: Item to find the parent of. The search will be performed among the children of the item pointed by `this`.
        :returns: nullptr if not found
        """
        ...
    def insn_is_epilog(self, insn: cinsn_t) -> bool:
        ...
    def is_epilog(self) -> Any:
        ...
    def is_expr(self) -> bool:
        r"""Is an expression?
        
        """
        ...
    def is_ordinary_flow(self) -> bool:
        r"""Check if the statement passes execution to the next statement. 
                
        :returns: false if the statement breaks the control flow (like goto, return, etc)
        """
        ...
    def new_insn(self, insn_ea: ida_idaapi.ea_t) -> cinsn_t:
        r"""Create a new statement. The current statement must be a block. The new statement will be appended to it. 
                
        :param insn_ea: statement address
        """
        ...
    def print1(self, func: cfunc_t) -> None:
        r"""Print the statement into one line. Currently this function is not available. 
                
        :param func: parent function. This argument is used to find out the referenced variable names.
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: cinsn_t) -> None:
        ...
    def value(self, i: int) -> uint64:
        ...
    def zero(self) -> None:
        r"""Overwrite with zeroes without cleaning memory or deleting children.
        
        """
        ...

class ccases_t(qvector_ccase_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ccases_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ccases_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: ccases_t) -> bool:
        ...
    def __init__(self) -> Any:
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
    def __le__(self, r: ccases_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: ccases_t) -> bool:
        ...
    def __ne__(self, r: ccases_t) -> bool:
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
    def __setitem__(self, i: size_t, v: ccase_t) -> None:
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
    def add_unique(self, x: ccase_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: ccases_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_ccase_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: ccase_t) -> bool:
        ...
    def inject(self, s: ccase_t, len: size_t) -> None:
        ...
    def insert(self, it: ccase_t, x: ccase_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_ccase_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class ccatch_t(cblock_t, cinsn_list_t):
    @property
    def exprs(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ccatch_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ccatch_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: ccatch_t) -> bool:
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
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
    def __le__(self, r: ccatch_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: ccatch_t) -> bool:
        ...
    def __ne__(self, r: ccatch_t) -> bool:
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
    def __setitem__(self, i: size_t, v: cinsn_t) -> None:
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
    def at(self, index: Any) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: ccatch_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def erase(self, p: cinsn_list_t_iterator) -> None:
        ...
    def find(self, item: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def index(self, item: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def is_catch_all(self) -> bool:
        ...
    def pop_back(self) -> None:
        ...
    def pop_front(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def push_front(self, x: cinsn_t) -> None:
        ...
    def rbegin(self, args: Any) -> qlist:
        ...
    def remove(self, v: cinsn_t) -> bool:
        ...
    def rend(self, args: Any) -> qlist:
        ...
    def size(self) -> int:
        ...
    def splice(self, pos: iterator, other: cinsn_list_t, first: iterator, last: iterator) -> None:
        ...
    def swap(self, r: ccatch_t) -> None:
        ...

class cdg_insn_iterator_t:
    @property
    def dslot(self) -> Any: ...
    @property
    def dslot_insn(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def end(self) -> Any: ...
    @property
    def is_likely_dslot(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def severed_branch(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def dslot_with_xrefs(self) -> bool:
        ...
    def has_dslot(self) -> bool:
        ...
    def is_severed_dslot(self) -> bool:
        ...
    def next(self, ins: insn_t) -> merror_t:
        ...
    def ok(self) -> bool:
        ...
    def start(self, rng: range_t) -> None:
        ...

class cdo_t(cloop_t, ceinsn_t):
    @property
    def body(self) -> Any: ...
    @property
    def expr(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cdo_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cdo_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cdo_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cdo_t) -> bool:
        ...
    def __lt__(self, r: cdo_t) -> bool:
        ...
    def __ne__(self, r: cdo_t) -> bool:
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
    def assign(self, r: cloop_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        ...
    def compare(self, r: cdo_t) -> int:
        ...

class ceinsn_t:
    @property
    def expr(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class cexpr_t(citem_t):
    op_to_typename: dict
    @property
    def a(self) -> Any: ...
    @property
    def cexpr(self) -> Any: ...
    @property
    def cinsn(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def exflags(self) -> Any: ...
    @property
    def fpc(self) -> Any: ...
    @property
    def helper(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    @property
    def insn(self) -> Any: ...
    @property
    def label_num(self) -> Any: ...
    @property
    def m(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def n(self) -> Any: ...
    @property
    def obj_ea(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    @property
    def operands(self) -> Any: ...
    @property
    def opname(self) -> Any: ...
    @property
    def ptrsize(self) -> Any: ...
    @property
    def refwidth(self) -> Any: ...
    @property
    def string(self) -> Any: ...
    @property
    def to_specific_type(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def v(self) -> Any: ...
    @property
    def x(self) -> Any: ...
    @property
    def y(self) -> Any: ...
    @property
    def z(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cexpr_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cexpr_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cexpr_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cexpr_t) -> bool:
        ...
    def __lt__(self, r: cexpr_t) -> bool:
        ...
    def __ne__(self, r: cexpr_t) -> bool:
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
    def assign(self, r: cexpr_t) -> cinsn_t:
        ...
    def calc_type(self, recursive: bool) -> None:
        r"""Calculate the type of the expression. Use this function to calculate the expression type when a new expression is built 
                
        :param recursive: if true, types of all children expression will be calculated before calculating our type
        """
        ...
    def cleanup(self) -> None:
        r"""Cleanup the expression. This function properly deletes all children and sets the item type to cot_empty. 
                
        """
        ...
    def compare(self, r: cexpr_t) -> int:
        ...
    def contains_comma(self, times: int = 1) -> bool:
        r"""Does the expression contain a comma operator?
        
        """
        ...
    def contains_comma_or_insn_or_label(self, maxcommas: int = 1) -> bool:
        r"""Does the expression contain a comma operator or an embedded statement operator or a label?
        
        """
        ...
    def contains_expr(self, e: cexpr_t) -> bool:
        r"""Does the item contain an expression?
        
        """
        ...
    def contains_insn(self, times: int = 1) -> bool:
        r"""Does the expression contain an embedded statement operator?
        
        """
        ...
    def contains_insn_or_label(self) -> bool:
        r"""Does the expression contain an embedded statement operator or a label?
        
        """
        ...
    def contains_label(self) -> bool:
        r"""Does the item contain a label?
        
        """
        ...
    def contains_operator(self, needed_op: ctype_t, times: int = 1) -> bool:
        r"""Check if the expression contains the specified operator. 
                
        :param needed_op: operator code to search for
        :param times: how many times the operator code should be present
        :returns: true if the expression has at least TIMES children with NEEDED_OP
        """
        ...
    def cpadone(self) -> bool:
        r"""Pointer arithmetic correction done for this expression?
        
        """
        ...
    def dstr(self) -> str:
        ...
    def equal_effect(self, r: cexpr_t) -> bool:
        r"""Compare two expressions. This function tries to compare two expressions in an 'intelligent' manner. For example, it knows about commutitive operators and can ignore useless casts. 
                
        :param r: the expression to compare against the current expression
        :returns: true expressions can be considered equal
        """
        ...
    def find_closest_addr(self, _ea: ida_idaapi.ea_t) -> citem_t:
        ...
    def find_num_op(self) -> cexpr_t:
        r"""Find the operand with a numeric value.
        
        """
        ...
    def find_op(self, _op: ctype_t) -> cexpr_t:
        r"""Find the child with the specified operator.
        
        """
        ...
    def find_parent_of(self, item: citem_t) -> citem_t:
        r"""Find parent of the specified item. 
                
        :param item: Item to find the parent of. The search will be performed among the children of the item pointed by `this`.
        :returns: nullptr if not found
        """
        ...
    def get_1num_op(self, o1: cexpr_t, o2: cexpr_t) -> bool:
        r"""Get pointers to operands. at last one operand should be a number o1 will be pointer to the number 
                
        """
        ...
    def get_const_value(self) -> bool:
        r"""Get expression value. 
                
        :returns: true if the expression is a number.
        """
        ...
    def get_high_nbit_bound(self) -> bit_bound_t:
        r"""Get max number of bits that can really be used by the expression. For example, x % 16 can yield only 4 non-zero bits, higher bits are zero 
                
        """
        ...
    def get_low_nbit_bound(self) -> int:
        r"""Get min number of bits that are certainly required to represent the expression. For example, constant 16 always uses 5 bits: 10000. 
                
        """
        ...
    def get_ptr_or_array(self) -> cexpr_t:
        r"""Find pointer or array child.
        
        """
        ...
    def get_type_sign(self) -> type_sign_t:
        r"""Get expression sign.
        
        """
        ...
    def get_v(self) -> var_ref_t:
        ...
    def has_side_effects(self) -> bool:
        r"""Check if the expression has side effects. Calls, pre/post inc/dec, and assignments have side effects. 
                
        """
        ...
    def is_call_arg_of(self, parent: citem_t) -> bool:
        r"""Is call argument? 
                
        :returns: true if our expression is a call argument of the specified parent expression.
        """
        ...
    def is_call_object_of(self, parent: citem_t) -> bool:
        r"""Is call object? 
                
        :returns: true if our expression is the call object of the specified parent expression.
        """
        ...
    def is_child_of(self, parent: citem_t) -> bool:
        r"""Verify if the specified item is our parent. 
                
        :param parent: possible parent item
        :returns: true if the specified item is our parent
        """
        ...
    def is_const_value(self, _v: uint64) -> bool:
        r"""Check if the expression is a number with the specified value.
        
        """
        ...
    def is_cstr(self) -> bool:
        ...
    def is_expr(self) -> bool:
        r"""Is an expression?
        
        """
        ...
    def is_fpop(self) -> bool:
        ...
    def is_jumpout(self) -> bool:
        ...
    def is_negative_const(self) -> bool:
        r"""Check if the expression is a negative number.
        
        """
        ...
    def is_nice_cond(self) -> bool:
        r"""Is nice condition?. Nice condition is a nice expression of the boolean type. 
                
        """
        ...
    def is_nice_expr(self) -> bool:
        r"""Is nice expression? Nice expressions do not contain comma operators, embedded statements, or labels. 
                
        """
        ...
    def is_non_negative_const(self) -> bool:
        r"""Check if the expression is a non-negative number.
        
        """
        ...
    def is_non_zero_const(self) -> bool:
        r"""Check if the expression is a non-zero number.
        
        """
        ...
    def is_odd_lvalue(self) -> bool:
        ...
    def is_type_signed(self) -> bool:
        r"""Is expression signed?
        
        """
        ...
    def is_type_unsigned(self) -> bool:
        r"""Is expression unsigned?
        
        """
        ...
    def is_undef_val(self) -> bool:
        ...
    def is_vftable(self) -> bool:
        ...
    def is_zero_const(self) -> bool:
        r"""Check if the expression is a zero.
        
        """
        ...
    def maybe_ptr(self) -> bool:
        r"""May the expression be a pointer?
        
        """
        ...
    def numval(self) -> uint64:
        r"""Get numeric value of the expression. This function can be called only on cot_num expressions! 
                
        """
        ...
    def print1(self, func: cfunc_t) -> None:
        r"""Print expression into one line. 
                
        :param func: parent function. This argument is used to find out the referenced variable names.
        """
        ...
    def put_number(self, args: Any) -> None:
        r"""Assign a number to the expression. 
                
        :param func: current function
        :param value: number value
        :param nbytes: size of the number in bytes
        :param sign: number sign
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def requires_lvalue(self, child: cexpr_t) -> bool:
        r"""Check if the expression requires an lvalue. 
                
        :param child: The function will check if this child of our expression must be an lvalue.
        :returns: true if child must be an lvalue.
        """
        ...
    def set_cpadone(self) -> None:
        ...
    def set_v(self, v: var_ref_t) -> None:
        ...
    def set_vftable(self) -> None:
        ...
    def swap(self, r: cexpr_t) -> None:
        ...
    def theother(self, what: cexpr_t) -> cexpr_t:
        r"""Get the other operand. This function returns the other operand (not the specified one) for binary expressions. 
                
        """
        ...

class cfor_t(cloop_t, ceinsn_t):
    @property
    def body(self) -> Any: ...
    @property
    def expr(self) -> Any: ...
    @property
    def init(self) -> Any: ...
    @property
    def step(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cfor_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cfor_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cfor_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cfor_t) -> bool:
        ...
    def __lt__(self, r: cfor_t) -> bool:
        ...
    def __ne__(self, r: cfor_t) -> bool:
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
    def assign(self, r: cloop_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        ...
    def compare(self, r: cfor_t) -> int:
        ...

class cfunc_parentee_t(ctree_parentee_t, ctree_visitor_t):
    @property
    def bposvec(self) -> Any: ...
    @property
    def cv_flags(self) -> Any: ...
    @property
    def func(self) -> Any: ...
    @property
    def parents(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, f: cfunc_t, post: bool = False) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply_to(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse ctree. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def apply_to_exprs(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse only expressions. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def calc_rvalue_type(self, target: tinfo_t, e: cexpr_t) -> bool:
        r"""Calculate rvalue type. This function tries to determine the type of the specified item based on its context. For example, if the current expression is the right side of an assignment operator, the type of its left side will be returned. This function can be used to determine the 'best' type of the specified expression. 
                
        :param target: 'best' type of the expression will be returned here
        :param e: expression to determine the desired type
        :returns: false if failed
        """
        ...
    def clr_prune(self) -> None:
        r"""Do not prune children. This is an internal function, no need to call it.
        
        """
        ...
    def clr_restart(self) -> None:
        r"""Do not restart. This is an internal function, no need to call it.
        
        """
        ...
    def is_postorder(self) -> bool:
        r"""Should the leave...() functions be called?
        
        """
        ...
    def leave_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def leave_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def maintain_parents(self) -> bool:
        r"""Should the parent information by maintained?
        
        """
        ...
    def must_prune(self) -> bool:
        r"""Should the traversal skip the children of the current item?
        
        """
        ...
    def must_restart(self) -> bool:
        r"""Should the traversal restart?
        
        """
        ...
    def only_insns(self) -> bool:
        r"""Should all expressions be automatically pruned?
        
        """
        ...
    def parent_expr(self) -> cexpr_t:
        r"""Get parent of the current item as an expression.
        
        """
        ...
    def parent_insn(self) -> cinsn_t:
        r"""Get parent of the current item as a statement.
        
        """
        ...
    def parent_item(self) -> citem_t:
        r"""Get parent of the current item as an item (statement or expression)
        
        """
        ...
    def prune_now(self) -> None:
        r"""Prune children. This function may be called by a visitor() to skip all children of the current item. 
                
        """
        ...
    def recalc_parent_types(self) -> bool:
        r"""Recalculate type of parent nodes. If a node type has been changed, the visitor must recalculate all parent types, otherwise the ctree becomes inconsistent. If during this recalculation a parent node is added/deleted, this function returns true. In this case the traversal must be stopped because the information about parent nodes is stale. 
                
        :returns: false-ok to continue the traversal, true-must stop.
        """
        ...
    def set_restart(self) -> None:
        r"""Restart the travesal. Meaningful only in apply_to_exprs()
        
        """
        ...
    def visit_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def visit_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...

class cfunc_t:
    @property
    def argidx(self) -> Any: ...
    @property
    def arguments(self) -> Any: ...
    @property
    def body(self) -> Any: ...
    @property
    def boundaries(self) -> Any: ...
    @property
    def eamap(self) -> Any: ...
    @property
    def entry_ea(self) -> Any: ...
    @property
    def hdrlines(self) -> Any: ...
    @property
    def lvars(self) -> Any: ...
    @property
    def maturity(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def numforms(self) -> Any: ...
    @property
    def pseudocode(self) -> Any: ...
    @property
    def refcnt(self) -> Any: ...
    @property
    def statebits(self) -> Any: ...
    @property
    def treeitems(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def user_cmts(self) -> Any: ...
    @property
    def user_iflags(self) -> Any: ...
    @property
    def user_labels(self) -> Any: ...
    @property
    def user_unions(self) -> Any: ...
    @property
    def warnings(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def __repr__(self) -> str:
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
    def build_c_tree(self) -> None:
        r"""Generate the function body. This function (re)generates the function body from the underlying microcode. 
                
        """
        ...
    def del_orphan_cmts(self) -> bool:
        r"""Delete all orphan comments. The save_user_cmts() function must be called after this call. 
                
        """
        ...
    def find_item_coords(self, args: Any) -> Any:
        r"""This method has the following signatures:
        
            1. find_item_coords(item: citem_t) -> Tuple[int, int]
            2. find_item_coords(item: citem_t, x: int_pointer, y: int_pointer) -> bool
        
        NOTE: The second form is retained for backward-compatibility,
        but we strongly recommend using the first.
        
        :param item: The item to find coordinates for in the pseudocode listing
        """
        ...
    def find_label(self, label: int) -> citem_t:
        r"""Find the label. 
                
        :returns: pointer to the ctree item with the specified label number.
        """
        ...
    def gather_derefs(self, ci: ctree_item_t, udm: udt_type_data_t = None) -> bool:
        ...
    def get_boundaries(self) -> boundaries_t:
        r"""Get pointer to map of instruction boundaries. This function initializes the boundary map if not done yet. 
                
        """
        ...
    def get_eamap(self) -> eamap_t:
        r"""Get pointer to ea->insn map. This function initializes eamap if not done yet. 
                
        """
        ...
    def get_func_type(self, type: tinfo_t) -> bool:
        r"""Get the function type. 
                
        :param type: variable where the function type is returned
        :returns: false if failure
        """
        ...
    def get_line_item(self, line: str, x: int, is_ctree_line: bool, phead: ctree_item_t, pitem: ctree_item_t, ptail: ctree_item_t) -> bool:
        r"""Get ctree item for the specified cursor position. 
                
        :param line: line of decompilation text (element of sv)
        :param x: x cursor coordinate in the line
        :param is_ctree_line: does the line belong to statement area? (if not, it is assumed to belong to the declaration area)
        :param phead: ptr to the first item on the line (used to attach block comments). May be nullptr
        :param pitem: ptr to the current item. May be nullptr
        :param ptail: ptr to the last item on the line (used to attach indented comments). May be nullptr
        :returns: false if failed to get the current item
        """
        ...
    def get_lvars(self) -> lvars_t:
        r"""Get vector of local variables. 
                
        :returns: pointer to the vector of local variables. If you modify this vector, the ctree must be regenerated in order to have correct cast operators. Use build_c_tree() for that. Removing lvars should be done carefully: all references in ctree and microcode must be corrected after that.
        """
        ...
    def get_pseudocode(self) -> strvec_t:
        r"""Get pointer to decompilation output: the pseudocode. This function generates pseudocode if not done yet. 
                
        """
        ...
    def get_stkoff_delta(self) -> int:
        r"""Get stack offset delta. The local variable stack offsets retrieved by v.location.stkoff() should be adjusted before being used as stack frame offsets in IDA. 
                
        :returns: the delta to apply. example: ida_stkoff = v.location.stkoff() - f->get_stkoff_delta()
        """
        ...
    def get_user_cmt(self, loc: treeloc_t, rt: cmt_retrieval_type_t) -> str:
        r"""Retrieve a user defined comment. 
                
        :param loc: ctree location
        :param rt: should already retrieved comments retrieved again?
        :returns: pointer to the comment string or nullptr
        """
        ...
    def get_user_iflags(self, loc: citem_locator_t) -> int:
        r"""Retrieve citem iflags. 
                
        :param loc: citem locator
        :returns: ctree item iflags bits or 0
        """
        ...
    def get_user_union_selection(self, ea: ida_idaapi.ea_t, path: intvec_t) -> bool:
        r"""Retrieve a user defined union field selection. 
                
        :param ea: address
        :param path: out: path describing the union selection.
        :returns: pointer to the path or nullptr
        """
        ...
    def get_warnings(self) -> hexwarns_t:
        r"""Get information about decompilation warnings. 
                
        :returns: reference to the vector of warnings
        """
        ...
    def has_orphan_cmts(self) -> bool:
        r"""Check if there are orphan comments.
        
        """
        ...
    def locked(self) -> bool:
        ...
    def print_dcl(self) -> None:
        r"""Print function prototype. 
                
        """
        ...
    def print_func(self, vp: vc_printer_t) -> None:
        r"""Print function text. 
                
        :param vp: printer helper class to receive the generated text.
        """
        ...
    def recalc_item_addresses(self) -> None:
        r"""Recalculate item adresses. This function may be required after shuffling ctree items. For example, when adding or removing statements of a block, or changing 'if' statements. 
                
        """
        ...
    def refresh_func_ctext(self) -> None:
        r"""Refresh ctext after a ctree modification. This function informs the decompiler that ctree (body) have been modified and ctext (sv) does not correspond to it anymore. It also refreshes the pseudocode windows if there is any. 
                
        """
        ...
    def release(self) -> None:
        ...
    def remove_unused_labels(self) -> None:
        r"""Remove unused labels. This function checks what labels are really used by the function and removes the unused ones. You must call it after deleting a goto statement. 
                
        """
        ...
    def save_user_cmts(self) -> None:
        r"""Save user-defined comments into the database.
        
        """
        ...
    def save_user_iflags(self) -> None:
        r"""Save user-defined iflags into the database.
        
        """
        ...
    def save_user_labels(self) -> None:
        r"""Save user-defined labels into the database.
        
        """
        ...
    def save_user_numforms(self) -> None:
        r"""Save user-defined number formats into the database.
        
        """
        ...
    def save_user_unions(self) -> None:
        r"""Save user-defined union field selections into the database.
        
        """
        ...
    def set_user_cmt(self, loc: treeloc_t, cmt: str) -> None:
        r"""Set a user defined comment. This function stores the specified comment in the cfunc_t structure. The save_user_cmts() function must be called after it. 
                
        :param loc: ctree location
        :param cmt: new comment. if empty or nullptr, then an existing comment is deleted.
        """
        ...
    def set_user_iflags(self, loc: citem_locator_t, iflags: int) -> None:
        r"""Set citem iflags. 
                
        :param loc: citem locator
        :param iflags: new iflags
        """
        ...
    def set_user_union_selection(self, ea: ida_idaapi.ea_t, path: intvec_t) -> None:
        r"""Set a union field selection. The save_user_unions() function must be called after calling this function. 
                
        :param ea: address
        :param path: in: path describing the union selection.
        """
        ...
    def verify(self, aul: allow_unused_labels_t, even_without_debugger: bool) -> None:
        r"""Verify the ctree. This function verifies the ctree. If the ctree is malformed, an internal error is generated. Use it to verify the ctree after your modifications. 
                
        :param aul: Are unused labels acceptable?
        :param even_without_debugger: if false and there is no debugger, the verification will be skipped
        """
        ...

class cfuncptr_t:
    @property
    def argidx(self) -> Any: ...
    @property
    def arguments(self) -> Any: ...
    @property
    def body(self) -> Any: ...
    @property
    def boundaries(self) -> Any: ...
    @property
    def eamap(self) -> Any: ...
    @property
    def entry_ea(self) -> Any: ...
    @property
    def hdrlines(self) -> Any: ...
    @property
    def lvars(self) -> Any: ...
    @property
    def maturity(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def numforms(self) -> Any: ...
    @property
    def pseudocode(self) -> Any: ...
    @property
    def refcnt(self) -> Any: ...
    @property
    def statebits(self) -> Any: ...
    @property
    def treeitems(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def user_cmts(self) -> Any: ...
    @property
    def user_iflags(self) -> Any: ...
    @property
    def user_labels(self) -> Any: ...
    @property
    def user_unions(self) -> Any: ...
    @property
    def warnings(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __deref__(self) -> cfunc_t:
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, other: Any) -> bool:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __ptrval__(self) -> int:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __ref__(self) -> int:
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
    def build_c_tree(self) -> None:
        ...
    def del_orphan_cmts(self) -> bool:
        ...
    def find_item_coords(self, args: Any) -> Any:
        r"""This method has the following signatures:
        
            1. find_item_coords(item: citem_t) -> Tuple[int, int]
            2. find_item_coords(item: citem_t, x: int_pointer, y: int_pointer) -> bool
        
        NOTE: The second form is retained for backward-compatibility,
        but we strongly recommend using the first.
        
        :param item: The item to find coordinates for in the pseudocode listing
        """
        ...
    def find_label(self, label: int) -> citem_t:
        ...
    def gather_derefs(self, ci: ctree_item_t, udm: udt_type_data_t = None) -> bool:
        ...
    def get_boundaries(self) -> boundaries_t:
        ...
    def get_eamap(self) -> eamap_t:
        ...
    def get_func_type(self, type: tinfo_t) -> bool:
        ...
    def get_line_item(self, line: str, x: int, is_ctree_line: bool, phead: ctree_item_t, pitem: ctree_item_t, ptail: ctree_item_t) -> bool:
        ...
    def get_lvars(self) -> lvars_t:
        ...
    def get_pseudocode(self) -> strvec_t:
        ...
    def get_stkoff_delta(self) -> int:
        ...
    def get_user_cmt(self, loc: treeloc_t, rt: cmt_retrieval_type_t) -> str:
        ...
    def get_user_iflags(self, loc: citem_locator_t) -> int:
        ...
    def get_user_union_selection(self, ea: ida_idaapi.ea_t, path: intvec_t) -> bool:
        ...
    def get_warnings(self) -> hexwarns_t:
        ...
    def has_orphan_cmts(self) -> bool:
        ...
    def locked(self) -> bool:
        ...
    def print_dcl(self) -> None:
        ...
    def print_func(self, vp: vc_printer_t) -> None:
        ...
    def recalc_item_addresses(self) -> None:
        ...
    def refresh_func_ctext(self) -> None:
        ...
    def release(self) -> None:
        ...
    def remove_unused_labels(self) -> None:
        ...
    def reset(self) -> None:
        ...
    def save_user_cmts(self) -> None:
        r"""Save user defined comments into the database. 
                
        """
        ...
    def save_user_iflags(self) -> None:
        r"""Save user defined citem iflags into the database. 
                
        """
        ...
    def save_user_labels(self) -> None:
        r"""Save user defined labels into the database. 
                
        """
        ...
    def save_user_numforms(self) -> None:
        r"""Save user defined number formats into the database. 
                
        """
        ...
    def save_user_unions(self) -> None:
        r"""Save user defined union field selections into the database. 
                
        """
        ...
    def set_user_cmt(self, loc: treeloc_t, cmt: str) -> None:
        ...
    def set_user_iflags(self, loc: citem_locator_t, iflags: int) -> None:
        ...
    def set_user_union_selection(self, ea: ida_idaapi.ea_t, path: intvec_t) -> None:
        ...
    def verify(self, aul: allow_unused_labels_t, even_without_debugger: bool) -> None:
        ...

class cgoto_t:
    @property
    def label_num(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cgoto_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cgoto_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cgoto_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cgoto_t) -> bool:
        ...
    def __lt__(self, r: cgoto_t) -> bool:
        ...
    def __ne__(self, r: cgoto_t) -> bool:
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
    def compare(self, r: cgoto_t) -> int:
        ...

class chain_keeper_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _gc: graph_chains_t) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def back(self) -> block_chains_t:
        ...
    def for_all_chains(self, cv: chain_visitor_t, gca: int) -> int:
        ...
    def front(self) -> block_chains_t:
        ...

class chain_t:
    @property
    def flags(self) -> Any: ...
    @property
    def varnum(self) -> Any: ...
    @property
    def width(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: intvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __lt__(self, r: chain_t) -> bool:
        ...
    def __ne__(self, r: intvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: int) -> None:
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
    def add_unique(self, x: int) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def append_list(self, mba: mba_t, list: mlist_t) -> None:
        r"""Append the contents of the chain to the specified list of locations.
        
        """
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def clear_varnum(self) -> None:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def endoff(self) -> voff_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: intvec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def get_reg(self) -> mreg_t:
        ...
    def get_stkoff(self) -> int:
        ...
    def has(self, x: int) -> bool:
        ...
    def includes(self, r: chain_t) -> bool:
        ...
    def inject(self, s: int, len: size_t) -> None:
        ...
    def insert(self, it: iterator, x: int) -> qvector:
        ...
    def is_fake(self) -> bool:
        ...
    def is_inited(self) -> bool:
        ...
    def is_overlapped(self) -> bool:
        ...
    def is_passreg(self) -> bool:
        ...
    def is_reg(self) -> bool:
        ...
    def is_replaced(self) -> bool:
        ...
    def is_stkoff(self) -> bool:
        ...
    def is_term(self) -> bool:
        ...
    def key(self) -> voff_t:
        ...
    def overlap(self, r: chain_t) -> bool:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def set_inited(self, b: bool) -> None:
        ...
    def set_overlapped(self, b: bool) -> None:
        ...
    def set_replaced(self, b: bool) -> None:
        ...
    def set_term(self, b: bool) -> None:
        ...
    def set_value(self, r: chain_t) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: intvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class chain_visitor_t:
    @property
    def parent(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def visit_chain(self, nblock: int, ch: chain_t) -> int:
        ...

class cif_t(ceinsn_t):
    @property
    def expr(self) -> Any: ...
    @property
    def ielse(self) -> Any: ...
    @property
    def ithen(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cif_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cif_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cif_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cif_t) -> bool:
        ...
    def __lt__(self, r: cif_t) -> bool:
        ...
    def __ne__(self, r: cif_t) -> bool:
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
    def assign(self, r: cif_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        ...
    def compare(self, r: cif_t) -> int:
        ...

class cinsn_list_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, x: cinsn_list_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, x: cinsn_list_t) -> bool:
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
    def __setitem__(self, i: size_t, v: cinsn_t) -> None:
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
    def at(self, index: Any) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def erase(self, p: cinsn_list_t_iterator) -> None:
        ...
    def find(self, item: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def index(self, item: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def pop_front(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def push_front(self, x: cinsn_t) -> None:
        ...
    def rbegin(self, args: Any) -> qlist:
        ...
    def remove(self, v: cinsn_t) -> bool:
        ...
    def rend(self, args: Any) -> qlist:
        ...
    def size(self) -> int:
        ...
    def splice(self, pos: iterator, other: cinsn_list_t, first: iterator, last: iterator) -> None:
        ...
    def swap(self, x: cinsn_list_t) -> None:
        ...

class cinsn_list_t_iterator:
    @property
    def cur(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, x: cinsn_list_t_iterator) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, x: cinsn_list_t_iterator) -> bool:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next__(self) -> None:
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
    def next(self) -> merror_t:
        ...

class cinsn_t(citem_t):
    op_to_typename: dict
    @property
    def casm(self) -> Any: ...
    @property
    def cblock(self) -> Any: ...
    @property
    def cdo(self) -> Any: ...
    @property
    def cexpr(self) -> Any: ...
    @property
    def cfor(self) -> Any: ...
    @property
    def cgoto(self) -> Any: ...
    @property
    def cif(self) -> Any: ...
    @property
    def cinsn(self) -> Any: ...
    @property
    def creturn(self) -> Any: ...
    @property
    def cswitch(self) -> Any: ...
    @property
    def cthrow(self) -> Any: ...
    @property
    def ctry(self) -> Any: ...
    @property
    def cwhile(self) -> Any: ...
    @property
    def details(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    @property
    def label_num(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    @property
    def opname(self) -> Any: ...
    @property
    def to_specific_type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cinsn_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cinsn_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cinsn_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cinsn_t) -> bool:
        ...
    def __lt__(self, r: cinsn_t) -> bool:
        ...
    def __ne__(self, r: cinsn_t) -> bool:
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
    def assign(self, r: cinsn_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        r"""Cleanup the statement. This function properly deletes all children and sets the item type to cit_empty. 
                
        """
        ...
    def collect_free_breaks(self, breaks: cinsnptrvec_t) -> bool:
        r"""Collect free `break` statements. This function finds all free `break` statements within the current statement. A `break` statement is free if it does not have a loop or switch parent that that is also within the current statement. 
                
        :param breaks: pointer to the variable where the vector of all found free `break` statements is returned. This argument can be nullptr.
        :returns: true if some free `break` statements have been found
        """
        ...
    def collect_free_continues(self, continues: cinsnptrvec_t) -> bool:
        r"""Collect free `continue` statements. This function finds all free `continue` statements within the current statement. A `continue` statement is free if it does not have a loop parent that that is also within the current statement. 
                
        :param continues: pointer to the variable where the vector of all found free `continue` statements is returned. This argument can be nullptr.
        :returns: true if some free `continue` statements have been found
        """
        ...
    def compare(self, r: cinsn_t) -> int:
        ...
    def contains_expr(self, e: cexpr_t) -> bool:
        r"""Does the item contain an expression?
        
        """
        ...
    def contains_free_break(self) -> bool:
        r"""Check if the statement has free `break` statements.
        
        """
        ...
    def contains_free_continue(self) -> bool:
        r"""Check if the statement has free `continue` statements.
        
        """
        ...
    def contains_insn(self, type: ctype_t, times: int = 1) -> bool:
        r"""Check if the statement contains a statement of the specified type. 
                
        :param type: statement opcode to look for
        :param times: how many times TYPE should be present
        :returns: true if the statement has at least TIMES children with opcode == TYPE
        """
        ...
    def contains_label(self) -> bool:
        r"""Does the item contain a label?
        
        """
        ...
    def create_if(self, cnd: cexpr_t) -> cif_t:
        r"""Create a new if-statement. The current statement must be a block. The new statement will be appended to it. 
                
        :param cnd: if condition. It will be deleted after being copied.
        """
        ...
    def dstr(self) -> str:
        ...
    def find_closest_addr(self, _ea: ida_idaapi.ea_t) -> citem_t:
        ...
    def find_parent_of(self, item: citem_t) -> citem_t:
        r"""Find parent of the specified item. 
                
        :param item: Item to find the parent of. The search will be performed among the children of the item pointed by `this`.
        :returns: nullptr if not found
        """
        ...
    def insn_is_epilog(self, insn: cinsn_t) -> bool:
        ...
    def is_epilog(self) -> Any:
        ...
    def is_expr(self) -> bool:
        r"""Is an expression?
        
        """
        ...
    def is_ordinary_flow(self) -> bool:
        r"""Check if the statement passes execution to the next statement. 
                
        :returns: false if the statement breaks the control flow (like goto, return, etc)
        """
        ...
    def new_insn(self, insn_ea: ida_idaapi.ea_t) -> cinsn_t:
        r"""Create a new statement. The current statement must be a block. The new statement will be appended to it. 
                
        :param insn_ea: statement address
        """
        ...
    def print1(self, func: cfunc_t) -> None:
        r"""Print the statement into one line. Currently this function is not available. 
                
        :param func: parent function. This argument is used to find out the referenced variable names.
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def swap(self, r: cinsn_t) -> None:
        ...
    def zero(self) -> None:
        r"""Overwrite with zeroes without cleaning memory or deleting children.
        
        """
        ...

class cinsnptrvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cinsnptrvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: cinsnptrvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: cinsn_t) -> None:
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
    def add_unique(self, x: cinsn_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: cinsnptrvec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, x: cinsn_t) -> bool:
        ...
    def inject(self, s: cinsn_t, len: size_t) -> None:
        ...
    def insert(self, it: iterator, x: cinsn_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: cinsnptrvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class citem_cmt_t:
    @property
    def used(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def c_str(self) -> str:
        ...

class citem_locator_t:
    @property
    def ea(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: citem_locator_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: citem_locator_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: citem_locator_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: citem_locator_t) -> bool:
        ...
    def __lt__(self, r: citem_locator_t) -> bool:
        ...
    def __ne__(self, r: citem_locator_t) -> bool:
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
    def compare(self, r: citem_locator_t) -> int:
        ...

class citem_t:
    @property
    def cexpr(self) -> Any: ...
    @property
    def cinsn(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    @property
    def label_num(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def op(self) -> Any: ...
    @property
    def to_specific_type(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, o: ctype_t = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def contains_expr(self, e: cexpr_t) -> bool:
        r"""Does the item contain an expression?
        
        """
        ...
    def contains_label(self) -> bool:
        r"""Does the item contain a label?
        
        """
        ...
    def find_closest_addr(self, _ea: ida_idaapi.ea_t) -> citem_t:
        ...
    def find_parent_of(self, item: citem_t) -> citem_t:
        r"""Find parent of the specified item. 
                
        :param item: Item to find the parent of. The search will be performed among the children of the item pointed by `this`.
        :returns: nullptr if not found
        """
        ...
    def is_expr(self) -> bool:
        r"""Is an expression?
        
        """
        ...
    def print1(self, func: cfunc_t) -> None:
        r"""Print item into one line. 
                
        :param func: parent function. This argument is used to find out the referenced variable names.
        :returns: length of the generated text.
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def swap(self, r: citem_t) -> None:
        r"""Swap two citem_t.
        
        """
        ...

class cloop_t(ceinsn_t):
    @property
    def body(self) -> Any: ...
    @property
    def expr(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def assign(self, r: cloop_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        ...

class cnumber_t:
    @property
    def nf(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cnumber_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cnumber_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cnumber_t) -> bool:
        ...
    def __init__(self, _opnum: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cnumber_t) -> bool:
        ...
    def __lt__(self, r: cnumber_t) -> bool:
        ...
    def __ne__(self, r: cnumber_t) -> bool:
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
    def assign(self, v: uint64, nbytes: int, sign: type_sign_t) -> cinsn_t:
        r"""Assign new value 
                
        :param v: new value
        :param nbytes: size of the new value in bytes
        :param sign: sign of the value
        """
        ...
    def compare(self, r: cnumber_t) -> int:
        ...
    def value(self, type: tinfo_t) -> uint64:
        r"""Get value. This function will properly extend the number sign to 64bits depending on the type sign. 
                
        """
        ...

class codegen_t:
    @property
    def ignore_micro(self) -> Any: ...
    @property
    def ii(self) -> Any: ...
    @property
    def insn(self) -> Any: ...
    @property
    def mb(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def analyze_prolog(self, fc: qflow_chart_t, reachable: bitset_t) -> merror_t:
        r"""Analyze prolog/epilog of the function to decompile. If prolog is found, allocate and fill 'mba->pi' structure. 
                
        :param fc: flow chart
        :param reachable: bitmap of reachable blocks
        :returns: error code
        """
        ...
    def clear(self) -> None:
        ...
    def emit(self, args: Any) -> minsn_t:
        r"""This function has the following signatures:
        
            0. emit(code: mcode_t, width: int, l: int, r: int, d: int, offsize: int) -> minsn_t *
            1. emit(code: mcode_t, l: const mop_t *, r: const mop_t *, d: const mop_t *) -> minsn_t *
        
        # 0: emit(code: mcode_t, width: int, l: int, r: int, d: int, offsize: int) -> minsn_t *
        
        Emit one microinstruction. The L, R, D arguments usually mean the register number. However, they depend on CODE. For example:
        * for m_goto and m_jcnd L is the target address
        * for m_ldc L is the constant value to load
        
        
        
        :returns: created microinstruction. can be nullptr if the instruction got immediately optimized away.
        
        # 1: emit(code: mcode_t, l: const mop_t *, r: const mop_t *, d: const mop_t *) -> minsn_t *
        
        Emit one microinstruction. This variant accepts pointers to operands. It is more difficult to use but permits to create virtually any instruction. Operands may be nullptr when it makes sense. 
                
        
        """
        ...
    def emit_micro_mvm(self, code: mcode_t, dtype: op_dtype_t, l: int, r: int, d: int, offsize: int) -> minsn_t:
        r"""Emit one microinstruction. This variant takes a data type not a size. 
                
        """
        ...
    def gen_micro(self) -> merror_t:
        r"""Generate microcode for one instruction. The instruction is in INSN 
                
        :returns: MERR_OK - all ok MERR_BLOCK - all ok, need to switch to new block MERR_BADBLK - delete current block and continue other error codes are fatal
        """
        ...
    def load_effective_address(self, n: int, flags: int = 0) -> mreg_t:
        r"""Generate microcode to calculate the address of a memory operand. 
                
        :param n: - number of INSN operand
        :param flags: - reserved for future use
        :returns: register containing the operand address. mr_none - failed (not a memory operand)
        """
        ...
    def load_operand(self, opnum: int, flags: int = 0) -> mreg_t:
        r"""Generate microcode to load one operand. 
                
        :param opnum: number of INSN operand
        :param flags: reserved for future use
        :returns: register containing the operand.
        """
        ...
    def microgen_completed(self) -> None:
        r"""This method is called when the microcode generation is done.
        
        """
        ...
    def prepare_gen_micro(self) -> merror_t:
        r"""Setup internal data to handle new instruction. This method should be called before calling gen_micro(). Usually gen_micro() is called by the decompiler. You have to call this function explicitly only if you yourself call gen_micro(). The instruction is in INSN 
                
        :returns: MERR_OK - all ok other error codes are fatal
        """
        ...
    def store_operand(self, n: int, mop: mop_t, flags: int = 0, outins: minsn_t = None) -> bool:
        r"""Generate microcode to store an operand. In case of success an arbitrary number of instructions can be generated (and even no instruction if the source and target are the same) 
                
        :param n: - number of target INSN operand
        :param mop: - operand to be stored
        :param flags: - reserved for future use
        :param outins: - (OUT) the last generated instruction
        :returns: success
        """
        ...

class creturn_t(ceinsn_t):
    @property
    def expr(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: creturn_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: creturn_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: creturn_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: creturn_t) -> bool:
        ...
    def __lt__(self, r: creturn_t) -> bool:
        ...
    def __ne__(self, r: creturn_t) -> bool:
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
    def compare(self, r: creturn_t) -> int:
        ...

class cswitch_t(ceinsn_t):
    @property
    def cases(self) -> Any: ...
    @property
    def expr(self) -> Any: ...
    @property
    def mvnf(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cswitch_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cswitch_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cswitch_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cswitch_t) -> bool:
        ...
    def __lt__(self, r: cswitch_t) -> bool:
        ...
    def __ne__(self, r: cswitch_t) -> bool:
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
    def compare(self, r: cswitch_t) -> int:
        ...

class ctext_position_t:
    @property
    def lnnum(self) -> Any: ...
    @property
    def x(self) -> Any: ...
    @property
    def y(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ctext_position_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ctext_position_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: ctext_position_t) -> bool:
        ...
    def __init__(self, _lnnum: int = -1, _x: int = 0, _y: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: ctext_position_t) -> bool:
        ...
    def __lt__(self, r: ctext_position_t) -> bool:
        ...
    def __ne__(self, r: ctext_position_t) -> bool:
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
    def compare(self, r: ctext_position_t) -> int:
        ...
    def in_ctree(self, hdrlines: int) -> bool:
        r"""Is the cursor in the variable/type declaration area? 
                
        :param hdrlines: Number of lines of the declaration area
        """
        ...

class cthrow_t(ceinsn_t):
    @property
    def expr(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cthrow_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cthrow_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cthrow_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cthrow_t) -> bool:
        ...
    def __lt__(self, r: cthrow_t) -> bool:
        ...
    def __ne__(self, r: cthrow_t) -> bool:
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
    def compare(self, r: cthrow_t) -> int:
        ...

class ctree_anchor_t:
    @property
    def value(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def get_index(self) -> int:
        ...
    def get_itp(self) -> item_preciser_t:
        ...
    def is_blkcmt_anchor(self) -> bool:
        ...
    def is_citem_anchor(self) -> bool:
        ...
    def is_itp_anchor(self) -> bool:
        ...
    def is_lvar_anchor(self) -> bool:
        ...
    def is_valid_anchor(self) -> bool:
        ...

class ctree_item_t:
    @property
    def citype(self) -> Any: ...
    @property
    def e(self) -> Any: ...
    @property
    def f(self) -> Any: ...
    @property
    def i(self) -> Any: ...
    @property
    def it(self) -> Any: ...
    @property
    def l(self) -> Any: ...
    @property
    def loc(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def dstr(self) -> str:
        ...
    def get_ea(self) -> ida_idaapi.ea_t:
        r"""Get address of the current item. Each ctree item has an address. 
                
        :returns: BADADDR if failed
        """
        ...
    def get_edm(self, parent: tinfo_t) -> int:
        r"""Get type of an enum member. If the current item is a symbolic constant, this function will return information about it. 
                
        :param parent: pointer to buffer for the enum type.
        :returns: member index or -1 if failed
        """
        ...
    def get_label_num(self, gln_flags: int) -> int:
        r"""Get label number of the current item. 
                
        :param gln_flags: Combination of get_label_num control bits
        :returns: -1 if failed or no label
        """
        ...
    def get_lvar(self) -> lvar_t:
        r"""Get pointer to local variable. If the current item is a local variable, this function will return pointer to its definition. 
                
        :returns: nullptr if failed
        """
        ...
    def get_udm(self, udm: udm_t = None, parent: tinfo_t = None, p_offset: uint64 = None) -> int:
        r"""Get type of a structure field. If the current item is a structure/union field, this function will return information about it. 
                
        :param udm: pointer to buffer for the udt member info.
        :param parent: pointer to buffer for the struct/union type.
        :param p_offset: pointer to the offset in bits inside udt.
        :returns: member index or -1 if failed Both output parameters can be nullptr.
        """
        ...
    def is_citem(self) -> bool:
        r"""Is the current item is a ctree item?
        
        """
        ...

class ctree_items_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ctree_items_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: ctree_items_t) -> bool:
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
    def __setitem__(self, i: size_t, v: citem_t) -> None:
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
    def add_unique(self, x: citem_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: ctree_items_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, x: citem_t) -> bool:
        ...
    def inject(self, s: citem_t, len: size_t) -> None:
        ...
    def insert(self, it: iterator, x: citem_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: ctree_items_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class ctree_parentee_t(ctree_visitor_t):
    @property
    def bposvec(self) -> Any: ...
    @property
    def cv_flags(self) -> Any: ...
    @property
    def parents(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, post: bool = False) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply_to(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse ctree. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def apply_to_exprs(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse only expressions. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def clr_prune(self) -> None:
        r"""Do not prune children. This is an internal function, no need to call it.
        
        """
        ...
    def clr_restart(self) -> None:
        r"""Do not restart. This is an internal function, no need to call it.
        
        """
        ...
    def is_postorder(self) -> bool:
        r"""Should the leave...() functions be called?
        
        """
        ...
    def leave_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def leave_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def maintain_parents(self) -> bool:
        r"""Should the parent information by maintained?
        
        """
        ...
    def must_prune(self) -> bool:
        r"""Should the traversal skip the children of the current item?
        
        """
        ...
    def must_restart(self) -> bool:
        r"""Should the traversal restart?
        
        """
        ...
    def only_insns(self) -> bool:
        r"""Should all expressions be automatically pruned?
        
        """
        ...
    def parent_expr(self) -> cexpr_t:
        r"""Get parent of the current item as an expression.
        
        """
        ...
    def parent_insn(self) -> cinsn_t:
        r"""Get parent of the current item as a statement.
        
        """
        ...
    def parent_item(self) -> citem_t:
        r"""Get parent of the current item as an item (statement or expression)
        
        """
        ...
    def prune_now(self) -> None:
        r"""Prune children. This function may be called by a visitor() to skip all children of the current item. 
                
        """
        ...
    def recalc_parent_types(self) -> bool:
        r"""Recalculate type of parent nodes. If a node type has been changed, the visitor must recalculate all parent types, otherwise the ctree becomes inconsistent. If during this recalculation a parent node is added/deleted, this function returns true. In this case the traversal must be stopped because the information about parent nodes is stale. 
                
        :returns: false-ok to continue the traversal, true-must stop.
        """
        ...
    def set_restart(self) -> None:
        r"""Restart the travesal. Meaningful only in apply_to_exprs()
        
        """
        ...
    def visit_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def visit_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...

class ctree_visitor_t:
    @property
    def bposvec(self) -> Any: ...
    @property
    def cv_flags(self) -> Any: ...
    @property
    def parents(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _flags: int) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply_to(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse ctree. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def apply_to_exprs(self, item: citem_t, parent: citem_t) -> int:
        r"""Traverse only expressions. The traversal will start at the specified item and continue until of one the visit_...() functions return a non-zero value. 
                
        :param item: root of the ctree to traverse
        :param parent: parent of the specified item. can be specified as nullptr.
        :returns: 0 or a non-zero value returned by a visit_...() function
        """
        ...
    def clr_prune(self) -> None:
        r"""Do not prune children. This is an internal function, no need to call it.
        
        """
        ...
    def clr_restart(self) -> None:
        r"""Do not restart. This is an internal function, no need to call it.
        
        """
        ...
    def is_postorder(self) -> bool:
        r"""Should the leave...() functions be called?
        
        """
        ...
    def leave_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def leave_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement after having visited its children. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs post-order traserval, i.e. an item is visited after its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def maintain_parents(self) -> bool:
        r"""Should the parent information by maintained?
        
        """
        ...
    def must_prune(self) -> bool:
        r"""Should the traversal skip the children of the current item?
        
        """
        ...
    def must_restart(self) -> bool:
        r"""Should the traversal restart?
        
        """
        ...
    def only_insns(self) -> bool:
        r"""Should all expressions be automatically pruned?
        
        """
        ...
    def parent_expr(self) -> cexpr_t:
        r"""Get parent of the current item as an expression.
        
        """
        ...
    def parent_insn(self) -> cinsn_t:
        r"""Get parent of the current item as a statement.
        
        """
        ...
    def parent_item(self) -> citem_t:
        r"""Get parent of the current item as an item (statement or expression)
        
        """
        ...
    def prune_now(self) -> None:
        r"""Prune children. This function may be called by a visitor() to skip all children of the current item. 
                
        """
        ...
    def set_restart(self) -> None:
        r"""Restart the travesal. Meaningful only in apply_to_exprs()
        
        """
        ...
    def visit_expr(self, arg0: cexpr_t) -> int:
        r"""Visit an expression. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...
    def visit_insn(self, arg0: cinsn_t) -> int:
        r"""Visit a statement. This is a visitor function which should be overridden by a derived class to do some useful work. This visitor performs pre-order traserval, i.e. an item is visited before its children. 
                
        :returns: 0 to continue the traversal, nonzero to stop.
        """
        ...

class ctry_t(cblock_t, cinsn_list_t):
    @property
    def catchs(self) -> Any: ...
    @property
    def is_wind(self) -> Any: ...
    @property
    def new_state(self) -> Any: ...
    @property
    def old_state(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ctry_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ctry_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, r: ctry_t) -> bool:
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
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
    def __le__(self, r: ctry_t) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, r: ctry_t) -> bool:
        ...
    def __ne__(self, r: ctry_t) -> bool:
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
    def __setitem__(self, i: size_t, v: cinsn_t) -> None:
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
    def at(self, index: Any) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: ctry_t) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def erase(self, p: cinsn_list_t_iterator) -> None:
        ...
    def find(self, item: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def index(self, item: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def pop_front(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def push_front(self, x: cinsn_t) -> None:
        ...
    def rbegin(self, args: Any) -> qlist:
        ...
    def remove(self, v: cinsn_t) -> bool:
        ...
    def rend(self, args: Any) -> qlist:
        ...
    def size(self) -> int:
        ...
    def splice(self, pos: iterator, other: cinsn_list_t, first: iterator, last: iterator) -> None:
        ...
    def swap(self, x: cinsn_list_t) -> None:
        ...

class cwhile_t(cloop_t, ceinsn_t):
    @property
    def body(self) -> Any: ...
    @property
    def expr(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: cwhile_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: cwhile_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: cwhile_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: cwhile_t) -> bool:
        ...
    def __lt__(self, r: cwhile_t) -> bool:
        ...
    def __ne__(self, r: cwhile_t) -> bool:
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
    def assign(self, r: cloop_t) -> cinsn_t:
        ...
    def cleanup(self) -> None:
        ...
    def compare(self, r: cwhile_t) -> int:
        ...

class eamap_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: eamap_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: eamap_iterator_t) -> bool:
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

class eamap_t:
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: int) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def keytype(self, *args: Any, **kwargs: Any) -> Any:
        r"""int([x]) -> integer
        int(x, base=10) -> integer
        
        Convert a number or string to an integer, or return 0 if no arguments
        are given.  If x is a number, return x.__int__().  For floating point
        numbers, this truncates towards zero.
        
        If x is not a number or if base is given, then x must be a string,
        bytes, or bytearray instance representing an integer literal in the
        given base.  The literal can be preceded by '+' or '-' and be surrounded
        by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
        Base 0 means to interpret the base from the string as an integer literal.
        >>> int('0b100', base=0)
        4
        """
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, args: Any) -> Any:
        ...

class fnumber_t:
    @property
    def fnum(self) -> Any: ...
    @property
    def nbytes(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: fnumber_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: fnumber_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: fnumber_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: fnumber_t) -> bool:
        ...
    def __lt__(self, r: fnumber_t) -> bool:
        ...
    def __ne__(self, r: fnumber_t) -> bool:
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
    def calc_max_exp(self) -> int:
        ...
    def compare(self, r: fnumber_t) -> int:
        ...
    def dereference_const_uint16(self) -> uint16:
        ...
    def dereference_uint16(self) -> uint16:
        ...
    def is_nan(self) -> bool:
        ...

class gco_info_t:
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def regnum(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def stkoff(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def append_to_list(self, list: mlist_t, mba: mba_t) -> bool:
        r"""Append operand info to LIST. This function converts IDA register number or stack offset to a decompiler list. 
                
        :param list: list to append to
        :param mba: microcode object
        """
        ...
    def cvt_to_ivl(self) -> vivl_t:
        r"""Convert operand info to VIVL. The returned VIVL can be used, for example, in a call of get_valranges(). 
                
        """
        ...
    def is_def(self) -> bool:
        ...
    def is_reg(self) -> bool:
        ...
    def is_use(self) -> bool:
        ...

class graph_chains_t(block_chains_vec_t):
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __ne__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: block_chains_t) -> None:
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
    def acquire(self) -> None:
        r"""Lock the chains.
        
        """
        ...
    def append(self, x: block_chains_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: block_chains_vec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def for_all_chains(self, cv: chain_visitor_t, gca_flags: int) -> int:
        r"""Visit all chains 
                
        :param cv: chain visitor
        :param gca_flags: combination of GCA_ bits
        """
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: block_chains_t, len: size_t) -> None:
        ...
    def insert(self, it: block_chains_t, x: block_chains_t) -> qvector:
        ...
    def is_locked(self) -> bool:
        r"""Are the chains locked? It is a good idea to lock the chains before using them. This ensures that they won't be recalculated and reallocated during the use. See the chain_keeper_t class for that. 
                
        """
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def release(self) -> None:
        r"""Unlock the chains.
        
        """
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: graph_chains_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class hexrays_failure_t:
    @property
    def code(self) -> Any: ...
    @property
    def errea(self) -> Any: ...
    @property
    def str(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def desc(self) -> str:
        ...

class hexwarn_t:
    @property
    def ea(self) -> Any: ...
    @property
    def id(self) -> Any: ...
    @property
    def text(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: hexwarn_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: hexwarn_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: hexwarn_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: hexwarn_t) -> bool:
        ...
    def __lt__(self, r: hexwarn_t) -> bool:
        ...
    def __ne__(self, r: hexwarn_t) -> bool:
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
    def compare(self, r: hexwarn_t) -> int:
        ...

class hexwarns_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: hexwarns_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: hexwarns_t) -> bool:
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
    def __setitem__(self, i: size_t, v: hexwarn_t) -> None:
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
    def add_unique(self, x: hexwarn_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: hexwarns_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: hexwarn_t) -> bool:
        ...
    def inject(self, s: hexwarn_t, len: size_t) -> None:
        ...
    def insert(self, it: hexwarn_t, x: hexwarn_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: hexwarns_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class history_item_t(ctext_position_t):
    @property
    def curr_ea(self) -> Any: ...
    @property
    def end(self) -> Any: ...
    @property
    def func_ea(self) -> Any: ...
    @property
    def lnnum(self) -> Any: ...
    @property
    def x(self) -> Any: ...
    @property
    def y(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ctext_position_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ctext_position_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: ctext_position_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: ctext_position_t) -> bool:
        ...
    def __lt__(self, r: ctext_position_t) -> bool:
        ...
    def __ne__(self, r: ctext_position_t) -> bool:
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
    def compare(self, r: ctext_position_t) -> int:
        ...
    def in_ctree(self, hdrlines: int) -> bool:
        r"""Is the cursor in the variable/type declaration area? 
                
        :param hdrlines: Number of lines of the declaration area
        """
        ...

class history_t(qvector_history_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_history_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
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
    def __ne__(self, r: qvector_history_t) -> bool:
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
    def __setitem__(self, i: size_t, v: history_item_t) -> None:
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
    def add_unique(self, x: history_item_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_history_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: history_item_t) -> bool:
        ...
    def inject(self, s: history_item_t, len: size_t) -> None:
        ...
    def insert(self, it: history_item_t, x: history_item_t) -> qvector:
        ...
    def pop(self) -> history_item_t:
        ...
    def pop_back(self) -> None:
        ...
    def push(self, v: history_item_t) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_history_t) -> None:
        ...
    def top(self, args: Any) -> history_item_t:
        ...
    def truncate(self) -> None:
        ...

class int64_emulator_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def get_mop_value(self, mop: mop_t) -> intval64_t:
        ...
    def minsn_value(self, insn: minsn_t) -> intval64_t:
        ...
    def mop_value(self, mop: mop_t) -> intval64_t:
        ...

class intval64_t:
    @property
    def size(self) -> Any: ...
    @property
    def val(self) -> Any: ...
    def __add__(self, o: intval64_t) -> intval64_t:
        ...
    def __and__(self, o: intval64_t) -> intval64_t:
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __div__(self, args: Any) -> Any:
        ...
    def __eq__(self, o: intval64_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, v: uint64 = 0, _s: int = 1) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __invert__(self) -> intval64_t:
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lshift__(self, o: intval64_t) -> intval64_t:
        ...
    def __lt__(self, o: intval64_t) -> bool:
        ...
    def __mod__(self, o: intval64_t) -> intval64_t:
        ...
    def __mul__(self, o: intval64_t) -> intval64_t:
        ...
    def __ne__(self, o: intval64_t) -> bool:
        ...
    def __neg__(self) -> intval64_t:
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __or__(self, o: intval64_t) -> intval64_t:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __rshift__(self, o: intval64_t) -> intval64_t:
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
    def __sub__(self, o: intval64_t) -> intval64_t:
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
    def __truediv__(self, args: Any) -> Any:
        ...
    def __xor__(self, o: intval64_t) -> intval64_t:
        ...
    def high(self, target_sz: int) -> intval64_t:
        ...
    def low(self, target_sz: int) -> intval64_t:
        ...
    def sar(self, o: intval64_t) -> intval64_t:
        ...
    def sdiv(self, o: intval64_t) -> intval64_t:
        ...
    def sext(self, target_sz: int) -> intval64_t:
        ...
    def smod(self, o: intval64_t) -> intval64_t:
        ...
    def sval(self) -> int64:
        ...
    def uval(self) -> uint64:
        ...
    def zext(self, target_sz: int) -> intval64_t:
        ...

class iterator:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, n: iterator) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, n: int = -1) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, n: iterator) -> bool:
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
    def __ref__(self) -> int:
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

class ivl_t(uval_ivl_t):
    @property
    def off(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ivl_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ivl_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: ivl_t) -> bool:
        ...
    def __init__(self, _off: int = 0, _size: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: ivl_t) -> bool:
        ...
    def __lt__(self, r: ivl_t) -> bool:
        ...
    def __ne__(self, r: ivl_t) -> bool:
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
    def clear(self) -> None:
        ...
    def compare(self, r: ivl_t) -> int:
        ...
    def contains(self, off2: int) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def extend_to_cover(self, r: ivl_t) -> bool:
        ...
    def includes(self, ivl: ivl_t) -> bool:
        ...
    def intersect(self, r: ivl_t) -> int:
        ...
    def last(self) -> int:
        ...
    def overlap(self, ivl: ivl_t) -> bool:
        ...
    def valid(self) -> bool:
        ...

class ivl_with_name_t:
    @property
    def ivl(self) -> Any: ...
    @property
    def part(self) -> Any: ...
    @property
    def whole(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class ivlset_t(uval_ivl_ivlset_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ivlset_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: ivlset_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: ivlset_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: ivlset_t) -> bool:
        ...
    def __lt__(self, r: ivlset_t) -> bool:
        ...
    def __ne__(self, r: ivlset_t) -> bool:
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
    def add(self, args: Any) -> voff_t:
        r"""This function has the following signatures:
        
            0. add(ivl: const ivl_t &) -> bool
            1. add(ea: ida_idaapi.ea_t, size: asize_t) -> bool
            2. add(ivs: const ivlset_t &) -> bool
        
        # 0: add(ivl: const ivl_t &) -> bool
        
        
        # 1: add(ea: ida_idaapi.ea_t, size: asize_t) -> bool
        
        
        # 2: add(ivs: const ivlset_t &) -> bool
        
        
        """
        ...
    def addmasked(self, ivs: ivlset_t, mask: ivl_t) -> bool:
        ...
    def all_values(self) -> bool:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: ivlset_t) -> int:
        ...
    def contains(self, off: int) -> bool:
        ...
    def count(self) -> int:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def getivl(self, idx: int) -> ivl_t:
        ...
    def has_common(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. has_common(ivl: const ivl_t &, strict: bool=false) -> bool
            1. has_common(ivs: const ivlset_t &) -> bool
        
        # 0: has_common(ivl: const ivl_t &, strict: bool=false) -> bool
        
        
        # 1: has_common(ivs: const ivlset_t &) -> bool
        
        
        """
        ...
    def includes(self, ivs: ivlset_t) -> bool:
        ...
    def intersect(self, ivs: ivlset_t) -> int:
        ...
    def lastivl(self) -> ivl_t:
        ...
    def nivls(self) -> int:
        ...
    def qclear(self) -> None:
        ...
    def set_all_values(self) -> None:
        ...
    def single_value(self, args: Any) -> bool:
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(ivl: const ivl_t &) -> bool
            1. sub(ea: ida_idaapi.ea_t, size: asize_t) -> bool
            2. sub(ivs: const ivlset_t &) -> bool
        
        # 0: sub(ivl: const ivl_t &) -> bool
        
        
        # 1: sub(ea: ida_idaapi.ea_t, size: asize_t) -> bool
        
        
        # 2: sub(ivs: const ivlset_t &) -> bool
        
        
        """
        ...
    def swap(self, r: uval_ivl_ivlset_t) -> None:
        ...

class lvar_locator_t:
    @property
    def defea(self) -> Any: ...
    @property
    def location(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: lvar_locator_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: lvar_locator_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: lvar_locator_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: lvar_locator_t) -> bool:
        ...
    def __lt__(self, r: lvar_locator_t) -> bool:
        ...
    def __ne__(self, r: lvar_locator_t) -> bool:
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
    def compare(self, r: lvar_locator_t) -> int:
        ...
    def get_reg1(self) -> mreg_t:
        r"""Get the register number of the variable.
        
        """
        ...
    def get_reg2(self) -> mreg_t:
        r"""Get the number of the second register (works only for ALOC_REG2 lvars)
        
        """
        ...
    def get_scattered(self) -> scattered_aloc_t:
        r"""Get information about scattered variable.
        
        """
        ...
    def get_stkoff(self) -> int:
        r"""Get offset of the varialbe in the stack frame. 
                
        :returns: a non-negative value for stack variables. The value is an offset from the bottom of the stack frame in terms of vd-offsets. negative values mean error (not a stack variable)
        """
        ...
    def is_reg1(self) -> bool:
        r"""Is variable located on one register?
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""Is variable located on two registers?
        
        """
        ...
    def is_reg_var(self) -> bool:
        r"""Is variable located on register(s)?
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""Is variable scattered?
        
        """
        ...
    def is_stk_var(self) -> bool:
        r"""Is variable located on the stack?
        
        """
        ...

class lvar_mapping_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: lvar_mapping_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: lvar_mapping_iterator_t) -> bool:
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

class lvar_mapping_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def at(self, _Keyval: lvar_locator_t) -> ivlset_t:
        ...
    def size(self) -> int:
        ...

class lvar_ref_t:
    @property
    def idx(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def off(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: lvar_ref_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: lvar_ref_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: lvar_ref_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: lvar_ref_t) -> bool:
        ...
    def __lt__(self, r: lvar_ref_t) -> bool:
        ...
    def __ne__(self, r: lvar_ref_t) -> bool:
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
    def compare(self, r: lvar_ref_t) -> int:
        ...
    def swap(self, r: lvar_ref_t) -> None:
        ...
    def var(self) -> lvar_t:
        r"""Retrieve the referenced variable.
        
        """
        ...

class lvar_saved_info_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def ll(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: lvar_saved_info_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: lvar_saved_info_t) -> bool:
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
    def clear_keep(self) -> None:
        ...
    def clr_nomap_lvar(self) -> None:
        ...
    def clr_noptr_lvar(self) -> None:
        ...
    def clr_split_lvar(self) -> None:
        ...
    def clr_unused_lvar(self) -> None:
        ...
    def has_info(self) -> bool:
        ...
    def is_kept(self) -> bool:
        ...
    def is_nomap_lvar(self) -> bool:
        ...
    def is_noptr_lvar(self) -> bool:
        ...
    def is_split_lvar(self) -> bool:
        ...
    def is_unused_lvar(self) -> bool:
        ...
    def set_keep(self) -> None:
        ...
    def set_nomap_lvar(self) -> None:
        ...
    def set_noptr_lvar(self) -> bool:
        ...
    def set_split_lvar(self) -> None:
        ...
    def set_unused_lvar(self) -> None:
        ...

class lvar_saved_infos_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: lvar_saved_infos_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: lvar_saved_infos_t) -> bool:
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
    def __setitem__(self, i: size_t, v: lvar_saved_info_t) -> None:
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
    def add_unique(self, x: lvar_saved_info_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: lvar_saved_infos_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: lvar_saved_info_t) -> bool:
        ...
    def inject(self, s: lvar_saved_info_t, len: size_t) -> None:
        ...
    def insert(self, it: lvar_saved_info_t, x: lvar_saved_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: lvar_saved_infos_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class lvar_t(lvar_locator_t):
    @property
    def cmt(self) -> Any: ...
    @property
    def defblk(self) -> Any: ...
    @property
    def defea(self) -> Any: ...
    @property
    def divisor(self) -> Any: ...
    @property
    def has_nice_name(self) -> Any: ...
    @property
    def has_user_info(self) -> Any: ...
    @property
    def has_user_name(self) -> Any: ...
    @property
    def has_user_type(self) -> Any: ...
    @property
    def is_arg_var(self) -> Any: ...
    @property
    def is_fake_var(self) -> Any: ...
    @property
    def is_floating_var(self) -> Any: ...
    @property
    def is_mapdst_var(self) -> Any: ...
    @property
    def is_overlapped_var(self) -> Any: ...
    @property
    def is_result_var(self) -> Any: ...
    @property
    def is_spoiled_var(self) -> Any: ...
    @property
    def is_unknown_width(self) -> Any: ...
    @property
    def location(self) -> Any: ...
    @property
    def mreg_done(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def tif(self) -> Any: ...
    @property
    def typed(self) -> Any: ...
    @property
    def used(self) -> Any: ...
    @property
    def width(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: lvar_locator_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: lvar_locator_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: lvar_locator_t) -> bool:
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: lvar_locator_t) -> bool:
        ...
    def __lt__(self, r: lvar_locator_t) -> bool:
        ...
    def __ne__(self, r: lvar_locator_t) -> bool:
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
    def accepts_type(self, t: tinfo_t, may_change_thisarg: bool = False) -> bool:
        r"""Check if the variable accept the specified type. Some types are forbidden (void, function types, wrong arrays, etc) 
                
        """
        ...
    def append_list(self, mba: mba_t, lst: mlist_t, pad_if_scattered: bool = False) -> None:
        r"""Append local variable to mlist. 
                
        :param mba: ptr to the current mba_t
        :param lst: list to append to
        :param pad_if_scattered: if true, append padding bytes in case of scattered lvar
        """
        ...
    def clear_used(self) -> None:
        ...
    def clr_arg_var(self) -> None:
        ...
    def clr_automapped(self) -> None:
        ...
    def clr_decl_unused(self) -> None:
        ...
    def clr_dummy_arg(self) -> None:
        ...
    def clr_fake_var(self) -> None:
        ...
    def clr_floating_var(self) -> None:
        ...
    def clr_mapdst_var(self) -> None:
        ...
    def clr_mreg_done(self) -> None:
        ...
    def clr_noptr_var(self) -> None:
        ...
    def clr_notarg(self) -> None:
        ...
    def clr_overlapped_var(self) -> None:
        ...
    def clr_scattered_arg(self) -> None:
        ...
    def clr_shared(self) -> None:
        ...
    def clr_split_var(self) -> None:
        ...
    def clr_spoiled_var(self) -> None:
        ...
    def clr_thisarg(self) -> None:
        ...
    def clr_unknown_width(self) -> None:
        ...
    def clr_used_byref(self) -> None:
        ...
    def clr_user_info(self) -> None:
        ...
    def clr_user_name(self) -> None:
        ...
    def clr_user_type(self) -> None:
        ...
    def compare(self, r: lvar_locator_t) -> int:
        ...
    def get_reg1(self) -> mreg_t:
        r"""Get the register number of the variable.
        
        """
        ...
    def get_reg2(self) -> mreg_t:
        r"""Get the number of the second register (works only for ALOC_REG2 lvars)
        
        """
        ...
    def get_scattered(self) -> scattered_aloc_t:
        r"""Get information about scattered variable.
        
        """
        ...
    def get_stkoff(self) -> int:
        r"""Get offset of the varialbe in the stack frame. 
                
        :returns: a non-negative value for stack variables. The value is an offset from the bottom of the stack frame in terms of vd-offsets. negative values mean error (not a stack variable)
        """
        ...
    def has_common(self, v: lvar_t) -> bool:
        r"""Do variables overlap?
        
        """
        ...
    def has_common_bit(self, loc: vdloc_t, width2: asize_t) -> bool:
        r"""Does the variable overlap with the specified location?
        
        """
        ...
    def has_regname(self) -> bool:
        r"""Has a register name? (like _RAX)
        
        """
        ...
    def in_asm(self) -> bool:
        r"""Is variable used in an instruction translated into __asm?
        
        """
        ...
    def is_aliasable(self, mba: mba_t) -> bool:
        r"""Is the variable aliasable? 
                
        :param mba: ptr to the current mba_t Aliasable variables may be modified indirectly (through a pointer)
        """
        ...
    def is_automapped(self) -> bool:
        r"""Was the variable automatically mapped to another variable?
        
        """
        ...
    def is_decl_unused(self) -> bool:
        r"""Was declared as __unused by the user? See CVAR_UNUSED.
        
        """
        ...
    def is_dummy_arg(self) -> bool:
        r"""Is a dummy argument (added to fill a hole in the argument list)
        
        """
        ...
    def is_noptr_var(self) -> bool:
        r"""Variable type should not be a pointer.
        
        """
        ...
    def is_notarg(self) -> bool:
        r"""Is a local variable? (local variable cannot be an input argument)
        
        """
        ...
    def is_reg1(self) -> bool:
        r"""Is variable located on one register?
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""Is variable located on two registers?
        
        """
        ...
    def is_reg_var(self) -> bool:
        r"""Is variable located on register(s)?
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""Is variable scattered?
        
        """
        ...
    def is_shared(self) -> bool:
        r"""Is lvar mapped to several chains.
        
        """
        ...
    def is_split_var(self) -> bool:
        r"""Is a split variable?
        
        """
        ...
    def is_stk_var(self) -> bool:
        r"""Is variable located on the stack?
        
        """
        ...
    def is_thisarg(self) -> bool:
        r"""Is 'this' argument of a C++ member function?
        
        """
        ...
    def is_used_byref(self) -> bool:
        r"""Was the address of the variable taken?
        
        """
        ...
    def set_arg_var(self) -> None:
        ...
    def set_automapped(self) -> None:
        ...
    def set_decl_unused(self) -> None:
        ...
    def set_dummy_arg(self) -> None:
        ...
    def set_fake_var(self) -> None:
        ...
    def set_final_lvar_type(self, t: tinfo_t) -> None:
        r"""Set final variable type.
        
        """
        ...
    def set_floating_var(self) -> None:
        ...
    def set_lvar_type(self, t: tinfo_t, may_fail: bool = False) -> bool:
        r"""Set variable type Note: this function does not modify the idb, only the lvar instance in the memory. For permanent changes see modify_user_lvars() Also, the variable type is not considered as final by the decompiler and may be modified later by the type derivation. In some cases set_final_var_type() may work better, but it does not do persistent changes to the database neither. 
                
        :param t: new type
        :param may_fail: if false and type is bad, interr
        :returns: success
        """
        ...
    def set_mapdst_var(self) -> None:
        ...
    def set_mreg_done(self) -> None:
        ...
    def set_non_typed(self) -> None:
        ...
    def set_noptr_var(self) -> None:
        ...
    def set_notarg(self) -> None:
        ...
    def set_overlapped_var(self) -> None:
        ...
    def set_scattered_arg(self) -> None:
        ...
    def set_shared(self) -> None:
        ...
    def set_split_var(self) -> None:
        ...
    def set_spoiled_var(self) -> None:
        ...
    def set_thisarg(self) -> None:
        ...
    def set_typed(self) -> None:
        ...
    def set_unknown_width(self) -> None:
        ...
    def set_used(self) -> None:
        ...
    def set_used_byref(self) -> None:
        ...
    def set_user_name(self) -> None:
        ...
    def set_user_type(self) -> None:
        ...
    def set_width(self, w: int, svw_flags: int = 0) -> bool:
        r"""Change the variable width. We call the variable size 'width', it is represents the number of bytes. This function may change the variable type using set_lvar_type(). 
                
        :param w: new width
        :param svw_flags: combination of SVW_... bits
        :returns: success
        """
        ...
    def type(self) -> tinfo_t:
        r"""Get variable type.
        
        """
        ...
    def was_scattered_arg(self) -> bool:
        r"""Was lvar transformed from a scattered argument?
        
        """
        ...

class lvar_uservec_t:
    @property
    def lmaps(self) -> Any: ...
    @property
    def lvvec(self) -> Any: ...
    @property
    def stkoff_delta(self) -> Any: ...
    @property
    def ulv_flags(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def find_info(self, vloc: lvar_locator_t) -> lvar_saved_info_t:
        r"""find saved user settings for given var
        
        """
        ...
    def keep_info(self, v: lvar_t) -> None:
        r"""Preserve user settings for given var.
        
        """
        ...
    def swap(self, r: lvar_uservec_t) -> None:
        ...

class lvars_t(qvector_lvar_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_lvar_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
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
    def __ne__(self, r: qvector_lvar_t) -> bool:
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
    def __setitem__(self, i: size_t, v: lvar_t) -> None:
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
    def add_unique(self, x: lvar_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_lvar_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, ll: lvar_locator_t) -> lvar_t:
        r"""Find a variable at the specified location. 
                
        :param ll: variable location
        :returns: pointer to variable or nullptr
        """
        ...
    def find_input_lvar(self, argloc: vdloc_t, _size: int) -> int:
        r"""Find an input variable at the specified location. 
                
        :param argloc: variable location
        :param _size: variable size in bytes
        :returns: -1 if failed, otherwise an index into 'vars'
        """
        ...
    def find_input_reg(self, reg: int, _size: int = 1) -> int:
        r"""Find an input register variable. 
                
        :param reg: register to find
        :param _size: variable size in bytes
        :returns: -1 if failed, otherwise an index into 'vars'
        """
        ...
    def find_lvar(self, location: vdloc_t, width: int, defblk: int = -1) -> int:
        r"""Find a variable at the specified location. 
                
        :param location: variable location
        :param width: variable size in bytes
        :param defblk: definition block of the lvar. -1 means any block
        :returns: -1 if failed, otherwise an index into 'vars'
        """
        ...
    def find_stkvar(self, spoff: int, width: int) -> int:
        r"""Find a stack variable at the specified location. 
                
        :param spoff: offset from the minimal sp
        :param width: variable size in bytes
        :returns: -1 if failed, otherwise an index into 'vars'
        """
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: lvar_t) -> bool:
        ...
    def inject(self, s: lvar_t, len: size_t) -> None:
        ...
    def insert(self, it: lvar_t, x: lvar_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_lvar_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class mba_range_iterator_t:
    @property
    def fii(self) -> Any: ...
    @property
    def rii(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def chunk(self) -> range_t:
        ...
    def is_snippet(self) -> bool:
        ...
    def next(self) -> merror_t:
        ...
    def set(self, mbr: mba_ranges_t) -> bool:
        ...

class mba_ranges_t:
    @property
    def pfn(self) -> Any: ...
    @property
    def ranges(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def is_fragmented(self) -> bool:
        ...
    def is_snippet(self) -> bool:
        ...
    def start(self) -> None:
        ...

class mba_t:
    @property
    def aliased_memory(self) -> Any: ...
    @property
    def argidx(self) -> Any: ...
    @property
    def blocks(self) -> Any: ...
    @property
    def cc(self) -> Any: ...
    @property
    def consumed_argregs(self) -> Any: ...
    @property
    def entry_ea(self) -> Any: ...
    @property
    def error_ea(self) -> Any: ...
    @property
    def error_strarg(self) -> Any: ...
    @property
    def final_type(self) -> Any: ...
    @property
    def first_epilog_ea(self) -> Any: ...
    @property
    def fpd(self) -> Any: ...
    @property
    def frregs(self) -> Any: ...
    @property
    def frsize(self) -> Any: ...
    @property
    def fti_flags(self) -> Any: ...
    @property
    def fullsize(self) -> Any: ...
    @property
    def gotoff_stkvars(self) -> Any: ...
    @property
    def idb_node(self) -> Any: ...
    @property
    def idb_spoiled(self) -> Any: ...
    @property
    def idb_type(self) -> Any: ...
    @property
    def inargoff(self) -> Any: ...
    @property
    def label(self) -> Any: ...
    @property
    def last_prolog_ea(self) -> Any: ...
    @property
    def maturity(self) -> Any: ...
    @property
    def mbr(self) -> Any: ...
    @property
    def minargref(self) -> Any: ...
    @property
    def minstkref(self) -> Any: ...
    @property
    def minstkref_ea(self) -> Any: ...
    @property
    def natural(self) -> Any: ...
    @property
    def nodel_memory(self) -> Any: ...
    @property
    def notes(self) -> Any: ...
    @property
    def npurged(self) -> Any: ...
    @property
    def occurred_warns(self) -> Any: ...
    @property
    def pfn_flags(self) -> Any: ...
    @property
    def qty(self) -> Any: ...
    @property
    def reqmat(self) -> Any: ...
    @property
    def restricted_memory(self) -> Any: ...
    @property
    def retsize(self) -> Any: ...
    @property
    def retvaridx(self) -> Any: ...
    @property
    def shadow_args(self) -> Any: ...
    @property
    def spd_adjust(self) -> Any: ...
    @property
    def spoiled_list(self) -> Any: ...
    @property
    def stacksize(self) -> Any: ...
    @property
    def std_ivls(self) -> Any: ...
    @property
    def tmpstk_size(self) -> Any: ...
    @property
    def vars(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def alloc_fict_ea(self, real_ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Allocate a fictional address. This function can be used to allocate a new unique address for a new instruction, if re-using any existing address leads to conflicts. For example, if the last instruction of the function modifies R0 and falls through to the next function, it will be a tail call: LDM R0!, {R4,R7} end of the function start of another function In this case R0 generates two different lvars at the same address:
        * one modified by LDM
        * another that represents the return value from the tail call
        
        
        Another example: a third-party plugin makes a copy of an instruction. This may lead to the generation of two variables at the same address. Example 3: fictional addresses can be used for new instructions created while modifying the microcode. This function can be used to allocate a new unique address for a new instruction or a variable. The fictional address is selected from an unallocated address range. 
                
        :param real_ea: real instruction address (BADADDR is ok too)
        :returns: a unique fictional address
        """
        ...
    def alloc_kreg(self, size: size_t, check_size: bool = True) -> mreg_t:
        r"""Allocate a kernel register. 
                
        :param size: size of the register in bytes
        :param check_size: if true, only the sizes that correspond to a size of a basic type will be accepted.
        :returns: allocated register. mr_none means failure.
        """
        ...
    def alloc_lvars(self) -> None:
        r"""Allocate local variables. Must be called only immediately after optimize_global(), with no modifications to the microcode. Converts registers, stack variables, and similar operands into mop_l. This call will not fail because all necessary checks were performed in optimize_global(). After this call the microcode reaches its final state. 
                
        """
        ...
    def analyze_calls(self, acflags: int) -> int:
        r"""Analyze calls and determine calling conventions. 
                
        :param acflags: permitted actions that are necessary for successful detection of calling conventions. See Bits for analyze_calls()
        :returns: number of calls. -1 means error.
        """
        ...
    def arg(self, n: int) -> lvar_t:
        r"""Get input argument of the decompiled function. 
                
        :param n: argument number (0..nargs-1)
        """
        ...
    def argbase(self) -> int:
        ...
    def argidx_ok(self) -> bool:
        ...
    def argidx_sorted(self) -> bool:
        ...
    def bad_call_sp_detected(self) -> bool:
        ...
    def build_graph(self) -> merror_t:
        r"""Build control flow graph. This function may be called only once. It calculates the type of each basic block and the adjacency list. optimize_local() calls this function if necessary. You need to call this function only before MMAT_LOCOPT. 
                
        :returns: error code
        """
        ...
    def calc_shins_flags(self) -> int:
        ...
    def callinfo_built(self) -> bool:
        ...
    def chain_varnums_ok(self) -> bool:
        ...
    def clr_cdtr(self) -> None:
        ...
    def clr_mba_flags(self, f: int) -> None:
        ...
    def clr_mba_flags2(self, f: int) -> None:
        ...
    def code16_bit_removed(self) -> bool:
        ...
    def common_stkvars_stkargs(self) -> bool:
        ...
    def copy_block(self, blk: mblock_t, new_serial: int, cpblk_flags: int = 3) -> mblock_t:
        r"""Make a copy of a block. This function makes a simple copy of the block. It does not fix the predecessor and successor lists, they must be fixed if necessary. 
                
        :param blk: block to copy
        :param new_serial: position of the copied block
        :param cpblk_flags: combination of Batch decompilation bits... bits
        :returns: pointer to the new copy
        """
        ...
    def create_helper_call(self, ea: ida_idaapi.ea_t, helper: str, rettype: tinfo_t = None, callargs: mcallargs_t = None, out: mop_t = None) -> minsn_t:
        r"""Create a call of a helper function. 
                
        :param ea: The desired address of the instruction
        :param helper: The helper name
        :param rettype: The return type (nullptr or empty type means 'void')
        :param callargs: The helper arguments (nullptr-no arguments)
        :param out: The operand where the call result should be stored. If this argument is not nullptr, "mov helper_call(), out" will be generated. Otherwise "call helper()" will be generated. Note: the size of this operand must be equal to the RETTYPE size
        :returns: pointer to the created instruction or nullptr if error
        """
        ...
    def deleted_pairs(self) -> bool:
        ...
    def deserialize(self, bytes: uchar) -> mba_t:
        r"""Deserialize a byte sequence into mbl array. 
                
        :param bytes: pointer to the beginning of the byte sequence.
        :returns: new mbl array
        """
        ...
    def display_numaddrs(self) -> bool:
        ...
    def display_valnums(self) -> bool:
        ...
    def dump(self) -> None:
        r"""Dump microcode to a file. The file will be created in the directory pointed by IDA_DUMPDIR envvar. Dump will be created only if IDA is run under debugger. 
                
        """
        ...
    def dump_mba(self, _verify: bool, title: str) -> None:
        ...
    def find_mop(self, ctx: op_parent_info_t, ea: ida_idaapi.ea_t, is_dest: bool, list: mlist_t) -> mop_t:
        r"""Find an operand in the microcode. This function tries to find the operand that matches LIST. Any operand that overlaps with LIST is considered as a match. 
                
        :param ctx: context information for the result
        :param ea: desired address of the operand. BADADDR means to accept any address.
        :param is_dest: search for destination operand? this argument may be ignored if the exact match could not be found
        :param list: list of locations the correspond to the operand
        :returns: pointer to the operand or nullptr.
        """
        ...
    def for_all_insns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all instructions. This function visits all instruction and subinstructions. 
                
        :param mv: instruction visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def for_all_ops(self, mv: mop_visitor_t) -> int:
        r"""Visit all operands of all instructions. 
                
        :param mv: operand visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def for_all_topinsns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all top level instructions. 
                
        :param mv: instruction visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def free_kreg(self, reg: mreg_t, size: size_t) -> None:
        r"""Free a kernel register. If wrong arguments are passed, this function will generate an internal error. 
                
        :param reg: a previously allocated kernel register
        :param size: size of the register in bytes
        """
        ...
    def generated_asserts(self) -> bool:
        ...
    def get_args_region(self) -> ivl_t:
        ...
    def get_curfunc(self) -> Optional[func_t]:
        ...
    def get_func_output_lists(self, args: Any) -> None:
        r"""Prepare the lists of registers & memory that are defined/killed by a function 
                
        :param return_regs: defined regs to return (eax,edx)
        :param spoiled: spoiled regs (flags,ecx,mem)
        :param type: the function type
        :param call_ea: the call insn address (if known)
        :param tail_call: is it the tail call?
        """
        ...
    def get_graph(self) -> mbl_graph_t:
        r"""Get control graph. Call build_graph() if you need the graph before MMAT_LOCOPT. 
                
        """
        ...
    def get_ida_argloc(self, v: lvar_t) -> argloc_t:
        ...
    def get_lvars_region(self) -> ivl_t:
        ...
    def get_mba_flags(self) -> int:
        ...
    def get_mba_flags2(self) -> int:
        ...
    def get_mblock(self, n: uint) -> mblock_t:
        r"""Get basic block by its serial number.
        
        """
        ...
    def get_shadow_region(self) -> ivl_t:
        ...
    def get_stack_region(self) -> ivl_t:
        ...
    def get_std_region(self, idx: memreg_index_t) -> ivl_t:
        r"""Get information about various memory regions. We map the stack frame to the global memory, to some unused range. 
                
        """
        ...
    def graph_insns(self) -> bool:
        ...
    def has_bad_sp(self) -> bool:
        ...
    def has_outlines(self) -> bool:
        ...
    def has_over_chains(self) -> bool:
        ...
    def has_passregs(self) -> bool:
        ...
    def has_stack_retval(self) -> bool:
        ...
    def idaloc2vd(self, loc: argloc_t, width: int) -> vdloc_t:
        ...
    def inline_func(self, cdg: codegen_t, blknum: int, ranges: mba_ranges_t, decomp_flags: int = 0, inline_flags: int = 0) -> merror_t:
        r"""Inline a range. This function may be called only during the initial microcode generation phase. 
                
        :param cdg: the codegenerator object
        :param blknum: the block contaning the call/jump instruction to inline
        :param ranges: the set of ranges to inline. in the case of multiple calls to inline_func(), ranges will be compared using their start addresses. if two ranges have the same address, they will be considered the same.
        :param decomp_flags: combination of decompile() flags bits
        :param inline_flags: combination of inline_func() flags bits
        :returns: error code
        """
        ...
    def insert_block(self, bblk: int) -> mblock_t:
        r"""Insert a block in the middle of the mbl array. The very first block of microcode must be empty, it is the entry block. The very last block of microcode must be BLT_STOP, it is the exit block. Therefore inserting a new block before the entry point or after the exit block is not a good idea. 
                
        :param bblk: the new block will be inserted before BBLK
        :returns: ptr to the new block
        """
        ...
    def is_cdtr(self) -> bool:
        ...
    def is_ctr(self) -> bool:
        ...
    def is_dtr(self) -> bool:
        ...
    def is_pattern(self) -> bool:
        ...
    def is_snippet(self) -> bool:
        ...
    def is_stkarg(self, v: lvar_t) -> bool:
        ...
    def is_thunk(self) -> bool:
        ...
    def loaded_gdl(self) -> bool:
        ...
    def locate_stkpnt(self, ea: ida_idaapi.ea_t) -> stkpnt_t:
        ...
    def lvar_names_ok(self) -> bool:
        ...
    def lvars_allocated(self) -> bool:
        ...
    def lvars_renamed(self) -> bool:
        ...
    def map_fict_ea(self, fict_ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Resolve a fictional address. This function provides a reverse of the mapping made by alloc_fict_ea(). 
                
        :param fict_ea: fictional definition address
        :returns: the real instruction address
        """
        ...
    def mark_chains_dirty(self) -> None:
        r"""Mark the microcode use-def chains dirty. Call this function is any inter-block data dependencies got changed because of your modifications to the microcode. Failing to do so may cause an internal error. 
                
        """
        ...
    def may_refine_rettype(self) -> bool:
        ...
    def merge_blocks(self) -> bool:
        r"""Merge blocks. This function merges blocks constituting linear flow. It calls remove_empty_and_unreachable_blocks() as well. 
                
        :returns: true if changed any blocks
        """
        ...
    def optimize_global(self) -> merror_t:
        r"""Optimize microcode globally. This function applies various optimization methods until we reach the fixed point. After that it preallocates lvars unless reqmat forbids it. 
                
        :returns: error code
        """
        ...
    def optimize_local(self, locopt_bits: int) -> int:
        r"""Optimize each basic block locally 
                
        :param locopt_bits: combination of Bits for optimize_local() bits
        :returns: number of changes. 0 means nothing changed This function is called by the decompiler, usually there is no need to call it explicitly.
        """
        ...
    def optimized(self) -> bool:
        ...
    def precise_defeas(self) -> bool:
        ...
    def prop_complex(self) -> bool:
        ...
    def propagated_asserts(self) -> bool:
        ...
    def really_alloc(self) -> bool:
        ...
    def regargs_is_not_aligned(self) -> bool:
        ...
    def remove_block(self, blk: mblock_t) -> bool:
        r"""Delete a block. 
                
        :param blk: block to delete
        :returns: true if at least one of the other blocks became empty or unreachable
        """
        ...
    def remove_blocks(self, start_blk: int, end_blk: int) -> bool:
        ...
    def remove_empty_and_unreachable_blocks(self) -> bool:
        r"""Delete all empty and unreachable blocks. Blocks marked with MBL_KEEP won't be deleted. 
                
        """
        ...
    def returns_fpval(self) -> bool:
        ...
    def rtype_refined(self) -> bool:
        ...
    def save_snapshot(self, description: str) -> None:
        r"""Create and save microcode snapshot.
        
        """
        ...
    def saverest_done(self) -> bool:
        ...
    def serialize(self) -> None:
        r"""Serialize mbl array into a sequence of bytes.
        
        """
        ...
    def set_lvar_name(self, v: lvar_t, name: str, flagbits: int) -> bool:
        ...
    def set_maturity(self, mat: mba_maturity_t) -> merror_t:
        r"""Set maturity level. 
                
        :param mat: new maturity level
        :returns: error code Plugins may use this function to skip some parts of the analysis. The maturity level cannot be decreased.
        """
        ...
    def set_mba_flags(self, f: int) -> None:
        ...
    def set_mba_flags2(self, f: int) -> None:
        ...
    def set_nice_lvar_name(self, v: lvar_t, name: str) -> bool:
        ...
    def set_user_lvar_name(self, v: lvar_t, name: str) -> bool:
        ...
    def short_display(self) -> bool:
        ...
    def should_beautify(self) -> bool:
        ...
    def show_reduction(self) -> bool:
        ...
    def split_block(self, blk: mblock_t, start_insn: minsn_t) -> mblock_t:
        r"""Split a block: insert a new one after the block, move some instructions to new block 
                
        :param blk: block to be split
        :param start_insn: all instructions to be moved to new block: starting with this one up to the end
        :returns: ptr to the new block
        """
        ...
    def stkoff_ida2vd(self, off: int) -> int:
        ...
    def stkoff_vd2ida(self, off: int) -> int:
        ...
    def term(self) -> None:
        ...
    def use_frame(self) -> bool:
        ...
    def use_wingraph32(self) -> bool:
        ...
    def valranges_done(self) -> bool:
        ...
    def vd2idaloc(self, args: Any) -> argloc_t:
        r"""This function has the following signatures:
        
            0. vd2idaloc(loc: const vdloc_t &, width: int) -> argloc_t
            1. vd2idaloc(loc: const vdloc_t &, width: int, spd: int) -> argloc_t
        
        # 0: vd2idaloc(loc: const vdloc_t &, width: int) -> argloc_t
        
        
        # 1: vd2idaloc(loc: const vdloc_t &, width: int, spd: int) -> argloc_t
        
        
        """
        ...
    def verify(self, always: bool) -> None:
        r"""Verify microcode consistency. 
                
        :param always: if false, the check will be performed only if ida runs under debugger If any inconsistency is discovered, an internal error will be generated. We strongly recommend you to call this function before returing control to the decompiler from your callbacks, in the case if you modified the microcode. If the microcode is inconsistent, this function will generate an internal error. We provide the source code of this function in the plugins/hexrays_sdk/verifier directory for your reference.
        """
        ...
    def write_to_const_detected(self) -> bool:
        ...

class mbl_array_t:
    @property
    def aliased_memory(self) -> Any: ...
    @property
    def argidx(self) -> Any: ...
    @property
    def blocks(self) -> Any: ...
    @property
    def cc(self) -> Any: ...
    @property
    def consumed_argregs(self) -> Any: ...
    @property
    def entry_ea(self) -> Any: ...
    @property
    def error_ea(self) -> Any: ...
    @property
    def error_strarg(self) -> Any: ...
    @property
    def final_type(self) -> Any: ...
    @property
    def first_epilog_ea(self) -> Any: ...
    @property
    def fpd(self) -> Any: ...
    @property
    def frregs(self) -> Any: ...
    @property
    def frsize(self) -> Any: ...
    @property
    def fti_flags(self) -> Any: ...
    @property
    def fullsize(self) -> Any: ...
    @property
    def gotoff_stkvars(self) -> Any: ...
    @property
    def idb_node(self) -> Any: ...
    @property
    def idb_spoiled(self) -> Any: ...
    @property
    def idb_type(self) -> Any: ...
    @property
    def inargoff(self) -> Any: ...
    @property
    def label(self) -> Any: ...
    @property
    def last_prolog_ea(self) -> Any: ...
    @property
    def maturity(self) -> Any: ...
    @property
    def mbr(self) -> Any: ...
    @property
    def minargref(self) -> Any: ...
    @property
    def minstkref(self) -> Any: ...
    @property
    def minstkref_ea(self) -> Any: ...
    @property
    def natural(self) -> Any: ...
    @property
    def nodel_memory(self) -> Any: ...
    @property
    def notes(self) -> Any: ...
    @property
    def npurged(self) -> Any: ...
    @property
    def occurred_warns(self) -> Any: ...
    @property
    def pfn_flags(self) -> Any: ...
    @property
    def qty(self) -> Any: ...
    @property
    def reqmat(self) -> Any: ...
    @property
    def restricted_memory(self) -> Any: ...
    @property
    def retsize(self) -> Any: ...
    @property
    def retvaridx(self) -> Any: ...
    @property
    def shadow_args(self) -> Any: ...
    @property
    def spd_adjust(self) -> Any: ...
    @property
    def spoiled_list(self) -> Any: ...
    @property
    def stacksize(self) -> Any: ...
    @property
    def std_ivls(self) -> Any: ...
    @property
    def tmpstk_size(self) -> Any: ...
    @property
    def vars(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def alloc_fict_ea(self, real_ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Allocate a fictional address. This function can be used to allocate a new unique address for a new instruction, if re-using any existing address leads to conflicts. For example, if the last instruction of the function modifies R0 and falls through to the next function, it will be a tail call: LDM R0!, {R4,R7} end of the function start of another function In this case R0 generates two different lvars at the same address:
        * one modified by LDM
        * another that represents the return value from the tail call
        
        
        Another example: a third-party plugin makes a copy of an instruction. This may lead to the generation of two variables at the same address. Example 3: fictional addresses can be used for new instructions created while modifying the microcode. This function can be used to allocate a new unique address for a new instruction or a variable. The fictional address is selected from an unallocated address range. 
                
        :param real_ea: real instruction address (BADADDR is ok too)
        :returns: a unique fictional address
        """
        ...
    def alloc_kreg(self, size: size_t, check_size: bool = True) -> mreg_t:
        r"""Allocate a kernel register. 
                
        :param size: size of the register in bytes
        :param check_size: if true, only the sizes that correspond to a size of a basic type will be accepted.
        :returns: allocated register. mr_none means failure.
        """
        ...
    def alloc_lvars(self) -> None:
        r"""Allocate local variables. Must be called only immediately after optimize_global(), with no modifications to the microcode. Converts registers, stack variables, and similar operands into mop_l. This call will not fail because all necessary checks were performed in optimize_global(). After this call the microcode reaches its final state. 
                
        """
        ...
    def analyze_calls(self, acflags: int) -> int:
        r"""Analyze calls and determine calling conventions. 
                
        :param acflags: permitted actions that are necessary for successful detection of calling conventions. See Bits for analyze_calls()
        :returns: number of calls. -1 means error.
        """
        ...
    def arg(self, n: int) -> lvar_t:
        r"""Get input argument of the decompiled function. 
                
        :param n: argument number (0..nargs-1)
        """
        ...
    def argbase(self) -> int:
        ...
    def argidx_ok(self) -> bool:
        ...
    def argidx_sorted(self) -> bool:
        ...
    def bad_call_sp_detected(self) -> bool:
        ...
    def build_graph(self) -> merror_t:
        r"""Build control flow graph. This function may be called only once. It calculates the type of each basic block and the adjacency list. optimize_local() calls this function if necessary. You need to call this function only before MMAT_LOCOPT. 
                
        :returns: error code
        """
        ...
    def calc_shins_flags(self) -> int:
        ...
    def callinfo_built(self) -> bool:
        ...
    def chain_varnums_ok(self) -> bool:
        ...
    def clr_cdtr(self) -> None:
        ...
    def clr_mba_flags(self, f: int) -> None:
        ...
    def clr_mba_flags2(self, f: int) -> None:
        ...
    def code16_bit_removed(self) -> bool:
        ...
    def common_stkvars_stkargs(self) -> bool:
        ...
    def copy_block(self, blk: mblock_t, new_serial: int, cpblk_flags: int = 3) -> mblock_t:
        r"""Make a copy of a block. This function makes a simple copy of the block. It does not fix the predecessor and successor lists, they must be fixed if necessary. 
                
        :param blk: block to copy
        :param new_serial: position of the copied block
        :param cpblk_flags: combination of Batch decompilation bits... bits
        :returns: pointer to the new copy
        """
        ...
    def create_helper_call(self, ea: ida_idaapi.ea_t, helper: str, rettype: tinfo_t = None, callargs: mcallargs_t = None, out: mop_t = None) -> minsn_t:
        r"""Create a call of a helper function. 
                
        :param ea: The desired address of the instruction
        :param helper: The helper name
        :param rettype: The return type (nullptr or empty type means 'void')
        :param callargs: The helper arguments (nullptr-no arguments)
        :param out: The operand where the call result should be stored. If this argument is not nullptr, "mov helper_call(), out" will be generated. Otherwise "call helper()" will be generated. Note: the size of this operand must be equal to the RETTYPE size
        :returns: pointer to the created instruction or nullptr if error
        """
        ...
    def deleted_pairs(self) -> bool:
        ...
    def deserialize(self, bytes: uchar) -> mba_t:
        r"""Deserialize a byte sequence into mbl array. 
                
        :param bytes: pointer to the beginning of the byte sequence.
        :returns: new mbl array
        """
        ...
    def display_numaddrs(self) -> bool:
        ...
    def display_valnums(self) -> bool:
        ...
    def dump(self) -> None:
        r"""Dump microcode to a file. The file will be created in the directory pointed by IDA_DUMPDIR envvar. Dump will be created only if IDA is run under debugger. 
                
        """
        ...
    def dump_mba(self, _verify: bool, title: str) -> None:
        ...
    def find_mop(self, ctx: op_parent_info_t, ea: ida_idaapi.ea_t, is_dest: bool, list: mlist_t) -> mop_t:
        r"""Find an operand in the microcode. This function tries to find the operand that matches LIST. Any operand that overlaps with LIST is considered as a match. 
                
        :param ctx: context information for the result
        :param ea: desired address of the operand. BADADDR means to accept any address.
        :param is_dest: search for destination operand? this argument may be ignored if the exact match could not be found
        :param list: list of locations the correspond to the operand
        :returns: pointer to the operand or nullptr.
        """
        ...
    def for_all_insns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all instructions. This function visits all instruction and subinstructions. 
                
        :param mv: instruction visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def for_all_ops(self, mv: mop_visitor_t) -> int:
        r"""Visit all operands of all instructions. 
                
        :param mv: operand visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def for_all_topinsns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all top level instructions. 
                
        :param mv: instruction visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def free_kreg(self, reg: mreg_t, size: size_t) -> None:
        r"""Free a kernel register. If wrong arguments are passed, this function will generate an internal error. 
                
        :param reg: a previously allocated kernel register
        :param size: size of the register in bytes
        """
        ...
    def generated_asserts(self) -> bool:
        ...
    def get_args_region(self) -> ivl_t:
        ...
    def get_curfunc(self) -> Optional[func_t]:
        ...
    def get_func_output_lists(self, args: Any) -> None:
        r"""Prepare the lists of registers & memory that are defined/killed by a function 
                
        :param return_regs: defined regs to return (eax,edx)
        :param spoiled: spoiled regs (flags,ecx,mem)
        :param type: the function type
        :param call_ea: the call insn address (if known)
        :param tail_call: is it the tail call?
        """
        ...
    def get_graph(self) -> mbl_graph_t:
        r"""Get control graph. Call build_graph() if you need the graph before MMAT_LOCOPT. 
                
        """
        ...
    def get_ida_argloc(self, v: lvar_t) -> argloc_t:
        ...
    def get_lvars_region(self) -> ivl_t:
        ...
    def get_mba_flags(self) -> int:
        ...
    def get_mba_flags2(self) -> int:
        ...
    def get_mblock(self, n: uint) -> mblock_t:
        r"""Get basic block by its serial number.
        
        """
        ...
    def get_shadow_region(self) -> ivl_t:
        ...
    def get_stack_region(self) -> ivl_t:
        ...
    def get_std_region(self, idx: memreg_index_t) -> ivl_t:
        r"""Get information about various memory regions. We map the stack frame to the global memory, to some unused range. 
                
        """
        ...
    def graph_insns(self) -> bool:
        ...
    def has_bad_sp(self) -> bool:
        ...
    def has_outlines(self) -> bool:
        ...
    def has_over_chains(self) -> bool:
        ...
    def has_passregs(self) -> bool:
        ...
    def has_stack_retval(self) -> bool:
        ...
    def idaloc2vd(self, loc: argloc_t, width: int) -> vdloc_t:
        ...
    def inline_func(self, cdg: codegen_t, blknum: int, ranges: mba_ranges_t, decomp_flags: int = 0, inline_flags: int = 0) -> merror_t:
        r"""Inline a range. This function may be called only during the initial microcode generation phase. 
                
        :param cdg: the codegenerator object
        :param blknum: the block contaning the call/jump instruction to inline
        :param ranges: the set of ranges to inline. in the case of multiple calls to inline_func(), ranges will be compared using their start addresses. if two ranges have the same address, they will be considered the same.
        :param decomp_flags: combination of decompile() flags bits
        :param inline_flags: combination of inline_func() flags bits
        :returns: error code
        """
        ...
    def insert_block(self, bblk: int) -> mblock_t:
        r"""Insert a block in the middle of the mbl array. The very first block of microcode must be empty, it is the entry block. The very last block of microcode must be BLT_STOP, it is the exit block. Therefore inserting a new block before the entry point or after the exit block is not a good idea. 
                
        :param bblk: the new block will be inserted before BBLK
        :returns: ptr to the new block
        """
        ...
    def is_cdtr(self) -> bool:
        ...
    def is_ctr(self) -> bool:
        ...
    def is_dtr(self) -> bool:
        ...
    def is_pattern(self) -> bool:
        ...
    def is_snippet(self) -> bool:
        ...
    def is_stkarg(self, v: lvar_t) -> bool:
        ...
    def is_thunk(self) -> bool:
        ...
    def loaded_gdl(self) -> bool:
        ...
    def locate_stkpnt(self, ea: ida_idaapi.ea_t) -> stkpnt_t:
        ...
    def lvar_names_ok(self) -> bool:
        ...
    def lvars_allocated(self) -> bool:
        ...
    def lvars_renamed(self) -> bool:
        ...
    def map_fict_ea(self, fict_ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Resolve a fictional address. This function provides a reverse of the mapping made by alloc_fict_ea(). 
                
        :param fict_ea: fictional definition address
        :returns: the real instruction address
        """
        ...
    def mark_chains_dirty(self) -> None:
        r"""Mark the microcode use-def chains dirty. Call this function is any inter-block data dependencies got changed because of your modifications to the microcode. Failing to do so may cause an internal error. 
                
        """
        ...
    def may_refine_rettype(self) -> bool:
        ...
    def merge_blocks(self) -> bool:
        r"""Merge blocks. This function merges blocks constituting linear flow. It calls remove_empty_and_unreachable_blocks() as well. 
                
        :returns: true if changed any blocks
        """
        ...
    def optimize_global(self) -> merror_t:
        r"""Optimize microcode globally. This function applies various optimization methods until we reach the fixed point. After that it preallocates lvars unless reqmat forbids it. 
                
        :returns: error code
        """
        ...
    def optimize_local(self, locopt_bits: int) -> int:
        r"""Optimize each basic block locally 
                
        :param locopt_bits: combination of Bits for optimize_local() bits
        :returns: number of changes. 0 means nothing changed This function is called by the decompiler, usually there is no need to call it explicitly.
        """
        ...
    def optimized(self) -> bool:
        ...
    def precise_defeas(self) -> bool:
        ...
    def prop_complex(self) -> bool:
        ...
    def propagated_asserts(self) -> bool:
        ...
    def really_alloc(self) -> bool:
        ...
    def regargs_is_not_aligned(self) -> bool:
        ...
    def remove_block(self, blk: mblock_t) -> bool:
        r"""Delete a block. 
                
        :param blk: block to delete
        :returns: true if at least one of the other blocks became empty or unreachable
        """
        ...
    def remove_blocks(self, start_blk: int, end_blk: int) -> bool:
        ...
    def remove_empty_and_unreachable_blocks(self) -> bool:
        r"""Delete all empty and unreachable blocks. Blocks marked with MBL_KEEP won't be deleted. 
                
        """
        ...
    def returns_fpval(self) -> bool:
        ...
    def rtype_refined(self) -> bool:
        ...
    def save_snapshot(self, description: str) -> None:
        r"""Create and save microcode snapshot.
        
        """
        ...
    def saverest_done(self) -> bool:
        ...
    def serialize(self) -> None:
        r"""Serialize mbl array into a sequence of bytes.
        
        """
        ...
    def set_lvar_name(self, v: lvar_t, name: str, flagbits: int) -> bool:
        ...
    def set_maturity(self, mat: mba_maturity_t) -> merror_t:
        r"""Set maturity level. 
                
        :param mat: new maturity level
        :returns: error code Plugins may use this function to skip some parts of the analysis. The maturity level cannot be decreased.
        """
        ...
    def set_mba_flags(self, f: int) -> None:
        ...
    def set_mba_flags2(self, f: int) -> None:
        ...
    def set_nice_lvar_name(self, v: lvar_t, name: str) -> bool:
        ...
    def set_user_lvar_name(self, v: lvar_t, name: str) -> bool:
        ...
    def short_display(self) -> bool:
        ...
    def should_beautify(self) -> bool:
        ...
    def show_reduction(self) -> bool:
        ...
    def split_block(self, blk: mblock_t, start_insn: minsn_t) -> mblock_t:
        r"""Split a block: insert a new one after the block, move some instructions to new block 
                
        :param blk: block to be split
        :param start_insn: all instructions to be moved to new block: starting with this one up to the end
        :returns: ptr to the new block
        """
        ...
    def stkoff_ida2vd(self, off: int) -> int:
        ...
    def stkoff_vd2ida(self, off: int) -> int:
        ...
    def term(self) -> None:
        ...
    def use_frame(self) -> bool:
        ...
    def use_wingraph32(self) -> bool:
        ...
    def valranges_done(self) -> bool:
        ...
    def vd2idaloc(self, args: Any) -> argloc_t:
        r"""This function has the following signatures:
        
            0. vd2idaloc(loc: const vdloc_t &, width: int) -> argloc_t
            1. vd2idaloc(loc: const vdloc_t &, width: int, spd: int) -> argloc_t
        
        # 0: vd2idaloc(loc: const vdloc_t &, width: int) -> argloc_t
        
        
        # 1: vd2idaloc(loc: const vdloc_t &, width: int, spd: int) -> argloc_t
        
        
        """
        ...
    def verify(self, always: bool) -> None:
        r"""Verify microcode consistency. 
                
        :param always: if false, the check will be performed only if ida runs under debugger If any inconsistency is discovered, an internal error will be generated. We strongly recommend you to call this function before returing control to the decompiler from your callbacks, in the case if you modified the microcode. If the microcode is inconsistent, this function will generate an internal error. We provide the source code of this function in the plugins/hexrays_sdk/verifier directory for your reference.
        """
        ...
    def write_to_const_detected(self) -> bool:
        ...

class mbl_graph_t(simple_graph_t):
    @property
    def colored_gdl_edges(self) -> Any: ...
    @property
    def title(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def begin(self) -> simple_graph_t:
        ...
    def compute_dominators(self, domin: array_of_node_bitset_t, post: bool = False) -> None:
        ...
    def compute_immediate_dominators(self, domin: array_of_node_bitset_t, idomin: intvec_t, post: bool = False) -> None:
        ...
    def depth_first_postorder(self, post: node_ordering_t) -> int:
        ...
    def depth_first_preorder(self, pre: node_ordering_t) -> int:
        ...
    def edge(self, node: int, i: int, ispred: bool) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def entry(self) -> int:
        ...
    def exists(self, node: int) -> bool:
        ...
    def exit(self) -> int:
        ...
    def front(self) -> block_chains_t:
        ...
    def get_chain_stamp(self) -> int:
        ...
    def get_du(self, gctype: gctype_t) -> graph_chains_t:
        r"""Get def-use chains.
        
        """
        ...
    def get_edge_color(self, i: int, j: int) -> bgcolor_t:
        ...
    def get_mblock(self, n: int) -> mblock_t:
        ...
    def get_node_color(self, n: int) -> bgcolor_t:
        ...
    def get_node_label(self, n: int) -> char:
        ...
    def get_ud(self, gctype: gctype_t) -> graph_chains_t:
        r"""Get use-def chains.
        
        """
        ...
    def goup(self, node: int) -> int:
        ...
    def inc(self, p: iterator, n: int = 1) -> None:
        ...
    def is_du_chain_dirty(self, gctype: gctype_t) -> bool:
        r"""Is the def-use chain of the specified kind dirty?
        
        """
        ...
    def is_redefined_globally(self, args: Any) -> bool:
        r"""Is LIST redefined in the graph?
        
        """
        ...
    def is_ud_chain_dirty(self, gctype: gctype_t) -> bool:
        r"""Is the use-def chain of the specified kind dirty?
        
        """
        ...
    def is_used_globally(self, args: Any) -> bool:
        r"""Is LIST used in the graph?
        
        """
        ...
    def nedge(self, node: int, ispred: bool) -> int:
        ...
    def node_qty(self) -> int:
        ...
    def npred(self, node: int) -> int:
        ...
    def nsucc(self, node: int) -> int:
        ...
    def pred(self, node: int, i: int) -> int:
        ...
    def print_edge(self, fp: FILE, i: int, j: int) -> bool:
        ...
    def print_graph_attributes(self, fp: FILE) -> None:
        ...
    def print_node(self, fp: FILE, n: int) -> bool:
        ...
    def print_node_attributes(self, fp: FILE, n: int) -> None:
        ...
    def size(self) -> int:
        ...
    def succ(self, node: int, i: int) -> int:
        ...

class mblock_t:
    @property
    def dead_at_start(self) -> Any: ...
    @property
    def dnu(self) -> Any: ...
    @property
    def end(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def head(self) -> Any: ...
    @property
    def maxbsp(self) -> Any: ...
    @property
    def maybdef(self) -> Any: ...
    @property
    def maybuse(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def minbargref(self) -> Any: ...
    @property
    def minbstkref(self) -> Any: ...
    @property
    def mustbdef(self) -> Any: ...
    @property
    def mustbuse(self) -> Any: ...
    @property
    def nextb(self) -> Any: ...
    @property
    def predset(self) -> Any: ...
    @property
    def prevb(self) -> Any: ...
    @property
    def serial(self) -> Any: ...
    @property
    def start(self) -> Any: ...
    @property
    def succset(self) -> Any: ...
    @property
    def tail(self) -> Any: ...
    @property
    def type(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def append_def_list(self, list: mlist_t, op: mop_t, maymust: maymust_t) -> None:
        r"""Append def-list of an operand. This function calculates list of locations that may or must be modified by the operand and appends it to LIST. 
                
        :param list: ptr to the output buffer. we will append to it.
        :param op: operand to calculate the def list of
        :param maymust: should we calculate 'may-def' or 'must-def' list? see maymust_t for more details.
        """
        ...
    def append_use_list(self, args: Any) -> None:
        r"""Append use-list of an operand. This function calculates list of locations that may or must be used by the operand and appends it to LIST. 
                
        :param list: ptr to the output buffer. we will append to it.
        :param op: operand to calculate the use list of
        :param maymust: should we calculate 'may-use' or 'must-use' list? see maymust_t for more details.
        :param mask: if only part of the operand should be considered, a bitmask can be used to specify which part. example: op=AX,mask=0xFF means that we will consider only AL.
        """
        ...
    def build_def_list(self, ins: minsn_t, maymust: maymust_t) -> mlist_t:
        r"""Build def-list of an instruction. This function calculates list of locations that may or must be modified by the instruction. Examples: "stx ebx.4, ds.2, eax.4", may-list: all aliasable memory "stx ebx.4, ds.2, eax.4", must-list: empty Since STX uses EAX for indirect access, it may modify any aliasable memory. On the other hand, we cannot tell for sure which memory cells will be modified, this is why the must-list is empty. 
                
        :param ins: instruction to calculate the def list of
        :param maymust: should we calculate 'may-def' or 'must-def' list? see maymust_t for more details.
        :returns: the calculated def-list
        """
        ...
    def build_lists(self, kill_deads: bool) -> int:
        r"""Build def-use lists and eliminate deads. 
                
        :param kill_deads: do delete dead instructions?
        :returns: the number of eliminated instructions Better mblock_t::call make_lists_ready() rather than this function.
        """
        ...
    def build_use_list(self, ins: minsn_t, maymust: maymust_t) -> mlist_t:
        r"""Build use-list of an instruction. This function calculates list of locations that may or must be used by the instruction. Examples: "ldx ds.2, eax.4, ebx.4", may-list: all aliasable memory "ldx ds.2, eax.4, ebx.4", must-list: empty Since LDX uses EAX for indirect access, it may access any aliasable memory. On the other hand, we cannot tell for sure which memory cells will be accessed, this is why the must-list is empty. 
                
        :param ins: instruction to calculate the use list of
        :param maymust: should we calculate 'may-use' or 'must-use' list? see maymust_t for more details.
        :returns: the calculated use-list
        """
        ...
    def dump(self) -> None:
        r"""Dump block info. This function is useful for debugging, see mba_t::dump for info 
                
        """
        ...
    def dump_block(self, title: str) -> None:
        ...
    def empty(self) -> bool:
        ...
    def find_access(self, op: mop_t, parent: minsn_t, mend: minsn_t, fdflags: int) -> minsn_t:
        r"""Find the instruction that accesses the specified operand. This function search inside one block. 
                
        :param op: operand to search for
        :param parent: ptr to ptr to a top level instruction. in: denotes the beginning of the search range. out: denotes the parent of the found instruction.
        :param mend: end instruction of the range (must be a top level insn) mend is excluded from the range. it can be specified as nullptr. parent and mend must belong to the same block.
        :param fdflags: combination of bits for mblock_t::find_access bits
        :returns: the instruction that accesses the operand. this instruction may be a sub-instruction. to find out the top level instruction, check out *parent. nullptr means 'not found'.
        """
        ...
    def find_def(self, op: mop_t, p_i1: minsn_t, i2: minsn_t, fdflags: int) -> minsn_t:
        ...
    def find_first_use(self, args: Any) -> minsn_t:
        r"""This function has the following signatures:
        
            0. find_first_use(list: mlist_t *, i1: const minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> const minsn_t *
            1. find_first_use(list: mlist_t *, i1: minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> minsn_t *
        
        # 0: find_first_use(list: mlist_t *, i1: const minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> const minsn_t *
        
        Find the first insn that uses the specified list in the insn range. 
                
        :returns: pointer to such instruction or nullptr. Upon return LIST will contain only locations not redefined by insns [i1..result]
        
        # 1: find_first_use(list: mlist_t *, i1: minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> minsn_t *
        
        
        """
        ...
    def find_redefinition(self, args: Any) -> minsn_t:
        r"""This function has the following signatures:
        
            0. find_redefinition(list: const mlist_t &, i1: const minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> const minsn_t *
            1. find_redefinition(list: const mlist_t &, i1: minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> minsn_t *
        
        # 0: find_redefinition(list: const mlist_t &, i1: const minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> const minsn_t *
        
        Find the first insn that redefines any part of the list in the insn range. 
                
        :returns: pointer to such instruction or nullptr.
        
        # 1: find_redefinition(list: const mlist_t &, i1: minsn_t *, i2: const minsn_t *, maymust: maymust_t=MAY_ACCESS) -> minsn_t *
        
        
        """
        ...
    def find_use(self, op: mop_t, p_i1: minsn_t, i2: minsn_t, fdflags: int) -> minsn_t:
        ...
    def for_all_insns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all instructions. This function visits subinstructions too. 
                
        :param mv: instruction visitor
        :returns: zero or the value returned by mv.visit_insn() See also mba_t::for_all_topinsns()
        """
        ...
    def for_all_ops(self, mv: mop_visitor_t) -> int:
        r"""Visit all operands. This function visit subinstruction operands too. 
                
        :param mv: operand visitor
        :returns: zero or the value returned by mv.visit_mop()
        """
        ...
    def for_all_uses(self, list: mlist_t, i1: minsn_t, i2: minsn_t, mmv: mlist_mop_visitor_t) -> int:
        r"""Visit all operands that use LIST. 
                
        :param list: ptr to the list of locations. it may be modified: parts that get redefined by the instructions in [i1,i2) will be deleted.
        :param i1: starting instruction. must be a top level insn.
        :param i2: ending instruction (excluded). must be a top level insn.
        :param mmv: operand visitor
        :returns: zero or the value returned by mmv.visit_mop()
        """
        ...
    def get_reginsn_qty(self) -> int:
        r"""Calculate number of regular instructions in the block. Assertions are skipped by this function. 
                
        :returns: Number of non-assertion instructions in the block.
        """
        ...
    def get_valranges(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. get_valranges(res: valrng_t *, vivl: const vivl_t &, vrflags: int) -> bool
            1. get_valranges(res: valrng_t *, vivl: const vivl_t &, m: const minsn_t *, vrflags: int) -> bool
        
        # 0: get_valranges(res: valrng_t *, vivl: const vivl_t &, vrflags: int) -> bool
        
        Find possible values for a block. 
                
        
        # 1: get_valranges(res: valrng_t *, vivl: const vivl_t &, m: const minsn_t *, vrflags: int) -> bool
        
        Find possible values for an instruction. 
                
        
        """
        ...
    def insert_into_block(self, nm: minsn_t, om: minsn_t) -> minsn_t:
        r"""Insert instruction into the doubly linked list 
                
        :param nm: new instruction
        :param om: existing instruction, part of the doubly linked list if nullptr, then the instruction will be inserted at the beginning of the list NM will be inserted immediately after OM
        :returns: pointer to NM
        """
        ...
    def is_branch(self) -> bool:
        ...
    def is_call_block(self) -> bool:
        ...
    def is_nway(self) -> bool:
        ...
    def is_redefined(self, args: Any) -> bool:
        r"""Is the list redefined by the specified instructions? 
                
        :param list: list of locations to check.
        :param i1: starting instruction of the range (must be a top level insn)
        :param i2: end instruction of the range (must be a top level insn) i2 is excluded from the range. it can be specified as nullptr. i1 and i2 must belong to the same block.
        :param maymust: should we search in 'may-access' or 'must-access' mode?
        """
        ...
    def is_rhs_redefined(self, ins: minsn_t, i1: minsn_t, i2: minsn_t) -> bool:
        r"""Is the right hand side of the instruction redefined the insn range? "right hand side" corresponds to the source operands of the instruction. 
                
        :param ins: instruction to consider
        :param i1: starting instruction of the range (must be a top level insn)
        :param i2: end instruction of the range (must be a top level insn) i2 is excluded from the range. it can be specified as nullptr. i1 and i2 must belong to the same block.
        """
        ...
    def is_simple_goto_block(self) -> bool:
        ...
    def is_simple_jcnd_block(self) -> bool:
        ...
    def is_unknown_call(self) -> bool:
        ...
    def is_used(self, args: Any) -> bool:
        r"""Is the list used by the specified instruction range? 
                
        :param list: list of locations. LIST may be modified by the function: redefined locations will be removed from it.
        :param i1: starting instruction of the range (must be a top level insn)
        :param i2: end instruction of the range (must be a top level insn) i2 is excluded from the range. it can be specified as nullptr. i1 and i2 must belong to the same block.
        :param maymust: should we search in 'may-access' or 'must-access' mode?
        """
        ...
    def lists_dirty(self) -> bool:
        ...
    def lists_ready(self) -> bool:
        ...
    def make_lists_ready(self) -> int:
        ...
    def make_nop(self, m: minsn_t) -> None:
        r"""Erase the instruction (convert it to nop) and mark the lists dirty. This is the recommended function to use because it also marks the block use-def lists dirty. 
                
        """
        ...
    def mark_lists_dirty(self) -> None:
        ...
    def needs_propagation(self) -> bool:
        ...
    def npred(self) -> int:
        r"""Get number of block predecessors.
        
        """
        ...
    def nsucc(self) -> int:
        r"""Get number of block successors.
        
        """
        ...
    def optimize_block(self) -> int:
        r"""Optimize a basic block. Usually there is no need to call this function explicitly because the decompiler will call it itself if optinsn_t::func or optblock_t::func return non-zero. 
                
        :returns: number of changes made to the block
        """
        ...
    def optimize_insn(self, args: Any) -> int:
        r"""Optimize one instruction in the context of the block. 
                
        :param m: pointer to a top level instruction
        :param optflags: combination of optimization flags bits
        :returns: number of changes made to the block This function may change other instructions in the block too. However, it will not destroy top level instructions (it may convert them to nop's). This function performs only intrablock modifications. See also minsn_t::optimize_solo()
        """
        ...
    def optimize_useless_jump(self) -> int:
        r"""Remove a jump at the end of the block if it is useless. This function preserves any side effects when removing a useless jump. Both conditional and unconditional jumps are handled (and jtbl too). This function deletes useless jumps, not only replaces them with a nop. (please note that \optimize_insn does not handle useless jumps). 
                
        :returns: number of changes made to the block
        """
        ...
    def pred(self, n: int) -> int:
        ...
    def preds(self) -> Any:
        r"""
                Iterates the list of predecessor blocks
                
        """
        ...
    def remove_from_block(self, m: minsn_t) -> minsn_t:
        r"""Remove instruction from the doubly linked list 
                
        :param m: instruction to remove The removed instruction is not deleted, the caller gets its ownership
        :returns: pointer to the next instruction
        """
        ...
    def request_demote64(self) -> None:
        ...
    def request_propagation(self) -> None:
        ...
    def succ(self, n: int) -> int:
        ...
    def succs(self) -> Any:
        r"""
                Iterates the list of successor blocks
                
        """
        ...

class mcallarg_t(mop_t):
    @property
    def a(self) -> Any: ...
    @property
    def argloc(self) -> Any: ...
    @property
    def b(self) -> Any: ...
    @property
    def c(self) -> Any: ...
    @property
    def cstr(self) -> Any: ...
    @property
    def d(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def f(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def fpc(self) -> Any: ...
    @property
    def g(self) -> Any: ...
    @property
    def helper(self) -> Any: ...
    @property
    def l(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def nnn(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def oprops(self) -> Any: ...
    @property
    def pair(self) -> Any: ...
    @property
    def r(self) -> Any: ...
    @property
    def s(self) -> Any: ...
    @property
    def scif(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def t(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def valnum(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, rop: mop_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, rop: mop_t) -> bool:
        ...
    def __ne__(self, rop: mop_t) -> bool:
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
    def apply_ld_mcode(self, mcode: mcode_t, ea: ida_idaapi.ea_t, newsize: int) -> None:
        r"""Apply a unary opcode to the operand. 
                
        :param mcode: opcode to apply. it must accept 'l' and 'd' operands but not 'r'. examples: m_low/m_high/m_xds/m_xdu
        :param ea: value of minsn_t::ea for the newly created insruction
        :param newsize: new operand size Example: apply_ld_mcode(m_low) will convert op => low(op)
        """
        ...
    def apply_xds(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def apply_xdu(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def assign(self, rop: mop_t) -> cinsn_t:
        ...
    def change_size(self, nsize: int, sideff: side_effect_t = 1) -> bool:
        r"""Change the operand size. Examples: change_size(AL.1, 2) -> AX.2 change_size(qword_00000008.8, 4) -> dword_00000008.4 change_size(xdu.8(op.4), 4) -> op.4 change_size(#0x12345678.4, 1) -> #0x78.1 
                
        :param nsize: new operand size
        :param sideff: may modify the database because of the size change?
        :returns: success
        """
        ...
    def copy_mop(self, op: mop_t) -> None:
        ...
    def create_from_insn(self, m: minsn_t) -> None:
        r"""Create operand from an instruction. This function creates a nested instruction that can be used as an operand. Example: if m="add x,y,z", our operand will be (t=mop_d,d=m). The destination operand of 'add' (z) is lost. 
                
        :param m: instruction to embed into operand. may not be nullptr.
        """
        ...
    def create_from_ivlset(self, mba: mba_t, ivs: ivlset_t, fullsize: int) -> bool:
        r"""Create operand from ivlset_t. Example: if IVS contains [glbvar..glbvar+4), our operand will be (t=mop_v, g=&glbvar, size=4) 
                
        :param mba: pointer to microcode
        :param ivs: set of memory intervals
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_mlist(self, mba: mba_t, lst: mlist_t, fullsize: int) -> bool:
        r"""Create operand from mlist_t. Example: if LST contains 4 bits for R0.4, our operand will be (t=mop_r, r=R0, size=4) 
                
        :param mba: pointer to microcode
        :param lst: list of locations
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_scattered_vdloc(self, mba: mba_t, name: str, type: tinfo_t, loc: vdloc_t) -> None:
        r"""Create operand from scattered vdloc_t. Example: if LOC is (ALOC_DIST, {EAX.4, EDX.4}) and TYPE is _LARGE_INTEGER, our operand will be (t=mop_sc, scif={EAX.4, EDX.4}) 
                
        :param mba: pointer to microcode
        :param name: name of the operand, if available
        :param type: type of the operand, must be present
        :param loc: a scattered location
        :returns: success
        """
        ...
    def create_from_vdloc(self, mba: mba_t, loc: vdloc_t, _size: int) -> None:
        r"""Create operand from vdloc_t. Example: if LOC contains (type=ALOC_REG1, r=R0), our operand will be (t=mop_r, r=R0, size=_SIZE) 
                
        :param mba: pointer to microcode
        :param loc: location
        :param _size: operand size Note: this function cannot handle scattered locations.
        :returns: success
        """
        ...
    def double_size(self, sideff: side_effect_t = 1) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def equal_mops(self, rop: mop_t, eqflags: int) -> bool:
        r"""Compare operands. This is the main comparison function for operands. 
                
        :param rop: operand to compare with
        :param eqflags: combination of comparison bits bits
        """
        ...
    def erase(self) -> None:
        ...
    def erase_but_keep_size(self) -> None:
        ...
    def for_all_ops(self, mv: mop_visitor_t, type: tinfo_t = None, is_target: bool = False) -> int:
        r"""Visit the operand and all its sub-operands. This function visits the current operand as well. 
                
        :param mv: visitor object
        :param type: operand type
        :param is_target: is a destination operand?
        """
        ...
    def for_all_scattered_submops(self, sv: scif_visitor_t) -> int:
        r"""Visit all sub-operands of a scattered operand. This function does not visit the current operand, only its sub-operands. All sub-operands are synthetic and are destroyed after the visitor. This function works only with scattered operands. 
                
        :param sv: visitor object
        """
        ...
    def get_insn(self, code: mcode_t) -> minsn_t:
        r"""Get subinstruction of the operand. If the operand has a subinstruction with the specified opcode, return it. 
                
        :param code: desired opcode
        :returns: pointer to the instruction or nullptr
        """
        ...
    def get_stkoff(self, p_vdoff: sval_t) -> int:
        r"""Get the referenced stack offset. This function can also handle mop_sc if it is entirely mapped into a continuous stack region. 
                
        :param p_vdoff: the output buffer
        :returns: success
        """
        ...
    def get_stkvar(self, udm: udm_t = None, p_idaoff: uval_t = None) -> ssize_t:
        r"""Retrieve the referenced stack variable. 
                
        :param udm: stkvar, may be nullptr
        :param p_idaoff: if specified, will hold IDA stkoff after the call.
        :returns: index of stkvar in the frame or -1
        """
        ...
    def has_side_effects(self, include_ldx_and_divs: bool = False) -> bool:
        r"""Has any side effects? 
                
        :param include_ldx_and_divs: consider ldx/div/mod as having side effects?
        """
        ...
    def is01(self) -> bool:
        r"""Are the possible values of the operand only 0 and 1? This function returns true for 0/1 constants, bit registers, the result of 'set' insns, etc. 
                
        """
        ...
    def is_arglist(self) -> bool:
        r"""Is a list of arguments?
        
        """
        ...
    def is_bit_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_bit_reg() -> bool
            1. is_bit_reg(reg: mreg_t) -> bool
        
        # 0: is_bit_reg() -> bool
        
        
        # 1: is_bit_reg(reg: mreg_t) -> bool
        
        Is a bit register? This includes condition codes and eventually other bit registers 
                
        
        """
        ...
    def is_cc(self) -> bool:
        r"""Is a condition code?
        
        """
        ...
    def is_ccflags(self) -> bool:
        ...
    def is_constant(self, is_signed: bool = True) -> bool:
        r"""Retrieve value of a constant integer operand. 
                
        :param is_signed: should treat the value as signed
        :returns: true if the operand is mop_n
        """
        ...
    def is_equal_to(self, n: uint64, is_signed: bool = True) -> bool:
        ...
    def is_extended_from(self, nbytes: int, is_signed: bool) -> bool:
        r"""Does the high part of the operand consist of zero or sign bytes?
        
        """
        ...
    def is_for_abi(self) -> bool:
        ...
    def is_glbaddr(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_glbaddr() -> bool
            1. is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        # 0: is_glbaddr() -> bool
        
        Is address of a global memory cell?
        
        
        # 1: is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        Is address of the specified global memory cell?
        
        
        """
        ...
    def is_glbaddr_from_fixup(self) -> bool:
        ...
    def is_glbvar(self) -> bool:
        r"""Is a global variable?
        
        """
        ...
    def is_impptr_done(self) -> bool:
        ...
    def is_insn(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_insn() -> bool
            1. is_insn(code: mcode_t) -> bool
        
        # 0: is_insn() -> bool
        
        Is a sub-instruction?
        
        
        # 1: is_insn(code: mcode_t) -> bool
        
        Is a sub-instruction with the specified opcode?
        
        
        """
        ...
    def is_kreg(self) -> bool:
        r"""Is a kernel register?
        
        """
        ...
    def is_lowaddr(self) -> bool:
        ...
    def is_mblock(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_mblock() -> bool
            1. is_mblock(serial: int) -> bool
        
        # 0: is_mblock() -> bool
        
        Is a block reference?
        
        
        # 1: is_mblock(serial: int) -> bool
        
        Is a block reference to the specified block?
        
        
        """
        ...
    def is_negative_constant(self) -> bool:
        ...
    def is_one(self) -> bool:
        ...
    def is_pcval(self) -> bool:
        ...
    def is_positive_constant(self) -> bool:
        ...
    def is_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_reg() -> bool
            1. is_reg(_r: mreg_t) -> bool
            2. is_reg(_r: mreg_t, _size: int) -> bool
        
        # 0: is_reg() -> bool
        
        Is a register operand? See also get_mreg_name() 
                
        
        # 1: is_reg(_r: mreg_t) -> bool
        
        Is the specified register?
        
        
        # 2: is_reg(_r: mreg_t, _size: int) -> bool
        
        Is the specified register of the specified size?
        
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""Is a scattered operand?
        
        """
        ...
    def is_sign_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of the sign bytes? 
                
        :param nbytes: number of bytes that were sign extended. the remaining size-nbytes high bytes must be sign bytes Example: is_sign_extended_from(xds.4(op.1), 1) -> true because the high 3 bytes are certainly sign bits
        """
        ...
    def is_stkaddr(self) -> bool:
        r"""Is address of a stack variable?
        
        """
        ...
    def is_stkvar(self) -> bool:
        r"""Is a stack variable?
        
        """
        ...
    def is_udt(self) -> bool:
        ...
    def is_undef_val(self) -> bool:
        ...
    def is_zero(self) -> bool:
        ...
    def is_zero_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of zero bytes? 
                
        :param nbytes: number of bytes that were zero extended. the remaining size-nbytes high bytes must be zero Example: is_zero_extended_from(xdu.8(op.1), 2) -> true because the high 6 bytes are certainly zero
        """
        ...
    def lexcompare(self, rop: mop_t) -> int:
        ...
    def make_blkref(self, blknum: int) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_first_half(self, width: int) -> bool:
        r"""Make the first part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_fpnum(self, bytes: void) -> bool:
        r"""Create a floating point constant operand. 
                
        :param bytes: pointer to the floating point value as used by the current processor (e.g. for x86 it must be in IEEE 754)
        :returns: success
        """
        ...
    def make_gvar(self, ea: ida_idaapi.ea_t) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_helper(self, name: str) -> None:
        r"""Create a helper operand. A helper operand usually keeps a built-in function name like "va_start" It is essentially just an arbitrary identifier without any additional info. 
                
        """
        ...
    def make_high_half(self, width: int) -> bool:
        r"""Make the high part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_insn(self, ins: minsn_t) -> None:
        r"""Create a nested instruction.
        
        """
        ...
    def make_int(self, val: int, val_ea: ida_idaapi.ea_t, opno: int = 0) -> None:
        ...
    def make_low_half(self, width: int) -> bool:
        r"""Make the low part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_number(self, args: Any) -> None:
        r"""Create an integer constant operand. 
                
        :param _value: value to store in the operand
        :param _size: size of the value in bytes (1,2,4,8)
        :param _ea: address of the processor instruction that made the value
        :param opnum: operand number of the processor instruction
        """
        ...
    def make_reg(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. make_reg(reg: mreg_t) -> None
            1. make_reg(reg: mreg_t, _size: int) -> None
        
        # 0: make_reg(reg: mreg_t) -> None
        
        Create a register operand.
        
        
        # 1: make_reg(reg: mreg_t, _size: int) -> None
        
        
        """
        ...
    def make_reg_pair(self, loreg: int, hireg: int, halfsize: int) -> None:
        r"""Create pair of registers. 
                
        :param loreg: register holding the low part of the value
        :param hireg: register holding the high part of the value
        :param halfsize: the size of each of loreg/hireg
        """
        ...
    def make_second_half(self, width: int) -> bool:
        r"""Make the second part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_stkvar(self, mba: mba_t, off: int) -> None:
        ...
    def make_uint(self, val: int, val_ea: ida_idaapi.ea_t, opno: int = 0) -> None:
        ...
    def may_use_aliased_memory(self) -> bool:
        r"""Is it possible for the operand to use aliased memory?
        
        """
        ...
    def preserve_side_effects(self, blk: mblock_t, top: minsn_t, moved_calls: bool = None) -> bool:
        r"""Move subinstructions with side effects out of the operand. If we decide to delete an instruction operand, it is a good idea to call this function. Alternatively we should skip such operands by calling mop_t::has_side_effects() For example, if we transform: jnz x, x, @blk => goto @blk then we must call this function before deleting the X operands. 
                
        :param blk: current block
        :param top: top level instruction that contains our operand
        :param moved_calls: pointer to the boolean that will track if all side effects get handled correctly. must be false initially.
        :returns: false failed to preserve a side effect, it is not safe to delete the operand true no side effects or successfully preserved them
        """
        ...
    def probably_floating(self) -> bool:
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def set_for_abi(self) -> None:
        ...
    def set_impptr_done(self) -> None:
        ...
    def set_lowaddr(self) -> None:
        ...
    def set_regarg(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. set_regarg(mr: mreg_t, sz: int, tif: const tinfo_t &) -> None
            1. set_regarg(mr: mreg_t, tif: const tinfo_t &) -> None
            2. set_regarg(mr: mreg_t, dt: char, sign: type_sign_t=type_unsigned) -> None
        
        # 0: set_regarg(mr: mreg_t, sz: int, tif: const tinfo_t &) -> None
        
        
        # 1: set_regarg(mr: mreg_t, tif: const tinfo_t &) -> None
        
        
        # 2: set_regarg(mr: mreg_t, dt: char, sign: type_sign_t=type_unsigned) -> None
        
        
        """
        ...
    def set_udt(self) -> None:
        ...
    def set_undef_val(self) -> None:
        ...
    def shift_mop(self, offset: int) -> bool:
        r"""Shift the operand. This function shifts only the beginning of the operand. The operand size will be changed. Examples: shift_mop(AH.1, -1) -> AX.2 shift_mop(qword_00000008.8, 4) -> dword_0000000C.4 shift_mop(xdu.8(op.4), 4) -> #0.4 shift_mop(#0x12345678.4, 3) -> #12.1 
                
        :param offset: shift count (the number of bytes to shift)
        :returns: success
        """
        ...
    def signed_value(self) -> int64:
        ...
    def swap(self, rop: mop_t) -> None:
        ...
    def unsigned_value(self) -> uint64:
        ...
    def update_numop_value(self, val: uint64) -> None:
        ...
    def value(self, is_signed: bool) -> uint64:
        r"""Retrieve value of a constant integer operand. These functions can be called only for mop_n operands. See is_constant() that can be called on any operand. 
                
        """
        ...
    def zero(self) -> None:
        ...

class mcallargs_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: mcallargs_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: mcallargs_t) -> bool:
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
    def __setitem__(self, i: size_t, v: mcallarg_t) -> None:
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
    def add_unique(self, x: mcallarg_t) -> bool:
        ...
    def append(self, x: mcallarg_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: mcallargs_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: mcallarg_t) -> bool:
        ...
    def inject(self, s: mcallarg_t, len: size_t) -> None:
        ...
    def insert(self, it: mcallarg_t, x: mcallarg_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: mcallargs_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class mcallinfo_t:
    @property
    def args(self) -> Any: ...
    @property
    def call_spd(self) -> Any: ...
    @property
    def callee(self) -> Any: ...
    @property
    def cc(self) -> Any: ...
    @property
    def dead_regs(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def fti_attrs(self) -> Any: ...
    @property
    def pass_regs(self) -> Any: ...
    @property
    def retregs(self) -> Any: ...
    @property
    def return_argloc(self) -> Any: ...
    @property
    def return_regs(self) -> Any: ...
    @property
    def return_type(self) -> Any: ...
    @property
    def role(self) -> Any: ...
    @property
    def solid_args(self) -> Any: ...
    @property
    def spoiled(self) -> Any: ...
    @property
    def stkargs_top(self) -> Any: ...
    @property
    def visible_memory(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def dstr(self) -> str:
        ...
    def get_type(self) -> tinfo_t:
        ...
    def is_vararg(self) -> bool:
        ...
    def lexcompare(self, f: mcallinfo_t) -> int:
        ...
    def set_type(self, type: tinfo_t) -> bool:
        ...

class mcases_t:
    @property
    def targets(self) -> Any: ...
    @property
    def values(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: mcases_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: mcases_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: mcases_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: mcases_t) -> bool:
        ...
    def __lt__(self, r: mcases_t) -> bool:
        ...
    def __ne__(self, r: mcases_t) -> bool:
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
    def compare(self, r: mcases_t) -> int:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def resize(self, s: int) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: mcases_t) -> None:
        ...

class microcode_filter_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply(self, cdg: codegen_t) -> bool:
        r"""generate microcode for an instruction 
                
        :returns: MERR_... code: MERR_OK - user-defined microcode generated, go to the next instruction MERR_INSN - not generated - the caller should try the standard way else - error
        """
        ...
    def match(self, cdg: codegen_t) -> bool:
        r"""check if the filter object is to be applied 
                
        :returns: success
        """
        ...

class minsn_t:
    @property
    def d(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def iprops(self) -> Any: ...
    @property
    def l(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def next(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def opcode(self) -> Any: ...
    @property
    def prev(self) -> Any: ...
    @property
    def r(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, ri: minsn_t) -> bool:
        ...
    def __ne__(self, value: Any) -> bool:
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
    def clr_assert(self) -> None:
        ...
    def clr_combinable(self) -> None:
        ...
    def clr_combined(self) -> None:
        ...
    def clr_fpinsn(self) -> None:
        ...
    def clr_ignlowsrc(self) -> None:
        ...
    def clr_multimov(self) -> None:
        ...
    def clr_noret_icall(self) -> None:
        ...
    def clr_propagatable(self) -> None:
        ...
    def clr_tailcall(self) -> None:
        ...
    def contains_call(self, with_helpers: bool = False) -> bool:
        r"""Does the instruction contain a call?
        
        """
        ...
    def contains_opcode(self, mcode: mcode_t) -> bool:
        r"""Does the instruction have the specified opcode? This function searches subinstructions as well. 
                
        :param mcode: opcode to search for.
        """
        ...
    def deserialize(self, bytes: uchar, format_version: int) -> mba_t:
        r"""Deserialize an instruction 
                
        :param bytes: pointer to serialized data
        :param format_version: serialization format version. this value is returned by minsn_t::serialize()
        :returns: success
        """
        ...
    def dstr(self) -> str:
        r"""Get displayable text without tags in a static buffer.
        
        """
        ...
    def equal_insns(self, m: minsn_t, eqflags: int) -> bool:
        r"""Compare instructions. This is the main comparison function for instructions. 
                
        :param m: instruction to compare with
        :param eqflags: combination of comparison bits bits
        """
        ...
    def find_call(self, with_helpers: bool = False) -> minsn_t:
        r"""Find a call instruction. Check for the current instruction and its subinstructions. 
                
        :param with_helpers: consider helper calls as well?
        """
        ...
    def find_ins_op(self, op: mcode_t = 0) -> minsn_t:
        r"""Find an operand that is a subinsruction with the specified opcode. This function checks only the 'l' and 'r' operands of the current insn. 
                
        :param op: opcode to search for
        :returns: &l or &r or nullptr
        """
        ...
    def find_num_op(self) -> cexpr_t:
        r"""Find a numeric operand of the current instruction. This function checks only the 'l' and 'r' operands of the current insn. 
                
        :returns: &l or &r or nullptr
        """
        ...
    def find_opcode(self, mcode: mcode_t) -> minsn_t:
        r"""Find a (sub)insruction with the specified opcode. 
                
        :param mcode: opcode to search for.
        """
        ...
    def for_all_insns(self, mv: minsn_visitor_t) -> int:
        r"""Visit all instructions. This function visits the instruction itself and all its subinstructions. 
                
        :param mv: instruction visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def for_all_ops(self, mv: mop_visitor_t) -> int:
        r"""Visit all instruction operands. This function visits subinstruction operands as well. 
                
        :param mv: operand visitor
        :returns: non-zero value returned by mv.visit_mop() or zero
        """
        ...
    def get_role(self) -> funcrole_t:
        r"""Get the function role of a call.
        
        """
        ...
    def get_split_size(self) -> int:
        ...
    def has_side_effects(self, include_ldx_and_divs: bool = False) -> bool:
        r"""Does the instruction have a side effect? 
                
        :param include_ldx_and_divs: consider ldx/div/mod as having side effects? stx is always considered as having side effects. Apart from ldx/std only call may have side effects.
        """
        ...
    def is_after(self, m: minsn_t) -> bool:
        r"""Is the instruction after the specified one? 
                
        :param m: the instruction to compare against in the list
        """
        ...
    def is_alloca(self) -> bool:
        ...
    def is_assert(self) -> bool:
        ...
    def is_between(self, m1: minsn_t, m2: minsn_t) -> bool:
        r"""Is the instruction in the specified range of instructions? 
                
        :param m1: beginning of the range in the doubly linked list
        :param m2: end of the range in the doubly linked list (excluded, may be nullptr) This function assumes that m1 and m2 belong to the same basic block and they are top level instructions.
        """
        ...
    def is_bswap(self) -> bool:
        ...
    def is_cleaning_pop(self) -> bool:
        ...
    def is_combinable(self) -> bool:
        ...
    def is_combined(self) -> bool:
        ...
    def is_extstx(self) -> bool:
        ...
    def is_farcall(self) -> bool:
        ...
    def is_fpinsn(self) -> bool:
        ...
    def is_helper(self, name: str) -> bool:
        r"""Is a helper call with the specified name? Helper calls usually have well-known function names (see Well known function names) but they may have any other name. The decompiler does not assume any special meaning for non-well-known names. 
                
        """
        ...
    def is_ignlowsrc(self) -> bool:
        ...
    def is_inverted_jx(self) -> bool:
        ...
    def is_like_move(self) -> bool:
        ...
    def is_mbarrier(self) -> bool:
        ...
    def is_memcpy(self) -> bool:
        ...
    def is_memset(self) -> bool:
        ...
    def is_mov(self) -> bool:
        ...
    def is_multimov(self) -> bool:
        ...
    def is_noret_call(self, flags: int = 0) -> bool:
        r"""Is a non-returing call? 
                
        :param flags: combination of NORET_... bits
        """
        ...
    def is_optional(self) -> bool:
        ...
    def is_persistent(self) -> bool:
        ...
    def is_propagatable(self) -> bool:
        ...
    def is_readflags(self) -> bool:
        ...
    def is_tailcall(self) -> bool:
        ...
    def is_unknown_call(self) -> bool:
        r"""Is an unknown call? Unknown calls are calls without the argument list (mcallinfo_t). Usually the argument lists are determined by mba_t::analyze_calls(). Unknown calls exist until the MMAT_CALLS maturity level. See also mblock_t::is_call_block 
                
        """
        ...
    def is_wild_match(self) -> bool:
        ...
    def lexcompare(self, ri: minsn_t) -> int:
        ...
    def may_use_aliased_memory(self) -> bool:
        r"""Is it possible for the instruction to use aliased memory?
        
        """
        ...
    def modifies_d(self) -> bool:
        r"""Does the instruction modify its 'd' operand? Some instructions (e.g. m_stx) do not modify the 'd' operand. 
                
        """
        ...
    def modifies_pair_mop(self) -> bool:
        ...
    def optimize_solo(self, optflags: int = 0) -> int:
        r"""Optimize one instruction without context. This function does not have access to the instruction context (the previous and next instructions in the list, the block number, etc). It performs only basic optimizations that are available without this info. 
                
        :param optflags: combination of optimization flags bits
        :returns: number of changes, 0-unchanged See also mblock_t::optimize_insn()
        """
        ...
    def optimize_subtree(self, blk: mblock_t, top: minsn_t, parent: minsn_t, converted_call: ea_t, optflags: int = 2) -> int:
        r"""Optimize instruction in its context. Do not use this function, use mblock_t::optimize() 
                
        """
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def serialize(self, b: bytevec_t) -> None:
        r"""Serialize an instruction 
                
        :param b: the output buffer
        :returns: the serialization format that was used to store info
        """
        ...
    def set_assert(self) -> None:
        ...
    def set_cleaning_pop(self) -> None:
        ...
    def set_combinable(self) -> None:
        ...
    def set_extstx(self) -> None:
        ...
    def set_farcall(self) -> None:
        ...
    def set_fpinsn(self) -> None:
        ...
    def set_ignlowsrc(self) -> None:
        ...
    def set_inverted_jx(self) -> None:
        ...
    def set_mbarrier(self) -> None:
        ...
    def set_multimov(self) -> None:
        ...
    def set_noret_icall(self) -> None:
        ...
    def set_optional(self) -> None:
        ...
    def set_persistent(self) -> None:
        ...
    def set_split_size(self, s: int) -> None:
        ...
    def set_tailcall(self) -> None:
        ...
    def set_unmerged(self) -> None:
        ...
    def set_wild_match(self) -> None:
        ...
    def setaddr(self, new_ea: ida_idaapi.ea_t) -> None:
        r"""Change the instruction address. This function modifies subinstructions as well. 
                
        """
        ...
    def swap(self, m: minsn_t) -> None:
        r"""Swap two instructions. The prev/next fields are not modified by this function because it would corrupt the doubly linked list. 
                
        """
        ...
    def was_noret_icall(self) -> bool:
        ...
    def was_split(self) -> bool:
        ...
    def was_unmerged(self) -> bool:
        ...
    def was_unpaired(self) -> bool:
        ...

class minsn_visitor_t(op_parent_info_t):
    @property
    def blk(self) -> Any: ...
    @property
    def curins(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def topins(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _mba: mba_t = None, _blk: mblock_t = None, _topins: minsn_t = None) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def visit_minsn(self) -> int:
        ...

class mlist_mop_visitor_t:
    @property
    def changed(self) -> Any: ...
    @property
    def curins(self) -> Any: ...
    @property
    def list(self) -> Any: ...
    @property
    def prune(self) -> Any: ...
    @property
    def topins(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def visit_mop(self, op: mop_t) -> int:
        ...

class mlist_t:
    @property
    def mem(self) -> Any: ...
    @property
    def reg(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: mlist_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: mlist_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: mlist_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: mlist_t) -> bool:
        ...
    def __lt__(self, r: mlist_t) -> bool:
        ...
    def __ne__(self, r: mlist_t) -> bool:
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
    def add(self, args: Any) -> voff_t:
        r"""This function has the following signatures:
        
            0. add(r: mreg_t, size: int) -> bool
            1. add(r: const rlist_t &) -> bool
            2. add(ivl: const ivl_t &) -> bool
            3. add(lst: const mlist_t &) -> bool
        
        # 0: add(r: mreg_t, size: int) -> bool
        
        
        # 1: add(r: const rlist_t &) -> bool
        
        
        # 2: add(ivl: const ivl_t &) -> bool
        
        
        # 3: add(lst: const mlist_t &) -> bool
        
        
        """
        ...
    def addmem(self, ea: ida_idaapi.ea_t, size: asize_t) -> bool:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: mlist_t) -> int:
        ...
    def count(self) -> int:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def has(self, r: mreg_t) -> bool:
        ...
    def has_all(self, r: mreg_t, size: int) -> bool:
        ...
    def has_any(self, r: mreg_t, size: int) -> bool:
        ...
    def has_common(self, lst: mlist_t) -> bool:
        ...
    def has_memory(self) -> bool:
        ...
    def includes(self, lst: mlist_t) -> bool:
        ...
    def intersect(self, lst: mlist_t) -> int:
        ...
    def is_subset_of(self, lst: mlist_t) -> bool:
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(r: mreg_t, size: int) -> bool
            1. sub(ivl: const ivl_t &) -> bool
            2. sub(lst: const mlist_t &) -> bool
        
        # 0: sub(r: mreg_t, size: int) -> bool
        
        
        # 1: sub(ivl: const ivl_t &) -> bool
        
        
        # 2: sub(lst: const mlist_t &) -> bool
        
        
        """
        ...
    def swap(self, r: mlist_t) -> None:
        ...

class mnumber_t(operand_locator_t):
    @property
    def ea(self) -> Any: ...
    @property
    def opnum(self) -> Any: ...
    @property
    def org_value(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: mnumber_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: mnumber_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: mnumber_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: mnumber_t) -> bool:
        ...
    def __lt__(self, r: mnumber_t) -> bool:
        ...
    def __ne__(self, r: mnumber_t) -> bool:
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
    def compare(self, r: mnumber_t) -> int:
        ...
    def update_value(self, val64: uint64) -> None:
        ...

class mop_addr_t(mop_t):
    @property
    def a(self) -> Any: ...
    @property
    def b(self) -> Any: ...
    @property
    def c(self) -> Any: ...
    @property
    def cstr(self) -> Any: ...
    @property
    def d(self) -> Any: ...
    @property
    def f(self) -> Any: ...
    @property
    def fpc(self) -> Any: ...
    @property
    def g(self) -> Any: ...
    @property
    def helper(self) -> Any: ...
    @property
    def insize(self) -> Any: ...
    @property
    def l(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def nnn(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def oprops(self) -> Any: ...
    @property
    def outsize(self) -> Any: ...
    @property
    def pair(self) -> Any: ...
    @property
    def r(self) -> Any: ...
    @property
    def s(self) -> Any: ...
    @property
    def scif(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def t(self) -> Any: ...
    @property
    def valnum(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, rop: mop_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, rop: mop_t) -> bool:
        ...
    def __ne__(self, rop: mop_t) -> bool:
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
    def apply_ld_mcode(self, mcode: mcode_t, ea: ida_idaapi.ea_t, newsize: int) -> None:
        r"""Apply a unary opcode to the operand. 
                
        :param mcode: opcode to apply. it must accept 'l' and 'd' operands but not 'r'. examples: m_low/m_high/m_xds/m_xdu
        :param ea: value of minsn_t::ea for the newly created insruction
        :param newsize: new operand size Example: apply_ld_mcode(m_low) will convert op => low(op)
        """
        ...
    def apply_xds(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def apply_xdu(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def assign(self, rop: mop_t) -> cinsn_t:
        ...
    def change_size(self, nsize: int, sideff: side_effect_t = 1) -> bool:
        r"""Change the operand size. Examples: change_size(AL.1, 2) -> AX.2 change_size(qword_00000008.8, 4) -> dword_00000008.4 change_size(xdu.8(op.4), 4) -> op.4 change_size(#0x12345678.4, 1) -> #0x78.1 
                
        :param nsize: new operand size
        :param sideff: may modify the database because of the size change?
        :returns: success
        """
        ...
    def create_from_insn(self, m: minsn_t) -> None:
        r"""Create operand from an instruction. This function creates a nested instruction that can be used as an operand. Example: if m="add x,y,z", our operand will be (t=mop_d,d=m). The destination operand of 'add' (z) is lost. 
                
        :param m: instruction to embed into operand. may not be nullptr.
        """
        ...
    def create_from_ivlset(self, mba: mba_t, ivs: ivlset_t, fullsize: int) -> bool:
        r"""Create operand from ivlset_t. Example: if IVS contains [glbvar..glbvar+4), our operand will be (t=mop_v, g=&glbvar, size=4) 
                
        :param mba: pointer to microcode
        :param ivs: set of memory intervals
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_mlist(self, mba: mba_t, lst: mlist_t, fullsize: int) -> bool:
        r"""Create operand from mlist_t. Example: if LST contains 4 bits for R0.4, our operand will be (t=mop_r, r=R0, size=4) 
                
        :param mba: pointer to microcode
        :param lst: list of locations
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_scattered_vdloc(self, mba: mba_t, name: str, type: tinfo_t, loc: vdloc_t) -> None:
        r"""Create operand from scattered vdloc_t. Example: if LOC is (ALOC_DIST, {EAX.4, EDX.4}) and TYPE is _LARGE_INTEGER, our operand will be (t=mop_sc, scif={EAX.4, EDX.4}) 
                
        :param mba: pointer to microcode
        :param name: name of the operand, if available
        :param type: type of the operand, must be present
        :param loc: a scattered location
        :returns: success
        """
        ...
    def create_from_vdloc(self, mba: mba_t, loc: vdloc_t, _size: int) -> None:
        r"""Create operand from vdloc_t. Example: if LOC contains (type=ALOC_REG1, r=R0), our operand will be (t=mop_r, r=R0, size=_SIZE) 
                
        :param mba: pointer to microcode
        :param loc: location
        :param _size: operand size Note: this function cannot handle scattered locations.
        :returns: success
        """
        ...
    def double_size(self, sideff: side_effect_t = 1) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def equal_mops(self, rop: mop_t, eqflags: int) -> bool:
        r"""Compare operands. This is the main comparison function for operands. 
                
        :param rop: operand to compare with
        :param eqflags: combination of comparison bits bits
        """
        ...
    def erase(self) -> None:
        ...
    def erase_but_keep_size(self) -> None:
        ...
    def for_all_ops(self, mv: mop_visitor_t, type: tinfo_t = None, is_target: bool = False) -> int:
        r"""Visit the operand and all its sub-operands. This function visits the current operand as well. 
                
        :param mv: visitor object
        :param type: operand type
        :param is_target: is a destination operand?
        """
        ...
    def for_all_scattered_submops(self, sv: scif_visitor_t) -> int:
        r"""Visit all sub-operands of a scattered operand. This function does not visit the current operand, only its sub-operands. All sub-operands are synthetic and are destroyed after the visitor. This function works only with scattered operands. 
                
        :param sv: visitor object
        """
        ...
    def get_insn(self, code: mcode_t) -> minsn_t:
        r"""Get subinstruction of the operand. If the operand has a subinstruction with the specified opcode, return it. 
                
        :param code: desired opcode
        :returns: pointer to the instruction or nullptr
        """
        ...
    def get_stkoff(self, p_vdoff: sval_t) -> int:
        r"""Get the referenced stack offset. This function can also handle mop_sc if it is entirely mapped into a continuous stack region. 
                
        :param p_vdoff: the output buffer
        :returns: success
        """
        ...
    def get_stkvar(self, udm: udm_t = None, p_idaoff: uval_t = None) -> ssize_t:
        r"""Retrieve the referenced stack variable. 
                
        :param udm: stkvar, may be nullptr
        :param p_idaoff: if specified, will hold IDA stkoff after the call.
        :returns: index of stkvar in the frame or -1
        """
        ...
    def has_side_effects(self, include_ldx_and_divs: bool = False) -> bool:
        r"""Has any side effects? 
                
        :param include_ldx_and_divs: consider ldx/div/mod as having side effects?
        """
        ...
    def is01(self) -> bool:
        r"""Are the possible values of the operand only 0 and 1? This function returns true for 0/1 constants, bit registers, the result of 'set' insns, etc. 
                
        """
        ...
    def is_arglist(self) -> bool:
        r"""Is a list of arguments?
        
        """
        ...
    def is_bit_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_bit_reg() -> bool
            1. is_bit_reg(reg: mreg_t) -> bool
        
        # 0: is_bit_reg() -> bool
        
        
        # 1: is_bit_reg(reg: mreg_t) -> bool
        
        Is a bit register? This includes condition codes and eventually other bit registers 
                
        
        """
        ...
    def is_cc(self) -> bool:
        r"""Is a condition code?
        
        """
        ...
    def is_ccflags(self) -> bool:
        ...
    def is_constant(self, is_signed: bool = True) -> bool:
        r"""Retrieve value of a constant integer operand. 
                
        :param is_signed: should treat the value as signed
        :returns: true if the operand is mop_n
        """
        ...
    def is_equal_to(self, n: uint64, is_signed: bool = True) -> bool:
        ...
    def is_extended_from(self, nbytes: int, is_signed: bool) -> bool:
        r"""Does the high part of the operand consist of zero or sign bytes?
        
        """
        ...
    def is_for_abi(self) -> bool:
        ...
    def is_glbaddr(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_glbaddr() -> bool
            1. is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        # 0: is_glbaddr() -> bool
        
        Is address of a global memory cell?
        
        
        # 1: is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        Is address of the specified global memory cell?
        
        
        """
        ...
    def is_glbaddr_from_fixup(self) -> bool:
        ...
    def is_glbvar(self) -> bool:
        r"""Is a global variable?
        
        """
        ...
    def is_impptr_done(self) -> bool:
        ...
    def is_insn(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_insn() -> bool
            1. is_insn(code: mcode_t) -> bool
        
        # 0: is_insn() -> bool
        
        Is a sub-instruction?
        
        
        # 1: is_insn(code: mcode_t) -> bool
        
        Is a sub-instruction with the specified opcode?
        
        
        """
        ...
    def is_kreg(self) -> bool:
        r"""Is a kernel register?
        
        """
        ...
    def is_lowaddr(self) -> bool:
        ...
    def is_mblock(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_mblock() -> bool
            1. is_mblock(serial: int) -> bool
        
        # 0: is_mblock() -> bool
        
        Is a block reference?
        
        
        # 1: is_mblock(serial: int) -> bool
        
        Is a block reference to the specified block?
        
        
        """
        ...
    def is_negative_constant(self) -> bool:
        ...
    def is_one(self) -> bool:
        ...
    def is_pcval(self) -> bool:
        ...
    def is_positive_constant(self) -> bool:
        ...
    def is_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_reg() -> bool
            1. is_reg(_r: mreg_t) -> bool
            2. is_reg(_r: mreg_t, _size: int) -> bool
        
        # 0: is_reg() -> bool
        
        Is a register operand? See also get_mreg_name() 
                
        
        # 1: is_reg(_r: mreg_t) -> bool
        
        Is the specified register?
        
        
        # 2: is_reg(_r: mreg_t, _size: int) -> bool
        
        Is the specified register of the specified size?
        
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""Is a scattered operand?
        
        """
        ...
    def is_sign_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of the sign bytes? 
                
        :param nbytes: number of bytes that were sign extended. the remaining size-nbytes high bytes must be sign bytes Example: is_sign_extended_from(xds.4(op.1), 1) -> true because the high 3 bytes are certainly sign bits
        """
        ...
    def is_stkaddr(self) -> bool:
        r"""Is address of a stack variable?
        
        """
        ...
    def is_stkvar(self) -> bool:
        r"""Is a stack variable?
        
        """
        ...
    def is_udt(self) -> bool:
        ...
    def is_undef_val(self) -> bool:
        ...
    def is_zero(self) -> bool:
        ...
    def is_zero_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of zero bytes? 
                
        :param nbytes: number of bytes that were zero extended. the remaining size-nbytes high bytes must be zero Example: is_zero_extended_from(xdu.8(op.1), 2) -> true because the high 6 bytes are certainly zero
        """
        ...
    def lexcompare(self, ra: mop_addr_t) -> int:
        ...
    def make_blkref(self, blknum: int) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_first_half(self, width: int) -> bool:
        r"""Make the first part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_fpnum(self, bytes: void) -> bool:
        r"""Create a floating point constant operand. 
                
        :param bytes: pointer to the floating point value as used by the current processor (e.g. for x86 it must be in IEEE 754)
        :returns: success
        """
        ...
    def make_gvar(self, ea: ida_idaapi.ea_t) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_helper(self, name: str) -> None:
        r"""Create a helper operand. A helper operand usually keeps a built-in function name like "va_start" It is essentially just an arbitrary identifier without any additional info. 
                
        """
        ...
    def make_high_half(self, width: int) -> bool:
        r"""Make the high part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_insn(self, ins: minsn_t) -> None:
        r"""Create a nested instruction.
        
        """
        ...
    def make_low_half(self, width: int) -> bool:
        r"""Make the low part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_number(self, args: Any) -> None:
        r"""Create an integer constant operand. 
                
        :param _value: value to store in the operand
        :param _size: size of the value in bytes (1,2,4,8)
        :param _ea: address of the processor instruction that made the value
        :param opnum: operand number of the processor instruction
        """
        ...
    def make_reg(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. make_reg(reg: mreg_t) -> None
            1. make_reg(reg: mreg_t, _size: int) -> None
        
        # 0: make_reg(reg: mreg_t) -> None
        
        Create a register operand.
        
        
        # 1: make_reg(reg: mreg_t, _size: int) -> None
        
        
        """
        ...
    def make_reg_pair(self, loreg: int, hireg: int, halfsize: int) -> None:
        r"""Create pair of registers. 
                
        :param loreg: register holding the low part of the value
        :param hireg: register holding the high part of the value
        :param halfsize: the size of each of loreg/hireg
        """
        ...
    def make_second_half(self, width: int) -> bool:
        r"""Make the second part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_stkvar(self, mba: mba_t, off: int) -> None:
        ...
    def may_use_aliased_memory(self) -> bool:
        r"""Is it possible for the operand to use aliased memory?
        
        """
        ...
    def preserve_side_effects(self, blk: mblock_t, top: minsn_t, moved_calls: bool = None) -> bool:
        r"""Move subinstructions with side effects out of the operand. If we decide to delete an instruction operand, it is a good idea to call this function. Alternatively we should skip such operands by calling mop_t::has_side_effects() For example, if we transform: jnz x, x, @blk => goto @blk then we must call this function before deleting the X operands. 
                
        :param blk: current block
        :param top: top level instruction that contains our operand
        :param moved_calls: pointer to the boolean that will track if all side effects get handled correctly. must be false initially.
        :returns: false failed to preserve a side effect, it is not safe to delete the operand true no side effects or successfully preserved them
        """
        ...
    def probably_floating(self) -> bool:
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def set_for_abi(self) -> None:
        ...
    def set_impptr_done(self) -> None:
        ...
    def set_lowaddr(self) -> None:
        ...
    def set_udt(self) -> None:
        ...
    def set_undef_val(self) -> None:
        ...
    def shift_mop(self, offset: int) -> bool:
        r"""Shift the operand. This function shifts only the beginning of the operand. The operand size will be changed. Examples: shift_mop(AH.1, -1) -> AX.2 shift_mop(qword_00000008.8, 4) -> dword_0000000C.4 shift_mop(xdu.8(op.4), 4) -> #0.4 shift_mop(#0x12345678.4, 3) -> #12.1 
                
        :param offset: shift count (the number of bytes to shift)
        :returns: success
        """
        ...
    def signed_value(self) -> int64:
        ...
    def swap(self, rop: mop_t) -> None:
        ...
    def unsigned_value(self) -> uint64:
        ...
    def update_numop_value(self, val: uint64) -> None:
        ...
    def value(self, is_signed: bool) -> uint64:
        r"""Retrieve value of a constant integer operand. These functions can be called only for mop_n operands. See is_constant() that can be called on any operand. 
                
        """
        ...
    def zero(self) -> None:
        ...

class mop_pair_t:
    @property
    def hop(self) -> Any: ...
    @property
    def lop(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class mop_t:
    @property
    def a(self) -> Any: ...
    @property
    def b(self) -> Any: ...
    @property
    def c(self) -> Any: ...
    @property
    def cstr(self) -> Any: ...
    @property
    def d(self) -> Any: ...
    @property
    def f(self) -> Any: ...
    @property
    def fpc(self) -> Any: ...
    @property
    def g(self) -> Any: ...
    @property
    def helper(self) -> Any: ...
    @property
    def l(self) -> Any: ...
    @property
    def meminfo(self) -> Any: ...
    @property
    def nnn(self) -> Any: ...
    @property
    def obj_id(self) -> Any: ...
    @property
    def oprops(self) -> Any: ...
    @property
    def pair(self) -> Any: ...
    @property
    def r(self) -> Any: ...
    @property
    def s(self) -> Any: ...
    @property
    def scif(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def t(self) -> Any: ...
    @property
    def valnum(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, rop: mop_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, rop: mop_t) -> bool:
        ...
    def __ne__(self, rop: mop_t) -> bool:
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
    def apply_ld_mcode(self, mcode: mcode_t, ea: ida_idaapi.ea_t, newsize: int) -> None:
        r"""Apply a unary opcode to the operand. 
                
        :param mcode: opcode to apply. it must accept 'l' and 'd' operands but not 'r'. examples: m_low/m_high/m_xds/m_xdu
        :param ea: value of minsn_t::ea for the newly created insruction
        :param newsize: new operand size Example: apply_ld_mcode(m_low) will convert op => low(op)
        """
        ...
    def apply_xds(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def apply_xdu(self, ea: ida_idaapi.ea_t, newsize: int) -> None:
        ...
    def assign(self, rop: mop_t) -> cinsn_t:
        ...
    def change_size(self, nsize: int, sideff: side_effect_t = 1) -> bool:
        r"""Change the operand size. Examples: change_size(AL.1, 2) -> AX.2 change_size(qword_00000008.8, 4) -> dword_00000008.4 change_size(xdu.8(op.4), 4) -> op.4 change_size(#0x12345678.4, 1) -> #0x78.1 
                
        :param nsize: new operand size
        :param sideff: may modify the database because of the size change?
        :returns: success
        """
        ...
    def create_from_insn(self, m: minsn_t) -> None:
        r"""Create operand from an instruction. This function creates a nested instruction that can be used as an operand. Example: if m="add x,y,z", our operand will be (t=mop_d,d=m). The destination operand of 'add' (z) is lost. 
                
        :param m: instruction to embed into operand. may not be nullptr.
        """
        ...
    def create_from_ivlset(self, mba: mba_t, ivs: ivlset_t, fullsize: int) -> bool:
        r"""Create operand from ivlset_t. Example: if IVS contains [glbvar..glbvar+4), our operand will be (t=mop_v, g=&glbvar, size=4) 
                
        :param mba: pointer to microcode
        :param ivs: set of memory intervals
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_mlist(self, mba: mba_t, lst: mlist_t, fullsize: int) -> bool:
        r"""Create operand from mlist_t. Example: if LST contains 4 bits for R0.4, our operand will be (t=mop_r, r=R0, size=4) 
                
        :param mba: pointer to microcode
        :param lst: list of locations
        :param fullsize: mba->fullsize
        :returns: success
        """
        ...
    def create_from_scattered_vdloc(self, mba: mba_t, name: str, type: tinfo_t, loc: vdloc_t) -> None:
        r"""Create operand from scattered vdloc_t. Example: if LOC is (ALOC_DIST, {EAX.4, EDX.4}) and TYPE is _LARGE_INTEGER, our operand will be (t=mop_sc, scif={EAX.4, EDX.4}) 
                
        :param mba: pointer to microcode
        :param name: name of the operand, if available
        :param type: type of the operand, must be present
        :param loc: a scattered location
        :returns: success
        """
        ...
    def create_from_vdloc(self, mba: mba_t, loc: vdloc_t, _size: int) -> None:
        r"""Create operand from vdloc_t. Example: if LOC contains (type=ALOC_REG1, r=R0), our operand will be (t=mop_r, r=R0, size=_SIZE) 
                
        :param mba: pointer to microcode
        :param loc: location
        :param _size: operand size Note: this function cannot handle scattered locations.
        :returns: success
        """
        ...
    def double_size(self, sideff: side_effect_t = 1) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def equal_mops(self, rop: mop_t, eqflags: int) -> bool:
        r"""Compare operands. This is the main comparison function for operands. 
                
        :param rop: operand to compare with
        :param eqflags: combination of comparison bits bits
        """
        ...
    def erase(self) -> None:
        ...
    def erase_but_keep_size(self) -> None:
        ...
    def for_all_ops(self, mv: mop_visitor_t, type: tinfo_t = None, is_target: bool = False) -> int:
        r"""Visit the operand and all its sub-operands. This function visits the current operand as well. 
                
        :param mv: visitor object
        :param type: operand type
        :param is_target: is a destination operand?
        """
        ...
    def for_all_scattered_submops(self, sv: scif_visitor_t) -> int:
        r"""Visit all sub-operands of a scattered operand. This function does not visit the current operand, only its sub-operands. All sub-operands are synthetic and are destroyed after the visitor. This function works only with scattered operands. 
                
        :param sv: visitor object
        """
        ...
    def get_insn(self, code: mcode_t) -> minsn_t:
        r"""Get subinstruction of the operand. If the operand has a subinstruction with the specified opcode, return it. 
                
        :param code: desired opcode
        :returns: pointer to the instruction or nullptr
        """
        ...
    def get_stkoff(self, p_vdoff: sval_t) -> int:
        r"""Get the referenced stack offset. This function can also handle mop_sc if it is entirely mapped into a continuous stack region. 
                
        :param p_vdoff: the output buffer
        :returns: success
        """
        ...
    def get_stkvar(self, udm: udm_t = None, p_idaoff: uval_t = None) -> ssize_t:
        r"""Retrieve the referenced stack variable. 
                
        :param udm: stkvar, may be nullptr
        :param p_idaoff: if specified, will hold IDA stkoff after the call.
        :returns: index of stkvar in the frame or -1
        """
        ...
    def has_side_effects(self, include_ldx_and_divs: bool = False) -> bool:
        r"""Has any side effects? 
                
        :param include_ldx_and_divs: consider ldx/div/mod as having side effects?
        """
        ...
    def is01(self) -> bool:
        r"""Are the possible values of the operand only 0 and 1? This function returns true for 0/1 constants, bit registers, the result of 'set' insns, etc. 
                
        """
        ...
    def is_arglist(self) -> bool:
        r"""Is a list of arguments?
        
        """
        ...
    def is_bit_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_bit_reg() -> bool
            1. is_bit_reg(reg: mreg_t) -> bool
        
        # 0: is_bit_reg() -> bool
        
        
        # 1: is_bit_reg(reg: mreg_t) -> bool
        
        Is a bit register? This includes condition codes and eventually other bit registers 
                
        
        """
        ...
    def is_cc(self) -> bool:
        r"""Is a condition code?
        
        """
        ...
    def is_ccflags(self) -> bool:
        ...
    def is_constant(self, is_signed: bool = True) -> bool:
        r"""Retrieve value of a constant integer operand. 
                
        :param is_signed: should treat the value as signed
        :returns: true if the operand is mop_n
        """
        ...
    def is_equal_to(self, n: uint64, is_signed: bool = True) -> bool:
        ...
    def is_extended_from(self, nbytes: int, is_signed: bool) -> bool:
        r"""Does the high part of the operand consist of zero or sign bytes?
        
        """
        ...
    def is_for_abi(self) -> bool:
        ...
    def is_glbaddr(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_glbaddr() -> bool
            1. is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        # 0: is_glbaddr() -> bool
        
        Is address of a global memory cell?
        
        
        # 1: is_glbaddr(ea: ida_idaapi.ea_t) -> bool
        
        Is address of the specified global memory cell?
        
        
        """
        ...
    def is_glbaddr_from_fixup(self) -> bool:
        ...
    def is_glbvar(self) -> bool:
        r"""Is a global variable?
        
        """
        ...
    def is_impptr_done(self) -> bool:
        ...
    def is_insn(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_insn() -> bool
            1. is_insn(code: mcode_t) -> bool
        
        # 0: is_insn() -> bool
        
        Is a sub-instruction?
        
        
        # 1: is_insn(code: mcode_t) -> bool
        
        Is a sub-instruction with the specified opcode?
        
        
        """
        ...
    def is_kreg(self) -> bool:
        r"""Is a kernel register?
        
        """
        ...
    def is_lowaddr(self) -> bool:
        ...
    def is_mblock(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_mblock() -> bool
            1. is_mblock(serial: int) -> bool
        
        # 0: is_mblock() -> bool
        
        Is a block reference?
        
        
        # 1: is_mblock(serial: int) -> bool
        
        Is a block reference to the specified block?
        
        
        """
        ...
    def is_negative_constant(self) -> bool:
        ...
    def is_one(self) -> bool:
        ...
    def is_pcval(self) -> bool:
        ...
    def is_positive_constant(self) -> bool:
        ...
    def is_reg(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. is_reg() -> bool
            1. is_reg(_r: mreg_t) -> bool
            2. is_reg(_r: mreg_t, _size: int) -> bool
        
        # 0: is_reg() -> bool
        
        Is a register operand? See also get_mreg_name() 
                
        
        # 1: is_reg(_r: mreg_t) -> bool
        
        Is the specified register?
        
        
        # 2: is_reg(_r: mreg_t, _size: int) -> bool
        
        Is the specified register of the specified size?
        
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""Is a scattered operand?
        
        """
        ...
    def is_sign_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of the sign bytes? 
                
        :param nbytes: number of bytes that were sign extended. the remaining size-nbytes high bytes must be sign bytes Example: is_sign_extended_from(xds.4(op.1), 1) -> true because the high 3 bytes are certainly sign bits
        """
        ...
    def is_stkaddr(self) -> bool:
        r"""Is address of a stack variable?
        
        """
        ...
    def is_stkvar(self) -> bool:
        r"""Is a stack variable?
        
        """
        ...
    def is_udt(self) -> bool:
        ...
    def is_undef_val(self) -> bool:
        ...
    def is_zero(self) -> bool:
        ...
    def is_zero_extended_from(self, nbytes: int) -> bool:
        r"""Does the high part of the operand consist of zero bytes? 
                
        :param nbytes: number of bytes that were zero extended. the remaining size-nbytes high bytes must be zero Example: is_zero_extended_from(xdu.8(op.1), 2) -> true because the high 6 bytes are certainly zero
        """
        ...
    def lexcompare(self, rop: mop_t) -> int:
        ...
    def make_blkref(self, blknum: int) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_first_half(self, width: int) -> bool:
        r"""Make the first part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_fpnum(self, bytes: void) -> bool:
        r"""Create a floating point constant operand. 
                
        :param bytes: pointer to the floating point value as used by the current processor (e.g. for x86 it must be in IEEE 754)
        :returns: success
        """
        ...
    def make_gvar(self, ea: ida_idaapi.ea_t) -> None:
        r"""Create a global variable operand.
        
        """
        ...
    def make_helper(self, name: str) -> None:
        r"""Create a helper operand. A helper operand usually keeps a built-in function name like "va_start" It is essentially just an arbitrary identifier without any additional info. 
                
        """
        ...
    def make_high_half(self, width: int) -> bool:
        r"""Make the high part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_insn(self, ins: minsn_t) -> None:
        r"""Create a nested instruction.
        
        """
        ...
    def make_low_half(self, width: int) -> bool:
        r"""Make the low part of the operand. This function takes into account the memory endianness (byte sex) 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_number(self, args: Any) -> None:
        r"""Create an integer constant operand. 
                
        :param _value: value to store in the operand
        :param _size: size of the value in bytes (1,2,4,8)
        :param _ea: address of the processor instruction that made the value
        :param opnum: operand number of the processor instruction
        """
        ...
    def make_reg(self, args: Any) -> None:
        r"""This function has the following signatures:
        
            0. make_reg(reg: mreg_t) -> None
            1. make_reg(reg: mreg_t, _size: int) -> None
        
        # 0: make_reg(reg: mreg_t) -> None
        
        Create a register operand.
        
        
        # 1: make_reg(reg: mreg_t, _size: int) -> None
        
        
        """
        ...
    def make_reg_pair(self, loreg: int, hireg: int, halfsize: int) -> None:
        r"""Create pair of registers. 
                
        :param loreg: register holding the low part of the value
        :param hireg: register holding the high part of the value
        :param halfsize: the size of each of loreg/hireg
        """
        ...
    def make_second_half(self, width: int) -> bool:
        r"""Make the second part of the operand. This function does not care about the memory endianness 
                
        :param width: the desired size of the operand part in bytes
        :returns: success
        """
        ...
    def make_stkvar(self, mba: mba_t, off: int) -> None:
        ...
    def may_use_aliased_memory(self) -> bool:
        r"""Is it possible for the operand to use aliased memory?
        
        """
        ...
    def preserve_side_effects(self, blk: mblock_t, top: minsn_t, moved_calls: bool = None) -> bool:
        r"""Move subinstructions with side effects out of the operand. If we decide to delete an instruction operand, it is a good idea to call this function. Alternatively we should skip such operands by calling mop_t::has_side_effects() For example, if we transform: jnz x, x, @blk => goto @blk then we must call this function before deleting the X operands. 
                
        :param blk: current block
        :param top: top level instruction that contains our operand
        :param moved_calls: pointer to the boolean that will track if all side effects get handled correctly. must be false initially.
        :returns: false failed to preserve a side effect, it is not safe to delete the operand true no side effects or successfully preserved them
        """
        ...
    def probably_floating(self) -> bool:
        ...
    def replace_by(self, o: Any) -> Any:
        ...
    def set_for_abi(self) -> None:
        ...
    def set_impptr_done(self) -> None:
        ...
    def set_lowaddr(self) -> None:
        ...
    def set_udt(self) -> None:
        ...
    def set_undef_val(self) -> None:
        ...
    def shift_mop(self, offset: int) -> bool:
        r"""Shift the operand. This function shifts only the beginning of the operand. The operand size will be changed. Examples: shift_mop(AH.1, -1) -> AX.2 shift_mop(qword_00000008.8, 4) -> dword_0000000C.4 shift_mop(xdu.8(op.4), 4) -> #0.4 shift_mop(#0x12345678.4, 3) -> #12.1 
                
        :param offset: shift count (the number of bytes to shift)
        :returns: success
        """
        ...
    def signed_value(self) -> int64:
        ...
    def swap(self, rop: mop_t) -> None:
        ...
    def unsigned_value(self) -> uint64:
        ...
    def update_numop_value(self, val: uint64) -> None:
        ...
    def value(self, is_signed: bool) -> uint64:
        r"""Retrieve value of a constant integer operand. These functions can be called only for mop_n operands. See is_constant() that can be called on any operand. 
                
        """
        ...
    def zero(self) -> None:
        ...

class mop_visitor_t(op_parent_info_t):
    @property
    def blk(self) -> Any: ...
    @property
    def curins(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def prune(self) -> Any: ...
    @property
    def topins(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _mba: mba_t = None, _blk: mblock_t = None, _topins: minsn_t = None) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def visit_mop(self, op: mop_t, type: tinfo_t, is_target: bool) -> int:
        ...

class mopvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: mopvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: mopvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: mop_t) -> None:
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
    def add_unique(self, x: mop_t) -> bool:
        ...
    def append(self, x: mop_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: mopvec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: mop_t) -> bool:
        ...
    def inject(self, s: mop_t, len: size_t) -> None:
        ...
    def insert(self, it: mop_t, x: mop_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: mopvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class node_bitset_t(bitset_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: bitset_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: bitset_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: bitset_t) -> bool:
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
        ...
    def __le__(self, r: bitset_t) -> bool:
        ...
    def __len__(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def __lt__(self, r: bitset_t) -> bool:
        ...
    def __ne__(self, r: bitset_t) -> bool:
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
    def add(self, args: Any) -> voff_t:
        r"""This function has the following signatures:
        
            0. add(bit: int) -> bool
            1. add(bit: int, width: int) -> bool
            2. add(ml: const bitset_t &) -> bool
        
        # 0: add(bit: int) -> bool
        
        
        # 1: add(bit: int, width: int) -> bool
        
        
        # 2: add(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: bitset_t) -> int:
        ...
    def copy(self, m: bitset_t) -> bitset_t:
        ...
    def count(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def cut_at(self, maxbit: int) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def fill_with_ones(self, maxbit: int) -> None:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, bit: int) -> bool:
        ...
    def has_all(self, bit: int, width: int) -> bool:
        ...
    def has_any(self, bit: int, width: int) -> bool:
        ...
    def has_common(self, ml: bitset_t) -> bool:
        ...
    def inc(self, p: iterator, n: int = 1) -> None:
        ...
    def includes(self, ml: bitset_t) -> bool:
        ...
    def intersect(self, ml: bitset_t) -> int:
        ...
    def is_subset_of(self, ml: bitset_t) -> bool:
        ...
    def itat(self, n: int) -> bitset_t:
        ...
    def itv(self, it: iterator) -> int:
        ...
    def last(self) -> int:
        ...
    def shift_down(self, shift: int) -> None:
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(bit: int) -> bool
            1. sub(bit: int, width: int) -> bool
            2. sub(ml: const bitset_t &) -> bool
        
        # 0: sub(bit: int) -> bool
        
        
        # 1: sub(bit: int, width: int) -> bool
        
        
        # 2: sub(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def swap(self, r: bitset_t) -> None:
        ...

class number_format_t:
    @property
    def flags(self) -> Any: ...
    @property
    def flags32(self) -> Any: ...
    @property
    def opnum(self) -> Any: ...
    @property
    def org_nbytes(self) -> Any: ...
    @property
    def props(self) -> Any: ...
    @property
    def serial(self) -> Any: ...
    @property
    def type_name(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _opnum: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def get_radix(self) -> int:
        r"""Get number radix 
                
        :returns: 2,8,10, or 16
        """
        ...
    def has_unmutable_type(self) -> bool:
        ...
    def is_char(self) -> bool:
        r"""Is a character constant?
        
        """
        ...
    def is_dec(self) -> bool:
        r"""Is a decimal number?
        
        """
        ...
    def is_enum(self) -> bool:
        r"""Is a symbolic constant?
        
        """
        ...
    def is_fixed(self) -> bool:
        r"""Is number representation fixed? Fixed representation cannot be modified by the decompiler 
                
        """
        ...
    def is_hex(self) -> bool:
        r"""Is a hexadecimal number?
        
        """
        ...
    def is_numop(self) -> bool:
        r"""Is a number?
        
        """
        ...
    def is_oct(self) -> bool:
        r"""Is a octal number?
        
        """
        ...
    def is_stroff(self) -> bool:
        r"""Is a structure field offset?
        
        """
        ...
    def needs_to_be_inverted(self) -> bool:
        r"""Does the number need to be negated or bitwise negated? Returns true if the user requested a negation but it is not done yet 
                
        """
        ...

class op_parent_info_t:
    @property
    def blk(self) -> Any: ...
    @property
    def curins(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def topins(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _mba: mba_t = None, _blk: mblock_t = None, _topins: minsn_t = None) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class operand_locator_t:
    @property
    def ea(self) -> Any: ...
    @property
    def opnum(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: operand_locator_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: operand_locator_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: operand_locator_t) -> bool:
        ...
    def __init__(self, _ea: ida_idaapi.ea_t, _opnum: int) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: operand_locator_t) -> bool:
        ...
    def __lt__(self, r: operand_locator_t) -> bool:
        ...
    def __ne__(self, r: operand_locator_t) -> bool:
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
    def compare(self, r: operand_locator_t) -> int:
        ...

class optblock_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def func(self, blk: mblock_t) -> int:
        r"""Optimize a block. This function usually performs the optimizations that require analyzing the entire block and/or its neighbors. For example it can recognize patterns and perform conversions like: b0: b0: ... ... jnz x, 0, @b2 => jnz x, 0, @b2 b1: b1: add x, 0, y mov x, y ... ... 
                
        :param blk: Basic block to optimize as a whole.
        :returns: number of changes made to the block. See also mark_lists_dirty.
        """
        ...
    def install(self) -> None:
        ...
    def remove(self) -> bool:
        ...

class optinsn_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def func(self, blk: mblock_t, ins: minsn_t, optflags: int) -> int:
        r"""Optimize an instruction. 
                
        :param blk: current basic block. maybe nullptr, which means that the instruction must be optimized without context
        :param ins: instruction to optimize; it is always a top-level instruction. the callback may not delete the instruction but may convert it into nop (see mblock_t::make_nop). to optimize sub-instructions, visit them using minsn_visitor_t. sub-instructions may not be converted into nop but can be converted to "mov x,x". for example: add x,0,x => mov x,x this callback may change other instructions in the block, but should do this with care, e.g. to no break the propagation algorithm if called with OPTI_NO_LDXOPT.
        :param optflags: combination of optimization flags bits
        :returns: number of changes made to the instruction. if after this call the instruction's use/def lists have changed, you must mark the block level lists as dirty (see mark_lists_dirty)
        """
        ...
    def install(self) -> None:
        ...
    def remove(self) -> bool:
        ...

class qstring_printer_t(vc_printer_t, vd_printer_t):
    @property
    def func(self) -> Any: ...
    @property
    def hdrlines(self) -> Any: ...
    @property
    def lastchar(self) -> Any: ...
    @property
    def s(self) -> Any: ...
    @property
    def tmpbuf(self) -> Any: ...
    @property
    def with_tags(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, f: cfunc_t, tags: bool) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def get_s(self) -> str:
        ...
    def oneliner(self) -> bool:
        r"""Are we generating one-line text representation? 
                
        :returns: `true` if the output will occupy one line without line breaks
        """
        ...

class qvector_carg_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_carg_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_carg_t) -> bool:
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
    def __setitem__(self, i: size_t, v: carg_t) -> None:
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
    def add_unique(self, x: carg_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_carg_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: carg_t) -> bool:
        ...
    def inject(self, s: carg_t, len: size_t) -> None:
        ...
    def insert(self, it: carg_t, x: carg_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_carg_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class qvector_catchexprs_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_catchexprs_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_catchexprs_t) -> bool:
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
    def __setitem__(self, i: size_t, v: catchexpr_t) -> None:
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
    def add_unique(self, x: catchexpr_t) -> bool:
        ...
    def append(self, x: catchexpr_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_catchexprs_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: catchexpr_t) -> bool:
        ...
    def inject(self, s: catchexpr_t, len: size_t) -> None:
        ...
    def insert(self, it: catchexpr_t, x: catchexpr_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_catchexprs_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class qvector_ccase_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_ccase_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_ccase_t) -> bool:
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
    def __setitem__(self, i: size_t, v: ccase_t) -> None:
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
    def add_unique(self, x: ccase_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_ccase_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: ccase_t) -> bool:
        ...
    def inject(self, s: ccase_t, len: size_t) -> None:
        ...
    def insert(self, it: ccase_t, x: ccase_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_ccase_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class qvector_ccatchvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_ccatchvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_ccatchvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: ccatch_t) -> None:
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
    def add_unique(self, x: ccatch_t) -> bool:
        ...
    def append(self, x: ccatch_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_ccatchvec_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: ccatch_t) -> bool:
        ...
    def inject(self, s: ccatch_t, len: size_t) -> None:
        ...
    def insert(self, it: ccatch_t, x: ccatch_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_ccatchvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class qvector_history_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_history_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_history_t) -> bool:
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
    def __setitem__(self, i: size_t, v: history_item_t) -> None:
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
    def add_unique(self, x: history_item_t) -> bool:
        ...
    def append(self, x: history_item_t) -> None:
        ...
    def at(self, _idx: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_history_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: history_item_t) -> bool:
        ...
    def inject(self, s: history_item_t, len: size_t) -> None:
        ...
    def insert(self, it: history_item_t, x: history_item_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_history_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class qvector_lvar_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: qvector_lvar_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: qvector_lvar_t) -> bool:
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
    def __setitem__(self, i: size_t, v: lvar_t) -> None:
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
    def add_unique(self, x: lvar_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: qvector_lvar_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: lvar_t) -> bool:
        ...
    def inject(self, s: lvar_t, len: size_t) -> None:
        ...
    def insert(self, it: lvar_t, x: lvar_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: qvector_lvar_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class rlist_t(bitset_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: bitset_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: bitset_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: bitset_t) -> bool:
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
        ...
    def __le__(self, r: bitset_t) -> bool:
        ...
    def __len__(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def __lt__(self, r: bitset_t) -> bool:
        ...
    def __ne__(self, r: bitset_t) -> bool:
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
    def add(self, args: Any) -> voff_t:
        r"""This function has the following signatures:
        
            0. add(bit: int) -> bool
            1. add(bit: int, width: int) -> bool
            2. add(ml: const bitset_t &) -> bool
        
        # 0: add(bit: int) -> bool
        
        
        # 1: add(bit: int, width: int) -> bool
        
        
        # 2: add(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: bitset_t) -> int:
        ...
    def copy(self, m: bitset_t) -> bitset_t:
        ...
    def count(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. count() -> int
            1. count(bit: int) -> int
        
        # 0: count() -> int
        
        
        # 1: count(bit: int) -> int
        
        
        """
        ...
    def cut_at(self, maxbit: int) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def fill_with_ones(self, maxbit: int) -> None:
        ...
    def front(self) -> block_chains_t:
        ...
    def has(self, bit: int) -> bool:
        ...
    def has_all(self, bit: int, width: int) -> bool:
        ...
    def has_any(self, bit: int, width: int) -> bool:
        ...
    def has_common(self, ml: bitset_t) -> bool:
        ...
    def inc(self, p: iterator, n: int = 1) -> None:
        ...
    def includes(self, ml: bitset_t) -> bool:
        ...
    def intersect(self, ml: bitset_t) -> int:
        ...
    def is_subset_of(self, ml: bitset_t) -> bool:
        ...
    def itat(self, n: int) -> bitset_t:
        ...
    def itv(self, it: iterator) -> int:
        ...
    def last(self) -> int:
        ...
    def shift_down(self, shift: int) -> None:
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(bit: int) -> bool
            1. sub(bit: int, width: int) -> bool
            2. sub(ml: const bitset_t &) -> bool
        
        # 0: sub(bit: int) -> bool
        
        
        # 1: sub(bit: int, width: int) -> bool
        
        
        # 2: sub(ml: const bitset_t &) -> bool
        
        
        """
        ...
    def swap(self, r: bitset_t) -> None:
        ...

class scif_t(vdloc_t):
    @property
    def mba(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: vdloc_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: vdloc_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: vdloc_t) -> bool:
        ...
    def __init__(self, _mba: mba_t, tif: tinfo_t, n: str = None) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: vdloc_t) -> bool:
        ...
    def __lt__(self, r: vdloc_t) -> bool:
        ...
    def __ne__(self, r: vdloc_t) -> bool:
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
    def advance(self, delta: int) -> bool:
        r"""Move the location to point 'delta' bytes further.
        
        """
        ...
    def align_reg_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set register offset to align it to the upper part of _SLOTSIZE.
        
        """
        ...
    def align_stkoff_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set stack offset to align to the upper part of _SLOTSIZE.
        
        """
        ...
    def atype(self) -> argloc_type_t:
        r"""Get type (Argument location types)
        
        """
        ...
    def calc_offset(self) -> int:
        r"""Calculate offset that can be used to compare 2 similar arglocs.
        
        """
        ...
    def compare(self, r: vdloc_t) -> int:
        ...
    def consume_rrel(self, p: rrel_t) -> None:
        r"""Set register-relative location - can't be nullptr.
        
        """
        ...
    def consume_scattered(self, p: scattered_aloc_t) -> None:
        r"""Set distributed argument location.
        
        """
        ...
    def get_biggest(self) -> biggest_t:
        r"""Get largest element in internal union.
        
        """
        ...
    def get_custom(self) -> None:
        r"""Get custom argloc info. Use if atype() == ALOC_CUSTOM 
                
        """
        ...
    def get_ea(self) -> ida_idaapi.ea_t:
        r"""Get the global address. Use when atype() == ALOC_STATIC 
                
        """
        ...
    def get_reginfo(self) -> int:
        r"""Get all register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def get_rrel(self) -> rrel_t:
        r"""Get register-relative info. Use when atype() == ALOC_RREL 
                
        """
        ...
    def has_reg(self) -> bool:
        r"""TRUE if argloc has a register part.
        
        """
        ...
    def has_stkoff(self) -> bool:
        r"""TRUE if argloc has a stack part.
        
        """
        ...
    def in_stack(self) -> bool:
        r"""TRUE if argloc is in stack entirely.
        
        """
        ...
    def is_aliasable(self, mb: mba_t, size: int) -> bool:
        ...
    def is_badloc(self) -> bool:
        r"""See ALOC_NONE.
        
        """
        ...
    def is_custom(self) -> bool:
        r"""See ALOC_CUSTOM.
        
        """
        ...
    def is_ea(self) -> bool:
        r"""See ALOC_STATIC.
        
        """
        ...
    def is_fragmented(self) -> bool:
        r"""is_scattered() || is_reg2()
        
        """
        ...
    def is_mixed_scattered(self) -> bool:
        r"""mixed scattered: consists of register and stack parts
        
        """
        ...
    def is_reg(self) -> bool:
        r"""is_reg1() || is_reg2()
        
        """
        ...
    def is_reg1(self) -> bool:
        r"""See ALOC_REG1.
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""See ALOC_REG2.
        
        """
        ...
    def is_rrel(self) -> bool:
        r"""See ALOC_RREL.
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""See ALOC_DIST.
        
        """
        ...
    def is_stkoff(self) -> bool:
        r"""See ALOC_STACK.
        
        """
        ...
    def reg1(self) -> int:
        ...
    def reg2(self) -> int:
        r"""Get info for the second register. Use when atype() == ALOC_REG2 
                
        """
        ...
    def regoff(self) -> int:
        r"""Get offset from the beginning of the register in bytes. Use when atype() == ALOC_REG1 
                
        """
        ...
    def scattered(self) -> scattered_aloc_t:
        r"""Get scattered argument info. Use when atype() == ALOC_DIST 
                
        """
        ...
    def set_badloc(self) -> None:
        r"""Set to invalid location.
        
        """
        ...
    def set_ea(self, _ea: ida_idaapi.ea_t) -> None:
        r"""Set static ea location.
        
        """
        ...
    def set_reg1(self, r1: int) -> None:
        ...
    def set_reg2(self, _reg1: int, _reg2: int) -> None:
        r"""Set secondary register location.
        
        """
        ...
    def set_stkoff(self, off: int) -> None:
        r"""Set stack offset location.
        
        """
        ...
    def stkoff(self) -> int:
        r"""Get the stack offset. Use if atype() == ALOC_STACK 
                
        """
        ...
    def swap(self, r: argloc_t) -> None:
        r"""Assign this == r and r == this.
        
        """
        ...

class scif_visitor_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def visit_scif_mop(self, r: mop_t, off: int) -> int:
        ...

class simple_graph_t:
    @property
    def colored_gdl_edges(self) -> Any: ...
    @property
    def title(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def begin(self) -> simple_graph_t:
        ...
    def compute_dominators(self, domin: array_of_node_bitset_t, post: bool = False) -> None:
        ...
    def compute_immediate_dominators(self, domin: array_of_node_bitset_t, idomin: intvec_t, post: bool = False) -> None:
        ...
    def depth_first_postorder(self, post: node_ordering_t) -> int:
        ...
    def depth_first_preorder(self, pre: node_ordering_t) -> int:
        ...
    def edge(self, node: int, i: int, ispred: bool) -> int:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> simple_graph_t:
        ...
    def entry(self) -> int:
        ...
    def exists(self, node: int) -> bool:
        ...
    def exit(self) -> int:
        ...
    def front(self) -> block_chains_t:
        ...
    def get_edge_color(self, i: int, j: int) -> bgcolor_t:
        ...
    def get_node_color(self, n: int) -> bgcolor_t:
        ...
    def get_node_label(self, n: int) -> char:
        ...
    def goup(self, node: int) -> int:
        ...
    def inc(self, p: iterator, n: int = 1) -> None:
        ...
    def nedge(self, node: int, ispred: bool) -> int:
        ...
    def node_qty(self) -> int:
        ...
    def npred(self, node: int) -> int:
        ...
    def nsucc(self, node: int) -> int:
        ...
    def pred(self, node: int, i: int) -> int:
        ...
    def print_edge(self, fp: FILE, i: int, j: int) -> bool:
        ...
    def print_graph_attributes(self, fp: FILE) -> None:
        ...
    def print_node(self, fp: FILE, n: int) -> bool:
        ...
    def print_node_attributes(self, fp: FILE, n: int) -> None:
        ...
    def size(self) -> int:
        ...
    def succ(self, node: int, i: int) -> int:
        ...

class stkvar_ref_t:
    @property
    def mba(self) -> Any: ...
    @property
    def off(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: stkvar_ref_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: stkvar_ref_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: stkvar_ref_t) -> bool:
        ...
    def __init__(self, m: mba_t, o: int) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: stkvar_ref_t) -> bool:
        ...
    def __lt__(self, r: stkvar_ref_t) -> bool:
        ...
    def __ne__(self, r: stkvar_ref_t) -> bool:
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
    def compare(self, r: stkvar_ref_t) -> int:
        ...
    def get_stkvar(self, udm: udm_t = None, p_idaoff: uval_t = None) -> ssize_t:
        r"""Retrieve the referenced stack variable. 
                
        :param udm: stkvar, may be nullptr
        :param p_idaoff: if specified, will hold IDA stkoff after the call.
        :returns: index of stkvar in the frame or -1
        """
        ...
    def swap(self, r: stkvar_ref_t) -> None:
        ...

class treeloc_t:
    @property
    def ea(self) -> Any: ...
    @property
    def itp(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: treeloc_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, r: treeloc_t) -> bool:
        ...
    def __ne__(self, value: Any) -> bool:
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

class udc_filter_t(microcode_filter_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply(self, cdg: codegen_t) -> bool:
        r"""generate microcode for an instruction 
                
        :returns: MERR_... code: MERR_OK - user-defined microcode generated, go to the next instruction MERR_INSN - not generated - the caller should try the standard way else - error
        """
        ...
    def cleanup(self) -> None:
        r"""Cleanup the filter This function properly clears type information associated to this filter. 
                
        """
        ...
    def empty(self) -> bool:
        ...
    def init(self, decl: str) -> bool:
        ...
    def install(self) -> None:
        ...
    def match(self, cdg: codegen_t) -> bool:
        r"""return true if the filter object should be applied to given instruction
        
        """
        ...
    def remove(self) -> bool:
        ...

class udcall_map_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: udcall_map_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: udcall_map_iterator_t) -> bool:
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

class udcall_t:
    @property
    def name(self) -> Any: ...
    @property
    def tif(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udcall_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: udcall_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: udcall_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: udcall_t) -> bool:
        ...
    def __lt__(self, r: udcall_t) -> bool:
        ...
    def __ne__(self, r: udcall_t) -> bool:
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
    def compare(self, r: udcall_t) -> int:
        ...
    def empty(self) -> bool:
        ...

class ui_stroff_applicator_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def apply(self, opnum: size_t, path: intvec_t, top_tif: tinfo_t, spath: str) -> bool:
        r""":param opnum: operand ordinal number, see below
        :param path: path describing the union selection, maybe empty
        :param top_tif: tinfo_t of the selected toplevel UDT
        :param spath: selected path
        """
        ...

class ui_stroff_op_t:
    @property
    def offset(self) -> Any: ...
    @property
    def text(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ui_stroff_op_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: ui_stroff_op_t) -> bool:
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

class ui_stroff_ops_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ui_stroff_ops_t) -> bool:
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
    def __getitem__(self, i: size_t) -> ivlset_t:
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
    def __ne__(self, r: ui_stroff_ops_t) -> bool:
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
    def __setitem__(self, i: size_t, v: ui_stroff_op_t) -> None:
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
    def add_unique(self, x: ui_stroff_op_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> ivlset_t:
        ...
    def back(self) -> block_chains_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def extend(self, x: ui_stroff_ops_t) -> None:
        ...
    def extract(self) -> ivlset_t:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def front(self) -> block_chains_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: ui_stroff_op_t) -> bool:
        ...
    def inject(self, s: ui_stroff_op_t, len: size_t) -> None:
        ...
    def insert(self, it: ui_stroff_op_t, x: ui_stroff_op_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ivlset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: ui_stroff_ops_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class user_cmts_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: user_cmts_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: user_cmts_iterator_t) -> bool:
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

class user_cmts_t:
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: treeloc_t) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def keytype(self) -> Any:
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, args: Any) -> Any:
        ...

class user_iflags_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: user_iflags_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: user_iflags_iterator_t) -> bool:
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

class user_iflags_t:
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: citem_locator_t) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def keytype(self, args: Any) -> Any:
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, *args: Any, **kwargs: Any) -> Any:
        r"""int([x]) -> integer
        int(x, base=10) -> integer
        
        Convert a number or string to an integer, or return 0 if no arguments
        are given.  If x is a number, return x.__int__().  For floating point
        numbers, this truncates towards zero.
        
        If x is not a number or if base is given, then x must be a string,
        bytes, or bytearray instance representing an integer literal in the
        given base.  The literal can be preceded by '+' or '-' and be surrounded
        by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
        Base 0 means to interpret the base from the string as an integer literal.
        >>> int('0b100', base=0)
        4
        """
        ...

class user_labels_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: user_labels_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: user_labels_iterator_t) -> bool:
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

class user_labels_t:
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def at(self, _Keyval: int) -> ivlset_t:
        ...
    def size(self) -> int:
        ...

class user_lvar_modifier_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def modify_lvars(self, lvinf: lvar_uservec_t) -> bool:
        r"""Modify lvar settings. Returns: true-modified 
                
        """
        ...

class user_numforms_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: user_numforms_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: user_numforms_iterator_t) -> bool:
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

class user_numforms_t:
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: operand_locator_t) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def keytype(self, _ea: ida_idaapi.ea_t, _opnum: int) -> Any:
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, _opnum: int = 0) -> Any:
        ...

class user_unions_iterator_t:
    @property
    def x(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, p: user_unions_iterator_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, p: user_unions_iterator_t) -> bool:
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

class user_unions_t:
    keytype: tuple  # (<class 'int'>,)
    def __begin(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __clear(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __contains__(self, key: Any) -> Any:
        r""" Returns true if the specified key exists in the . """
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __delitem__(self, key: Any) -> Any:
        r""" Removes the value associated with the provided key. """
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __end(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
        ...
    def __erase(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __find(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __first(self, *args: Any, **kwargs: Any) -> Any:
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
    def __getitem__(self, key: Any) -> ivlset_t:
        r""" Returns the value associated with the provided key. """
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __insert(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __iter__(self) -> Any:
        r""" Iterate over dictionary keys. """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __next(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        ...
    def __second(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def __setattr__(self, name: Any, value: Any) -> Any:
        r"""Implement setattr(self, name, value)."""
        ...
    def __setitem__(self, key: Any, value: Any) -> None:
        r""" Returns the value associated with the provided key. """
        ...
    def __size(self, *args: Any, **kwargs: Any) -> Any:
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
    def at(self, _Keyval: int) -> ivlset_t:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def copy(self) -> bitset_t:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def erase(self, args: Any) -> None:
        ...
    def find(self, args: Any) -> lvar_t:
        ...
    def first(self, args: Any) -> Any:
        ...
    def get(self, key: Any, default: Any = None) -> Any:
        ...
    def has_key(self, key: Any) -> Any:
        ...
    def insert(self, args: Any) -> qvector:
        ...
    def items(self) -> Any:
        ...
    def iteritems(self) -> Any:
        ...
    def iterkeys(self) -> Any:
        ...
    def itervalues(self) -> Any:
        ...
    def keys(self) -> Any:
        ...
    def next(self, args: Any) -> merror_t:
        ...
    def pop(self, key: Any) -> history_item_t:
        r""" Sets the value associated with the provided key. """
        ...
    def popitem(self) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def second(self, args: Any) -> Any:
        ...
    def setdefault(self, key: Any, default: Any = None) -> Any:
        r""" Sets the value associated with the provided key. """
        ...
    def size(self, args: Any) -> int:
        ...
    def values(self) -> Any:
        ...
    def valuetype(self, args: Any) -> Any:
        ...

class uval_ivl_ivlset_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, v: ivl_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, v: ivl_t) -> bool:
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
    def all_values(self) -> bool:
        ...
    def begin(self, args: Any) -> simple_graph_t:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> simple_graph_t:
        ...
    def getivl(self, idx: int) -> ivl_t:
        ...
    def lastivl(self) -> ivl_t:
        ...
    def nivls(self) -> int:
        ...
    def qclear(self) -> None:
        ...
    def set_all_values(self) -> None:
        ...
    def single_value(self, args: Any) -> bool:
        ...
    def swap(self, r: uval_ivl_ivlset_t) -> None:
        ...

class uval_ivl_t:
    @property
    def off(self) -> Any: ...
    @property
    def size(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _off: int, _size: int) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def end(self) -> simple_graph_t:
        ...
    def last(self) -> int:
        ...
    def valid(self) -> bool:
        ...

class valrng_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: valrng_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: valrng_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: valrng_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: valrng_t) -> bool:
        ...
    def __lt__(self, r: valrng_t) -> bool:
        ...
    def __ne__(self, r: valrng_t) -> bool:
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
    def all_values(self) -> bool:
        ...
    def compare(self, r: valrng_t) -> int:
        ...
    def cvt_to_cmp(self) -> bool:
        ...
    def cvt_to_single_value(self) -> bool:
        ...
    def dstr(self) -> str:
        ...
    def empty(self) -> bool:
        ...
    def get_size(self) -> int:
        ...
    def has(self, v: uvlr_t) -> bool:
        ...
    def intersect_with(self, r: valrng_t) -> bool:
        ...
    def inverse(self) -> None:
        ...
    def is_unknown(self) -> bool:
        ...
    def max_svalue(self) -> uvlr_t:
        ...
    def max_value(self) -> uvlr_t:
        ...
    def min_svalue(self) -> uvlr_t:
        ...
    def reduce_size(self, new_size: int) -> bool:
        ...
    def set_all(self) -> None:
        ...
    def set_cmp(self, cmp: cmpop_t, _value: uvlr_t) -> None:
        ...
    def set_eq(self, v: uvlr_t) -> None:
        ...
    def set_none(self) -> None:
        ...
    def set_unk(self) -> None:
        ...
    def swap(self, r: valrng_t) -> None:
        ...
    def unite_with(self, r: valrng_t) -> bool:
        ...

class var_ref_t:
    @property
    def idx(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: var_ref_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: var_ref_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: var_ref_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: var_ref_t) -> bool:
        ...
    def __lt__(self, r: var_ref_t) -> bool:
        ...
    def __ne__(self, r: var_ref_t) -> bool:
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
    def compare(self, r: var_ref_t) -> int:
        ...
    def getv(self) -> lvar_t:
        ...

class vc_printer_t(vd_printer_t):
    @property
    def func(self) -> Any: ...
    @property
    def hdrlines(self) -> Any: ...
    @property
    def lastchar(self) -> Any: ...
    @property
    def tmpbuf(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, f: cfunc_t) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def oneliner(self) -> bool:
        r"""Are we generating one-line text representation? 
                
        :returns: `true` if the output will occupy one line without line breaks
        """
        ...

class vd_failure_t:
    @property
    def hf(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def desc(self) -> str:
        ...

class vd_interr_t(vd_failure_t):
    @property
    def hf(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, ea: ida_idaapi.ea_t, buf: str) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def desc(self) -> str:
        ...

class vd_printer_t:
    @property
    def hdrlines(self) -> Any: ...
    @property
    def tmpbuf(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
        ...
    def __eq__(self, value: Any) -> bool:
        r"""Return self==value."""
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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

class vdloc_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: vdloc_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: vdloc_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: vdloc_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: vdloc_t) -> bool:
        ...
    def __lt__(self, r: vdloc_t) -> bool:
        ...
    def __ne__(self, r: vdloc_t) -> bool:
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
    def advance(self, delta: int) -> bool:
        r"""Move the location to point 'delta' bytes further.
        
        """
        ...
    def align_reg_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set register offset to align it to the upper part of _SLOTSIZE.
        
        """
        ...
    def align_stkoff_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set stack offset to align to the upper part of _SLOTSIZE.
        
        """
        ...
    def atype(self) -> argloc_type_t:
        r"""Get type (Argument location types)
        
        """
        ...
    def calc_offset(self) -> int:
        r"""Calculate offset that can be used to compare 2 similar arglocs.
        
        """
        ...
    def compare(self, r: vdloc_t) -> int:
        ...
    def consume_rrel(self, p: rrel_t) -> None:
        r"""Set register-relative location - can't be nullptr.
        
        """
        ...
    def consume_scattered(self, p: scattered_aloc_t) -> None:
        r"""Set distributed argument location.
        
        """
        ...
    def get_biggest(self) -> biggest_t:
        r"""Get largest element in internal union.
        
        """
        ...
    def get_custom(self) -> None:
        r"""Get custom argloc info. Use if atype() == ALOC_CUSTOM 
                
        """
        ...
    def get_ea(self) -> ida_idaapi.ea_t:
        r"""Get the global address. Use when atype() == ALOC_STATIC 
                
        """
        ...
    def get_reginfo(self) -> int:
        r"""Get all register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def get_rrel(self) -> rrel_t:
        r"""Get register-relative info. Use when atype() == ALOC_RREL 
                
        """
        ...
    def has_reg(self) -> bool:
        r"""TRUE if argloc has a register part.
        
        """
        ...
    def has_stkoff(self) -> bool:
        r"""TRUE if argloc has a stack part.
        
        """
        ...
    def in_stack(self) -> bool:
        r"""TRUE if argloc is in stack entirely.
        
        """
        ...
    def is_aliasable(self, mb: mba_t, size: int) -> bool:
        ...
    def is_badloc(self) -> bool:
        r"""See ALOC_NONE.
        
        """
        ...
    def is_custom(self) -> bool:
        r"""See ALOC_CUSTOM.
        
        """
        ...
    def is_ea(self) -> bool:
        r"""See ALOC_STATIC.
        
        """
        ...
    def is_fragmented(self) -> bool:
        r"""is_scattered() || is_reg2()
        
        """
        ...
    def is_mixed_scattered(self) -> bool:
        r"""mixed scattered: consists of register and stack parts
        
        """
        ...
    def is_reg(self) -> bool:
        r"""is_reg1() || is_reg2()
        
        """
        ...
    def is_reg1(self) -> bool:
        r"""See ALOC_REG1.
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""See ALOC_REG2.
        
        """
        ...
    def is_rrel(self) -> bool:
        r"""See ALOC_RREL.
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""See ALOC_DIST.
        
        """
        ...
    def is_stkoff(self) -> bool:
        r"""See ALOC_STACK.
        
        """
        ...
    def reg1(self) -> int:
        ...
    def reg2(self) -> int:
        r"""Get info for the second register. Use when atype() == ALOC_REG2 
                
        """
        ...
    def regoff(self) -> int:
        r"""Get offset from the beginning of the register in bytes. Use when atype() == ALOC_REG1 
                
        """
        ...
    def scattered(self) -> scattered_aloc_t:
        r"""Get scattered argument info. Use when atype() == ALOC_DIST 
                
        """
        ...
    def set_badloc(self) -> None:
        r"""Set to invalid location.
        
        """
        ...
    def set_ea(self, _ea: ida_idaapi.ea_t) -> None:
        r"""Set static ea location.
        
        """
        ...
    def set_reg1(self, r1: int) -> None:
        ...
    def set_reg2(self, _reg1: int, _reg2: int) -> None:
        r"""Set secondary register location.
        
        """
        ...
    def set_stkoff(self, off: int) -> None:
        r"""Set stack offset location.
        
        """
        ...
    def stkoff(self) -> int:
        r"""Get the stack offset. Use if atype() == ALOC_STACK 
                
        """
        ...
    def swap(self, r: argloc_t) -> None:
        r"""Assign this == r and r == this.
        
        """
        ...

class vdui_t:
    @property
    def cfunc(self) -> Any: ...
    @property
    def cpos(self) -> Any: ...
    @property
    def ct(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def head(self) -> Any: ...
    @property
    def item(self) -> Any: ...
    @property
    def last_code(self) -> Any: ...
    @property
    def mba(self) -> Any: ...
    @property
    def tail(self) -> Any: ...
    @property
    def toplevel(self) -> Any: ...
    @property
    def view_idx(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, args: Any, kwargs: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
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
    def calc_cmt_type(self, lnnum: size_t, cmttype: cmt_type_t) -> cmt_type_t:
        r"""Check if the specified line can have a comment. Due to the coordinate system for comments: ([https://hex-rays.com/blog/coordinate-system-for-hex-rays](https://hex-rays.com/blog/coordinate-system-for-hex-rays)) some function lines cannot have comments. This function checks if a comment can be attached to the specified line. 
                
        :param lnnum: line number (0 based)
        :param cmttype: comment types to check
        :returns: possible comment types
        """
        ...
    def clear(self) -> None:
        r"""Clear the pseudocode window. It deletes the current function and microcode. 
                
        """
        ...
    def collapse_item(self, hide: bool) -> bool:
        r"""Collapse/uncollapse item. This function collapses the current item. 
                
        :returns: false if failed.
        """
        ...
    def collapse_lvars(self, hide: bool) -> bool:
        r"""Collapse/uncollapse local variable declarations. 
                
        :returns: false if failed.
        """
        ...
    def ctree_to_disasm(self) -> bool:
        r"""Jump to disassembly. This function jumps to the address in the disassembly window which corresponds to the current item. The current item is determined based on the current keyboard cursor position. 
                
        :returns: false if failed
        """
        ...
    def del_orphan_cmts(self) -> bool:
        r"""Delete all orphan comments. Delete all orphan comments and refresh the screen. 
                
        :returns: true
        """
        ...
    def edit_cmt(self, loc: treeloc_t) -> bool:
        r"""Edit an indented comment. This function displays a dialog box and allows the user to edit the comment for the specified ctree location. 
                
        :param loc: comment location
        :returns: false if failed or cancelled
        """
        ...
    def edit_func_cmt(self) -> bool:
        r"""Edit a function comment. This function displays a dialog box and allows the user to edit the function comment. 
                
        :returns: false if failed or cancelled
        """
        ...
    def get_current_item(self, idv: input_device_t) -> bool:
        r"""Get current item. This function refreshes the cpos, item, tail fields. 
                
        :param idv: keyboard or mouse
        :returns: false if failed
        """
        ...
    def get_current_label(self) -> int:
        r"""Get current label. If there is a label under the cursor, return its number. 
                
        :returns: -1 if there is no label under the cursor. prereq: get_current_item() has been called
        """
        ...
    def get_number(self) -> cnumber_t:
        r"""Get current number. If the current item is a number, return pointer to it. 
                
        :returns: nullptr if the current item is not a number This function returns non-null for the cases of a 'switch' statement Also, if the current item is a casted number, then this function will succeed.
        """
        ...
    def in_ctree(self) -> bool:
        r"""Is the current item a statement?
        
        :returns: false if the cursor is in the local variable/type declaration area
         true if the cursor is in the statement area
        """
        ...
    def invert_bits(self) -> bool:
        r"""Bitwise negate a number. This function inverts all bits of the current number. 
                
        :returns: false if failed.
        """
        ...
    def invert_sign(self) -> bool:
        r"""Negate a number. This function negates the current number. 
                
        :returns: false if failed.
        """
        ...
    def jump_enter(self, idv: input_device_t, omflags: int) -> bool:
        r"""Process the Enter key. This function jumps to the definition of the item under the cursor. If the current item is a function, it will be decompiled. If the current item is a global data, its disassemly text will be displayed. 
                
        :param idv: what cursor must be used, the keyboard or the mouse
        :param omflags: OM_NEWWIN: new pseudocode window will open, 0: reuse the existing window
        :returns: false if failed
        """
        ...
    def locked(self) -> bool:
        r"""Does the pseudocode window contain valid code? We lock windows before modifying them, to avoid recursion due to the events generated by the IDA kernel. 
                
        :returns: true: The window is locked and may have stale info
        """
        ...
    def map_lvar(self, frm: lvar_t, to: lvar_t) -> bool:
        r"""Map a local variable to another. This function permanently maps one lvar to another. All occurrences of the mapped variable are replaced by the new variable 
                
        :param to: the variable to map to. if nullptr, unmaps the variable
        :returns: false if failed
        """
        ...
    def refresh_cpos(self, idv: input_device_t) -> bool:
        r"""Refresh the current position. This function refreshes the cpos field. 
                
        :param idv: keyboard or mouse
        :returns: false if failed
        """
        ...
    def refresh_ctext(self, activate: bool = True) -> None:
        r"""Refresh pseudocode window. This function refreshes the pseudocode window by regenerating its text from cfunc_t. Instead of this function use refresh_func_ctext(), which refreshes all pseudocode windows for the function. 
                
        """
        ...
    def refresh_view(self, redo_mba: bool) -> None:
        r"""Refresh pseudocode window. This is the highest level refresh function. It causes the most profound refresh possible and can lead to redecompilation of the current function. Please consider using refresh_ctext() if you need a more superficial refresh. 
                
        :param redo_mba: true means to redecompile the current function
         false means to rebuild ctree without regenerating microcode
        """
        ...
    def rename_global(self, ea: ida_idaapi.ea_t) -> bool:
        r"""Rename global item. This function displays a dialog box and allows the user to rename a global item (data or function). 
                
        :param ea: address of the global item
        :returns: false if failed or cancelled
        """
        ...
    def rename_label(self, label: int) -> bool:
        r"""Rename a label. This function displays a dialog box and allows the user to rename a statement label. 
                
        :param label: label number
        :returns: false if failed or cancelled
        """
        ...
    def rename_lvar(self, v: lvar_t, name: str, is_user_name: bool) -> bool:
        r"""Rename local variable. This function permanently renames a local variable. 
                
        :param v: pointer to local variable
        :param name: new variable name
        :param is_user_name: use true to save the new name into the database. use false to delete the saved name.
        :returns: false if failed
        """
        ...
    def rename_udm(self, udt_type: tinfo_t, udm_idx: int) -> bool:
        r"""Rename structure field. This function displays a dialog box and allows the user to rename a structure field. 
                
        :param udt_type: structure/union type
        :param udm_idx: index of the structure/union member
        :returns: false if failed or cancelled
        """
        ...
    def set_global_type(self, ea: ida_idaapi.ea_t) -> bool:
        r"""Set global item type. This function displays a dialog box and allows the user to change the type of a global item (data or function). 
                
        :param ea: address of the global item
        :returns: false if failed or cancelled
        """
        ...
    def set_locked(self, v: bool) -> bool:
        ...
    def set_lvar_cmt(self, v: lvar_t, cmt: str) -> bool:
        r"""Set local variable comment. This function permanently sets a variable comment. 
                
        :param v: pointer to local variable
        :param cmt: new comment
        :returns: false if failed
        """
        ...
    def set_lvar_type(self, v: lvar_t, type: tinfo_t) -> bool:
        r"""Set local variable type. This function permanently sets a local variable type and clears NOPTR flag if it was set before by function 'set_noptr_lvar' 
                
        :param v: pointer to local variable
        :param type: new variable type
        :returns: false if failed
        """
        ...
    def set_noptr_lvar(self, v: lvar_t) -> bool:
        r"""Inform that local variable should have a non-pointer type This function permanently sets a corresponding variable flag (NOPTR) and removes type if it was set before by function 'set_lvar_type' 
                
        :param v: pointer to local variable
        :returns: false if failed
        """
        ...
    def set_num_enum(self) -> bool:
        r"""Convert number to symbolic constant. This function displays a dialog box and allows the user to select a symbolic constant to represent the number. 
                
        :returns: false if failed or cancelled
        """
        ...
    def set_num_radix(self, base: int) -> bool:
        r"""Change number base. This function changes the current number representation. 
                
        :param base: number radix (10 or 16)
         0 means a character constant
        :returns: false if failed
        """
        ...
    def set_num_stroff(self) -> bool:
        r"""Convert number to structure field offset. Currently not implemented. 
                
        :returns: false if failed or cancelled
        """
        ...
    def set_udm_type(self, udt_type: tinfo_t, udm_idx: int) -> bool:
        r"""Set structure field type. This function displays a dialog box and allows the user to change the type of a structure field. 
                
        :param udt_type: structure/union type
        :param udm_idx: index of the structure/union member
        :returns: false if failed or cancelled
        """
        ...
    def set_valid(self, v: bool) -> None:
        ...
    def set_visible(self, v: bool) -> None:
        ...
    def split_item(self, split: bool) -> bool:
        r"""Split/unsplit item. This function splits the current assignment expression. 
                
        :returns: false if failed.
        """
        ...
    def switch_to(self, f: cfuncptr_t, activate: bool) -> None:
        r"""Display the specified pseudocode. This function replaces the pseudocode window contents with the specified cfunc_t. 
                
        :param f: pointer to the function to display.
        :param activate: should the pseudocode window get focus?
        """
        ...
    def ui_edit_lvar_cmt(self, v: lvar_t) -> bool:
        r"""Set local variable comment. This function displays a dialog box and allows the user to edit the comment of a local variable. 
                
        :param v: pointer to local variable
        :returns: false if failed or cancelled
        """
        ...
    def ui_map_lvar(self, v: lvar_t) -> bool:
        r"""Map a local variable to another. This function displays a variable list and allows the user to select mapping. 
                
        :param v: pointer to local variable
        :returns: false if failed or cancelled
        """
        ...
    def ui_rename_lvar(self, v: lvar_t) -> bool:
        r"""Rename local variable. This function displays a dialog box and allows the user to rename a local variable. 
                
        :param v: pointer to local variable
        :returns: false if failed or cancelled
        """
        ...
    def ui_set_call_type(self, e: cexpr_t) -> bool:
        r"""Set type of a function call This function displays a dialog box and allows the user to change the type of a function call 
                
        :param e: pointer to call expression
        :returns: false if failed or cancelled
        """
        ...
    def ui_set_lvar_type(self, v: lvar_t) -> bool:
        r"""Set local variable type. This function displays a dialog box and allows the user to change the type of a local variable. 
                
        :param v: pointer to local variable
        :returns: false if failed or cancelled
        """
        ...
    def ui_unmap_lvar(self, v: lvar_t) -> bool:
        r"""Unmap a local variable. This function displays list of variables mapped to the specified variable and allows the user to select a variable to unmap. 
                
        :param v: pointer to local variable
        :returns: false if failed or cancelled
        """
        ...
    def valid(self) -> bool:
        r"""Does the pseudocode window contain valid code? It can become invalid if the function type gets changed in IDA. 
                
        """
        ...
    def visible(self) -> bool:
        r"""Is the pseudocode window visible? if not, it might be invisible or destroyed 
                
        """
        ...

class vivl_t(voff_t):
    @property
    def off(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, args: Any) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: vivl_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: vivl_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: vivl_t) -> bool:
        ...
    def __lt__(self, r: vivl_t) -> bool:
        ...
    def __ne__(self, r: vivl_t) -> bool:
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
    def add(self, width: int) -> voff_t:
        ...
    def compare(self, r: vivl_t) -> int:
        ...
    def contains(self, voff2: voff_t) -> bool:
        r"""Does our value interval contain the specified value offset?
        
        """
        ...
    def defined(self) -> bool:
        ...
    def diff(self, r: voff_t) -> int:
        ...
    def dstr(self) -> str:
        ...
    def extend_to_cover(self, r: vivl_t) -> bool:
        r"""Extend a value interval using another value interval of the same type 
                
        :returns: success
        """
        ...
    def get_reg(self) -> mreg_t:
        ...
    def get_stkoff(self) -> int:
        ...
    def inc(self, delta: int) -> None:
        ...
    def includes(self, r: vivl_t) -> bool:
        r"""Does our value interval include another?
        
        """
        ...
    def intersect(self, r: vivl_t) -> int:
        r"""Intersect value intervals the same type 
                
        :returns: size of the resulting intersection
        """
        ...
    def is_reg(self) -> bool:
        ...
    def is_stkoff(self) -> bool:
        ...
    def overlap(self, r: vivl_t) -> bool:
        r"""Do two value intervals overlap?
        
        """
        ...
    def set(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. set(_type: mopt_t, _off: int, _size: int=0) -> None
            1. set(voff: const voff_t &, _size: int) -> None
        
        # 0: set(_type: mopt_t, _off: int, _size: int=0) -> None
        
        
        # 1: set(voff: const voff_t &, _size: int) -> None
        
        
        """
        ...
    def set_reg(self, mreg: mreg_t, sz: int = 0) -> None:
        ...
    def set_stkoff(self, stkoff: int, sz: int = 0) -> None:
        ...
    def undef(self) -> None:
        ...

class voff_t:
    @property
    def off(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: voff_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: voff_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: voff_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: voff_t) -> bool:
        ...
    def __lt__(self, r: voff_t) -> bool:
        ...
    def __ne__(self, r: voff_t) -> bool:
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
    def add(self, width: int) -> voff_t:
        ...
    def compare(self, r: voff_t) -> int:
        ...
    def defined(self) -> bool:
        ...
    def diff(self, r: voff_t) -> int:
        ...
    def get_reg(self) -> mreg_t:
        ...
    def get_stkoff(self) -> int:
        ...
    def inc(self, delta: int) -> None:
        ...
    def is_reg(self) -> bool:
        ...
    def is_stkoff(self) -> bool:
        ...
    def set(self, _type: mopt_t, _off: int) -> bool:
        ...
    def set_reg(self, mreg: mreg_t) -> None:
        ...
    def set_stkoff(self, stkoff: int) -> None:
        ...
    def undef(self) -> None:
        ...

def accepts_small_udts(op: ctype_t) -> bool:
    r"""Is the operator allowed on small structure or union?
    
    """
    ...

def accepts_udts(op: ctype_t) -> bool:
    ...

def arglocs_overlap(loc1: vdloc_t, w1: size_t, loc2: vdloc_t, w2: size_t) -> bool:
    r"""Do two arglocs overlap?
    
    """
    ...

def asgop(cop: ctype_t) -> ctype_t:
    r"""Convert plain operator into assignment operator. For example, cot_add returns cot_asgadd.
    
    """
    ...

def asgop_revert(cop: ctype_t) -> ctype_t:
    r"""Convert assignment operator into plain operator. For example, cot_asgadd returns cot_add 
            
    :returns: cot_empty is the input operator is not an assignment operator.
    """
    ...

def block_chains_begin(set: block_chains_t) -> block_chains_iterator_t:
    r"""Get iterator pointing to the beginning of block_chains_t.
    
    """
    ...

def block_chains_clear(set: block_chains_t) -> None:
    r"""Clear block_chains_t.
    
    """
    ...

def block_chains_end(set: block_chains_t) -> block_chains_iterator_t:
    r"""Get iterator pointing to the end of block_chains_t.
    
    """
    ...

def block_chains_erase(set: block_chains_t, p: block_chains_iterator_t) -> None:
    r"""Erase current element from block_chains_t.
    
    """
    ...

def block_chains_find(set: block_chains_t, val: chain_t) -> block_chains_iterator_t:
    r"""Find the specified key in set block_chains_t.
    
    """
    ...

def block_chains_free(set: block_chains_t) -> None:
    r"""Delete block_chains_t instance.
    
    """
    ...

def block_chains_get(p: block_chains_iterator_t) -> chain_t:
    r"""Get reference to the current set value.
    
    """
    ...

def block_chains_insert(set: block_chains_t, val: chain_t) -> block_chains_iterator_t:
    r"""Insert new (chain_t) into set block_chains_t.
    
    """
    ...

def block_chains_new() -> block_chains_t:
    r"""Create a new block_chains_t instance.
    
    """
    ...

def block_chains_next(p: block_chains_iterator_t) -> block_chains_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def block_chains_prev(p: block_chains_iterator_t) -> block_chains_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def block_chains_size(set: block_chains_t) -> int:
    r"""Get size of block_chains_t.
    
    """
    ...

def boundaries_begin(map: boundaries_t) -> boundaries_iterator_t:
    r"""Get iterator pointing to the beginning of boundaries_t.
    
    """
    ...

def boundaries_clear(map: boundaries_t) -> None:
    r"""Clear boundaries_t.
    
    """
    ...

def boundaries_end(map: boundaries_t) -> boundaries_iterator_t:
    r"""Get iterator pointing to the end of boundaries_t.
    
    """
    ...

def boundaries_erase(map: boundaries_t, p: boundaries_iterator_t) -> None:
    r"""Erase current element from boundaries_t.
    
    """
    ...

def boundaries_find(map: boundaries_t, key: cinsn_t) -> boundaries_iterator_t:
    r"""Find the specified key in boundaries_t.
    
    """
    ...

def boundaries_first(p: boundaries_iterator_t) -> cinsn_t:
    r"""Get reference to the current map key.
    
    """
    ...

def boundaries_free(map: boundaries_t) -> None:
    r"""Delete boundaries_t instance.
    
    """
    ...

def boundaries_insert(map: boundaries_t, key: cinsn_t, val: rangeset_t) -> boundaries_iterator_t:
    r"""Insert new (cinsn_t *, rangeset_t) pair into boundaries_t.
    
    """
    ...

def boundaries_new() -> boundaries_t:
    r"""Create a new boundaries_t instance.
    
    """
    ...

def boundaries_next(p: boundaries_iterator_t) -> boundaries_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def boundaries_prev(p: boundaries_iterator_t) -> boundaries_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def boundaries_second(p: boundaries_iterator_t) -> rangeset_t:
    r"""Get reference to the current map value.
    
    """
    ...

def boundaries_size(map: boundaries_t) -> int:
    r"""Get size of boundaries_t.
    
    """
    ...

def call_helper(rettype: Any, args: Any, rest: Any) -> Any:
    r"""Create a helper call.
    
    """
    ...

def cexpr_operands(self: Any) -> Any:
    r""" return a dictionary with the operands of a cexpr_t. """
    ...

def cfunc_type(self: Any) -> Any:
    r""" Get the function's return type tinfo_t object. """
    ...

def change_hexrays_config(directive: str) -> bool:
    r"""Parse DIRECTIVE and update the current configuration variables. For the syntax see hexrays.cfg 
            
    """
    ...

def cinsn_details(self: Any) -> Any:
    r"""
        return the details pointer for the cinsn_t object depending on the value of its op member.     this is one of the cblock_t, cif_t, etc. objects.
        
    """
    ...

def citem_to_specific_type(self: Any) -> Any:
    r""" cast the citem_t object to its more specific type, either cexpr_t or cinsn_t. """
    ...

def clear_cached_cfuncs() -> None:
    r"""Flush all cached decompilation results.
    
    """
    ...

def close_hexrays_waitbox() -> None:
    r"""Close the waitbox displayed by the decompiler. Useful if DECOMP_NO_HIDE was used during decompilation. 
            
    """
    ...

def close_pseudocode(f: TWidget) -> bool:
    r"""Close pseudocode window. 
            
    :param f: pointer to window
    :returns: false if failed
    """
    ...

def convert_to_user_call(udc: udcall_t, cdg: codegen_t) -> merror_t:
    r"""try to generate user-defined call for an instruction 
            
    :returns: Microcode error code code: MERR_OK - user-defined call generated else - error (MERR_INSN == inacceptable udc.tif)
    """
    ...

def create_cfunc(mba: mba_t) -> cfuncptr_t:
    r"""Create a new cfunc_t object. 
            
    :param mba: microcode object. After creating the cfunc object it takes the ownership of MBA.
    """
    ...

def create_empty_mba(mbr: mba_ranges_t, hf: hexrays_failure_t = None) -> mba_t:
    r"""Create an empty microcode object.
    
    """
    ...

def create_field_name(args: Any) -> str:
    ...

def create_helper(args: Any) -> Any:
    r"""Create a helper object..
    
    """
    ...

def create_typedef(args: Any) -> tinfo_t:
    r"""This function has the following signatures:
    
        0. create_typedef(name: str) -> tinfo_t
        1. create_typedef(n: int) -> tinfo_t
    
    # 0: create_typedef(name: str) -> tinfo_t
    
    Create a reference to a named type. 
            
    :returns: type which refers to the specified name. For example, if name is "DWORD", the type info which refers to "DWORD" is created.
    
    # 1: create_typedef(n: int) -> tinfo_t
    
    Create a reference to an ordinal type. 
            
    :returns: type which refers to the specified ordinal. For example, if n is 1, the type info which refers to ordinal type 1 is created.
    
    """
    ...

def debug_hexrays_ctree(level: int, msg: str) -> None:
    ...

def decompile(ea: Any, hf: Any = None, flags: Any = 0) -> cfuncptr_t:
    r"""Decompile a snippet or a function. 
            
    :param hf: extended error information (if failed)
    :returns: pointer to the decompilation result (a reference counted pointer). nullptr if failed.
    """
    ...

def decompile_func(pfn: func_t, hf: hexrays_failure_t = None, decomp_flags: int = 0) -> cfuncptr_t:
    r"""Decompile a function. Multiple decompilations of the same function return the same object. 
            
    :param pfn: pointer to function to decompile
    :param hf: extended error information (if failed)
    :param decomp_flags: bitwise combination of decompile() flags... bits
    :returns: pointer to the decompilation result (a reference counted pointer). nullptr if failed.
    """
    ...

def decompile_many(outfile: str, funcaddrs: uint64vec_t, flags: int) -> bool:
    r"""Batch decompilation. Decompile all or the specified functions 
            
    :param outfile: name of the output file
    :param funcaddrs: list of functions to decompile. If nullptr or empty, then decompile all nonlib functions
    :param flags: Batch decompilation bits
    :returns: true if no internal error occurred and the user has not cancelled decompilation
    """
    ...

def dereference(e: Any, ptrsize: Any, is_float: Any = False) -> Any:
    r"""Dereference a pointer. This function dereferences a pointer expression. It performs the following conversion: "ptr" => "*ptr" It can handle discrepancies in the pointer type and the access size. 
            
    :param e: expression to deference
    :param ptrsize: access size
    :returns: dereferenced expression
    """
    ...

def dstr(tif: tinfo_t) -> str:
    r"""Print the specified type info. This function can be used from a debugger by typing "tif->dstr()" 
            
    """
    ...

def dummy_ptrtype(ptrsize: int, isfp: bool) -> tinfo_t:
    r"""Generate a dummy pointer type 
            
    :param ptrsize: size of pointed object
    :param isfp: is floating point object?
    """
    ...

def eamap_begin(map: eamap_t) -> eamap_iterator_t:
    r"""Get iterator pointing to the beginning of eamap_t.
    
    """
    ...

def eamap_clear(map: eamap_t) -> None:
    r"""Clear eamap_t.
    
    """
    ...

def eamap_end(map: eamap_t) -> eamap_iterator_t:
    r"""Get iterator pointing to the end of eamap_t.
    
    """
    ...

def eamap_erase(map: eamap_t, p: eamap_iterator_t) -> None:
    r"""Erase current element from eamap_t.
    
    """
    ...

def eamap_find(map: eamap_t, key: ea_t) -> eamap_iterator_t:
    r"""Find the specified key in eamap_t.
    
    """
    ...

def eamap_first(p: eamap_iterator_t) -> int:
    r"""Get reference to the current map key.
    
    """
    ...

def eamap_free(map: eamap_t) -> None:
    r"""Delete eamap_t instance.
    
    """
    ...

def eamap_insert(map: eamap_t, key: ea_t, val: cinsnptrvec_t) -> eamap_iterator_t:
    r"""Insert new (ea_t, cinsnptrvec_t) pair into eamap_t.
    
    """
    ...

def eamap_new() -> eamap_t:
    r"""Create a new eamap_t instance.
    
    """
    ...

def eamap_next(p: eamap_iterator_t) -> eamap_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def eamap_prev(p: eamap_iterator_t) -> eamap_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def eamap_second(p: eamap_iterator_t) -> cinsnptrvec_t:
    r"""Get reference to the current map value.
    
    """
    ...

def eamap_size(map: eamap_t) -> int:
    r"""Get size of eamap_t.
    
    """
    ...

def gen_microcode(mbr: mba_ranges_t, hf: hexrays_failure_t = None, retlist: mlist_t = None, decomp_flags: int = 0, reqmat: mba_maturity_t = 7) -> mba_t:
    r"""Generate microcode of an arbitrary code snippet 
            
    :param mbr: snippet ranges
    :param hf: extended error information (if failed)
    :param retlist: list of registers the snippet returns
    :param decomp_flags: bitwise combination of decompile() flags... bits
    :param reqmat: required microcode maturity
    :returns: pointer to the microcode, nullptr if failed.
    """
    ...

def get_ctype_name(op: ctype_t) -> str:
    ...

def get_current_operand(out: gco_info_t) -> bool:
    r"""Get the instruction operand under the cursor. This function determines the operand that is under the cursor in the active disassembly listing. If the operand refers to a register or stack variable, it returns true. 
            
    """
    ...

def get_float_type(width: int) -> tinfo_t:
    r"""Get a type of a floating point value with the specified width 
            
    :param width: width of the desired type
    :returns: type info object
    """
    ...

def get_hexrays_version() -> str:
    r"""Get decompiler version. The returned string is of the form <major>.<minor>.<revision>.<build-date> 
            
    :returns: pointer to version string. For example: "2.0.0.140605"
    """
    ...

def get_int_type_by_width_and_sign(srcwidth: int, sign: type_sign_t) -> tinfo_t:
    r"""Create a type info by width and sign. Returns a simple type (examples: int, short) with the given width and sign. 
            
    :param srcwidth: size of the type in bytes
    :param sign: sign of the type
    """
    ...

def get_merror_desc(code: merror_t, mba: mba_t) -> str:
    r"""Get textual description of an error code 
            
    :param code: Microcode error code
    :param mba: the microcode array
    :returns: the error address
    """
    ...

def get_mreg_name(reg: mreg_t, width: int, ud: void = None) -> str:
    r"""Get the microregister name. 
            
    :param reg: microregister number
    :param width: size of microregister in bytes. may be bigger than the real register size.
    :param ud: reserved, must be nullptr
    :returns: width of the printed register. this value may be less than the WIDTH argument.
    """
    ...

def get_op_signness(op: ctype_t) -> type_sign_t:
    r"""Get operator sign. Meaningful for sign-dependent operators, like cot_sdiv.
    
    """
    ...

def get_signed_mcode(code: mcode_t) -> mcode_t:
    ...

def get_temp_regs() -> mlist_t:
    r"""Get list of temporary registers. Tempregs are temporary registers that are used during code generation. They do not map to regular processor registers. They are used only to store temporary values during execution of one instruction. Tempregs may not be used to pass a value from one block to another. In other words, at the end of a block all tempregs must be dead. 
            
    """
    ...

def get_type(id: int, tif: tinfo_t, guess: type_source_t) -> tinfo_t:
    r"""Get a global type. Global types are types of addressable objects and struct/union/enum types 
            
    :param id: address or id of the object
    :param tif: buffer for the answer
    :param guess: what kind of types to consider
    :returns: success
    """
    ...

def get_unk_type(size: int) -> tinfo_t:
    r"""Create a partial type info by width. Returns a partially defined type (examples: _DWORD, _BYTE) with the given width. 
            
    :param size: size of the type in bytes
    """
    ...

def get_unsigned_mcode(code: mcode_t) -> mcode_t:
    ...

def get_widget_vdui(f: TWidget) -> vdui_t:
    r"""Get the vdui_t instance associated to the TWidget 
            
    :param f: pointer to window
    :returns: a vdui_t *, or nullptr
    """
    ...

def getb_reginsn(ins: minsn_t) -> minsn_t:
    r"""Skip assertions backward.
    
    """
    ...

def getf_reginsn(ins: minsn_t) -> minsn_t:
    r"""Skip assertions forward.
    
    """
    ...

def has_cached_cfunc(ea: ida_idaapi.ea_t) -> bool:
    r"""Do we have a cached decompilation result for 'ea'?
    
    """
    ...

def has_mcode_seloff(op: mcode_t) -> bool:
    ...

def hexrays_alloc(size: size_t) -> None:
    ...

def hexrays_free(ptr: void) -> None:
    ...

def init_hexrays_plugin(flags: int = 0) -> bool:
    r"""Check that your plugin is compatible with hex-rays decompiler. This function must be called before calling any other decompiler function. 
            
    :param flags: reserved, must be 0
    :returns: true if the decompiler exists and is compatible with your plugin
    """
    ...

def install_hexrays_callback(callback: Any) -> Any:
    r"""Install handler for decompiler events. 
            
    :param callback: handler to install
    :returns: false if failed
    """
    ...

def install_microcode_filter(filter: microcode_filter_t, install: bool = True) -> bool:
    r"""register/unregister non-standard microcode generator 
            
    :param filter: - microcode generator object
    :param install: - TRUE - register the object, FALSE - unregister
    :returns: success
    """
    ...

def is_additive(op: ctype_t) -> bool:
    r"""Is additive operator?
    
    """
    ...

def is_allowed_on_small_struni(op: ctype_t) -> bool:
    r"""Is the operator allowed on small structure or union?
    
    """
    ...

def is_assignment(op: ctype_t) -> bool:
    r"""Is assignment operator?
    
    """
    ...

def is_binary(op: ctype_t) -> bool:
    r"""Is binary operator?
    
    """
    ...

def is_bitop(op: ctype_t) -> bool:
    r"""Is bit related operator?
    
    """
    ...

def is_bool_type(type: tinfo_t) -> bool:
    r"""Is a boolean type? 
            
    :returns: true if the type is a boolean type
    """
    ...

def is_break_consumer(op: ctype_t) -> bool:
    r"""Does a break statement influence the specified statement code?
    
    """
    ...

def is_cmpop_with_eq(cmpop: cmpop_t) -> bool:
    ...

def is_cmpop_without_eq(cmpop: cmpop_t) -> bool:
    ...

def is_commutative(op: ctype_t) -> bool:
    r"""Is commutative operator?
    
    """
    ...

def is_inplace_def(type: tinfo_t) -> bool:
    r"""Is struct/union/enum definition (not declaration)?
    
    """
    ...

def is_kreg(r: mreg_t) -> bool:
    r"""Is a kernel register? Kernel registers are temporary registers that can be used freely. They may be used to store values that cross instruction or basic block boundaries. Kernel registers do not map to regular processor registers. See also mba_t::alloc_kreg() 
            
    """
    ...

def is_logical(op: ctype_t) -> bool:
    r"""Is logical operator?
    
    """
    ...

def is_loop(op: ctype_t) -> bool:
    r"""Is loop statement code?
    
    """
    ...

def is_lvalue(op: ctype_t) -> bool:
    r"""Is Lvalue operator?
    
    """
    ...

def is_may_access(maymust: maymust_t) -> bool:
    ...

def is_mcode_addsub(mcode: mcode_t) -> bool:
    ...

def is_mcode_call(mcode: mcode_t) -> bool:
    ...

def is_mcode_commutative(mcode: mcode_t) -> bool:
    ...

def is_mcode_convertible_to_jmp(mcode: mcode_t) -> bool:
    ...

def is_mcode_convertible_to_set(mcode: mcode_t) -> bool:
    ...

def is_mcode_divmod(op: mcode_t) -> bool:
    ...

def is_mcode_fpu(mcode: mcode_t) -> bool:
    ...

def is_mcode_j1(mcode: mcode_t) -> bool:
    ...

def is_mcode_jcond(mcode: mcode_t) -> bool:
    ...

def is_mcode_propagatable(mcode: mcode_t) -> bool:
    r"""May opcode be propagated? Such opcodes can be used in sub-instructions (nested instructions) There is a handful of non-propagatable opcodes, like jumps, ret, nop, etc All other regular opcodes are propagatable and may appear in a nested instruction. 
            
    """
    ...

def is_mcode_set(mcode: mcode_t) -> bool:
    ...

def is_mcode_set1(mcode: mcode_t) -> bool:
    ...

def is_mcode_shift(mcode: mcode_t) -> bool:
    ...

def is_mcode_xdsu(mcode: mcode_t) -> bool:
    ...

def is_multiplicative(op: ctype_t) -> bool:
    r"""Is multiplicative operator?
    
    """
    ...

def is_nonbool_type(type: tinfo_t) -> bool:
    r"""Is definitely a non-boolean type? 
            
    :returns: true if the type is a non-boolean type (non bool and well defined)
    """
    ...

def is_paf(t: type_t) -> bool:
    r"""Is a pointer, array, or function type?
    
    """
    ...

def is_prepost(op: ctype_t) -> bool:
    r"""Is pre/post increment/decrement operator?
    
    """
    ...

def is_ptr_or_array(t: type_t) -> bool:
    r"""Is a pointer or array type?
    
    """
    ...

def is_relational(op: ctype_t) -> bool:
    r"""Is comparison operator?
    
    """
    ...

def is_signed_cmpop(cmpop: cmpop_t) -> bool:
    ...

def is_signed_mcode(code: mcode_t) -> bool:
    ...

def is_small_struni(tif: tinfo_t) -> bool:
    r"""Is a small structure or union? 
            
    :returns: true if the type is a small UDT (user defined type). Small UDTs fit into a register (or pair or registers) as a rule.
    """
    ...

def is_small_udt(tif: tinfo_t) -> bool:
    r"""Is a small structure or union? 
            
    :returns: true if the type is a small UDT (user defined type). Small UDTs fit into a register (or pair or registers) as a rule.
    """
    ...

def is_type_correct(ptr: type_t) -> bool:
    r"""Verify a type string. 
            
    :returns: true if type string is correct
    """
    ...

def is_unary(op: ctype_t) -> bool:
    r"""Is unary operator?
    
    """
    ...

def is_unsigned_cmpop(cmpop: cmpop_t) -> bool:
    ...

def is_unsigned_mcode(code: mcode_t) -> bool:
    ...

def jcnd2set(code: mcode_t) -> mcode_t:
    ...

def lexcompare(a: mop_t, b: mop_t) -> int:
    ...

def lnot(e: Any) -> Any:
    r"""Logically negate the specified expression. The specified expression will be logically negated. For example, "x == y" is converted into "x != y" by this function. 
            
    :param e: expression to negate. After the call, e must not be used anymore because it can be changed by the function. The function return value must be used to refer to the expression.
    :returns: logically negated expression.
    """
    ...

def locate_lvar(out: lvar_locator_t, func_ea: ida_idaapi.ea_t, varname: str) -> bool:
    r"""Find a variable by name. 
            
    :param out: output buffer for the variable locator
    :param func_ea: function start address
    :param varname: variable name
    :returns: success Since VARNAME is not always enough to find the variable, it may decompile the function.
    """
    ...

def lvar_mapping_begin(map: lvar_mapping_t) -> lvar_mapping_iterator_t:
    r"""Get iterator pointing to the beginning of lvar_mapping_t.
    
    """
    ...

def lvar_mapping_clear(map: lvar_mapping_t) -> None:
    r"""Clear lvar_mapping_t.
    
    """
    ...

def lvar_mapping_end(map: lvar_mapping_t) -> lvar_mapping_iterator_t:
    r"""Get iterator pointing to the end of lvar_mapping_t.
    
    """
    ...

def lvar_mapping_erase(map: lvar_mapping_t, p: lvar_mapping_iterator_t) -> None:
    r"""Erase current element from lvar_mapping_t.
    
    """
    ...

def lvar_mapping_find(map: lvar_mapping_t, key: lvar_locator_t) -> lvar_mapping_iterator_t:
    r"""Find the specified key in lvar_mapping_t.
    
    """
    ...

def lvar_mapping_first(p: lvar_mapping_iterator_t) -> lvar_locator_t:
    r"""Get reference to the current map key.
    
    """
    ...

def lvar_mapping_free(map: lvar_mapping_t) -> None:
    r"""Delete lvar_mapping_t instance.
    
    """
    ...

def lvar_mapping_insert(map: lvar_mapping_t, key: lvar_locator_t, val: lvar_locator_t) -> lvar_mapping_iterator_t:
    r"""Insert new (lvar_locator_t, lvar_locator_t) pair into lvar_mapping_t.
    
    """
    ...

def lvar_mapping_new() -> lvar_mapping_t:
    r"""Create a new lvar_mapping_t instance.
    
    """
    ...

def lvar_mapping_next(p: lvar_mapping_iterator_t) -> lvar_mapping_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def lvar_mapping_prev(p: lvar_mapping_iterator_t) -> lvar_mapping_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def lvar_mapping_second(p: lvar_mapping_iterator_t) -> lvar_locator_t:
    r"""Get reference to the current map value.
    
    """
    ...

def lvar_mapping_size(map: lvar_mapping_t) -> int:
    r"""Get size of lvar_mapping_t.
    
    """
    ...

def make_num(args: Any) -> Any:
    r"""Create a number expression 
            
    :param n: value
    :param func: current function
    :param ea: definition address of the number
    :param opnum: operand number of the number (in the disassembly listing)
    :param sign: number sign
    :param size: size of number in bytes Please note that the type of the resulting expression can be anything because it can be inherited from the disassembly listing or taken from the user specified number representation in the pseudocode view.
    """
    ...

def make_pointer(type: tinfo_t) -> tinfo_t:
    r"""Create a pointer type. This function performs the following conversion: "type" -> "type*" 
            
    :param type: object type.
    :returns: "type*". for example, if 'char' is passed as the argument,
    """
    ...

def make_ref(e: Any) -> Any:
    r"""Create a reference. This function performs the following conversion: "obj" => "&obj". It can handle casts, annihilate "&*", and process other special cases. 
            
    """
    ...

def mark_cfunc_dirty(ea: ida_idaapi.ea_t, close_views: bool = False) -> bool:
    r"""Flush the cached decompilation results. Erases a cache entry for the specified function. 
            
    :param ea: function to erase from the cache
    :param close_views: close pseudocode windows that show the function
    :returns: if a cache entry existed.
    """
    ...

def max_vlr_svalue(size: int) -> uvlr_t:
    ...

def max_vlr_value(size: int) -> uvlr_t:
    ...

def mcode_modifies_d(mcode: mcode_t) -> bool:
    ...

def min_vlr_svalue(size: int) -> uvlr_t:
    ...

def modify_user_lvar_info(func_ea: ida_idaapi.ea_t, mli_flags: uint, info: lvar_saved_info_t) -> bool:
    r"""Modify saved local variable settings of one variable. 
            
    :param func_ea: function start address
    :param mli_flags: bits that specify which attrs defined by INFO are to be set
    :param info: local variable info attrs
    :returns: true if modified, false if invalid MLI_FLAGS passed
    """
    ...

def modify_user_lvars(entry_ea: ida_idaapi.ea_t, mlv: user_lvar_modifier_t) -> bool:
    r"""Modify saved local variable settings. 
            
    :param entry_ea: function start address
    :param mlv: local variable modifier
    :returns: true if modified variables
    """
    ...

def mreg2reg(reg: mreg_t, width: int) -> int:
    r"""Map a microregister to a processor register. 
            
    :param reg: microregister number
    :param width: size of microregister in bytes
    :returns: processor register id or -1
    """
    ...

def must_mcode_close_block(mcode: mcode_t, including_calls: bool) -> bool:
    r"""Must an instruction with the given opcode be the last one in a block? Such opcodes are called closing opcodes. 
            
    :param mcode: instruction opcode
    :param including_calls: should m_call/m_icall be considered as the closing opcodes? If this function returns true, the opcode cannot appear in the middle of a block. Calls are a special case: unknown calls (is_unknown_call) are considered as closing opcodes.
    """
    ...

def negate_mcode_relation(code: mcode_t) -> mcode_t:
    ...

def negated_relation(op: ctype_t) -> ctype_t:
    r"""Negate a comparison operator. For example, cot_sge becomes cot_slt.
    
    """
    ...

def new_block() -> Any:
    r"""Create a new block-statement.
    
    """
    ...

def op_uses_x(op: ctype_t) -> bool:
    r"""Does operator use the 'x' field of cexpr_t?
    
    """
    ...

def op_uses_y(op: ctype_t) -> bool:
    r"""Does operator use the 'y' field of cexpr_t?
    
    """
    ...

def op_uses_z(op: ctype_t) -> bool:
    r"""Does operator use the 'z' field of cexpr_t?
    
    """
    ...

def open_pseudocode(ea: ida_idaapi.ea_t, flags: int) -> vdui_t:
    r"""Open pseudocode window. The specified function is decompiled and the pseudocode window is opened. 
            
    :param ea: function to decompile
    :param flags: a combination of OPF_ flags
    :returns: false if failed
    """
    ...

def parse_user_call(udc: udcall_t, decl: str, silent: bool) -> bool:
    r"""Convert function type declaration into internal structure 
            
    :param udc: - pointer to output structure
    :param decl: - function type declaration
    :param silent: - if TRUE: do not show warning in case of incorrect type
    :returns: success
    """
    ...

def partial_type_num(type: tinfo_t) -> int:
    r"""Calculate number of partial subtypes. 
            
    :returns: number of partial subtypes. The bigger is this number, the uglier is the type.
    """
    ...

def print_vdloc(loc: vdloc_t, nbytes: int) -> str:
    r"""Print vdloc. Since vdloc does not always carry the size info, we pass it as NBYTES.. 
            
    """
    ...

def property_op_to_typename(self: Any) -> Any:
    ...

def qswap(a: cinsn_t, b: cinsn_t) -> None:
    ...

def reg2mreg(reg: int) -> mreg_t:
    r"""Map a processor register to a microregister. 
            
    :param reg: processor register number
    :returns: microregister register id or mr_none
    """
    ...

def remitem(e: citem_t) -> None:
    ...

def remove_hexrays_callback(callback: Any) -> Any:
    r"""Uninstall handler for decompiler events. 
            
    :param callback: handler to uninstall
    :returns: number of uninstalled handlers.
    """
    ...

def rename_lvar(func_ea: ida_idaapi.ea_t, oldname: str, newname: str) -> bool:
    r"""Rename a local variable. 
            
    :param func_ea: function start address
    :param oldname: old name of the variable
    :param newname: new name of the variable
    :returns: success This is a convenience function. For bulk renaming consider using modify_user_lvars.
    """
    ...

def restore_user_cmts(func_ea: ida_idaapi.ea_t) -> user_cmts_t:
    r"""Restore user defined comments from the database. 
            
    :param func_ea: the entry address of the function
    :returns: collection of user defined comments. The returned object must be deleted by the caller using delete_user_cmts()
    """
    ...

def restore_user_defined_calls(udcalls: udcall_map_t, func_ea: ida_idaapi.ea_t) -> bool:
    r"""Restore user defined function calls from the database. 
            
    :param udcalls: ptr to output buffer
    :param func_ea: entry address of the function
    :returns: success
    """
    ...

def restore_user_iflags(func_ea: ida_idaapi.ea_t) -> user_iflags_t:
    r"""Restore user defined citem iflags from the database. 
            
    :param func_ea: the entry address of the function
    :returns: collection of user defined iflags. The returned object must be deleted by the caller using delete_user_iflags()
    """
    ...

def restore_user_labels(func_ea: ida_idaapi.ea_t, func: cfunc_t = None) -> user_labels_t:
    r"""Restore user defined labels from the database. 
            
    :param func_ea: the entry address of the function, ignored if FUNC != nullptr
    :param func: pointer to current function
    :returns: collection of user defined labels. The returned object must be deleted by the caller using delete_user_labels()
    """
    ...

def restore_user_lvar_settings(lvinf: lvar_uservec_t, func_ea: ida_idaapi.ea_t) -> bool:
    r"""Restore user defined local variable settings in the database. 
            
    :param lvinf: ptr to output buffer
    :param func_ea: entry address of the function
    :returns: success
    """
    ...

def restore_user_numforms(func_ea: ida_idaapi.ea_t) -> user_numforms_t:
    r"""Restore user defined number formats from the database. 
            
    :param func_ea: the entry address of the function
    :returns: collection of user defined number formats. The returned object must be deleted by the caller using delete_user_numforms()
    """
    ...

def restore_user_unions(func_ea: ida_idaapi.ea_t) -> user_unions_t:
    r"""Restore user defined union field selections from the database. 
            
    :param func_ea: the entry address of the function
    :returns: collection of union field selections The returned object must be deleted by the caller using delete_user_unions()
    """
    ...

def save_user_cmts(func_ea: ida_idaapi.ea_t, user_cmts: user_cmts_t) -> None:
    r"""Save user defined comments into the database. 
            
    :param func_ea: the entry address of the function
    :param user_cmts: collection of user defined comments
    """
    ...

def save_user_defined_calls(func_ea: ida_idaapi.ea_t, udcalls: udcall_map_t) -> None:
    r"""Save user defined local function calls into the database. 
            
    :param func_ea: entry address of the function
    :param udcalls: user-specified info about user defined function calls
    """
    ...

def save_user_iflags(func_ea: ida_idaapi.ea_t, iflags: user_iflags_t) -> None:
    r"""Save user defined citem iflags into the database. 
            
    :param func_ea: the entry address of the function
    :param iflags: collection of user defined citem iflags
    """
    ...

def save_user_labels(func_ea: ida_idaapi.ea_t, user_labels: user_labels_t, func: cfunc_t = None) -> None:
    r"""Save user defined labels into the database. 
            
    :param func_ea: the entry address of the function, ignored if FUNC != nullptr
    :param user_labels: collection of user defined labels
    :param func: pointer to current function, if FUNC != nullptr, then save labels using a more stable method that preserves them even when the decompiler output drastically changes
    """
    ...

def save_user_lvar_settings(func_ea: ida_idaapi.ea_t, lvinf: lvar_uservec_t) -> None:
    r"""Save user defined local variable settings into the database. 
            
    :param func_ea: entry address of the function
    :param lvinf: user-specified info about local variables
    """
    ...

def save_user_numforms(func_ea: ida_idaapi.ea_t, numforms: user_numforms_t) -> None:
    r"""Save user defined number formats into the database. 
            
    :param func_ea: the entry address of the function
    :param numforms: collection of user defined comments
    """
    ...

def save_user_unions(func_ea: ida_idaapi.ea_t, unions: user_unions_t) -> None:
    r"""Save user defined union field selections into the database. 
            
    :param func_ea: the entry address of the function
    :param unions: collection of union field selections
    """
    ...

def select_udt_by_offset(udts: qvector, ops: ui_stroff_ops_t, applicator: ui_stroff_applicator_t) -> int:
    r"""Select UDT 
            
    :param udts: list of UDT tinfo_t for the selection, if nullptr or empty then UDTs from the "Local types" will be used
    :param ops: operands
    :param applicator: callback will be called to apply the selection for every operand
    """
    ...

def send_database(err: hexrays_failure_t, silent: bool) -> None:
    r"""Send the database to Hex-Rays. This function sends the current database to the Hex-Rays server. The database is sent in the compressed form over an encrypted (SSL) connection. 
            
    :param err: failure description object. Empty hexrays_failure_t object can be used if error information is not available.
    :param silent: if false, a dialog box will be displayed before sending the database.
    """
    ...

def set2jcnd(code: mcode_t) -> mcode_t:
    ...

def set_type(id: int, tif: tinfo_t, source: type_source_t, force: bool = False) -> bool:
    r"""Set a global type. 
            
    :param id: address or id of the object
    :param tif: new type info
    :param source: where the type comes from
    :param force: true means to set the type as is, false means to merge the new type with the possibly existing old type info.
    :returns: success
    """
    ...

def swap_mcode_relation(code: mcode_t) -> mcode_t:
    ...

def swapped_relation(op: ctype_t) -> ctype_t:
    r"""Swap a comparison operator. For example, cot_sge becomes cot_sle.
    
    """
    ...

def term_hexrays_plugin() -> None:
    r"""Stop working with hex-rays decompiler.
    
    """
    ...

def udcall_map_begin(map: udcall_map_t) -> udcall_map_iterator_t:
    r"""Get iterator pointing to the beginning of udcall_map_t.
    
    """
    ...

def udcall_map_clear(map: udcall_map_t) -> None:
    r"""Clear udcall_map_t.
    
    """
    ...

def udcall_map_end(map: udcall_map_t) -> udcall_map_iterator_t:
    r"""Get iterator pointing to the end of udcall_map_t.
    
    """
    ...

def udcall_map_erase(map: udcall_map_t, p: udcall_map_iterator_t) -> None:
    r"""Erase current element from udcall_map_t.
    
    """
    ...

def udcall_map_find(map: udcall_map_t, key: ea_t) -> udcall_map_iterator_t:
    r"""Find the specified key in udcall_map_t.
    
    """
    ...

def udcall_map_first(p: udcall_map_iterator_t) -> int:
    r"""Get reference to the current map key.
    
    """
    ...

def udcall_map_free(map: udcall_map_t) -> None:
    r"""Delete udcall_map_t instance.
    
    """
    ...

def udcall_map_insert(map: udcall_map_t, key: ea_t, val: udcall_t) -> udcall_map_iterator_t:
    r"""Insert new (ea_t, udcall_t) pair into udcall_map_t.
    
    """
    ...

def udcall_map_new() -> udcall_map_t:
    r"""Create a new udcall_map_t instance.
    
    """
    ...

def udcall_map_next(p: udcall_map_iterator_t) -> udcall_map_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def udcall_map_prev(p: udcall_map_iterator_t) -> udcall_map_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def udcall_map_second(p: udcall_map_iterator_t) -> udcall_t:
    r"""Get reference to the current map value.
    
    """
    ...

def udcall_map_size(map: udcall_map_t) -> int:
    r"""Get size of udcall_map_t.
    
    """
    ...

def user_cmts_begin(map: user_cmts_t) -> user_cmts_iterator_t:
    r"""Get iterator pointing to the beginning of user_cmts_t.
    
    """
    ...

def user_cmts_clear(map: user_cmts_t) -> None:
    r"""Clear user_cmts_t.
    
    """
    ...

def user_cmts_end(map: user_cmts_t) -> user_cmts_iterator_t:
    r"""Get iterator pointing to the end of user_cmts_t.
    
    """
    ...

def user_cmts_erase(map: user_cmts_t, p: user_cmts_iterator_t) -> None:
    r"""Erase current element from user_cmts_t.
    
    """
    ...

def user_cmts_find(map: user_cmts_t, key: treeloc_t) -> user_cmts_iterator_t:
    r"""Find the specified key in user_cmts_t.
    
    """
    ...

def user_cmts_first(p: user_cmts_iterator_t) -> treeloc_t:
    r"""Get reference to the current map key.
    
    """
    ...

def user_cmts_free(map: user_cmts_t) -> None:
    r"""Delete user_cmts_t instance.
    
    """
    ...

def user_cmts_insert(map: user_cmts_t, key: treeloc_t, val: citem_cmt_t) -> user_cmts_iterator_t:
    r"""Insert new (treeloc_t, citem_cmt_t) pair into user_cmts_t.
    
    """
    ...

def user_cmts_new() -> user_cmts_t:
    r"""Create a new user_cmts_t instance.
    
    """
    ...

def user_cmts_next(p: user_cmts_iterator_t) -> user_cmts_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def user_cmts_prev(p: user_cmts_iterator_t) -> user_cmts_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def user_cmts_second(p: user_cmts_iterator_t) -> citem_cmt_t:
    r"""Get reference to the current map value.
    
    """
    ...

def user_cmts_size(map: user_cmts_t) -> int:
    r"""Get size of user_cmts_t.
    
    """
    ...

def user_iflags_begin(map: user_iflags_t) -> user_iflags_iterator_t:
    r"""Get iterator pointing to the beginning of user_iflags_t.
    
    """
    ...

def user_iflags_clear(map: user_iflags_t) -> None:
    r"""Clear user_iflags_t.
    
    """
    ...

def user_iflags_end(map: user_iflags_t) -> user_iflags_iterator_t:
    r"""Get iterator pointing to the end of user_iflags_t.
    
    """
    ...

def user_iflags_erase(map: user_iflags_t, p: user_iflags_iterator_t) -> None:
    r"""Erase current element from user_iflags_t.
    
    """
    ...

def user_iflags_find(map: user_iflags_t, key: citem_locator_t) -> user_iflags_iterator_t:
    r"""Find the specified key in user_iflags_t.
    
    """
    ...

def user_iflags_first(p: user_iflags_iterator_t) -> citem_locator_t:
    r"""Get reference to the current map key.
    
    """
    ...

def user_iflags_free(map: user_iflags_t) -> None:
    r"""Delete user_iflags_t instance.
    
    """
    ...

def user_iflags_insert(map: user_iflags_t, key: citem_locator_t, val: int32) -> user_iflags_iterator_t:
    r"""Insert new (citem_locator_t, int32) pair into user_iflags_t.
    
    """
    ...

def user_iflags_new() -> user_iflags_t:
    r"""Create a new user_iflags_t instance.
    
    """
    ...

def user_iflags_next(p: user_iflags_iterator_t) -> user_iflags_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def user_iflags_prev(p: user_iflags_iterator_t) -> user_iflags_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def user_iflags_second(p: user_iflags_iterator_t) -> int32:
    r"""Get reference to the current map value.
    
    """
    ...

def user_iflags_size(map: user_iflags_t) -> int:
    r"""Get size of user_iflags_t.
    
    """
    ...

def user_labels_begin(map: user_labels_t) -> user_labels_iterator_t:
    r"""Get iterator pointing to the beginning of user_labels_t.
    
    """
    ...

def user_labels_clear(map: user_labels_t) -> None:
    r"""Clear user_labels_t.
    
    """
    ...

def user_labels_end(map: user_labels_t) -> user_labels_iterator_t:
    r"""Get iterator pointing to the end of user_labels_t.
    
    """
    ...

def user_labels_erase(map: user_labels_t, p: user_labels_iterator_t) -> None:
    r"""Erase current element from user_labels_t.
    
    """
    ...

def user_labels_find(map: user_labels_t, key: int) -> user_labels_iterator_t:
    r"""Find the specified key in user_labels_t.
    
    """
    ...

def user_labels_first(p: user_labels_iterator_t) -> int:
    r"""Get reference to the current map key.
    
    """
    ...

def user_labels_free(map: user_labels_t) -> None:
    r"""Delete user_labels_t instance.
    
    """
    ...

def user_labels_insert(map: user_labels_t, key: int, val: str) -> user_labels_iterator_t:
    r"""Insert new (int, qstring) pair into user_labels_t.
    
    """
    ...

def user_labels_new() -> user_labels_t:
    r"""Create a new user_labels_t instance.
    
    """
    ...

def user_labels_next(p: user_labels_iterator_t) -> user_labels_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def user_labels_prev(p: user_labels_iterator_t) -> user_labels_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def user_labels_second(p: user_labels_iterator_t) -> str:
    r"""Get reference to the current map value.
    
    """
    ...

def user_labels_size(map: user_labels_t) -> int:
    r"""Get size of user_labels_t.
    
    """
    ...

def user_numforms_begin(map: user_numforms_t) -> user_numforms_iterator_t:
    r"""Get iterator pointing to the beginning of user_numforms_t.
    
    """
    ...

def user_numforms_clear(map: user_numforms_t) -> None:
    r"""Clear user_numforms_t.
    
    """
    ...

def user_numforms_end(map: user_numforms_t) -> user_numforms_iterator_t:
    r"""Get iterator pointing to the end of user_numforms_t.
    
    """
    ...

def user_numforms_erase(map: user_numforms_t, p: user_numforms_iterator_t) -> None:
    r"""Erase current element from user_numforms_t.
    
    """
    ...

def user_numforms_find(map: user_numforms_t, key: operand_locator_t) -> user_numforms_iterator_t:
    r"""Find the specified key in user_numforms_t.
    
    """
    ...

def user_numforms_first(p: user_numforms_iterator_t) -> operand_locator_t:
    r"""Get reference to the current map key.
    
    """
    ...

def user_numforms_free(map: user_numforms_t) -> None:
    r"""Delete user_numforms_t instance.
    
    """
    ...

def user_numforms_insert(map: user_numforms_t, key: operand_locator_t, val: number_format_t) -> user_numforms_iterator_t:
    r"""Insert new (operand_locator_t, number_format_t) pair into user_numforms_t.
    
    """
    ...

def user_numforms_new() -> user_numforms_t:
    r"""Create a new user_numforms_t instance.
    
    """
    ...

def user_numforms_next(p: user_numforms_iterator_t) -> user_numforms_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def user_numforms_prev(p: user_numforms_iterator_t) -> user_numforms_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def user_numforms_second(p: user_numforms_iterator_t) -> number_format_t:
    r"""Get reference to the current map value.
    
    """
    ...

def user_numforms_size(map: user_numforms_t) -> int:
    r"""Get size of user_numforms_t.
    
    """
    ...

def user_unions_begin(map: user_unions_t) -> user_unions_iterator_t:
    r"""Get iterator pointing to the beginning of user_unions_t.
    
    """
    ...

def user_unions_clear(map: user_unions_t) -> None:
    r"""Clear user_unions_t.
    
    """
    ...

def user_unions_end(map: user_unions_t) -> user_unions_iterator_t:
    r"""Get iterator pointing to the end of user_unions_t.
    
    """
    ...

def user_unions_erase(map: user_unions_t, p: user_unions_iterator_t) -> None:
    r"""Erase current element from user_unions_t.
    
    """
    ...

def user_unions_find(map: user_unions_t, key: ea_t) -> user_unions_iterator_t:
    r"""Find the specified key in user_unions_t.
    
    """
    ...

def user_unions_first(p: user_unions_iterator_t) -> int:
    r"""Get reference to the current map key.
    
    """
    ...

def user_unions_free(map: user_unions_t) -> None:
    r"""Delete user_unions_t instance.
    
    """
    ...

def user_unions_insert(map: user_unions_t, key: ea_t, val: intvec_t) -> user_unions_iterator_t:
    r"""Insert new (ea_t, intvec_t) pair into user_unions_t.
    
    """
    ...

def user_unions_new() -> user_unions_t:
    r"""Create a new user_unions_t instance.
    
    """
    ...

def user_unions_next(p: user_unions_iterator_t) -> user_unions_iterator_t:
    r"""Move to the next element.
    
    """
    ...

def user_unions_prev(p: user_unions_iterator_t) -> user_unions_iterator_t:
    r"""Move to the previous element.
    
    """
    ...

def user_unions_second(p: user_unions_iterator_t) -> intvec_t:
    r"""Get reference to the current map value.
    
    """
    ...

def user_unions_size(map: user_unions_t) -> int:
    r"""Get size of user_unions_t.
    
    """
    ...

ACFL_BLKOPT: int  # 2
ACFL_GLBDEL: int  # 8
ACFL_GLBPROP: int  # 4
ACFL_GUESS: int  # 16
ACFL_LOCOPT: int  # 1
ALLOW_UNUSED_LABELS: int  # 1
ANCHOR_BLKCMT: int  # 536870912
ANCHOR_CITEM: int  # 0
ANCHOR_INDEX: int  # 536870911
ANCHOR_ITP: int  # -2147483648
ANCHOR_LVAR: int  # 1073741824
ANCHOR_MASK: int  # -1073741824
ANY_FPSIZE: int  # 256
ANY_REGSIZE: int  # 128
BLT_0WAY: int  # 2
BLT_1WAY: int  # 3
BLT_2WAY: int  # 4
BLT_NONE: int  # 0
BLT_NWAY: int  # 5
BLT_STOP: int  # 1
BLT_XTRN: int  # 6
CALC_CURLY_BRACES: int  # 0
CALL_SPOILS_ONLY_ARGS: int  # 16384
CFL_FINAL: int  # 1
CFL_HELPER: int  # 2
CFL_NORET: int  # 4
CFS_BOUNDS: int  # 1
CFS_LOCKED: int  # 8
CFS_LVARS_HIDDEN: int  # 4
CFS_TEXT: int  # 2
CHF_FAKE: int  # 8
CHF_INITED: int  # 1
CHF_OVER: int  # 4
CHF_PASSTHRU: int  # 16
CHF_REPLACED: int  # 2
CHF_TERM: int  # 32
CIT_COLLAPSED: int  # 1
CMAT_BUILT: int  # 1
CMAT_CASTED: int  # 7
CMAT_CPA: int  # 5
CMAT_FINAL: int  # 8
CMAT_NICE: int  # 3
CMAT_TRANS1: int  # 2
CMAT_TRANS2: int  # 4
CMAT_TRANS3: int  # 6
CMAT_ZERO: int  # 0
CMP_A: int  # 4
CMP_AE: int  # 2
CMP_B: int  # 3
CMP_BE: int  # 5
CMP_GE: int  # 7
CMP_GT: int  # 6
CMP_LE: int  # 9
CMP_LT: int  # 8
CMP_NZ: int  # 0
CMP_Z: int  # 1
CMT_ALL: int  # 31
CMT_BLOCK1: int  # 2
CMT_BLOCK2: int  # 4
CMT_FUNC: int  # 16
CMT_LVAR: int  # 8
CMT_NONE: int  # 0
CMT_TAIL: int  # 1
CPBLK_FAST: int  # 0
CPBLK_MINREF: int  # 1
CPBLK_OPTJMP: int  # 2
CV_FAST: int  # 0
CV_INSNS: int  # 16
CV_PARENTS: int  # 2
CV_POST: int  # 4
CV_PRUNE: int  # 1
CV_RESTART: int  # 8
DECOMP_ALL_BLKS: int  # 16
DECOMP_GXREFS_DEFLT: int  # 0
DECOMP_GXREFS_FORCE: int  # 128
DECOMP_GXREFS_NOUPD: int  # 64
DECOMP_NO_CACHE: int  # 2
DECOMP_NO_FRAME: int  # 4
DECOMP_NO_HIDE: int  # 32
DECOMP_NO_WAIT: int  # 1
DECOMP_OUTLINE: int  # -2147483648
DECOMP_VOID_MBA: int  # 256
DECOMP_WARNINGS: int  # 8
EQ_CMPDEST: int  # 4
EQ_IGNCODE: int  # 2
EQ_IGNSIZE: int  # 1
EQ_OPTINSN: int  # 8
EXCLUDE_PASS_REGS: int  # 128
EXCLUDE_VOLATILE: int  # 1024
EXFL_ALL: int  # 511
EXFL_ALONE: int  # 8
EXFL_CPADONE: int  # 1
EXFL_CSTR: int  # 16
EXFL_FPOP: int  # 4
EXFL_JUMPOUT: int  # 128
EXFL_LVALUE: int  # 2
EXFL_PARTIAL: int  # 32
EXFL_UNDEF: int  # 64
EXFL_VFTABLE: int  # 256
FCI_DEAD: int  # 2
FCI_EXPLOCS: int  # 1024
FCI_FINAL: int  # 4
FCI_HASCALL: int  # 128
FCI_HASFMT: int  # 256
FCI_NORET: int  # 8
FCI_NOSIDE: int  # 32
FCI_PROP: int  # 1
FCI_PURE: int  # 16
FCI_SPLOK: int  # 64
FD_BACKWARD: int  # 0
FD_DEF: int  # 2
FD_DIRTY: int  # 4
FD_FORWARD: int  # 1
FD_USE: int  # 0
FORBID_UNUSED_LABELS: int  # 0
FULL_XDSU: int  # 256
FUNC_NAME_CONTAINING_RECORD: str  # CONTAINING_RECORD
FUNC_NAME_EMPTY: str  # $empty
FUNC_NAME_MEMCPY: str  # memcpy
FUNC_NAME_MEMSET: str  # memset
FUNC_NAME_MEMSET32: str  # memset32
FUNC_NAME_MEMSET64: str  # memset64
FUNC_NAME_MORESTACK: str  # runtime_morestack
FUNC_NAME_PRESENT: str  # $present
FUNC_NAME_STRCAT: str  # strcat
FUNC_NAME_STRCPY: str  # strcpy
FUNC_NAME_STRLEN: str  # strlen
FUNC_NAME_TAIL: str  # tail
FUNC_NAME_VA_ARG: str  # va_arg
FUNC_NAME_WCSCAT: str  # wcscat
FUNC_NAME_WCSCPY: str  # wcscpy
FUNC_NAME_WCSLEN: str  # wcslen
FUNC_NAME_WMEMCPY: str  # wmemcpy
FUNC_NAME_WMEMSET: str  # wmemset
GCA_ALLOC: int  # 4
GCA_EMPTY: int  # 1
GCA_NALLOC: int  # 8
GCA_OFIRST: int  # 16
GCA_OLAST: int  # 32
GCA_SPEC: int  # 2
GCO_DEF: int  # 4
GCO_REG: int  # 1
GCO_STK: int  # 0
GCO_USE: int  # 2
GC_ASR: int  # 1
GC_DIRTY_ALL: int  # 63
GC_END: int  # 3
GC_REGS_AND_STKVARS: int  # 0
GC_XDSU: int  # 2
GLN_ALL: int  # 3
GLN_CURRENT: int  # 1
GLN_GOTO_TARGET: int  # 2
GUESSED_DATA: int  # 3
GUESSED_FUNC: int  # 2
GUESSED_NONE: int  # 0
GUESSED_WEAK: int  # 1
HEXRAYS_API_MAGIC: int  # 62699504545038340
INCLUDE_DEAD_RETREGS: int  # 4096
INCLUDE_RESTRICTED: int  # 8192
INCLUDE_SPOILED_REGS: int  # 64
INCLUDE_UNUSED_SRC: int  # 2048
INLINE_DONTCOPY: int  # 2
INLINE_EXTFRAME: int  # 1
IPROP_ASSERT: int  # 128
IPROP_CLNPOP: int  # 8
IPROP_COMBINED: int  # 2048
IPROP_DONT_COMB: int  # 262144
IPROP_DONT_PROP: int  # 131072
IPROP_EXTSTX: int  # 4096
IPROP_FARCALL: int  # 32
IPROP_FPINSN: int  # 16
IPROP_IGNLOWSRC: int  # 8192
IPROP_INV_JX: int  # 16384
IPROP_MBARRIER: int  # 524288
IPROP_MULTI_MOV: int  # 65536
IPROP_OPTIONAL: int  # 1
IPROP_PERSIST: int  # 2
IPROP_SPLIT: int  # 1792
IPROP_SPLIT1: int  # 256
IPROP_SPLIT2: int  # 512
IPROP_SPLIT4: int  # 768
IPROP_SPLIT8: int  # 1024
IPROP_TAILCALL: int  # 64
IPROP_UNMERGED: int  # 1048576
IPROP_UNPAIRED: int  # 2097152
IPROP_WAS_NORET: int  # 32768
IPROP_WILDMATCH: int  # 4
ITP_ARG1: int  # 1
ITP_ARG64: int  # 64
ITP_ASM: int  # 66
ITP_BLOCK1: int  # 74
ITP_BLOCK2: int  # 75
ITP_BRACE1: int  # 65
ITP_BRACE2: int  # 72
ITP_CASE: int  # 1073741824
ITP_COLON: int  # 73
ITP_CURLY1: int  # 70
ITP_CURLY2: int  # 71
ITP_DO: int  # 68
ITP_ELSE: int  # 67
ITP_EMPTY: int  # 0
ITP_INNER_LAST: int  # 65
ITP_SEMI: int  # 69
ITP_SIGN: int  # 536870912
ITP_TRY: int  # 76
LOCOPT_ALL: int  # 1
LOCOPT_REFINE: int  # 2
LOCOPT_REFINE2: int  # 4
LVINF_KEEP: int  # 1
LVINF_NOMAP: int  # 8
LVINF_NOPTR: int  # 4
LVINF_SPLIT: int  # 2
LVINF_UNUSED: int  # 16
MAX_SUPPORTED_STACK_SIZE: int  # 1048576
MAX_VLR_SIZE: int  # 8
MAX_VLR_SVALUE: int  # 9223372036854775807
MAX_VLR_VALUE: int  # 18446744073709551615
MAYMUST_ACCESS_MASK: int  # 1
MAY_ACCESS: int  # 1
MBA2_ALL_FLAGS: int  # 131071
MBA2_ARGIDX_OK: int  # 64
MBA2_ARGIDX_SORTED: int  # 1024
MBA2_CODE16_BIT: int  # 2048
MBA2_DONT_VERIFY: int  # -2147483648
MBA2_HAS_OUTLINES: int  # 8192
MBA2_INITIAL_FLAGS: int  # 3
MBA2_IS_CTR: int  # 16
MBA2_IS_DTR: int  # 32
MBA2_LVARNAMES_OK: int  # 1
MBA2_LVARS_RENAMED: int  # 2
MBA2_NO_DUP_CALLS: int  # 128
MBA2_NO_DUP_LVARS: int  # 256
MBA2_NO_FRAME: int  # 16384
MBA2_OVER_CHAINS: int  # 4
MBA2_PROP_COMPLEX: int  # 32768
MBA2_STACK_RETVAL: int  # 4096
MBA2_UNDEF_RETVAR: int  # 512
MBA2_VALRNG_DONE: int  # 8
MBA_ASRPROP: int  # 8192
MBA_ASRTOK: int  # 2048
MBA_CALLS: int  # 4096
MBA_CHVARS: int  # 2097152
MBA_CMBBLK: int  # 1024
MBA_CMNSTK: int  # 256
MBA_COLGDL: int  # 8388608
MBA_DELPAIRS: int  # 1048576
MBA_GLBOPT: int  # 65536
MBA_INITIAL_FLAGS: int  # 1459618817
MBA_INSGDL: int  # 16777216
MBA_LOADED: int  # 8
MBA_LVARS0: int  # 262144
MBA_LVARS1: int  # 524288
MBA_NICE: int  # 33554432
MBA_NOFUNC: int  # 2
MBA_NUMADDR: int  # 536870912
MBA_PASSREGS: int  # 64
MBA_PATTERN: int  # 4
MBA_PRCDEFS: int  # 1
MBA_PREOPT: int  # 512
MBA_REFINE: int  # 67108864
MBA_RETFP: int  # 16
MBA_RETREF: int  # 32768
MBA_SAVRST: int  # 16384
MBA_SHORT: int  # 4194304
MBA_SPLINFO: int  # 32
MBA_THUNK: int  # 128
MBA_VALNUM: int  # 1073741824
MBA_WINGR32: int  # 268435456
MBL_BACKPROP: int  # 4096
MBL_CALL: int  # 2048
MBL_COMB: int  # 64
MBL_DEAD: int  # 256
MBL_DMT64: int  # 32
MBL_DSLOT: int  # 16384
MBL_EXTFRAME: int  # 262144
MBL_FAKE: int  # 2
MBL_GOTO: int  # 4
MBL_INCONST: int  # 1024
MBL_INLINED: int  # 131072
MBL_KEEP: int  # 65536
MBL_LIST: int  # 512
MBL_NONFAKE: int  # 0
MBL_NORET: int  # 8192
MBL_PRIV: int  # 1
MBL_PROP: int  # 128
MBL_PUSH: int  # 16
MBL_TCAL: int  # 8
MBL_VALRANGES: int  # 32768
MERR_BADARCH: int  # -31
MERR_BADBLK: int  # -4
MERR_BADCALL: int  # -12
MERR_BADFRAME: int  # -13
MERR_BADIDB: int  # -15
MERR_BADRANGES: int  # -30
MERR_BADSP: int  # -5
MERR_BITNESS: int  # -11
MERR_BLOCK: int  # 1
MERR_BUSY: int  # -26
MERR_CANCELED: int  # -18
MERR_CLOUD: int  # -34
MERR_COMPLEX: int  # -22
MERR_DSLOT: int  # -32
MERR_EMULATOR: int  # -35
MERR_EXCEPTION: int  # -8
MERR_EXTERN: int  # -28
MERR_FARPTR: int  # -27
MERR_FUNCSIZE: int  # -29
MERR_HUGESTACK: int  # -9
MERR_INSN: int  # -2
MERR_INTERR: int  # -1
MERR_LICENSE: int  # -23
MERR_LOOP: int  # -36
MERR_LVARS: int  # -10
MERR_MAX_ERR: int  # 35
MERR_MEM: int  # -3
MERR_OK: int  # 0
MERR_ONLY32: int  # -24
MERR_ONLY64: int  # -25
MERR_OVERLAP: int  # -20
MERR_PARTINIT: int  # -21
MERR_PROLOG: int  # -6
MERR_RECDEPTH: int  # -19
MERR_REDO: int  # -17
MERR_SIZEOF: int  # -16
MERR_STOP: int  # -33
MERR_SWITCH: int  # -7
MERR_UNKTYPE: int  # -14
MIN_VLR_SVALUE: int  # -9223372036854775808
MLI_CLR_FLAGS: int  # 16
MLI_CMT: int  # 4
MLI_NAME: int  # 1
MLI_SET_FLAGS: int  # 8
MLI_TYPE: int  # 2
MMAT_CALLS: int  # 4
MMAT_GENERATED: int  # 1
MMAT_GLBOPT1: int  # 5
MMAT_GLBOPT2: int  # 6
MMAT_GLBOPT3: int  # 7
MMAT_LOCOPT: int  # 3
MMAT_LVARS: int  # 8
MMAT_PREOPTIMIZED: int  # 2
MMAT_ZERO: int  # 0
MMIDX_ARGS: int  # 4
MMIDX_GLBHIGH: int  # 5
MMIDX_GLBLOW: int  # 0
MMIDX_LVARS: int  # 1
MMIDX_RETADDR: int  # 2
MMIDX_SHADOW: int  # 3
MUST_ACCESS: int  # 0
NALT_VD: int  # 2
NF_BINVDONE: int  # 4
NF_BITNOT: int  # 16
NF_FIXED: int  # 1
NF_NEGATE: int  # 8
NF_NEGDONE: int  # 2
NF_VALID: int  # 32
NORET_FORBID_ANALYSIS: int  # 2
NORET_IGNORE_WAS_NORET_ICALL: int  # 1
NOSIZE: int  # -1
NO_CURLY_BRACES: int  # 1
NO_SIDEFF: int  # 0
ONE_ACCESS_TYPE: int  # 32
ONLY_SIDEFF: int  # 2
OPF_NEW_WINDOW: int  # 1
OPF_NO_WAIT: int  # 8
OPF_REUSE: int  # 0
OPF_REUSE_ACTIVE: int  # 2
OPF_WINDOW_MGMT_MASK: int  # 7
OPROP_ABI: int  # 64
OPROP_CCFLAGS: int  # 8
OPROP_FLOAT: int  # 4
OPROP_IMPDONE: int  # 1
OPROP_LOWADDR: int  # 32
OPROP_UDEFVAL: int  # 16
OPROP_UDT: int  # 2
OPTI_ADDREXPRS: int  # 1
OPTI_COMBINSNS: int  # 4
OPTI_MINSTKREF: int  # 2
OPTI_NO_LDXOPT: int  # 8
OPTI_NO_VALRNG: int  # 16
RETRIEVE_ALWAYS: int  # 1
RETRIEVE_ONCE: int  # 0
ROLE_3WAYCMP0: int  # 32
ROLE_3WAYCMP1: int  # 33
ROLE_ABS: int  # 31
ROLE_ALLOCA: int  # 11
ROLE_BITTEST: int  # 19
ROLE_BITTESTANDCOMPLEMENT: int  # 22
ROLE_BITTESTANDRESET: int  # 21
ROLE_BITTESTANDSET: int  # 20
ROLE_BSWAP: int  # 12
ROLE_BUG: int  # 10
ROLE_CFSUB3: int  # 29
ROLE_CONTAINING_RECORD: int  # 14
ROLE_EMPTY: int  # 1
ROLE_FASTFAIL: int  # 15
ROLE_IS_MUL_OK: int  # 17
ROLE_MEMCPY: int  # 5
ROLE_MEMSET: int  # 2
ROLE_MEMSET32: int  # 3
ROLE_MEMSET64: int  # 4
ROLE_OFSUB3: int  # 30
ROLE_PRESENT: int  # 13
ROLE_READFLAGS: int  # 16
ROLE_ROL: int  # 27
ROLE_ROR: int  # 28
ROLE_SATURATED_MUL: int  # 18
ROLE_SSE_CMP4: int  # 39
ROLE_SSE_CMP8: int  # 40
ROLE_STRCAT: int  # 8
ROLE_STRCPY: int  # 6
ROLE_STRLEN: int  # 7
ROLE_TAIL: int  # 9
ROLE_UNK: int  # 0
ROLE_VA_ARG: int  # 23
ROLE_VA_COPY: int  # 24
ROLE_VA_END: int  # 26
ROLE_VA_START: int  # 25
ROLE_WCSCAT: int  # 38
ROLE_WCSCPY: int  # 36
ROLE_WCSLEN: int  # 37
ROLE_WMEMCPY: int  # 34
ROLE_WMEMSET: int  # 35
SHINS_LDXEA: int  # 8
SHINS_NUMADDR: int  # 1
SHINS_SHORT: int  # 4
SHINS_VALNUM: int  # 2
SIZEOF_BLOCK_CHAINS: int  # 56
SVW_FLOAT: int  # 1
SVW_INT: int  # 0
SVW_SOFT: int  # 2
SWIG_PYTHON_LEGACY_BOOL: int  # 1
TS_DONTREF: int  # 33554432
TS_MASK: int  # 234881024
TS_NOELL: int  # 134217728
TS_SHRINK: int  # 67108864
ULV_PRECISE_DEFEA: int  # 1
USE_CURLY_BRACES: int  # 2
USE_KEYBOARD: int  # 0
USE_MOUSE: int  # 1
VDI_EXPR: int  # 1
VDI_FUNC: int  # 3
VDI_LVAR: int  # 2
VDI_NONE: int  # 0
VDI_TAIL: int  # 4
VDRUN_APPEND: int  # 1
VDRUN_CMDLINE: int  # 32
VDRUN_LUMINA: int  # 128
VDRUN_MAYSTOP: int  # 16
VDRUN_NEWFILE: int  # 0
VDRUN_ONLYNEW: int  # 2
VDRUN_PERF: int  # 2097152
VDRUN_SENDIDB: int  # 8
VDRUN_SILENT: int  # 4
VDRUN_STATS: int  # 64
VDUI_VALID: int  # 2
VDUI_VISIBLE: int  # 1
VR_AT_END: int  # 1
VR_AT_START: int  # 0
VR_EXACT: int  # 2
WARN_ADDR_OUTARGS: int  # 6
WARN_ARRAY_INARG: int  # 21
WARN_BAD_CALL_SP: int  # 38
WARN_BAD_FIELD_TYPE: int  # 23
WARN_BAD_INSN: int  # 49
WARN_BAD_MAPDST: int  # 48
WARN_BAD_PURGED: int  # 12
WARN_BAD_RETVAR: int  # 25
WARN_BAD_SHADOW: int  # 45
WARN_BAD_SP: int  # 40
WARN_BAD_STD_TYPE: int  # 37
WARN_BAD_STKPNT: int  # 41
WARN_BAD_STROFF: int  # 33
WARN_BAD_VALRNG: int  # 44
WARN_BAD_VARSIZE: int  # 34
WARN_CBUILD_LOOPS: int  # 13
WARN_CR_BADOFF: int  # 32
WARN_CR_NOFIELD: int  # 31
WARN_DEP_UNK_CALLS: int  # 7
WARN_EXP_LINVAR: int  # 10
WARN_FIXED_INSN: int  # 29
WARN_FRAG_LVAR: int  # 26
WARN_GUESSED_TYPE: int  # 9
WARN_HUGE_STKOFF: int  # 27
WARN_ILL_ELLIPSIS: int  # 8
WARN_ILL_FPU_STACK: int  # 18
WARN_ILL_FUNCTYPE: int  # 2
WARN_ILL_PURGED: int  # 1
WARN_JUMPOUT: int  # 43
WARN_MAX: int  # 57
WARN_MAX_ARGS: int  # 22
WARN_MISSED_SWITCH: int  # 39
WARN_MUST_RET_FP: int  # 17
WARN_NO_SAVE_REST: int  # 14
WARN_ODD_ABI: int  # 50
WARN_ODD_ADDR_USE: int  # 16
WARN_ODD_INPUT_REG: int  # 15
WARN_OPT_USELESS_JCND: int  # 54
WARN_OPT_VALRNG: int  # 46
WARN_OPT_VALRNG2: int  # 52
WARN_OPT_VALRNG3: int  # 53
WARN_OPT_VALRNG4: int  # 56
WARN_RET_LOCREF: int  # 47
WARN_SELFREF_PROP: int  # 19
WARN_SUBFRAME_OVERFLOW: int  # 55
WARN_UNALIGNED_ARG: int  # 36
WARN_UNBALANCED_STACK: int  # 51
WARN_UNDEF_LVAR: int  # 42
WARN_UNINITED_REG: int  # 28
WARN_UNSUPP_REG: int  # 35
WARN_VARARG_MANY: int  # 5
WARN_VARARG_NOSTK: int  # 4
WARN_VARARG_REGS: int  # 0
WARN_VARARG_TCAL: int  # 3
WARN_WIDEN_CHAINS: int  # 11
WARN_WOULD_OVERLAP: int  # 20
WARN_WRITE_CONST: int  # 24
WARN_WRONG_VA_OFF: int  # 30
WITH_ASSERTS: int  # 512
WITH_SIDEFF: int  # 1
annotations: _Feature
bitset_align: int  # 63
bitset_shift: int  # 6
bitset_width: int  # 64
cc_count: int  # 5
cit_asm: int  # 82
cit_block: int  # 71
cit_break: int  # 78
cit_continue: int  # 79
cit_do: int  # 76
cit_empty: int  # 70
cit_end: int  # 85
cit_expr: int  # 72
cit_for: int  # 74
cit_goto: int  # 81
cit_if: int  # 73
cit_return: int  # 80
cit_switch: int  # 77
cit_throw: int  # 84
cit_try: int  # 83
cit_while: int  # 75
cot_add: int  # 35
cot_asg: int  # 2
cot_asgadd: int  # 6
cot_asgband: int  # 5
cot_asgbor: int  # 3
cot_asgmul: int  # 8
cot_asgsdiv: int  # 12
cot_asgshl: int  # 11
cot_asgsmod: int  # 14
cot_asgsshr: int  # 9
cot_asgsub: int  # 7
cot_asgudiv: int  # 13
cot_asgumod: int  # 15
cot_asgushr: int  # 10
cot_asgxor: int  # 4
cot_band: int  # 21
cot_bnot: int  # 50
cot_bor: int  # 19
cot_call: int  # 57
cot_cast: int  # 48
cot_comma: int  # 1
cot_empty: int  # 0
cot_eq: int  # 22
cot_fadd: int  # 42
cot_fdiv: int  # 45
cot_fmul: int  # 44
cot_fneg: int  # 46
cot_fnum: int  # 62
cot_fsub: int  # 43
cot_helper: int  # 68
cot_idx: int  # 58
cot_insn: int  # 66
cot_land: int  # 18
cot_last: int  # 69
cot_lnot: int  # 49
cot_lor: int  # 17
cot_memptr: int  # 60
cot_memref: int  # 59
cot_mul: int  # 37
cot_ne: int  # 23
cot_neg: int  # 47
cot_num: int  # 61
cot_obj: int  # 64
cot_postdec: int  # 54
cot_postinc: int  # 53
cot_predec: int  # 56
cot_preinc: int  # 55
cot_ptr: int  # 51
cot_ref: int  # 52
cot_sdiv: int  # 38
cot_sge: int  # 24
cot_sgt: int  # 28
cot_shl: int  # 34
cot_sizeof: int  # 67
cot_sle: int  # 26
cot_slt: int  # 30
cot_smod: int  # 40
cot_sshr: int  # 32
cot_str: int  # 63
cot_sub: int  # 36
cot_tern: int  # 16
cot_type: int  # 69
cot_udiv: int  # 39
cot_uge: int  # 25
cot_ugt: int  # 29
cot_ule: int  # 27
cot_ult: int  # 31
cot_umod: int  # 41
cot_ushr: int  # 33
cot_var: int  # 65
cot_xor: int  # 20
cvar: swigvarlink
hx_arglocs_overlap: int  # 177
hx_asgop: int  # 448
hx_asgop_revert: int  # 449
hx_bitset_t_add: int  # 205
hx_bitset_t_add_: int  # 206
hx_bitset_t_add__: int  # 207
hx_bitset_t_bitset_t: int  # 203
hx_bitset_t_compare: int  # 226
hx_bitset_t_copy: int  # 204
hx_bitset_t_count: int  # 218
hx_bitset_t_count_: int  # 219
hx_bitset_t_cut_at: int  # 211
hx_bitset_t_dstr: int  # 216
hx_bitset_t_empty: int  # 217
hx_bitset_t_fill_gaps: int  # 222
hx_bitset_t_fill_with_ones: int  # 221
hx_bitset_t_goup: int  # 227
hx_bitset_t_has: int  # 213
hx_bitset_t_has_all: int  # 214
hx_bitset_t_has_any: int  # 215
hx_bitset_t_has_common: int  # 223
hx_bitset_t_intersect: int  # 224
hx_bitset_t_is_subset_of: int  # 225
hx_bitset_t_last: int  # 220
hx_bitset_t_shift_down: int  # 212
hx_bitset_t_sub: int  # 208
hx_bitset_t_sub_: int  # 209
hx_bitset_t_sub__: int  # 210
hx_block_chains_begin: int  # 117
hx_block_chains_clear: int  # 125
hx_block_chains_end: int  # 118
hx_block_chains_erase: int  # 124
hx_block_chains_find: int  # 122
hx_block_chains_free: int  # 127
hx_block_chains_get: int  # 121
hx_block_chains_insert: int  # 123
hx_block_chains_new: int  # 128
hx_block_chains_next: int  # 119
hx_block_chains_prev: int  # 120
hx_block_chains_size: int  # 126
hx_block_chains_t_dstr: int  # 330
hx_block_chains_t_get_chain: int  # 328
hx_block_chains_t_print: int  # 329
hx_boundaries_begin: int  # 104
hx_boundaries_clear: int  # 113
hx_boundaries_end: int  # 105
hx_boundaries_erase: int  # 112
hx_boundaries_find: int  # 110
hx_boundaries_first: int  # 108
hx_boundaries_free: int  # 115
hx_boundaries_insert: int  # 111
hx_boundaries_new: int  # 116
hx_boundaries_next: int  # 106
hx_boundaries_prev: int  # 107
hx_boundaries_second: int  # 109
hx_boundaries_size: int  # 114
hx_carglist_t_compare: int  # 504
hx_casm_t_compare: int  # 489
hx_catchexpr_t_compare: int  # 616
hx_cblock_t_compare: int  # 503
hx_ccase_t_compare: int  # 505
hx_ccases_t_compare: int  # 506
hx_ccatch_t_compare: int  # 508
hx_cdg_insn_iterator_t_next: int  # 430
hx_cdo_t_compare: int  # 485
hx_cexpr_t_assign: int  # 464
hx_cexpr_t_calc_type: int  # 470
hx_cexpr_t_cleanup: int  # 467
hx_cexpr_t_compare: int  # 465
hx_cexpr_t_contains_operator: int  # 473
hx_cexpr_t_dstr: int  # 479
hx_cexpr_t_equal_effect: int  # 471
hx_cexpr_t_get_high_nbit_bound: int  # 474
hx_cexpr_t_get_low_nbit_bound: int  # 475
hx_cexpr_t_has_side_effects: int  # 477
hx_cexpr_t_is_child_of: int  # 472
hx_cexpr_t_maybe_ptr: int  # 478
hx_cexpr_t_print1: int  # 469
hx_cexpr_t_put_number: int  # 468
hx_cexpr_t_replace_by: int  # 466
hx_cexpr_t_requires_lvalue: int  # 476
hx_cfor_t_compare: int  # 483
hx_cfunc_parentee_t_calc_rvalue_type: int  # 458
hx_cfunc_t_build_c_tree: int  # 534
hx_cfunc_t_cleanup: int  # 564
hx_cfunc_t_del_orphan_cmts: int  # 548
hx_cfunc_t_find_item_coords: int  # 563
hx_cfunc_t_find_label: int  # 541
hx_cfunc_t_gather_derefs: int  # 562
hx_cfunc_t_get_boundaries: int  # 559
hx_cfunc_t_get_eamap: int  # 558
hx_cfunc_t_get_func_type: int  # 538
hx_cfunc_t_get_line_item: int  # 556
hx_cfunc_t_get_lvars: int  # 539
hx_cfunc_t_get_pseudocode: int  # 560
hx_cfunc_t_get_stkoff_delta: int  # 540
hx_cfunc_t_get_user_cmt: int  # 543
hx_cfunc_t_get_user_iflags: int  # 545
hx_cfunc_t_get_user_union_selection: int  # 549
hx_cfunc_t_get_warnings: int  # 557
hx_cfunc_t_has_orphan_cmts: int  # 547
hx_cfunc_t_print_dcl: int  # 536
hx_cfunc_t_print_func: int  # 537
hx_cfunc_t_recalc_item_addresses: int  # 619
hx_cfunc_t_refresh_func_ctext: int  # 561
hx_cfunc_t_remove_unused_labels: int  # 542
hx_cfunc_t_save_user_cmts: int  # 552
hx_cfunc_t_save_user_iflags: int  # 554
hx_cfunc_t_save_user_labels: int  # 551
hx_cfunc_t_save_user_numforms: int  # 553
hx_cfunc_t_save_user_unions: int  # 555
hx_cfunc_t_set_user_cmt: int  # 544
hx_cfunc_t_set_user_iflags: int  # 546
hx_cfunc_t_set_user_union_selection: int  # 550
hx_cfunc_t_verify: int  # 535
hx_cgoto_t_compare: int  # 488
hx_chain_t_append_list: int  # 326
hx_chain_t_append_list_: int  # 327
hx_chain_t_dstr: int  # 325
hx_chain_t_print: int  # 324
hx_change_hexrays_config: int  # 434
hx_cif_t_assign: int  # 480
hx_cif_t_compare: int  # 481
hx_cinsn_t_assign: int  # 490
hx_cinsn_t_cleanup: int  # 493
hx_cinsn_t_collect_free_breaks: int  # 500
hx_cinsn_t_collect_free_continues: int  # 501
hx_cinsn_t_compare: int  # 491
hx_cinsn_t_contains_insn: int  # 499
hx_cinsn_t_create_if: int  # 495
hx_cinsn_t_dstr: int  # 502
hx_cinsn_t_is_ordinary_flow: int  # 498
hx_cinsn_t_new_insn: int  # 494
hx_cinsn_t_print: int  # 496
hx_cinsn_t_print1: int  # 497
hx_cinsn_t_replace_by: int  # 492
hx_citem_locator_t_compare: int  # 459
hx_citem_t_contains_expr: int  # 460
hx_citem_t_contains_label: int  # 461
hx_citem_t_find_closest_addr: int  # 463
hx_citem_t_find_parent_of: int  # 462
hx_clear_cached_cfuncs: int  # 570
hx_cloop_t_assign: int  # 482
hx_close_hexrays_waitbox: int  # 565
hx_close_pseudocode: int  # 437
hx_cnumber_t_assign: int  # 452
hx_cnumber_t_compare: int  # 453
hx_cnumber_t_print: int  # 450
hx_cnumber_t_value: int  # 451
hx_codegen_t_clear: int  # 431
hx_codegen_t_emit: int  # 432
hx_codegen_t_emit_: int  # 433
hx_convert_to_user_call: int  # 198
hx_create_cfunc: int  # 568
hx_create_field_name: int  # 573
hx_create_typedef: int  # 170
hx_creturn_t_compare: int  # 486
hx_cswitch_t_compare: int  # 507
hx_cthrow_t_compare: int  # 487
hx_ctree_item_t_dstr: int  # 516
hx_ctree_item_t_get_ea: int  # 513
hx_ctree_item_t_get_edm: int  # 511
hx_ctree_item_t_get_label_num: int  # 514
hx_ctree_item_t_get_lvar: int  # 512
hx_ctree_item_t_get_udm: int  # 510
hx_ctree_item_t_print: int  # 515
hx_ctree_parentee_t_recalc_parent_types: int  # 457
hx_ctree_visitor_t_apply_to: int  # 455
hx_ctree_visitor_t_apply_to_exprs: int  # 456
hx_ctry_t_compare: int  # 509
hx_cwhile_t_compare: int  # 484
hx_decompile: int  # 566
hx_decompile_many: int  # 439
hx_dereference: int  # 523
hx_dstr: int  # 158
hx_dummy_ptrtype: int  # 167
hx_eamap_begin: int  # 91
hx_eamap_clear: int  # 100
hx_eamap_end: int  # 92
hx_eamap_erase: int  # 99
hx_eamap_find: int  # 97
hx_eamap_first: int  # 95
hx_eamap_free: int  # 102
hx_eamap_insert: int  # 98
hx_eamap_new: int  # 103
hx_eamap_next: int  # 93
hx_eamap_prev: int  # 94
hx_eamap_second: int  # 96
hx_eamap_size: int  # 101
hx_file_printer_t_print: int  # 156
hx_fnumber_t_dstr: int  # 272
hx_fnumber_t_print: int  # 271
hx_gco_info_t_append_to_list: int  # 442
hx_gen_microcode: int  # 567
hx_get_ctype_name: int  # 572
hx_get_current_operand: int  # 443
hx_get_float_type: int  # 164
hx_get_hexrays_version: int  # 435
hx_get_int_type_by_width_and_sign: int  # 165
hx_get_member_type: int  # 168
hx_get_merror_desc: int  # 146
hx_get_mreg_name: int  # 254
hx_get_op_signness: int  # 447
hx_get_signed_mcode: int  # 151
hx_get_temp_regs: int  # 250
hx_get_type: int  # 171
hx_get_unk_type: int  # 166
hx_get_unsigned_mcode: int  # 152
hx_get_widget_vdui: int  # 438
hx_getb_reginsn: int  # 359
hx_getf_reginsn: int  # 358
hx_graph_chains_t_for_all_chains: int  # 331
hx_graph_chains_t_release: int  # 332
hx_has_cached_cfunc: int  # 571
hx_hexrays_alloc: int  # 129
hx_hexrays_failure_t_desc: int  # 440
hx_hexrays_free: int  # 130
hx_install_hexrays_callback: int  # 574
hx_install_microcode_filter: int  # 199
hx_install_optblock_handler: int  # 257
hx_install_optinsn_handler: int  # 255
hx_int64_emulator_t_minsn_value: int  # 621
hx_int64_emulator_t_mop_value: int  # 620
hx_is_bool_type: int  # 162
hx_is_kreg: int  # 251
hx_is_mcode_propagatable: int  # 148
hx_is_nonbool_type: int  # 161
hx_is_small_udt: int  # 160
hx_is_type_correct: int  # 159
hx_ivl_t_compare: int  # 229
hx_ivl_t_dstr: int  # 228
hx_ivlset_t_add: int  # 230
hx_ivlset_t_add_: int  # 231
hx_ivlset_t_addmasked: int  # 232
hx_ivlset_t_compare: int  # 243
hx_ivlset_t_contains: int  # 240
hx_ivlset_t_count: int  # 238
hx_ivlset_t_dstr: int  # 237
hx_ivlset_t_has_common: int  # 235
hx_ivlset_t_has_common_: int  # 239
hx_ivlset_t_includes: int  # 241
hx_ivlset_t_intersect: int  # 242
hx_ivlset_t_print: int  # 236
hx_ivlset_t_sub: int  # 233
hx_ivlset_t_sub_: int  # 234
hx_lnot: int  # 517
hx_locate_lvar: int  # 194
hx_lvar_locator_t_compare: int  # 178
hx_lvar_locator_t_dstr: int  # 179
hx_lvar_mapping_begin: int  # 13
hx_lvar_mapping_clear: int  # 22
hx_lvar_mapping_end: int  # 14
hx_lvar_mapping_erase: int  # 21
hx_lvar_mapping_find: int  # 19
hx_lvar_mapping_first: int  # 17
hx_lvar_mapping_free: int  # 24
hx_lvar_mapping_insert: int  # 20
hx_lvar_mapping_new: int  # 25
hx_lvar_mapping_next: int  # 15
hx_lvar_mapping_prev: int  # 16
hx_lvar_mapping_second: int  # 18
hx_lvar_mapping_size: int  # 23
hx_lvar_ref_t_compare: int  # 267
hx_lvar_ref_t_var: int  # 268
hx_lvar_t_accepts_type: int  # 182
hx_lvar_t_append_list: int  # 185
hx_lvar_t_append_list_: int  # 186
hx_lvar_t_dstr: int  # 180
hx_lvar_t_is_promoted_arg: int  # 181
hx_lvar_t_set_lvar_type: int  # 183
hx_lvar_t_set_width: int  # 184
hx_lvars_t_find: int  # 188
hx_lvars_t_find_lvar: int  # 189
hx_lvars_t_find_stkvar: int  # 187
hx_make_num: int  # 521
hx_make_pointer: int  # 169
hx_make_ref: int  # 522
hx_mark_cfunc_dirty: int  # 569
hx_mba_ranges_t_range_contains: int  # 384
hx_mba_t_alloc_fict_ea: int  # 417
hx_mba_t_alloc_kreg: int  # 422
hx_mba_t_alloc_lvars: int  # 399
hx_mba_t_analyze_calls: int  # 397
hx_mba_t_arg: int  # 416
hx_mba_t_build_graph: int  # 395
hx_mba_t_copy_block: int  # 407
hx_mba_t_create_helper_call: int  # 414
hx_mba_t_deserialize: int  # 420
hx_mba_t_dump: int  # 400
hx_mba_t_find_mop: int  # 413
hx_mba_t_for_all_insns: int  # 411
hx_mba_t_for_all_ops: int  # 410
hx_mba_t_for_all_topinsns: int  # 412
hx_mba_t_free_kreg: int  # 423
hx_mba_t_get_curfunc: int  # 392
hx_mba_t_get_func_output_lists: int  # 415
hx_mba_t_get_graph: int  # 396
hx_mba_t_idaloc2vd: int  # 387
hx_mba_t_idaloc2vd_: int  # 388
hx_mba_t_inline_func: int  # 424
hx_mba_t_insert_block: int  # 405
hx_mba_t_locate_stkpnt: int  # 425
hx_mba_t_map_fict_ea: int  # 418
hx_mba_t_mark_chains_dirty: int  # 404
hx_mba_t_merge_blocks: int  # 409
hx_mba_t_optimize_global: int  # 398
hx_mba_t_optimize_local: int  # 394
hx_mba_t_print: int  # 402
hx_mba_t_remove_block: int  # 406
hx_mba_t_remove_blocks: int  # 618
hx_mba_t_remove_empty_and_unreachable_blocks: int  # 408
hx_mba_t_save_snapshot: int  # 421
hx_mba_t_serialize: int  # 419
hx_mba_t_set_lvar_name: int  # 426
hx_mba_t_set_maturity: int  # 393
hx_mba_t_split_block: int  # 617
hx_mba_t_stkoff_ida2vd: int  # 386
hx_mba_t_stkoff_vd2ida: int  # 385
hx_mba_t_term: int  # 391
hx_mba_t_vd2idaloc: int  # 389
hx_mba_t_vd2idaloc_: int  # 390
hx_mba_t_vdump_mba: int  # 401
hx_mba_t_verify: int  # 403
hx_mbl_graph_t_get_du: int  # 429
hx_mbl_graph_t_get_ud: int  # 428
hx_mbl_graph_t_is_accessed_globally: int  # 427
hx_mblock_t_append_def_list: int  # 374
hx_mblock_t_append_use_list: int  # 373
hx_mblock_t_build_def_list: int  # 376
hx_mblock_t_build_lists: int  # 371
hx_mblock_t_build_use_list: int  # 375
hx_mblock_t_dump: int  # 362
hx_mblock_t_find_access: int  # 380
hx_mblock_t_find_first_use: int  # 377
hx_mblock_t_find_redefinition: int  # 378
hx_mblock_t_for_all_insns: int  # 366
hx_mblock_t_for_all_ops: int  # 367
hx_mblock_t_for_all_uses: int  # 368
hx_mblock_t_get_reginsn_qty: int  # 383
hx_mblock_t_get_valranges: int  # 381
hx_mblock_t_get_valranges_: int  # 382
hx_mblock_t_init: int  # 360
hx_mblock_t_insert_into_block: int  # 364
hx_mblock_t_is_rhs_redefined: int  # 379
hx_mblock_t_optimize_block: int  # 370
hx_mblock_t_optimize_insn: int  # 369
hx_mblock_t_optimize_useless_jump: int  # 372
hx_mblock_t_print: int  # 361
hx_mblock_t_remove_from_block: int  # 365
hx_mblock_t_vdump_block: int  # 363
hx_mcallarg_t_dstr: int  # 310
hx_mcallarg_t_print: int  # 309
hx_mcallarg_t_set_regarg: int  # 311
hx_mcallinfo_t_dstr: int  # 316
hx_mcallinfo_t_get_type: int  # 314
hx_mcallinfo_t_lexcompare: int  # 312
hx_mcallinfo_t_print: int  # 315
hx_mcallinfo_t_set_type: int  # 313
hx_mcases_t_compare: int  # 317
hx_mcases_t_dstr: int  # 319
hx_mcases_t_print: int  # 318
hx_mcode_modifies_d: int  # 153
hx_minsn_t__make_nop: int  # 343
hx_minsn_t_copy: int  # 334
hx_minsn_t_deserialize: int  # 357
hx_minsn_t_dstr: int  # 338
hx_minsn_t_equal_insns: int  # 344
hx_minsn_t_find_call: int  # 348
hx_minsn_t_find_ins_op: int  # 351
hx_minsn_t_find_num_op: int  # 352
hx_minsn_t_find_opcode: int  # 350
hx_minsn_t_for_all_insns: int  # 342
hx_minsn_t_for_all_ops: int  # 341
hx_minsn_t_has_side_effects: int  # 349
hx_minsn_t_init: int  # 333
hx_minsn_t_is_between: int  # 354
hx_minsn_t_is_helper: int  # 347
hx_minsn_t_is_noret_call: int  # 346
hx_minsn_t_lexcompare: int  # 345
hx_minsn_t_may_use_aliased_memory: int  # 355
hx_minsn_t_modifies_d: int  # 353
hx_minsn_t_optimize_subtree: int  # 340
hx_minsn_t_print: int  # 337
hx_minsn_t_serialize: int  # 356
hx_minsn_t_set_combined: int  # 335
hx_minsn_t_setaddr: int  # 339
hx_minsn_t_swap: int  # 336
hx_mlist_t_addmem: int  # 246
hx_mlist_t_compare: int  # 249
hx_mlist_t_dstr: int  # 248
hx_mlist_t_print: int  # 247
hx_modify_user_lvar_info: int  # 193
hx_modify_user_lvars: int  # 192
hx_mop_t__make_gvar: int  # 286
hx_mop_t_apply_ld_mcode: int  # 308
hx_mop_t_assign: int  # 274
hx_mop_t_change_size: int  # 306
hx_mop_t_copy: int  # 273
hx_mop_t_create_from_insn: int  # 283
hx_mop_t_create_from_ivlset: int  # 280
hx_mop_t_create_from_mlist: int  # 279
hx_mop_t_create_from_scattered_vdloc: int  # 282
hx_mop_t_create_from_vdloc: int  # 281
hx_mop_t_dstr: int  # 278
hx_mop_t_equal_mops: int  # 295
hx_mop_t_erase: int  # 276
hx_mop_t_for_all_ops: int  # 297
hx_mop_t_for_all_scattered_submops: int  # 298
hx_mop_t_get_stkoff: int  # 300
hx_mop_t_is01: int  # 292
hx_mop_t_is_bit_reg: int  # 290
hx_mop_t_is_constant: int  # 299
hx_mop_t_is_sign_extended_from: int  # 293
hx_mop_t_is_zero_extended_from: int  # 294
hx_mop_t_lexcompare: int  # 296
hx_mop_t_make_first_half: int  # 303
hx_mop_t_make_fpnum: int  # 285
hx_mop_t_make_gvar: int  # 287
hx_mop_t_make_helper: int  # 289
hx_mop_t_make_high_half: int  # 302
hx_mop_t_make_low_half: int  # 301
hx_mop_t_make_number: int  # 284
hx_mop_t_make_reg_pair: int  # 288
hx_mop_t_make_second_half: int  # 304
hx_mop_t_may_use_aliased_memory: int  # 291
hx_mop_t_preserve_side_effects: int  # 307
hx_mop_t_print: int  # 277
hx_mop_t_shift_mop: int  # 305
hx_mop_t_swap: int  # 275
hx_mreg2reg: int  # 253
hx_must_mcode_close_block: int  # 147
hx_mutable_graph_t_del_edge: int  # 266
hx_mutable_graph_t_goup: int  # 265
hx_mutable_graph_t_resize: int  # 264
hx_negate_mcode_relation: int  # 149
hx_negated_relation: int  # 445
hx_new_block: int  # 518
hx_open_pseudocode: int  # 436
hx_operand_locator_t_compare: int  # 154
hx_parse_user_call: int  # 197
hx_partial_type_num: int  # 163
hx_print_vdloc: int  # 176
hx_qstring_printer_t_print: int  # 157
hx_reg2mreg: int  # 252
hx_remitem: int  # 444
hx_remove_hexrays_callback: int  # 575
hx_remove_optblock_handler: int  # 258
hx_remove_optinsn_handler: int  # 256
hx_restore_user_cmts: int  # 530
hx_restore_user_defined_calls: int  # 195
hx_restore_user_iflags: int  # 532
hx_restore_user_labels: int  # 529
hx_restore_user_lvar_settings: int  # 190
hx_restore_user_numforms: int  # 531
hx_restore_user_unions: int  # 533
hx_rlist_t_dstr: int  # 245
hx_rlist_t_print: int  # 244
hx_save_user_cmts: int  # 525
hx_save_user_defined_calls: int  # 196
hx_save_user_iflags: int  # 527
hx_save_user_labels: int  # 524
hx_save_user_lvar_settings: int  # 191
hx_save_user_numforms: int  # 526
hx_save_user_unions: int  # 528
hx_select_udt_by_offset: int  # 615
hx_send_database: int  # 441
hx_set_type: int  # 172
hx_simple_graph_t_compute_dominators: int  # 259
hx_simple_graph_t_compute_immediate_dominators: int  # 260
hx_simple_graph_t_depth_first_postorder: int  # 262
hx_simple_graph_t_depth_first_preorder: int  # 261
hx_simple_graph_t_goup: int  # 263
hx_stkvar_ref_t_compare: int  # 269
hx_stkvar_ref_t_get_stkvar: int  # 270
hx_swap_mcode_relation: int  # 150
hx_swapped_relation: int  # 446
hx_udc_filter_t_apply: int  # 202
hx_udc_filter_t_cleanup: int  # 200
hx_udc_filter_t_init: int  # 201
hx_udcall_map_begin: int  # 26
hx_udcall_map_clear: int  # 35
hx_udcall_map_end: int  # 27
hx_udcall_map_erase: int  # 34
hx_udcall_map_find: int  # 32
hx_udcall_map_first: int  # 30
hx_udcall_map_free: int  # 37
hx_udcall_map_insert: int  # 33
hx_udcall_map_new: int  # 38
hx_udcall_map_next: int  # 28
hx_udcall_map_prev: int  # 29
hx_udcall_map_second: int  # 31
hx_udcall_map_size: int  # 36
hx_user_cmts_begin: int  # 39
hx_user_cmts_clear: int  # 48
hx_user_cmts_end: int  # 40
hx_user_cmts_erase: int  # 47
hx_user_cmts_find: int  # 45
hx_user_cmts_first: int  # 43
hx_user_cmts_free: int  # 50
hx_user_cmts_insert: int  # 46
hx_user_cmts_new: int  # 51
hx_user_cmts_next: int  # 41
hx_user_cmts_prev: int  # 42
hx_user_cmts_second: int  # 44
hx_user_cmts_size: int  # 49
hx_user_iflags_begin: int  # 52
hx_user_iflags_clear: int  # 61
hx_user_iflags_end: int  # 53
hx_user_iflags_erase: int  # 60
hx_user_iflags_find: int  # 58
hx_user_iflags_first: int  # 56
hx_user_iflags_free: int  # 63
hx_user_iflags_insert: int  # 59
hx_user_iflags_new: int  # 64
hx_user_iflags_next: int  # 54
hx_user_iflags_prev: int  # 55
hx_user_iflags_second: int  # 57
hx_user_iflags_size: int  # 62
hx_user_labels_begin: int  # 78
hx_user_labels_clear: int  # 87
hx_user_labels_end: int  # 79
hx_user_labels_erase: int  # 86
hx_user_labels_find: int  # 84
hx_user_labels_first: int  # 82
hx_user_labels_free: int  # 89
hx_user_labels_insert: int  # 85
hx_user_labels_new: int  # 90
hx_user_labels_next: int  # 80
hx_user_labels_prev: int  # 81
hx_user_labels_second: int  # 83
hx_user_labels_size: int  # 88
hx_user_numforms_begin: int  # 0
hx_user_numforms_clear: int  # 9
hx_user_numforms_end: int  # 1
hx_user_numforms_erase: int  # 8
hx_user_numforms_find: int  # 6
hx_user_numforms_first: int  # 4
hx_user_numforms_free: int  # 11
hx_user_numforms_insert: int  # 7
hx_user_numforms_new: int  # 12
hx_user_numforms_next: int  # 2
hx_user_numforms_prev: int  # 3
hx_user_numforms_second: int  # 5
hx_user_numforms_size: int  # 10
hx_user_unions_begin: int  # 65
hx_user_unions_clear: int  # 74
hx_user_unions_end: int  # 66
hx_user_unions_erase: int  # 73
hx_user_unions_find: int  # 71
hx_user_unions_first: int  # 69
hx_user_unions_free: int  # 76
hx_user_unions_insert: int  # 72
hx_user_unions_new: int  # 77
hx_user_unions_next: int  # 67
hx_user_unions_prev: int  # 68
hx_user_unions_second: int  # 70
hx_user_unions_size: int  # 75
hx_valrng_t_assign: int  # 133
hx_valrng_t_clear: int  # 131
hx_valrng_t_compare: int  # 134
hx_valrng_t_copy: int  # 132
hx_valrng_t_cvt_to_cmp: int  # 145
hx_valrng_t_cvt_to_single_value: int  # 144
hx_valrng_t_dstr: int  # 143
hx_valrng_t_has: int  # 141
hx_valrng_t_intersect_with: int  # 138
hx_valrng_t_inverse: int  # 140
hx_valrng_t_print: int  # 142
hx_valrng_t_reduce_size: int  # 137
hx_valrng_t_set_cmp: int  # 136
hx_valrng_t_set_eq: int  # 135
hx_valrng_t_unite_with: int  # 139
hx_var_ref_t_compare: int  # 454
hx_vcall_helper: int  # 520
hx_vcreate_helper: int  # 519
hx_vd_printer_t_print: int  # 155
hx_vdloc_t_compare: int  # 174
hx_vdloc_t_dstr: int  # 173
hx_vdloc_t_is_aliasable: int  # 175
hx_vdui_t_calc_cmt_type: int  # 603
hx_vdui_t_clear: int  # 582
hx_vdui_t_collapse_item: int  # 612
hx_vdui_t_collapse_lvars: int  # 613
hx_vdui_t_ctree_to_disasm: int  # 602
hx_vdui_t_del_orphan_cmts: int  # 606
hx_vdui_t_edit_cmt: int  # 604
hx_vdui_t_edit_func_cmt: int  # 605
hx_vdui_t_get_current_item: int  # 584
hx_vdui_t_get_current_label: int  # 581
hx_vdui_t_get_number: int  # 580
hx_vdui_t_invert_bits: int  # 611
hx_vdui_t_invert_sign: int  # 610
hx_vdui_t_jump_enter: int  # 601
hx_vdui_t_map_lvar: int  # 595
hx_vdui_t_refresh_cpos: int  # 583
hx_vdui_t_refresh_ctext: int  # 578
hx_vdui_t_refresh_view: int  # 577
hx_vdui_t_rename_global: int  # 599
hx_vdui_t_rename_label: int  # 600
hx_vdui_t_rename_lvar: int  # 586
hx_vdui_t_rename_udm: int  # 597
hx_vdui_t_set_global_type: int  # 598
hx_vdui_t_set_locked: int  # 576
hx_vdui_t_set_lvar_cmt: int  # 592
hx_vdui_t_set_lvar_type: int  # 589
hx_vdui_t_set_noptr_lvar: int  # 590
hx_vdui_t_set_num_enum: int  # 608
hx_vdui_t_set_num_radix: int  # 607
hx_vdui_t_set_num_stroff: int  # 609
hx_vdui_t_set_udm_type: int  # 596
hx_vdui_t_split_item: int  # 614
hx_vdui_t_switch_to: int  # 579
hx_vdui_t_ui_edit_lvar_cmt: int  # 591
hx_vdui_t_ui_map_lvar: int  # 593
hx_vdui_t_ui_rename_lvar: int  # 585
hx_vdui_t_ui_set_call_type: int  # 587
hx_vdui_t_ui_set_lvar_type: int  # 588
hx_vdui_t_ui_unmap_lvar: int  # 594
hx_vivl_t_dstr: int  # 323
hx_vivl_t_extend_to_cover: int  # 320
hx_vivl_t_intersect: int  # 321
hx_vivl_t_print: int  # 322
hxe_begin_inlining: int  # 19
hxe_build_callinfo: int  # 16
hxe_callinfo_built: int  # 17
hxe_calls_done: int  # 18
hxe_close_pseudocode: int  # 103
hxe_cmt_changed: int  # 115
hxe_collect_warnings: int  # 22
hxe_combine: int  # 12
hxe_create_hint: int  # 108
hxe_curpos: int  # 107
hxe_double_click: int  # 106
hxe_flowchart: int  # 0
hxe_func_printed: int  # 14
hxe_glbopt: int  # 7
hxe_inlined_func: int  # 21
hxe_inlining_func: int  # 20
hxe_interr: int  # 11
hxe_keyboard: int  # 104
hxe_locopt: int  # 5
hxe_maturity: int  # 10
hxe_mba_maturity: int  # 116
hxe_microcode: int  # 3
hxe_open_pseudocode: int  # 100
hxe_populating_popup: int  # 110
hxe_pre_structural: int  # 8
hxe_prealloc: int  # 6
hxe_preoptimized: int  # 4
hxe_print_func: int  # 13
hxe_prolog: int  # 2
hxe_refresh_pseudocode: int  # 102
hxe_resolve_stkaddrs: int  # 15
hxe_right_click: int  # 105
hxe_stkpnts: int  # 1
hxe_structural: int  # 9
hxe_switch_pseudocode: int  # 101
hxe_text_ready: int  # 109
ida_funcs: module
ida_gdl: module
ida_idaapi: module
ida_idp: module
ida_pro: module
ida_range: module
ida_typeinf: module
ida_xref: module
k: str  # voff_t_undef
lxe_lvar_cmt_changed: int  # 113
lxe_lvar_mapping_changed: int  # 114
lxe_lvar_name_changed: int  # 111
lxe_lvar_type_changed: int  # 112
m_add: int  # 12
m_and: int  # 20
m_bnot: int  # 7
m_call: int  # 56
m_cfadd: int  # 25
m_cfshl: int  # 27
m_cfshr: int  # 28
m_ext: int  # 62
m_f2f: int  # 67
m_f2i: int  # 63
m_f2u: int  # 64
m_fadd: int  # 69
m_fdiv: int  # 72
m_fmul: int  # 71
m_fneg: int  # 68
m_fsub: int  # 70
m_goto: int  # 55
m_high: int  # 11
m_i2f: int  # 65
m_icall: int  # 57
m_ijmp: int  # 54
m_ja: int  # 47
m_jae: int  # 45
m_jb: int  # 46
m_jbe: int  # 48
m_jcnd: int  # 42
m_jg: int  # 49
m_jge: int  # 50
m_jl: int  # 51
m_jle: int  # 52
m_jnz: int  # 43
m_jtbl: int  # 53
m_jz: int  # 44
m_ldc: int  # 3
m_ldx: int  # 2
m_lnot: int  # 6
m_low: int  # 10
m_mov: int  # 4
m_mul: int  # 14
m_neg: int  # 5
m_nop: int  # 0
m_ofadd: int  # 26
m_or: int  # 19
m_pop: int  # 60
m_push: int  # 59
m_ret: int  # 58
m_sar: int  # 24
m_sdiv: int  # 16
m_seta: int  # 36
m_setae: int  # 34
m_setb: int  # 35
m_setbe: int  # 37
m_setg: int  # 38
m_setge: int  # 39
m_setl: int  # 40
m_setle: int  # 41
m_setnz: int  # 32
m_seto: int  # 30
m_setp: int  # 31
m_sets: int  # 29
m_setz: int  # 33
m_shl: int  # 22
m_shr: int  # 23
m_smod: int  # 18
m_stx: int  # 1
m_sub: int  # 13
m_u2f: int  # 66
m_udiv: int  # 15
m_umod: int  # 17
m_und: int  # 61
m_xds: int  # 8
m_xdu: int  # 9
m_xor: int  # 21
mop_S: int  # 5
mop_a: int  # 10
mop_b: int  # 7
mop_c: int  # 12
mop_d: int  # 4
mop_f: int  # 8
mop_fn: int  # 13
mop_h: int  # 11
mop_l: int  # 9
mop_n: int  # 2
mop_p: int  # 14
mop_r: int  # 1
mop_sc: int  # 15
mop_str: int  # 3
mop_v: int  # 6
mop_z: int  # 0
mr_cc: int  # 5
mr_cf: int  # 0
mr_first: int  # 8
mr_none: int  # -1
mr_of: int  # 3
mr_pf: int  # 4
mr_sf: int  # 2
mr_zf: int  # 1
weakref: module