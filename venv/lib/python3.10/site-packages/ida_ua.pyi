from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with the disassembling of program instructions.

There are 2 kinds of functions:
* functions that are called from the kernel to disassemble an instruction. These functions call IDP module for it.
* functions that are called from IDP module to disassemble an instruction. We will call them 'helper functions'.


Disassembly of an instruction is made in three steps:
0. analysis: ana.cpp
1. emulation: emu.cpp
2. conversion to text: out.cpp


The kernel calls the IDP module to perform these steps. At first, the kernel always calls the analysis. The analyzer must decode the instruction and fill the insn_t instance that it receives through its callback. It must not change anything in the database.
The second step, the emulation, is called for each instruction. This step must make necessary changes to the database, plan analysis of subsequent instructions, track register values, memory contents, etc. Please keep in mind that the kernel may call the emulation step for any address in the program - there is no ordering of addresses. Usually, the emulation is called for consecutive addresses but this is not guaranteed.
The last step, conversion to text, is called each time an instruction is displayed on the screen. The kernel will always call the analysis step before calling the text conversion step. The emulation and the text conversion steps should use the information stored in the insn_t instance they receive. They should not access the bytes of the instruction and decode it again - this should only be done in the analysis step. 
    
"""

class insn_t:
    @property
    def Op1(self) -> Any: ...
    @property
    def Op2(self) -> Any: ...
    @property
    def Op3(self) -> Any: ...
    @property
    def Op4(self) -> Any: ...
    @property
    def Op5(self) -> Any: ...
    @property
    def Op6(self) -> Any: ...
    @property
    def Op7(self) -> Any: ...
    @property
    def Op8(self) -> Any: ...
    @property
    def auxpref(self) -> Any: ...
    @property
    def auxpref_u16(self) -> Any: ...
    @property
    def auxpref_u8(self) -> Any: ...
    @property
    def cs(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def insnpref(self) -> Any: ...
    @property
    def ip(self) -> Any: ...
    @property
    def itype(self) -> Any: ...
    @property
    def ops(self) -> Any: ...
    @property
    def segpref(self) -> Any: ...
    @property
    def size(self) -> Any: ...
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
    def __get_auxpref__(self) -> int:
        ...
    def __get_operand__(self, n: int) -> op_t:
        ...
    def __get_ops__(self) -> Any:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, idx: Any) -> op_t:
        r"""
                Operands can be accessed directly as indexes
        
                :returns: an operand of type op_t
                
        """
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
    def __iter__(self) -> Any:
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
    def __set_auxpref__(self, v: int) -> None:
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
    def add_cref(self, to: ida_idaapi.ea_t, opoff: int, type: cref_t) -> None:
        r"""Add a code cross-reference from the instruction. 
                
        :param to: target linear address
        :param opoff: offset of the operand from the start of instruction. if the offset is unknown, then 0.
        :param type: type of xref
        """
        ...
    def add_dref(self, to: ida_idaapi.ea_t, opoff: int, type: dref_t) -> None:
        r"""Add a data cross-reference from the instruction. See add_off_drefs() - usually it can be used in most cases. 
                
        :param to: target linear address
        :param opoff: offset of the operand from the start of instruction if the offset is unknown, then 0
        :param type: type of xref
        """
        ...
    def add_off_drefs(self, x: op_t, type: dref_t, outf: int) -> ida_idaapi.ea_t:
        r"""Add xrefs for an operand of the instruction. This function creates all cross references for 'enum', 'offset' and 'structure offset' operands. Use add_off_drefs() in the presence of negative offsets. 
                
        :param x: reference to operand
        :param type: type of xref
        :param outf: out_value() flags. These flags should match the flags used to output the operand
        :returns: if: is_off(): the reference target address (the same as calc_reference_data).
        :returns: if: is_stroff(): BADADDR because for stroffs the target address is unknown
        :returns: otherwise: BADADDR because enums do not represent addresses
        """
        ...
    def assign(self, other: insn_t) -> None:
        ...
    def create_op_data(self, args: Any) -> bool:
        ...
    def create_stkvar(self, x: op_t, v: adiff_t, flags_: int) -> bool:
        ...
    def get_canon_feature(self, args: Any) -> int:
        r"""see instruc_t::feature
        
        """
        ...
    def get_canon_mnem(self, args: Any) -> str:
        r"""see instruc_t::name
        
        """
        ...
    def get_next_byte(self) -> uint8:
        ...
    def get_next_dword(self) -> int:
        ...
    def get_next_qword(self) -> uint64:
        ...
    def get_next_word(self) -> uint16:
        ...
    def is_64bit(self) -> bool:
        r"""Belongs to a 64bit segment?
        
        """
        ...
    def is_canon_insn(self, args: Any) -> bool:
        r"""see processor_t::is_canon_insn()
        
        """
        ...
    def is_macro(self) -> bool:
        r"""Is a macro instruction?
        
        """
        ...

class macro_constructor_t:
    @property
    def reserved(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def build_macro(self, insn: insn_t, may_go_forward: bool) -> bool:
        r"""Try to extend the instruction.
        This function may modify 'insn' and return false; these changes will be accepted by the kernel but the instruction will not be considered as a macro.
        
        :param insn: Instruction to modify, usually the first instruction of the macro
        :param may_go_forward: Is it ok to consider the next instruction for the macro? This argument may be false, for example, if there is a cross reference to the end of INSN. In this case creating a macro is not desired. However, it may still be useful to perform minor tweaks to the instruction using the information about the surrounding instructions.
        :returns: true if created an macro instruction.
        """
        ...
    def construct_macro(self, insn: insn_t, enable: bool) -> bool:
        r"""Construct a macro instruction. This function may be called from ana() to generate a macro instruction.
        The real work is done by the 'build_macro()' virtual function. It must be defined by the processor module.
        construct_macro() modifies the database using the info provided by build_macro(). It verifies if the instruction can really be created (for example, that other items do not hinder), may plan to reanalyze the macro, etc. If the macro instructions are disabled by the user, construct_macro() will destroy the macro instruction. Note: if INSN_MODMAC is not set in insn.flags, the database will not be modified.
        
        :param insn: the instruction to modify into a macro
        :param enable: enable macro generation
        :returns: true: the macro instruction is generated in 'insn'
        :returns: false: did not create a macro
        """
        ...

class op_t:
    @property
    def addr(self) -> Any: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def n(self) -> Any: ...
    @property
    def offb(self) -> Any: ...
    @property
    def offo(self) -> Any: ...
    @property
    def phrase(self) -> Any: ...
    @property
    def reg(self) -> Any: ...
    @property
    def specflag1(self) -> Any: ...
    @property
    def specflag2(self) -> Any: ...
    @property
    def specflag3(self) -> Any: ...
    @property
    def specflag4(self) -> Any: ...
    @property
    def specval(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    @property
    def value64(self) -> Any: ...
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
    def __get_addr__(self) -> ida_idaapi.ea_t:
        ...
    def __get_reg_phrase__(self) -> uint16:
        ...
    def __get_specval__(self) -> ida_idaapi.ea_t:
        ...
    def __get_value64__(self) -> uint64:
        ...
    def __get_value__(self) -> ida_idaapi.ea_t:
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
    def __set_addr__(self, v: ida_idaapi.ea_t) -> None:
        ...
    def __set_reg_phrase__(self, r: uint16) -> None:
        ...
    def __set_specval__(self, v: ida_idaapi.ea_t) -> None:
        ...
    def __set_value64__(self, v: uint64) -> None:
        ...
    def __set_value__(self, v: ida_idaapi.ea_t) -> None:
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
    def assign(self, other: op_t) -> None:
        ...
    def clr_shown(self) -> None:
        r"""Set operand to hidden.
        
        """
        ...
    def has_reg(self, r: Any) -> Any:
        r"""Checks if the operand accesses the given processor register"""
        ...
    def is_imm(self, v: int) -> bool:
        r"""Is immediate operand?
        
        """
        ...
    def is_reg(self, r: int) -> bool:
        r"""Is register operand?
        
        """
        ...
    def set_shown(self) -> None:
        r"""Set operand to be shown.
        
        """
        ...
    def shown(self) -> bool:
        r"""Is operand set to be shown?
        
        """
        ...

class operands_array:
    @property
    def bytes(self) -> Any: ...
    @property
    def data(self) -> Any: ...
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
    def __getitem__(self, i: size_t) -> op_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, data: Any) -> Any:
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
    def __le__(self, value: Any) -> Any:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
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
    def __setitem__(self, i: size_t, v: op_t) -> None:
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

class outctx_base_t:
    @property
    def F32(self) -> Any: ...
    @property
    def default_lnnum(self) -> Any: ...
    @property
    def insn_ea(self) -> Any: ...
    @property
    def outbuf(self) -> Any: ...
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
    def __init__(self, args: Any, kwargs: Any) -> Any:
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
    def close_comment(self) -> None:
        ...
    def clr_gen_label(self) -> None:
        ...
    def display_hidden(self) -> bool:
        ...
    def display_voids(self) -> bool:
        ...
    def flush_buf(self, buf: str, indent: int = -1) -> bool:
        r"""Append contents of 'buf' to the line array. Behaves like flush_outbuf but accepts an arbitrary buffer 
                
        """
        ...
    def flush_outbuf(self, indent: int = -1) -> bool:
        r"""------------------------------------------------------------------------- Functions to populate the output line array (lnar) Move the contents of the output buffer to the line array (outbuf->lnar) The kernel augments the outbuf contents with additional text like the line prefix, user-defined comments, xrefs, etc at this call. 
                
        """
        ...
    def forbid_annotations(self) -> int:
        ...
    def force_code(self) -> bool:
        ...
    def gen_block_cmt(self, cmt: str, color: color_t) -> bool:
        r"""Generate big non-indented comment lines. 
                
        :param cmt: comment text. may contain \n characters to denote new lines. should not contain comment character (;)
        :param color: color of comment text (one of Color tags)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_border_line(self, solid: bool = False) -> bool:
        r"""Generate thin border line. This function does nothing if generation of border lines is disabled. 
                
        :param solid: generate solid border line (with =), otherwise with -
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_cmt_line(self, format: str) -> bool:
        r"""Generate one non-indented comment line, colored with COLOR_AUTOCMT. 
                
        :param format: printf() style format line. The resulting comment line should not include comment character (;)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_collapsed_line(self, format: str) -> bool:
        r"""Generate one non-indented comment line, colored with COLOR_COLLAPSED. 
                
        :param format: printf() style format line. The resulting comment line should not include comment character (;)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_empty_line(self) -> bool:
        r"""Generate empty line. This function does nothing if generation of empty lines is disabled. 
                
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_empty_line_without_annotations(self) -> None:
        ...
    def gen_printf(self, indent: int, format: str) -> bool:
        r"""printf-like function to add lines to the line array. 
                
        :param indent: indention of the line. if indent == -1, the kernel will indent the line at idainfo::indent. if indent < 0, -indent will be used for indention. The first line printed with indent < 0 is considered as the most important line at the current address. Usually it is the line with the instruction itself. This line will be displayed in the cross-reference lists and other places. If you need to output an additional line before the main line then pass DEFAULT_INDENT instead of -1. The kernel will know that your line is not the most important one.
        :param format: printf style colored line to generate
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_xref_lines(self) -> bool:
        ...
    def getF(self) -> flags64_t:
        ...
    def get_stkvar(self, x: op_t, v: int, vv: sval_t, is_sp_based: int, _frame: tinfo_t) -> ssize_t:
        ...
    def init_lines_array(self, answers: qstrvec_t, maxsize: int) -> None:
        ...
    def multiline(self) -> bool:
        ...
    def only_main_line(self) -> bool:
        ...
    def out_addr_tag(self, ea: ida_idaapi.ea_t) -> None:
        r"""Output "address" escape sequence.
        
        """
        ...
    def out_btoa(self, Word: int, radix: char = 0) -> None:
        r"""Output a number with the specified base (binary, octal, decimal, hex) The number is output without color codes. see also out_long() 
                
        """
        ...
    def out_char(self, c: char) -> None:
        r"""Output one character. The character is output without color codes. see also out_symbol() 
                
        """
        ...
    def out_chars(self, c: char, n: int) -> None:
        r"""Append a character multiple times.
        
        """
        ...
    def out_colored_register_line(self, str: str) -> None:
        r"""Output a colored line with register names in it. The register names will be substituted by user-defined names (regvar_t) Please note that out_tagoff tries to make substitutions too (when called with COLOR_REG) 
                
        """
        ...
    def out_keyword(self, str: str) -> None:
        r"""Output a string with COLOR_KEYWORD color.
        
        """
        ...
    def out_line(self, str: str, color: color_t = 0) -> None:
        r"""Output a string with the specified color.
        
        """
        ...
    def out_long(self, v: int, radix: char) -> None:
        r"""Output a number with appropriate color. Low level function. Use out_value() if you can. if 'suspop' is set then this function uses COLOR_VOIDOP instead of COLOR_NUMBER. 'suspop' is initialized:
        * in out_one_operand()
        * in ..\ida\gl.cpp (before calling processor_t::d_out())
        
        
        
        :param v: value to output
        :param radix: base (2,8,10,16)
        """
        ...
    def out_lvar(self, name: str, width: int = -1) -> None:
        r"""Output local variable name with COLOR_LOCNAME color.
        
        """
        ...
    def out_name_expr(self, args: Any) -> bool:
        r"""Output a name expression. 
                
        :param x: instruction operand referencing the name expression
        :param ea: address to convert to name expression
        :param off: the value of name expression. this parameter is used only to check that the name expression will have the wanted value. You may pass BADADDR for this parameter but I discourage it because it prohibits checks.
        :returns: true if the name expression has been produced
        """
        ...
    def out_printf(self, format: str) -> int:
        r"""------------------------------------------------------------------------- Functions to append text to the current output buffer (outbuf) Append a formatted string to the output string. 
                
        :returns: the number of characters appended
        """
        ...
    def out_register(self, str: str) -> None:
        r"""Output a character with COLOR_REG color.
        
        """
        ...
    def out_spaces(self, len: ssize_t) -> None:
        r"""Appends spaces to outbuf until its tag_strlen becomes 'len'.
        
        """
        ...
    def out_symbol(self, c: char) -> None:
        r"""Output a character with COLOR_SYMBOL color.
        
        """
        ...
    def out_tagoff(self, tag: color_t) -> None:
        r"""Output "turn color off" escape sequence.
        
        """
        ...
    def out_tagon(self, tag: color_t) -> None:
        r"""Output "turn color on" escape sequence.
        
        """
        ...
    def out_value(self, x: op_t, outf: int = 0) -> flags64_t:
        r"""Output immediate value. Try to use this function to output all constants of instruction operands. This function outputs a number from x.addr or x.value in the form determined by F. It outputs colored text. 
                
        :param x: value to output
        :param outf: Output value flags
        :returns: flags of the output value, otherwise:
        :returns: -1: if printed a number with COLOR_ERROR
        :returns: 0: if printed a nice number or character or segment or enum
        """
        ...
    def print_label_now(self) -> bool:
        ...
    def restore_ctxflags(self, saved_flags: int) -> None:
        ...
    def retrieve_cmt(self) -> ssize_t:
        ...
    def retrieve_name(self, arg2: str, arg3: color_t) -> ssize_t:
        ...
    def set_comment_addr(self, ea: ida_idaapi.ea_t) -> None:
        ...
    def set_dlbind_opnd(self) -> None:
        ...
    def set_gen_cmt(self, on: bool = True) -> None:
        ...
    def set_gen_demangled_label(self) -> None:
        ...
    def set_gen_label(self) -> None:
        ...
    def set_gen_xrefs(self, on: bool = True) -> None:
        ...
    def setup_outctx(self, prefix: str, makeline_flags: int) -> None:
        r"""Initialization; normally used only by the kernel.
        
        """
        ...
    def stack_view(self) -> bool:
        ...
    def term_outctx(self, prefix: str = None) -> int:
        r"""Finalize the output context. 
                
        :returns: the number of generated lines.
        """
        ...

class outctx_t(outctx_base_t):
    @property
    def F32(self) -> Any: ...
    @property
    def ash(self) -> Any: ...
    @property
    def bin_ea(self) -> Any: ...
    @property
    def bin_state(self) -> Any: ...
    @property
    def bin_width(self) -> Any: ...
    @property
    def curlabel(self) -> Any: ...
    @property
    def default_lnnum(self) -> Any: ...
    @property
    def gl_bpsize(self) -> Any: ...
    @property
    def insn(self) -> Any: ...
    @property
    def insn_ea(self) -> Any: ...
    @property
    def next_line_ea(self) -> Any: ...
    @property
    def outbuf(self) -> Any: ...
    @property
    def ph(self) -> Any: ...
    @property
    def prefix_ea(self) -> Any: ...
    @property
    def procmod(self) -> Any: ...
    @property
    def saved_immvals(self) -> Any: ...
    @property
    def wif(self) -> Any: ...
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
    def __init__(self, args: Any, kwargs: Any) -> Any:
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
    def close_comment(self) -> None:
        ...
    def clr_gen_label(self) -> None:
        ...
    def display_hidden(self) -> bool:
        ...
    def display_voids(self) -> bool:
        ...
    def flush_buf(self, buf: str, indent: int = -1) -> bool:
        r"""Append contents of 'buf' to the line array. Behaves like flush_outbuf but accepts an arbitrary buffer 
                
        """
        ...
    def flush_outbuf(self, indent: int = -1) -> bool:
        r"""------------------------------------------------------------------------- Functions to populate the output line array (lnar) Move the contents of the output buffer to the line array (outbuf->lnar) The kernel augments the outbuf contents with additional text like the line prefix, user-defined comments, xrefs, etc at this call. 
                
        """
        ...
    def forbid_annotations(self) -> int:
        ...
    def force_code(self) -> bool:
        ...
    def gen_block_cmt(self, cmt: str, color: color_t) -> bool:
        r"""Generate big non-indented comment lines. 
                
        :param cmt: comment text. may contain \n characters to denote new lines. should not contain comment character (;)
        :param color: color of comment text (one of Color tags)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_border_line(self, solid: bool = False) -> bool:
        r"""Generate thin border line. This function does nothing if generation of border lines is disabled. 
                
        :param solid: generate solid border line (with =), otherwise with -
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_cmt_line(self, format: str) -> bool:
        r"""Generate one non-indented comment line, colored with COLOR_AUTOCMT. 
                
        :param format: printf() style format line. The resulting comment line should not include comment character (;)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_collapsed_line(self, format: str) -> bool:
        r"""Generate one non-indented comment line, colored with COLOR_COLLAPSED. 
                
        :param format: printf() style format line. The resulting comment line should not include comment character (;)
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_empty_line(self) -> bool:
        r"""Generate empty line. This function does nothing if generation of empty lines is disabled. 
                
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_empty_line_without_annotations(self) -> None:
        ...
    def gen_func_footer(self, pfn: func_t) -> None:
        ...
    def gen_func_header(self, pfn: func_t) -> None:
        ...
    def gen_header(self, args: Any) -> None:
        ...
    def gen_header_extra(self) -> None:
        ...
    def gen_printf(self, indent: int, format: str) -> bool:
        r"""printf-like function to add lines to the line array. 
                
        :param indent: indention of the line. if indent == -1, the kernel will indent the line at idainfo::indent. if indent < 0, -indent will be used for indention. The first line printed with indent < 0 is considered as the most important line at the current address. Usually it is the line with the instruction itself. This line will be displayed in the cross-reference lists and other places. If you need to output an additional line before the main line then pass DEFAULT_INDENT instead of -1. The kernel will know that your line is not the most important one.
        :param format: printf style colored line to generate
        :returns: overflow, lnar_maxsize has been reached
        """
        ...
    def gen_xref_lines(self) -> bool:
        ...
    def getF(self) -> flags64_t:
        ...
    def get_stkvar(self, x: op_t, v: int, vv: sval_t, is_sp_based: int, _frame: tinfo_t) -> ssize_t:
        ...
    def init_lines_array(self, answers: qstrvec_t, maxsize: int) -> None:
        ...
    def multiline(self) -> bool:
        ...
    def only_main_line(self) -> bool:
        ...
    def out_addr_tag(self, ea: ida_idaapi.ea_t) -> None:
        r"""Output "address" escape sequence.
        
        """
        ...
    def out_btoa(self, Word: int, radix: char = 0) -> None:
        r"""Output a number with the specified base (binary, octal, decimal, hex) The number is output without color codes. see also out_long() 
                
        """
        ...
    def out_char(self, c: char) -> None:
        r"""Output one character. The character is output without color codes. see also out_symbol() 
                
        """
        ...
    def out_chars(self, c: char, n: int) -> None:
        r"""Append a character multiple times.
        
        """
        ...
    def out_colored_register_line(self, str: str) -> None:
        r"""Output a colored line with register names in it. The register names will be substituted by user-defined names (regvar_t) Please note that out_tagoff tries to make substitutions too (when called with COLOR_REG) 
                
        """
        ...
    def out_custom_mnem(self, mnem: str, width: int = 8, postfix: str = None) -> None:
        r"""Output custom mnemonic for 'insn'. E.g. if it should differ from the one in 'ph.instruc'. This function outputs colored text. See out_mnem 
                
        :param mnem: custom mnemonic
        :param width: width of field with mnemonic. if < 0, then 'postfix' will be output before the mnemonic, i.e. as a prefix
        :param postfix: optional postfix added to 'mnem'
        """
        ...
    def out_data(self, analyze_only: bool) -> None:
        ...
    def out_fcref_names(self) -> None:
        r"""Print addresses referenced *from* the specified address as commented symbolic names. This function is used to show, for example, multiple callees of an indirect call. This function outputs colored text. 
                
        """
        ...
    def out_immchar_cmts(self) -> None:
        r"""Print all operand values as commented character constants. This function is used to comment void operands with their representation in the form of character constants. This function outputs colored text. 
                
        """
        ...
    def out_keyword(self, str: str) -> None:
        r"""Output a string with COLOR_KEYWORD color.
        
        """
        ...
    def out_line(self, str: str, color: color_t = 0) -> None:
        r"""Output a string with the specified color.
        
        """
        ...
    def out_long(self, v: int, radix: char) -> None:
        r"""Output a number with appropriate color. Low level function. Use out_value() if you can. if 'suspop' is set then this function uses COLOR_VOIDOP instead of COLOR_NUMBER. 'suspop' is initialized:
        * in out_one_operand()
        * in ..\ida\gl.cpp (before calling processor_t::d_out())
        
        
        
        :param v: value to output
        :param radix: base (2,8,10,16)
        """
        ...
    def out_lvar(self, name: str, width: int = -1) -> None:
        r"""Output local variable name with COLOR_LOCNAME color.
        
        """
        ...
    def out_mnem(self, width: int = 8, postfix: str = None) -> None:
        r"""Output instruction mnemonic for 'insn' using information in 'ph.instruc' array. This function outputs colored text. It should be called from processor_t::ev_out_insn() or processor_t::ev_out_mnem() handler. It will output at least one space after the instruction. mnemonic even if the specified 'width' is not enough. 
                
        :param width: width of field with mnemonic. if < 0, then 'postfix' will be output before the mnemonic, i.e. as a prefix
        :param postfix: optional postfix added to the instruction mnemonic
        """
        ...
    def out_mnemonic(self) -> None:
        r"""Output instruction mnemonic using information in 'insn'. It should be called from processor_t::ev_out_insn() and it will call processor_t::ev_out_mnem() or out_mnem. This function outputs colored text. 
                
        """
        ...
    def out_name_expr(self, args: Any) -> bool:
        r"""Output a name expression. 
                
        :param x: instruction operand referencing the name expression
        :param ea: address to convert to name expression
        :param off: the value of name expression. this parameter is used only to check that the name expression will have the wanted value. You may pass BADADDR for this parameter but I discourage it because it prohibits checks.
        :returns: true if the name expression has been produced
        """
        ...
    def out_one_operand(self, n: int) -> bool:
        r"""Use this function to output an operand of an instruction. This function checks for the existence of a manually defined operand and will output it if it exists. It should be called from processor_t::ev_out_insn() and it will call processor_t::ev_out_operand(). This function outputs colored text. 
                
        :param n: 0..UA_MAXOP-1 operand number
        :returns: 1: operand is displayed
        :returns: 0: operand is hidden
        """
        ...
    def out_printf(self, format: str) -> int:
        r"""------------------------------------------------------------------------- Functions to append text to the current output buffer (outbuf) Append a formatted string to the output string. 
                
        :returns: the number of characters appended
        """
        ...
    def out_register(self, str: str) -> None:
        r"""Output a character with COLOR_REG color.
        
        """
        ...
    def out_spaces(self, len: ssize_t) -> None:
        r"""Appends spaces to outbuf until its tag_strlen becomes 'len'.
        
        """
        ...
    def out_specea(self, segtype: uchar) -> bool:
        ...
    def out_symbol(self, c: char) -> None:
        r"""Output a character with COLOR_SYMBOL color.
        
        """
        ...
    def out_tagoff(self, tag: color_t) -> None:
        r"""Output "turn color off" escape sequence.
        
        """
        ...
    def out_tagon(self, tag: color_t) -> None:
        r"""Output "turn color on" escape sequence.
        
        """
        ...
    def out_value(self, x: op_t, outf: int = 0) -> flags64_t:
        r"""Output immediate value. Try to use this function to output all constants of instruction operands. This function outputs a number from x.addr or x.value in the form determined by F. It outputs colored text. 
                
        :param x: value to output
        :param outf: Output value flags
        :returns: flags of the output value, otherwise:
        :returns: -1: if printed a number with COLOR_ERROR
        :returns: 0: if printed a nice number or character or segment or enum
        """
        ...
    def print_label_now(self) -> bool:
        ...
    def restore_ctxflags(self, saved_flags: int) -> None:
        ...
    def retrieve_cmt(self) -> ssize_t:
        ...
    def retrieve_name(self, arg2: str, arg3: color_t) -> ssize_t:
        ...
    def set_bin_state(self, value: int) -> None:
        ...
    def set_comment_addr(self, ea: ida_idaapi.ea_t) -> None:
        ...
    def set_dlbind_opnd(self) -> None:
        ...
    def set_gen_cmt(self, on: bool = True) -> None:
        ...
    def set_gen_demangled_label(self) -> None:
        ...
    def set_gen_label(self) -> None:
        ...
    def set_gen_xrefs(self, on: bool = True) -> None:
        ...
    def setup_outctx(self, prefix: str, flags: int) -> None:
        r"""Initialization; normally used only by the kernel.
        
        """
        ...
    def stack_view(self) -> bool:
        ...
    def term_outctx(self, prefix: str = None) -> int:
        r"""Finalize the output context. 
                
        :returns: the number of generated lines.
        """
        ...

def calc_dataseg(insn: insn_t, n: int = -1, rgnum: int = -1) -> ida_idaapi.ea_t:
    ...

def can_decode(ea: ida_idaapi.ea_t) -> bool:
    r"""Can the bytes at address 'ea' be decoded as instruction? 
            
    :param ea: linear address
    :returns: whether or not the contents at that address could be a valid instruction
    """
    ...

def construct_macro(args: Any) -> bool:
    r"""See ua.hpp's construct_macro().
    
    This function has the following signatures
    
        1. construct_macro(insn: insn_t, enable: bool, build_macro: callable) -> bool
        2. construct_macro(constuctor: macro_constructor_t, insn: insn_t, enable: bool) -> bool
    
    :param insn: the instruction to build the macro for
    :param enable: enable macro generation
    :param build_macro: a callable with 2 arguments: an insn_t, and
                        whether it is ok to consider the next instruction
                        for the macro
    :param constructor: a macro_constructor_t implementation
    :returns: success
    """
    ...

def create_insn(ea: ida_idaapi.ea_t, out: insn_t = None) -> int:
    r"""Create an instruction at the specified address. This function checks if an instruction is present at the specified address and will try to create one if there is none. It will fail if there is a data item or other items hindering the creation of the new instruction. This function will also fill the 'out' structure. 
            
    :param ea: linear address
    :param out: the resulting instruction
    :returns: the length of the instruction or 0
    """
    ...

def create_outctx(ea: ida_idaapi.ea_t, F: flags64_t = 0, suspop: int = 0) -> outctx_base_t:
    r"""Create a new output context. To delete it, just use "delete pctx" 
            
    """
    ...

def decode_insn(out: insn_t, ea: ida_idaapi.ea_t) -> int:
    r"""Analyze the specified address and fill 'out'. This function does not modify the database. It just tries to interpret the specified address as an instruction and fills the 'out' structure. 
            
    :param out: the resulting instruction
    :param ea: linear address
    :returns: the length of the (possible) instruction or 0
    """
    ...

def decode_preceding_insn(out: insn_t, ea: ida_idaapi.ea_t) -> Any:
    r"""Decodes the preceding instruction.
    
    :param out: instruction storage
    :param ea: current ea
    :returns: tuple(preceeding_ea or BADADDR, farref = Boolean)
    """
    ...

def decode_prev_insn(out: insn_t, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Decode previous instruction if it exists, fill 'out'. 
            
    :param out: the resulting instruction
    :param ea: the address to decode the previous instruction from
    :returns: the previous instruction address (BADADDR-no such insn)
    """
    ...

def get_dtype_by_size(size: asize_t) -> int:
    r"""Get op_t::dtype from size.
    
    """
    ...

def get_dtype_flag(dtype: op_dtype_t) -> flags64_t:
    r"""Get flags for op_t::dtype field.
    
    """
    ...

def get_dtype_size(dtype: op_dtype_t) -> int:
    r"""Get size of opt_::dtype field.
    
    """
    ...

def get_immvals(ea: ida_idaapi.ea_t, n: int, F: flags64_t = 0) -> Any:
    r"""Get immediate values at the specified address. This function decodes instruction at the specified address or inspects the data item. It finds immediate values and copies them to 'out'. This function will store the original value of the operands in 'out', unless the last bits of 'F' are "...0 11111111", in which case the transformed values (as needed for printing) will be stored instead. 
            
    :param ea: address to analyze
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all the operands
    :param F: flags for the specified address
    :returns: number of immediate values (0..2*UA_MAXOP)
    """
    ...

def get_lookback() -> int:
    r"""Number of instructions to look back. This variable is not used by the kernel. Its value may be specified in ida.cfg: LOOKBACK = <number>. IDP may use it as you like it. (TMS module uses it) 
            
    """
    ...

def get_printable_immvals(ea: ida_idaapi.ea_t, n: int, F: flags64_t = 0) -> Any:
    r"""Get immediate ready-to-print values at the specified address 
            
    :param ea: address to analyze
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all the operands
    :param F: flags for the specified address
    :returns: number of immediate values (0..2*UA_MAXOP)
    """
    ...

def insn_add_cref(insn: insn_t, to: ida_idaapi.ea_t, opoff: int, type: cref_t) -> None:
    ...

def insn_add_dref(insn: insn_t, to: ida_idaapi.ea_t, opoff: int, type: dref_t) -> None:
    ...

def insn_add_off_drefs(insn: insn_t, x: op_t, type: dref_t, outf: int) -> ida_idaapi.ea_t:
    ...

def insn_create_stkvar(insn: insn_t, x: op_t, v: adiff_t, flags: int) -> bool:
    ...

def insn_t__from_ptrval__(ptrval: size_t) -> Optional[insn_t]:
    ...

def is_floating_dtype(dtype: op_dtype_t) -> bool:
    r"""Is a floating type operand?
    
    """
    ...

def map_code_ea(args: Any) -> ida_idaapi.ea_t:
    ...

def map_data_ea(args: Any) -> ida_idaapi.ea_t:
    ...

def map_ea(args: Any) -> ida_idaapi.ea_t:
    ...

def op_t__from_ptrval__(ptrval: size_t) -> op_t:
    ...

def outctx_base_t__from_ptrval__(ptrval: size_t) -> outctx_base_t:
    ...

def outctx_t__from_ptrval__(ptrval: size_t) -> outctx_t:
    ...

def print_insn_mnem(ea: ida_idaapi.ea_t) -> str:
    r"""Print instruction mnemonics. 
            
    :param ea: linear address of the instruction
    :returns: success
    """
    ...

def print_operand(ea: ida_idaapi.ea_t, n: int, getn_flags: int = 0, newtype: printop_t = None) -> str:
    r"""Generate text representation for operand #n. This function will generate the text representation of the specified operand (includes color codes.) 
            
    :param ea: the item address (instruction or data)
    :param n: 0..UA_MAXOP-1 operand number, meaningful only for instructions
    :param getn_flags: Name expression flags Currently only GETN_NODUMMY is accepted.
    :param newtype: if specified, print the operand using the specified type
    :returns: success
    """
    ...

def ua_mnem(ea: ida_idaapi.ea_t) -> str:
    r"""Print instruction mnemonics. 
            
    :param ea: linear address of the instruction
    :returns: success
    """
    ...

BINOPSTATE_DONE: int  # 524288
BINOPSTATE_GO: int  # 262144
BINOPSTATE_NONE: int  # 0
COMMSTATE_DONE: int  # 512
COMMSTATE_GO: int  # 256
COMMSTATE_NONE: int  # 0
CTXF_BINOP_STATE: int  # 786432
CTXF_BIT_PREFIX: int  # 2097152
CTXF_CMT_STATE: int  # 768
CTXF_CODE: int  # 4
CTXF_DBLIND_OPND: int  # 131072
CTXF_DEMANGLED_LABEL: int  # 4096
CTXF_DEMANGLED_OK: int  # 16384
CTXF_GEN_CMT: int  # 128
CTXF_GEN_XREFS: int  # 16
CTXF_HIDDEN_ADDR: int  # 1048576
CTXF_LABEL_OK: int  # 8192
CTXF_MAIN: int  # 1
CTXF_MULTI: int  # 2
CTXF_NORMAL_LABEL: int  # 2048
CTXF_OUTCTX_T: int  # 65536
CTXF_OVSTORE_PRNT: int  # 32768
CTXF_STACK: int  # 8
CTXF_UNHIDE: int  # 4194304
CTXF_VOIDS: int  # 1024
CTXF_XREF_STATE: int  # 96
DEFAULT_INDENT: int  # 65535
FCBF_CONT: int  # 1
FCBF_DELIM: int  # 8
FCBF_ERR_REPL: int  # 2
FCBF_FF_LIT: int  # 4
GH_BYTESEX_HAS_HIGHBYTE: int  # 16
GH_PRINT_ALL: int  # 15
GH_PRINT_ALL_BUT_BYTESEX: int  # 11
GH_PRINT_ASM: int  # 2
GH_PRINT_BYTESEX: int  # 4
GH_PRINT_HEADER: int  # 8
GH_PRINT_PROC: int  # 1
GH_PRINT_PROC_AND_ASM: int  # 3
GH_PRINT_PROC_ASM_AND_BYTESEX: int  # 7
INSN_64BIT: int  # 4
INSN_MACRO: int  # 1
INSN_MODMAC: int  # 2
MAKELINE_BINPREF: int  # 1
MAKELINE_NONE: int  # 0
MAKELINE_STACK: int  # 4
MAKELINE_VOID: int  # 2
OF_NO_BASE_DISP: int  # 128
OF_NUMBER: int  # 16
OF_OUTER_DISP: int  # 64
OF_SHOW: int  # 8
OOFS_IFSIGN: int  # 0
OOFS_NEEDSIGN: int  # 2
OOFS_NOSIGN: int  # 1
OOFW_16: int  # 32
OOFW_24: int  # 48
OOFW_32: int  # 64
OOFW_64: int  # 80
OOFW_8: int  # 16
OOFW_IMM: int  # 0
OOF_ADDR: int  # 128
OOF_ANYSERIAL: int  # 4096
OOF_LZEROES: int  # 8192
OOF_NOBNOT: int  # 1024
OOF_NO_LZEROES: int  # 16384
OOF_NUMBER: int  # 8
OOF_OUTER: int  # 256
OOF_SIGNED: int  # 4
OOF_SIGNMASK: int  # 3
OOF_SPACES: int  # 2048
OOF_WIDTHMASK: int  # 112
OOF_ZSTROFF: int  # 512
PACK_FORM_DEF: int  # 32
STKVAR_KEEP_EXISTING: int  # 2
STKVAR_VALID_SIZE: int  # 1
SWIG_PYTHON_LEGACY_BOOL: int  # 1
XREFSTATE_DONE: int  # 64
XREFSTATE_GO: int  # 32
XREFSTATE_NONE: int  # 0
annotations: _Feature
cvar: swigvarlink
dt_bitfild: int  # 12
dt_byte: int  # 0
dt_byte16: int  # 8
dt_byte32: int  # 16
dt_byte64: int  # 17
dt_code: int  # 9
dt_double: int  # 4
dt_dword: int  # 2
dt_float: int  # 3
dt_fword: int  # 11
dt_half: int  # 18
dt_ldbl: int  # 15
dt_packreal: int  # 6
dt_qword: int  # 7
dt_string: int  # 13
dt_tbyte: int  # 5
dt_unicode: int  # 14
dt_void: int  # 10
dt_word: int  # 1
ida_idaapi: module
o_displ: int  # 4
o_far: int  # 6
o_idpspec0: int  # 8
o_idpspec1: int  # 9
o_idpspec2: int  # 10
o_idpspec3: int  # 11
o_idpspec4: int  # 12
o_idpspec5: int  # 13
o_imm: int  # 5
o_mem: int  # 2
o_near: int  # 7
o_phrase: int  # 3
o_reg: int  # 1
o_void: int  # 0
weakref: module