from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Contains the ::inf structure definition and some functions common to the whole IDA project.

The ::inf structure is saved in the database and contains information specific to the current program being disassembled. Initially it is filled with values from ida.cfg.
Although it is not a good idea to change values in ::inf structure (because you will overwrite values taken from ida.cfg), you are allowed to do it if you feel it necessary. 
    
"""

class compiler_info_t:
    @property
    def cm(self) -> Any: ...
    @property
    def defalign(self) -> Any: ...
    @property
    def id(self) -> Any: ...
    @property
    def size_b(self) -> Any: ...
    @property
    def size_e(self) -> Any: ...
    @property
    def size_i(self) -> Any: ...
    @property
    def size_l(self) -> Any: ...
    @property
    def size_ldbl(self) -> Any: ...
    @property
    def size_ll(self) -> Any: ...
    @property
    def size_s(self) -> Any: ...
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
    def get_cc(self) -> callcnv_t:
        ...
    def set_cc(self, cc: callcnv_t) -> None:
        ...

class idainfo:
    @property
    def abibits(self) -> Any: ...
    @property
    def abiname(self) -> Any: ...
    @property
    def af(self) -> Any: ...
    @property
    def af2(self) -> Any: ...
    @property
    def appcall_options(self) -> Any: ...
    @property
    def apptype(self) -> Any: ...
    @property
    def asmtype(self) -> Any: ...
    @property
    def baseaddr(self) -> Any: ...
    @property
    def bin_prefix_size(self) -> Any: ...
    @property
    def cc(self) -> Any: ...
    @property
    def cmt_indent(self) -> Any: ...
    @property
    def database_change_count(self) -> Any: ...
    @property
    def datatypes(self) -> Any: ...
    @property
    def demnames(self) -> Any: ...
    @property
    def filetype(self) -> Any: ...
    @property
    def highoff(self) -> Any: ...
    @property
    def indent(self) -> Any: ...
    @property
    def lenxref(self) -> Any: ...
    @property
    def lflags(self) -> Any: ...
    @property
    def listnames(self) -> Any: ...
    @property
    def long_demnames(self) -> Any: ...
    @property
    def lowoff(self) -> Any: ...
    @property
    def main(self) -> Any: ...
    @property
    def margin(self) -> Any: ...
    @property
    def maxEA(self) -> Any: ...
    @property
    def max_autoname_len(self) -> Any: ...
    @property
    def max_ea(self) -> Any: ...
    @property
    def maxref(self) -> Any: ...
    @property
    def minEA(self) -> Any: ...
    @property
    def min_ea(self) -> Any: ...
    @property
    def nametype(self) -> Any: ...
    @property
    def omax_ea(self) -> Any: ...
    @property
    def omin_ea(self) -> Any: ...
    @property
    def ostype(self) -> Any: ...
    @property
    def outflags(self) -> Any: ...
    @property
    def procName(self) -> Any: ...
    @property
    def procname(self) -> Any: ...
    @property
    def refcmtnum(self) -> Any: ...
    @property
    def s_cmtflg(self) -> Any: ...
    @property
    def s_genflags(self) -> Any: ...
    @property
    def s_limiter(self) -> Any: ...
    @property
    def s_prefflag(self) -> Any: ...
    @property
    def s_xrefflag(self) -> Any: ...
    @property
    def short_demnames(self) -> Any: ...
    @property
    def specsegs(self) -> Any: ...
    @property
    def start_cs(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def start_ip(self) -> Any: ...
    @property
    def start_sp(self) -> Any: ...
    @property
    def start_ss(self) -> Any: ...
    @property
    def strlit_break(self) -> Any: ...
    @property
    def strlit_flags(self) -> Any: ...
    @property
    def strlit_pref(self) -> Any: ...
    @property
    def strlit_sernum(self) -> Any: ...
    @property
    def strlit_zeroes(self) -> Any: ...
    @property
    def strtype(self) -> Any: ...
    @property
    def tag(self) -> Any: ...
    @property
    def type_xrefnum(self) -> Any: ...
    @property
    def version(self) -> Any: ...
    @property
    def xrefnum(self) -> Any: ...
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
    def big_arg_align(self, args: Any) -> Any:
        ...
    def gen_lzero(self, args: Any) -> Any:
        ...
    def gen_null(self, args: Any) -> Any:
        ...
    def gen_tryblks(self, args: Any) -> Any:
        ...
    def get_abiname(self) -> str:
        ...
    def get_demname_form(self, args: Any) -> Any:
        ...
    def get_pack_mode(self, args: Any) -> Any:
        ...
    def is_32bit(self, args: Any) -> Any:
        ...
    def is_64bit(self, args: Any) -> Any:
        ...
    def is_auto_enabled(self, args: Any) -> Any:
        ...
    def is_be(self, args: Any) -> Any:
        ...
    def is_dll(self, args: Any) -> Any:
        ...
    def is_flat_off32(self, args: Any) -> Any:
        ...
    def is_graph_view(self, args: Any) -> Any:
        ...
    def is_hard_float(self, args: Any) -> Any:
        ...
    def is_kernel_mode(self, args: Any) -> Any:
        ...
    def is_mem_aligned4(self, args: Any) -> Any:
        ...
    def is_snapshot(self, args: Any) -> Any:
        ...
    def is_wide_high_byte_first(self, args: Any) -> Any:
        ...
    def like_binary(self, args: Any) -> Any:
        ...
    def line_pref_with_seg(self, args: Any) -> Any:
        ...
    def loading_idc(self, args: Any) -> Any:
        ...
    def map_stkargs(self, args: Any) -> Any:
        ...
    def pack_stkargs(self, args: Any) -> Any:
        ...
    def readonly_idb(self, args: Any) -> Any:
        ...
    def set_64bit(self, args: Any) -> Any:
        ...
    def set_auto_enabled(self, args: Any) -> Any:
        ...
    def set_be(self, args: Any) -> Any:
        ...
    def set_gen_lzero(self, args: Any) -> Any:
        ...
    def set_gen_null(self, args: Any) -> Any:
        ...
    def set_gen_tryblks(self, args: Any) -> Any:
        ...
    def set_graph_view(self, args: Any) -> Any:
        ...
    def set_line_pref_with_seg(self, args: Any) -> Any:
        ...
    def set_pack_mode(self, args: Any) -> Any:
        ...
    def set_show_auto(self, args: Any) -> Any:
        ...
    def set_show_line_pref(self, args: Any) -> Any:
        ...
    def set_show_void(self, args: Any) -> Any:
        ...
    def set_wide_high_byte_first(self, args: Any) -> Any:
        ...
    def show_auto(self, args: Any) -> Any:
        ...
    def show_line_pref(self, args: Any) -> Any:
        ...
    def show_void(self, args: Any) -> Any:
        ...
    def stack_ldbl(self, args: Any) -> Any:
        ...
    def stack_varargs(self, args: Any) -> Any:
        ...
    def use_allasm(self, args: Any) -> Any:
        ...
    def use_gcc_layout(self, args: Any) -> Any:
        ...

class idbattr_info_t:
    @property
    def bitmask(self) -> Any: ...
    @property
    def idi_flags(self) -> Any: ...
    @property
    def individual_node(self) -> Any: ...
    @property
    def maxsize(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def offset(self) -> Any: ...
    @property
    def tag(self) -> Any: ...
    @property
    def vmap(self) -> Any: ...
    @property
    def width(self) -> Any: ...
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
    def __init__(self, name: str, offset: uintptr_t, width: size_t, bitmask: uint64 = 0, tag: uchar = 0, idi_flags: uint = 0) -> Any:
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
    def __lt__(self, r: idbattr_info_t) -> bool:
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
    def has_individual_node(self) -> bool:
        ...
    def hashname(self) -> str:
        ...
    def is_bitfield(self) -> bool:
        ...
    def is_bitmap(self) -> bool:
        ...
    def is_boolean(self) -> bool:
        ...
    def is_buf_var(self) -> bool:
        ...
    def is_bytearray(self) -> bool:
        ...
    def is_cstr(self) -> bool:
        ...
    def is_decimal(self) -> bool:
        ...
    def is_hash(self) -> bool:
        ...
    def is_hexadecimal(self) -> bool:
        ...
    def is_incremented(self) -> bool:
        ...
    def is_node_altval(self) -> bool:
        ...
    def is_node_blob(self) -> bool:
        ...
    def is_node_supval(self) -> bool:
        ...
    def is_node_valobj(self) -> bool:
        ...
    def is_node_var(self) -> bool:
        ...
    def is_onoff(self) -> bool:
        ...
    def is_qstring(self) -> bool:
        ...
    def is_readonly_var(self) -> bool:
        ...
    def is_scalar_var(self) -> bool:
        ...
    def is_struc_field(self) -> bool:
        ...
    def is_val_mapped(self) -> bool:
        ...
    def ridx(self) -> int:
        ...
    def str_false(self) -> str:
        ...
    def str_true(self) -> str:
        ...
    def use_hlpstruc(self) -> bool:
        ...

class idbattr_valmap_t:
    @property
    def valname(self) -> Any: ...
    @property
    def value(self) -> Any: ...
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

def calc_default_idaplace_flags() -> int:
    r"""Get default disassembly line options.
    
    """
    ...

def delinf(tag: inftag_t) -> bool:
    r"""Undefine a program specific information 
            
    :param tag: one of inftag_t constants
    :returns: success
    """
    ...

def get_dbctx_id() -> ssize_t:
    r"""Get the current database context ID 
            
    :returns: the database context ID, or -1 if no current database
    """
    ...

def get_dbctx_qty() -> int:
    r"""Get number of database contexts 
            
    :returns: number of database contexts
    """
    ...

def getinf_str(tag: inftag_t) -> str:
    r"""Get program specific information (a non-scalar value) 
            
    :param tag: one of inftag_t constants
    :returns: number of bytes stored in the buffer (<0 - not defined)
    """
    ...

def idainfo_big_arg_align(args: Any) -> bool:
    ...

def idainfo_comment_get() -> uchar:
    ...

def idainfo_comment_set(_v: uchar) -> bool:
    ...

def idainfo_gen_lzero() -> bool:
    ...

def idainfo_gen_null() -> bool:
    ...

def idainfo_gen_tryblks() -> bool:
    ...

def idainfo_get_demname_form() -> uchar:
    r"""Get DEMNAM_MASK bits of #demnames.
    
    """
    ...

def idainfo_get_pack_mode() -> int:
    ...

def idainfo_is_32bit() -> Any:
    ...

def idainfo_is_64bit() -> bool:
    ...

def idainfo_is_auto_enabled() -> bool:
    ...

def idainfo_is_be() -> bool:
    ...

def idainfo_is_dll() -> bool:
    ...

def idainfo_is_flat_off32() -> bool:
    ...

def idainfo_is_graph_view() -> bool:
    ...

def idainfo_is_hard_float() -> bool:
    ...

def idainfo_is_kernel_mode() -> bool:
    ...

def idainfo_is_mem_aligned4() -> bool:
    ...

def idainfo_is_snapshot() -> bool:
    ...

def idainfo_is_wide_high_byte_first() -> bool:
    ...

def idainfo_like_binary() -> bool:
    ...

def idainfo_line_pref_with_seg() -> bool:
    ...

def idainfo_loading_idc() -> bool:
    ...

def idainfo_map_stkargs() -> bool:
    ...

def idainfo_pack_stkargs(args: Any) -> bool:
    ...

def idainfo_readonly_idb() -> bool:
    ...

def idainfo_set_64bit(_v: bool = True) -> bool:
    ...

def idainfo_set_auto_enabled(_v: bool = True) -> bool:
    ...

def idainfo_set_be(_v: bool = True) -> bool:
    ...

def idainfo_set_gen_lzero(_v: bool = True) -> bool:
    ...

def idainfo_set_gen_null(_v: bool = True) -> bool:
    ...

def idainfo_set_gen_tryblks(_v: bool = True) -> bool:
    ...

def idainfo_set_graph_view(_v: bool = True) -> bool:
    ...

def idainfo_set_line_pref_with_seg(_v: bool = True) -> bool:
    ...

def idainfo_set_pack_mode(pack_mode: int) -> int:
    ...

def idainfo_set_show_auto(_v: bool = True) -> bool:
    ...

def idainfo_set_show_line_pref(_v: bool = True) -> bool:
    ...

def idainfo_set_show_void(_v: bool = True) -> bool:
    ...

def idainfo_set_store_user_info(args: Any) -> Any:
    ...

def idainfo_set_wide_high_byte_first(_v: bool = True) -> bool:
    ...

def idainfo_show_auto() -> bool:
    ...

def idainfo_show_line_pref() -> bool:
    ...

def idainfo_show_void() -> bool:
    ...

def idainfo_stack_ldbl() -> bool:
    ...

def idainfo_stack_varargs() -> bool:
    ...

def idainfo_use_allasm() -> bool:
    ...

def idainfo_use_gcc_layout() -> bool:
    ...

def inf_abi_set_by_user() -> bool:
    ...

def inf_allow_non_matched_ops() -> bool:
    ...

def inf_allow_sigmulti() -> bool:
    ...

def inf_append_sigcmt() -> bool:
    ...

def inf_big_arg_align(args: Any) -> bool:
    ...

def inf_check_manual_ops() -> bool:
    ...

def inf_check_unicode_strlits() -> bool:
    ...

def inf_coagulate_code() -> bool:
    ...

def inf_coagulate_data() -> bool:
    ...

def inf_compress_idb() -> bool:
    ...

def inf_create_all_xrefs() -> bool:
    ...

def inf_create_func_from_call() -> bool:
    ...

def inf_create_func_from_ptr() -> bool:
    ...

def inf_create_func_tails() -> bool:
    ...

def inf_create_jump_tables() -> bool:
    ...

def inf_create_off_on_dref() -> bool:
    ...

def inf_create_off_using_fixup() -> bool:
    ...

def inf_create_strlit_on_xref() -> bool:
    ...

def inf_data_offset() -> bool:
    ...

def inf_dbg_no_store_path() -> bool:
    ...

def inf_decode_fpp() -> bool:
    ...

def inf_del_no_xref_insns() -> bool:
    ...

def inf_final_pass() -> bool:
    ...

def inf_full_sp_ana() -> bool:
    ...

def inf_gen_assume() -> bool:
    ...

def inf_gen_lzero() -> bool:
    ...

def inf_gen_null() -> bool:
    ...

def inf_gen_org() -> bool:
    ...

def inf_gen_tryblks() -> bool:
    ...

def inf_get_abibits() -> int:
    ...

def inf_get_af() -> int:
    ...

def inf_get_af2() -> int:
    ...

def inf_get_af2_low() -> ushort:
    r"""Get/set low 16bit half of inf.af2.
    
    """
    ...

def inf_get_af_high() -> ushort:
    ...

def inf_get_af_low() -> ushort:
    r"""Get/set low/high 16bit halves of inf.af.
    
    """
    ...

def inf_get_app_bitness() -> uint:
    ...

def inf_get_appcall_options() -> int:
    ...

def inf_get_apptype() -> ushort:
    ...

def inf_get_asmtype() -> uchar:
    ...

def inf_get_baseaddr() -> int:
    ...

def inf_get_bin_prefix_size() -> short:
    ...

def inf_get_callcnv() -> callcnv_t:
    ...

def inf_get_cc(out: compiler_info_t) -> bool:
    ...

def inf_get_cc_cm() -> cm_t:
    ...

def inf_get_cc_defalign() -> uchar:
    ...

def inf_get_cc_id() -> comp_t:
    ...

def inf_get_cc_size_b() -> uchar:
    ...

def inf_get_cc_size_e() -> uchar:
    ...

def inf_get_cc_size_i() -> uchar:
    ...

def inf_get_cc_size_l() -> uchar:
    ...

def inf_get_cc_size_ldbl() -> uchar:
    ...

def inf_get_cc_size_ll() -> uchar:
    ...

def inf_get_cc_size_s() -> uchar:
    ...

def inf_get_cmt_indent() -> uchar:
    ...

def inf_get_cmtflg() -> uchar:
    ...

def inf_get_comment() -> uchar:
    ...

def inf_get_database_change_count() -> int:
    ...

def inf_get_datatypes() -> int:
    ...

def inf_get_demname_form() -> uchar:
    r"""Get DEMNAM_MASK bits of #demnames.
    
    """
    ...

def inf_get_demnames() -> uchar:
    ...

def inf_get_filetype() -> filetype_t:
    ...

def inf_get_genflags() -> ushort:
    ...

def inf_get_highoff() -> ida_idaapi.ea_t:
    ...

def inf_get_indent() -> uchar:
    ...

def inf_get_lenxref() -> ushort:
    ...

def inf_get_lflags() -> int:
    ...

def inf_get_limiter() -> uchar:
    ...

def inf_get_listnames() -> uchar:
    ...

def inf_get_long_demnames() -> int:
    ...

def inf_get_lowoff() -> ida_idaapi.ea_t:
    ...

def inf_get_main() -> ida_idaapi.ea_t:
    ...

def inf_get_margin() -> ushort:
    ...

def inf_get_max_autoname_len() -> ushort:
    ...

def inf_get_max_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_maxref() -> int:
    ...

def inf_get_min_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_nametype() -> char:
    ...

def inf_get_netdelta() -> int:
    ...

def inf_get_omax_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_omin_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_ostype() -> ushort:
    ...

def inf_get_outflags() -> int:
    ...

def inf_get_pack_mode() -> int:
    ...

def inf_get_prefflag() -> uchar:
    ...

def inf_get_privrange(args: Any) -> range_t:
    r"""This function has the following signatures:
    
        0. inf_get_privrange(out: range_t *) -> bool
        1. inf_get_privrange() -> range_t
    
    # 0: inf_get_privrange(out: range_t *) -> bool
    
    
    # 1: inf_get_privrange() -> range_t
    
    
    """
    ...

def inf_get_privrange_end_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_privrange_start_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_procname() -> str:
    ...

def inf_get_refcmtnum() -> uchar:
    ...

def inf_get_short_demnames() -> int:
    ...

def inf_get_specsegs() -> uchar:
    ...

def inf_get_start_cs() -> sel_t:
    ...

def inf_get_start_ea() -> ida_idaapi.ea_t:
    ...

def inf_get_start_ip() -> ida_idaapi.ea_t:
    ...

def inf_get_start_sp() -> ida_idaapi.ea_t:
    ...

def inf_get_start_ss() -> sel_t:
    ...

def inf_get_strlit_break() -> uchar:
    ...

def inf_get_strlit_flags() -> uchar:
    ...

def inf_get_strlit_pref() -> str:
    ...

def inf_get_strlit_sernum() -> int:
    ...

def inf_get_strlit_zeroes() -> char:
    ...

def inf_get_strtype() -> int:
    ...

def inf_get_type_xrefnum() -> uchar:
    ...

def inf_get_version() -> ushort:
    ...

def inf_get_xrefflag() -> uchar:
    ...

def inf_get_xrefnum() -> uchar:
    ...

def inf_guess_func_type() -> bool:
    ...

def inf_handle_eh() -> bool:
    ...

def inf_handle_rtti() -> bool:
    ...

def inf_hide_comments() -> bool:
    ...

def inf_hide_libfuncs() -> bool:
    ...

def inf_huge_arg_align(args: Any) -> bool:
    ...

def inf_inc_database_change_count(cnt: int = 1) -> None:
    ...

def inf_is_16bit() -> bool:
    ...

def inf_is_32bit_exactly() -> bool:
    ...

def inf_is_32bit_or_higher() -> bool:
    ...

def inf_is_64bit() -> bool:
    ...

def inf_is_auto_enabled() -> bool:
    ...

def inf_is_be() -> bool:
    ...

def inf_is_dll() -> bool:
    ...

def inf_is_flat_off32() -> bool:
    ...

def inf_is_graph_view() -> bool:
    ...

def inf_is_hard_float() -> bool:
    ...

def inf_is_ilp32() -> bool:
    ...

def inf_is_kernel_mode() -> bool:
    ...

def inf_is_limiter_empty() -> bool:
    ...

def inf_is_limiter_thick() -> bool:
    ...

def inf_is_limiter_thin() -> bool:
    ...

def inf_is_mem_aligned4() -> bool:
    ...

def inf_is_snapshot() -> bool:
    ...

def inf_is_wide_high_byte_first() -> bool:
    ...

def inf_like_binary() -> bool:
    ...

def inf_line_pref_with_seg() -> bool:
    ...

def inf_loading_idc() -> bool:
    ...

def inf_macros_enabled() -> bool:
    ...

def inf_map_stkargs() -> bool:
    ...

def inf_mark_code() -> bool:
    ...

def inf_merge_strlits() -> bool:
    ...

def inf_no_store_user_info() -> bool:
    ...

def inf_noflow_to_data() -> bool:
    ...

def inf_noret_ana() -> bool:
    ...

def inf_op_offset() -> bool:
    ...

def inf_pack_idb() -> bool:
    ...

def inf_pack_stkargs(args: Any) -> bool:
    ...

def inf_postinc_strlit_sernum(cnt: int = 1) -> int:
    ...

def inf_prefix_show_funcoff() -> bool:
    ...

def inf_prefix_show_segaddr() -> bool:
    ...

def inf_prefix_show_stack() -> bool:
    ...

def inf_prefix_truncate_opcode_bytes() -> bool:
    ...

def inf_propagate_regargs() -> bool:
    ...

def inf_propagate_stkargs() -> bool:
    ...

def inf_readonly_idb() -> bool:
    ...

def inf_rename_jumpfunc() -> bool:
    ...

def inf_rename_nullsub() -> bool:
    ...

def inf_set_32bit(_v: bool = True) -> bool:
    ...

def inf_set_64bit(_v: bool = True) -> bool:
    ...

def inf_set_abi_set_by_user(_v: bool = True) -> bool:
    ...

def inf_set_abibits(_v: int) -> bool:
    ...

def inf_set_af(_v: int) -> bool:
    ...

def inf_set_af2(_v: int) -> bool:
    ...

def inf_set_af2_low(saf: ushort) -> None:
    ...

def inf_set_af_high(saf2: ushort) -> None:
    ...

def inf_set_af_low(saf: ushort) -> None:
    ...

def inf_set_allow_non_matched_ops(_v: bool = True) -> bool:
    ...

def inf_set_allow_sigmulti(_v: bool = True) -> bool:
    ...

def inf_set_app_bitness(bitness: uint) -> None:
    ...

def inf_set_appcall_options(_v: int) -> bool:
    ...

def inf_set_append_sigcmt(_v: bool = True) -> bool:
    ...

def inf_set_apptype(_v: ushort) -> bool:
    ...

def inf_set_asmtype(_v: uchar) -> bool:
    ...

def inf_set_auto_enabled(_v: bool = True) -> bool:
    ...

def inf_set_baseaddr(_v: int) -> bool:
    ...

def inf_set_be(_v: bool = True) -> bool:
    ...

def inf_set_big_arg_align(_v: bool = True) -> bool:
    ...

def inf_set_bin_prefix_size(_v: short) -> bool:
    ...

def inf_set_callcnv(_v: callcnv_t) -> bool:
    ...

def inf_set_cc(_v: compiler_info_t) -> bool:
    ...

def inf_set_cc_cm(_v: cm_t) -> bool:
    ...

def inf_set_cc_defalign(_v: uchar) -> bool:
    ...

def inf_set_cc_id(_v: comp_t) -> bool:
    ...

def inf_set_cc_size_b(_v: uchar) -> bool:
    ...

def inf_set_cc_size_e(_v: uchar) -> bool:
    ...

def inf_set_cc_size_i(_v: uchar) -> bool:
    ...

def inf_set_cc_size_l(_v: uchar) -> bool:
    ...

def inf_set_cc_size_ldbl(_v: uchar) -> bool:
    ...

def inf_set_cc_size_ll(_v: uchar) -> bool:
    ...

def inf_set_cc_size_s(_v: uchar) -> bool:
    ...

def inf_set_check_manual_ops(_v: bool = True) -> bool:
    ...

def inf_set_check_unicode_strlits(_v: bool = True) -> bool:
    ...

def inf_set_cmt_indent(_v: uchar) -> bool:
    ...

def inf_set_cmtflg(_v: uchar) -> bool:
    ...

def inf_set_coagulate_code(_v: bool = True) -> bool:
    ...

def inf_set_coagulate_data(_v: bool = True) -> bool:
    ...

def inf_set_comment(_v: uchar) -> bool:
    ...

def inf_set_compress_idb(_v: bool = True) -> bool:
    ...

def inf_set_create_all_xrefs(_v: bool = True) -> bool:
    ...

def inf_set_create_func_from_call(_v: bool = True) -> bool:
    ...

def inf_set_create_func_from_ptr(_v: bool = True) -> bool:
    ...

def inf_set_create_func_tails(_v: bool = True) -> bool:
    ...

def inf_set_create_jump_tables(_v: bool = True) -> bool:
    ...

def inf_set_create_off_on_dref(_v: bool = True) -> bool:
    ...

def inf_set_create_off_using_fixup(_v: bool = True) -> bool:
    ...

def inf_set_create_strlit_on_xref(_v: bool = True) -> bool:
    ...

def inf_set_data_offset(_v: bool = True) -> bool:
    ...

def inf_set_database_change_count(_v: int) -> bool:
    ...

def inf_set_datatypes(_v: int) -> bool:
    ...

def inf_set_dbg_no_store_path(_v: bool = True) -> bool:
    ...

def inf_set_decode_fpp(_v: bool = True) -> bool:
    ...

def inf_set_del_no_xref_insns(_v: bool = True) -> bool:
    ...

def inf_set_demnames(_v: uchar) -> bool:
    ...

def inf_set_dll(_v: bool = True) -> bool:
    ...

def inf_set_filetype(_v: filetype_t) -> bool:
    ...

def inf_set_final_pass(_v: bool = True) -> bool:
    ...

def inf_set_flat_off32(_v: bool = True) -> bool:
    ...

def inf_set_full_sp_ana(_v: bool = True) -> bool:
    ...

def inf_set_gen_assume(_v: bool = True) -> bool:
    ...

def inf_set_gen_lzero(_v: bool = True) -> bool:
    ...

def inf_set_gen_null(_v: bool = True) -> bool:
    ...

def inf_set_gen_org(_v: bool = True) -> bool:
    ...

def inf_set_gen_tryblks(_v: bool = True) -> bool:
    ...

def inf_set_genflags(_v: ushort) -> bool:
    ...

def inf_set_graph_view(_v: bool = True) -> bool:
    ...

def inf_set_guess_func_type(_v: bool = True) -> bool:
    ...

def inf_set_handle_eh(_v: bool = True) -> bool:
    ...

def inf_set_handle_rtti(_v: bool = True) -> bool:
    ...

def inf_set_hard_float(_v: bool = True) -> bool:
    ...

def inf_set_hide_comments(_v: bool = True) -> bool:
    ...

def inf_set_hide_libfuncs(_v: bool = True) -> bool:
    ...

def inf_set_highoff(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_huge_arg_align(_v: bool = True) -> bool:
    ...

def inf_set_ilp32(_v: bool = True) -> bool:
    ...

def inf_set_indent(_v: uchar) -> bool:
    ...

def inf_set_kernel_mode(_v: bool = True) -> bool:
    ...

def inf_set_lenxref(_v: ushort) -> bool:
    ...

def inf_set_lflags(_v: int) -> bool:
    ...

def inf_set_limiter(_v: uchar) -> bool:
    ...

def inf_set_limiter_empty(_v: bool = True) -> bool:
    ...

def inf_set_limiter_thick(_v: bool = True) -> bool:
    ...

def inf_set_limiter_thin(_v: bool = True) -> bool:
    ...

def inf_set_line_pref_with_seg(_v: bool = True) -> bool:
    ...

def inf_set_listnames(_v: uchar) -> bool:
    ...

def inf_set_loading_idc(_v: bool = True) -> bool:
    ...

def inf_set_long_demnames(_v: int) -> bool:
    ...

def inf_set_lowoff(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_macros_enabled(_v: bool = True) -> bool:
    ...

def inf_set_main(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_map_stkargs(_v: bool = True) -> bool:
    ...

def inf_set_margin(_v: ushort) -> bool:
    ...

def inf_set_mark_code(_v: bool = True) -> bool:
    ...

def inf_set_max_autoname_len(_v: ushort) -> bool:
    ...

def inf_set_max_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_maxref(_v: int) -> bool:
    ...

def inf_set_mem_aligned4(_v: bool = True) -> bool:
    ...

def inf_set_merge_strlits(_v: bool = True) -> bool:
    ...

def inf_set_min_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_nametype(_v: char) -> bool:
    ...

def inf_set_netdelta(_v: int) -> bool:
    ...

def inf_set_no_store_user_info(_v: bool = True) -> bool:
    ...

def inf_set_noflow_to_data(_v: bool = True) -> bool:
    ...

def inf_set_noret_ana(_v: bool = True) -> bool:
    ...

def inf_set_omax_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_omin_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_op_offset(_v: bool = True) -> bool:
    ...

def inf_set_ostype(_v: ushort) -> bool:
    ...

def inf_set_outflags(_v: int) -> bool:
    ...

def inf_set_pack_idb(_v: bool = True) -> bool:
    ...

def inf_set_pack_mode(pack_mode: int) -> int:
    ...

def inf_set_pack_stkargs(_v: bool = True) -> bool:
    ...

def inf_set_prefflag(_v: uchar) -> bool:
    ...

def inf_set_prefix_show_funcoff(_v: bool = True) -> bool:
    ...

def inf_set_prefix_show_segaddr(_v: bool = True) -> bool:
    ...

def inf_set_prefix_show_stack(_v: bool = True) -> bool:
    ...

def inf_set_prefix_truncate_opcode_bytes(_v: bool = True) -> bool:
    ...

def inf_set_privrange(_v: range_t) -> bool:
    ...

def inf_set_privrange_end_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_privrange_start_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_procname(args: Any) -> bool:
    ...

def inf_set_propagate_regargs(_v: bool = True) -> bool:
    ...

def inf_set_propagate_stkargs(_v: bool = True) -> bool:
    ...

def inf_set_readonly_idb(_v: bool = True) -> bool:
    ...

def inf_set_refcmtnum(_v: uchar) -> bool:
    ...

def inf_set_rename_jumpfunc(_v: bool = True) -> bool:
    ...

def inf_set_rename_nullsub(_v: bool = True) -> bool:
    ...

def inf_set_short_demnames(_v: int) -> bool:
    ...

def inf_set_should_create_stkvars(_v: bool = True) -> bool:
    ...

def inf_set_should_trace_sp(_v: bool = True) -> bool:
    ...

def inf_set_show_all_comments(_v: bool = True) -> bool:
    ...

def inf_set_show_auto(_v: bool = True) -> bool:
    ...

def inf_set_show_hidden_funcs(_v: bool = True) -> bool:
    ...

def inf_set_show_hidden_insns(_v: bool = True) -> bool:
    ...

def inf_set_show_hidden_segms(_v: bool = True) -> bool:
    ...

def inf_set_show_line_pref(_v: bool = True) -> bool:
    ...

def inf_set_show_repeatables(_v: bool = True) -> bool:
    ...

def inf_set_show_src_linnum(_v: bool = True) -> bool:
    ...

def inf_set_show_void(_v: bool = True) -> bool:
    ...

def inf_set_show_xref_fncoff(_v: bool = True) -> bool:
    ...

def inf_set_show_xref_seg(_v: bool = True) -> bool:
    ...

def inf_set_show_xref_tmarks(_v: bool = True) -> bool:
    ...

def inf_set_show_xref_val(_v: bool = True) -> bool:
    ...

def inf_set_snapshot(_v: bool = True) -> bool:
    ...

def inf_set_specsegs(_v: uchar) -> bool:
    ...

def inf_set_stack_ldbl(_v: bool = True) -> bool:
    ...

def inf_set_stack_varargs(_v: bool = True) -> bool:
    ...

def inf_set_start_cs(_v: sel_t) -> bool:
    ...

def inf_set_start_ea(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_start_ip(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_start_sp(_v: ida_idaapi.ea_t) -> bool:
    ...

def inf_set_start_ss(_v: sel_t) -> bool:
    ...

def inf_set_strlit_autocmt(_v: bool = True) -> bool:
    ...

def inf_set_strlit_break(_v: uchar) -> bool:
    ...

def inf_set_strlit_flags(_v: uchar) -> bool:
    ...

def inf_set_strlit_name_bit(_v: bool = True) -> bool:
    ...

def inf_set_strlit_names(_v: bool = True) -> bool:
    ...

def inf_set_strlit_pref(args: Any) -> bool:
    ...

def inf_set_strlit_savecase(_v: bool = True) -> bool:
    ...

def inf_set_strlit_serial_names(_v: bool = True) -> bool:
    ...

def inf_set_strlit_sernum(_v: int) -> bool:
    ...

def inf_set_strlit_zeroes(_v: char) -> bool:
    ...

def inf_set_strtype(_v: int) -> bool:
    ...

def inf_set_trace_flow(_v: bool = True) -> bool:
    ...

def inf_set_truncate_on_del(_v: bool = True) -> bool:
    ...

def inf_set_type_xrefnum(_v: uchar) -> bool:
    ...

def inf_set_unicode_strlits(_v: bool = True) -> bool:
    ...

def inf_set_use_allasm(_v: bool = True) -> bool:
    ...

def inf_set_use_flirt(_v: bool = True) -> bool:
    ...

def inf_set_use_gcc_layout(_v: bool = True) -> bool:
    ...

def inf_set_version(_v: ushort) -> bool:
    ...

def inf_set_wide_high_byte_first(_v: bool = True) -> bool:
    ...

def inf_set_xrefflag(_v: uchar) -> bool:
    ...

def inf_set_xrefnum(_v: uchar) -> bool:
    ...

def inf_should_create_stkvars() -> bool:
    ...

def inf_should_trace_sp() -> bool:
    ...

def inf_show_all_comments() -> bool:
    ...

def inf_show_auto() -> bool:
    ...

def inf_show_hidden_funcs() -> bool:
    ...

def inf_show_hidden_insns() -> bool:
    ...

def inf_show_hidden_segms() -> bool:
    ...

def inf_show_line_pref() -> bool:
    ...

def inf_show_repeatables() -> bool:
    ...

def inf_show_src_linnum() -> bool:
    ...

def inf_show_void() -> bool:
    ...

def inf_show_xref_fncoff() -> bool:
    ...

def inf_show_xref_seg() -> bool:
    ...

def inf_show_xref_tmarks() -> bool:
    ...

def inf_show_xref_val() -> bool:
    ...

def inf_stack_ldbl() -> bool:
    ...

def inf_stack_varargs() -> bool:
    ...

def inf_strlit_autocmt() -> bool:
    ...

def inf_strlit_name_bit() -> bool:
    ...

def inf_strlit_names() -> bool:
    ...

def inf_strlit_savecase() -> bool:
    ...

def inf_strlit_serial_names() -> bool:
    ...

def inf_test_mode() -> bool:
    ...

def inf_trace_flow() -> bool:
    ...

def inf_truncate_on_del() -> bool:
    ...

def inf_unicode_strlits() -> bool:
    ...

def inf_use_allasm() -> bool:
    ...

def inf_use_flirt() -> bool:
    ...

def inf_use_gcc_layout() -> bool:
    ...

def is_database_busy() -> bool:
    r"""Check if the database is busy (e.g. performing some critical operations and cannot be safely accessed) 
            
    """
    ...

def is_filetype_like_binary(ft: filetype_t) -> bool:
    r"""Is unstructured input file?
    
    """
    ...

def macros_enabled() -> bool:
    ...

def move_privrange(new_privrange_start: ida_idaapi.ea_t) -> bool:
    r"""Move privrange to the specified address 
            
    :param new_privrange_start: new start address of the privrange
    :returns: success
    """
    ...

def should_create_stkvars() -> bool:
    ...

def should_trace_sp() -> bool:
    ...

def show_all_comments() -> bool:
    ...

def show_comments(args: Any) -> Any:
    ...

def show_repeatables() -> bool:
    ...

def switch_dbctx(idx: size_t) -> dbctx_t:
    r"""Switch to the database with the provided context ID 
            
    :param idx: the index of the database to switch to
    :returns: the current dbctx_t instance or nullptr
    """
    ...

def to_ea(reg_cs: sel_t, reg_ip: int) -> ida_idaapi.ea_t:
    r"""Convert (sel,off) value to a linear address.
    
    """
    ...

def validate_idb(vld_flags: int = 0) -> int:
    r"""Validate the database 
            
    :param vld_flags: combination of VLD_.. constants
    :returns: number of corrupted/fixed records
    """
    ...

ABI_8ALIGN4: int  # 1
ABI_BIGARG_ALIGN: int  # 4
ABI_GCC_LAYOUT: int  # 128
ABI_HARD_FLOAT: int  # 32
ABI_HUGEARG_ALIGN: int  # 512
ABI_MAP_STKARGS: int  # 256
ABI_PACK_STKARGS: int  # 2
ABI_SET_BY_USER: int  # 64
ABI_STACK_LDBL: int  # 8
ABI_STACK_VARARGS: int  # 16
AF2_DOEH: int  # 1
AF2_DORTTI: int  # 2
AF2_MACRO: int  # 4
AF2_MERGESTR: int  # 8
AF_ANORET: int  # 16384
AF_CHKUNI: int  # 262144
AF_CODE: int  # 1
AF_DATOFF: int  # 4194304
AF_DOCODE: int  # 1073741824
AF_DODATA: int  # 536870912
AF_DREFOFF: int  # 1048576
AF_FINAL: int  # 2147483648
AF_FIXUP: int  # 524288
AF_FLIRT: int  # 8388608
AF_FTAIL: int  # 256
AF_HFLIRT: int  # 67108864
AF_IMMOFF: int  # 2097152
AF_JFUNC: int  # 134217728
AF_JUMPTBL: int  # 4
AF_LVAR: int  # 512
AF_MARKCODE: int  # 2
AF_MEMFUNC: int  # 32768
AF_NULLSUB: int  # 268435456
AF_PROC: int  # 128
AF_PROCPTR: int  # 64
AF_PURDAT: int  # 8
AF_REGARG: int  # 2048
AF_SIGCMT: int  # 16777216
AF_SIGMLT: int  # 33554432
AF_STKARG: int  # 1024
AF_STRLIT: int  # 131072
AF_TRACE: int  # 4096
AF_TRFUNC: int  # 65536
AF_UNK: int  # 32
AF_USED: int  # 16
AF_VERSP: int  # 8192
DEMNAM_CMNT: int  # 0
DEMNAM_FIRST: int  # 8
DEMNAM_GCC3: int  # 4
DEMNAM_MASK: int  # 3
DEMNAM_NAME: int  # 1
DEMNAM_NONE: int  # 2
IDAINFO_PROCNAME_SIZE: int  # 16
IDAINFO_STRLIT_PREF_SIZE: int  # 16
IDAINFO_TAG_SIZE: int  # 3
IDB_COMPRESSED: int  # 2
IDB_EXT: str  # i64
IDB_EXT32: str  # idb
IDB_EXT64: str  # i64
IDB_PACKED: int  # 1
IDB_UNPACKED: int  # 0
IDI_ALTVAL: int  # 1
IDI_BITMAP: int  # 16384
IDI_BLOB: int  # 8
IDI_BUFVAR: int  # 16496
IDI_BYTEARRAY: int  # 64
IDI_CSTR: int  # 16
IDI_DEC: int  # 128
IDI_EA_HEX: int  # 0
IDI_HASH: int  # 2048
IDI_HEX: int  # 256
IDI_HLPSTRUC: int  # 4096
IDI_INC: int  # 512
IDI_MAP_VAL: int  # 1024
IDI_NODEVAL: int  # 15
IDI_NOMERGE: int  # 65536
IDI_ONOFF: int  # 32768
IDI_QSTRING: int  # 32
IDI_READONLY: int  # 8192
IDI_SCALAR: int  # 0
IDI_STRUCFLD: int  # 0
IDI_SUPVAL: int  # 2
IDI_VALOBJ: int  # 4
INFFL_ALLASM: int  # 2
INFFL_AUTO: int  # 1
INFFL_CHKOPS: int  # 32
INFFL_GRAPH_VIEW: int  # 128
INFFL_LOADIDC: int  # 4
INFFL_NMOPS: int  # 64
INFFL_NOUSER: int  # 8
INFFL_READONLY: int  # 16
INF_ABIBITS: int  # 67
INF_ABINAME: int  # 81
INF_AF: int  # 10
INF_AF2: int  # 11
INF_APPCALL_OPTIONS: int  # 68
INF_APPTYPE: int  # 7
INF_ARCHIVE_PATH: int  # 82
INF_ASMTYPE: int  # 8
INF_BASEADDR: int  # 12
INF_BIN_PREFIX_SIZE: int  # 47
INF_CALLCNV: int  # 99
INF_CC_CM: int  # 58
INF_CC_DEFALIGN: int  # 62
INF_CC_ID: int  # 57
INF_CC_SIZE_B: int  # 60
INF_CC_SIZE_E: int  # 61
INF_CC_SIZE_I: int  # 59
INF_CC_SIZE_L: int  # 64
INF_CC_SIZE_LDBL: int  # 66
INF_CC_SIZE_LL: int  # 65
INF_CC_SIZE_S: int  # 63
INF_CMTFLG: int  # 45
INF_CMT_INDENT: int  # 41
INF_COMPILER_INFO: int  # 98
INF_CRC32: int  # 92
INF_CTIME: int  # 89
INF_C_MACROS: int  # 72
INF_DATABASE_CHANGE_COUNT: int  # 4
INF_DATATYPES: int  # 55
INF_DBG_BINPATHS: int  # 79
INF_DEMNAMES: int  # 38
INF_DUALOP_GRAPH: int  # 74
INF_DUALOP_TEXT: int  # 75
INF_ELAPSED: int  # 90
INF_FILETYPE: int  # 5
INF_FILE_FORMAT_NAME: int  # 69
INF_FSIZE: int  # 95
INF_GENFLAGS: int  # 2
INF_GROUPS: int  # 70
INF_HIGHOFF: int  # 24
INF_H_PATH: int  # 71
INF_IDA_VERSION: int  # 77
INF_IDSNODE: int  # 94
INF_IMAGEBASE: int  # 93
INF_INCLUDE: int  # 73
INF_INDENT: int  # 40
INF_INITIAL_VERSION: int  # 88
INF_INPUT_FILE_PATH: int  # 97
INF_LAST: int  # 100
INF_LENXREF: int  # 43
INF_LFLAGS: int  # 3
INF_LIMITER: int  # 46
INF_LISTNAMES: int  # 39
INF_LONG_DEMNAMES: int  # 37
INF_LOWOFF: int  # 23
INF_MAIN: int  # 18
INF_MARGIN: int  # 42
INF_MAXREF: int  # 25
INF_MAX_AUTONAME_LEN: int  # 34
INF_MAX_EA: int  # 20
INF_MD5: int  # 76
INF_MIN_EA: int  # 19
INF_NAMETYPE: int  # 35
INF_NETDELTA: int  # 29
INF_NOPENS: int  # 91
INF_NOTEPAD: int  # 85
INF_OBSOLETE_CC: int  # 56
INF_OMAX_EA: int  # 22
INF_OMIN_EA: int  # 21
INF_OSTYPE: int  # 6
INF_OUTFILEENC: int  # 96
INF_OUTFLAGS: int  # 44
INF_PREFFLAG: int  # 48
INF_PRIVRANGE: int  # 26
INF_PRIVRANGE_END_EA: int  # 28
INF_PRIVRANGE_START_EA: int  # 27
INF_PROBLEMS: int  # 83
INF_PROCNAME: int  # 1
INF_REFCMTNUM: int  # 32
INF_SELECTORS: int  # 84
INF_SHA256: int  # 80
INF_SHORT_DEMNAMES: int  # 36
INF_SPECSEGS: int  # 9
INF_SRCDBG_PATHS: int  # 86
INF_SRCDBG_UNDESIRED: int  # 87
INF_START_CS: int  # 14
INF_START_EA: int  # 16
INF_START_IP: int  # 15
INF_START_SP: int  # 17
INF_START_SS: int  # 13
INF_STRLIT_BREAK: int  # 50
INF_STRLIT_FLAGS: int  # 49
INF_STRLIT_PREF: int  # 53
INF_STRLIT_SERNUM: int  # 54
INF_STRLIT_ZEROES: int  # 51
INF_STRTYPE: int  # 52
INF_STR_ENCODINGS: int  # 78
INF_TYPE_XREFNUM: int  # 31
INF_VERSION: int  # 0
INF_XREFFLAG: int  # 33
INF_XREFNUM: int  # 30
LFLG_64BIT: int  # 4
LFLG_COMPRESS: int  # 1024
LFLG_DBG_NOPATH: int  # 128
LFLG_FLAT_OFF32: int  # 16
LFLG_ILP32: int  # 4096
LFLG_IS_DLL: int  # 8
LFLG_KERNMODE: int  # 2048
LFLG_MSF: int  # 32
LFLG_PACK: int  # 512
LFLG_PC_FLAT: int  # 2
LFLG_PC_FPP: int  # 1
LFLG_SNAPSHOT: int  # 256
LFLG_WIDE_HBF: int  # 64
LMT_EMPTY: int  # 4
LMT_THICK: int  # 2
LMT_THIN: int  # 1
LN_AUTO: int  # 4
LN_NORMAL: int  # 1
LN_PUBLIC: int  # 2
LN_WEAK: int  # 8
MAXADDR: int  # 0
NM_EA: int  # 6
NM_EA4: int  # 7
NM_EA8: int  # 8
NM_NAM_EA: int  # 5
NM_NAM_OFF: int  # 2
NM_PTR_EA: int  # 4
NM_PTR_OFF: int  # 1
NM_REL_EA: int  # 3
NM_REL_OFF: int  # 0
NM_SERIAL: int  # 10
NM_SHORT: int  # 9
OFLG_GEN_ASSUME: int  # 512
OFLG_GEN_NULL: int  # 16
OFLG_GEN_ORG: int  # 256
OFLG_GEN_TRYBLKS: int  # 1024
OFLG_LZERO: int  # 128
OFLG_PREF_SEG: int  # 64
OFLG_SHOW_AUTO: int  # 4
OFLG_SHOW_PREF: int  # 32
OFLG_SHOW_VOID: int  # 2
PREF_FNCOFF: int  # 2
PREF_PFXTRUNC: int  # 8
PREF_SEGADR: int  # 1
PREF_STACK: int  # 4
SCF_ALLCMT: int  # 2
SCF_LINNUM: int  # 8
SCF_NOCMT: int  # 4
SCF_RPTCMT: int  # 1
SCF_SHHID_FUNC: int  # 64
SCF_SHHID_ITEM: int  # 32
SCF_SHHID_SEGM: int  # 128
SCF_TESTMODE: int  # 16
STRF_AUTO: int  # 2
STRF_COMMENT: int  # 16
STRF_GEN: int  # 1
STRF_SAVECASE: int  # 32
STRF_SERIAL: int  # 4
STRF_UNICODE: int  # 8
STT_CUR: int  # -1
STT_DBG: int  # 2
STT_MM: int  # 1
STT_VA: int  # 0
SWIG_PYTHON_LEGACY_BOOL: int  # 1
SW_SEGXRF: int  # 1
SW_XRFFNC: int  # 4
SW_XRFMRK: int  # 2
SW_XRFVAL: int  # 8
UA_MAXOP: int  # 8
VLD_AUTO_REPAIR: int  # 1
VLD_DIALOG: int  # 2
VLD_SILENT: int  # 4
annotations: _Feature
f_AIXAR: int  # 24
f_AOUT: int  # 20
f_AR: int  # 16
f_BIN: int  # 2
f_COFF: int  # 10
f_COM: int  # 23
f_COM_old: int  # 1
f_DRV: int  # 3
f_ELF: int  # 18
f_EXE: int  # 22
f_EXE_old: int  # 0
f_HEX: int  # 5
f_LE: int  # 8
f_LOADER: int  # 17
f_LX: int  # 7
f_MACHO: int  # 25
f_MD1IMG: int  # 27
f_MEX: int  # 6
f_NLM: int  # 9
f_OMF: int  # 12
f_OMFLIB: int  # 15
f_PE: int  # 11
f_PRC: int  # 21
f_PSXOBJ: int  # 26
f_SREC: int  # 13
f_W32RUN: int  # 19
f_WIN: int  # 4
f_ZIP: int  # 14
ida_idaapi: module
sys: module  # <module 'sys' (built-in)>
weakref: module