from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Contains definition of the interface to IDP modules.

The interface consists of two structures:
* definition of target assembler: ::ash
* definition of current processor: ::ph


These structures contain information about target processor and assembler features.
It also defines two groups of kernel events:
* processor_t::event_t processor related events
* idb_event:event_code_t database related events


The processor related events are used to communicate with the processor module. The database related events are used to inform any interested parties, like plugins or processor modules, about the changes in the database. 
    
"""

class IDB_Hooks:
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
    def adding_segm(self, s: segment_t) -> None:
        r"""A segment is being created. 
                  
        :param s: (segment_t *)
        """
        ...
    def allsegs_moved(self, info: segm_move_infos_t) -> None:
        r"""Program rebasing is complete. This event is generated after series of segm_moved events 
                  
        :param info: (segm_move_infos_t *)
        """
        ...
    def auto_empty(self) -> None:
        r"""Info: all analysis queues are empty. This callback is called once when the initial analysis is finished. If the queue is not empty upon the return from this callback, it will be called later again. 
                  
        """
        ...
    def auto_empty_finally(self) -> None:
        r"""Info: all analysis queues are empty definitively. This callback is called only once. 
                  
        """
        ...
    def bookmark_changed(self, index: int, pos: lochist_entry_t, desc: str, operation: int) -> None:
        r"""Boomarked position changed. 
                  
        :param index: (uint32)
        :param pos: (::const lochist_entry_t *)
        :param desc: (::const char *)
        :param operation: (int) 0-added, 1-updated, 2-deleted if desc==nullptr, then the bookmark was deleted.
        """
        ...
    def byte_patched(self, ea: ida_idaapi.ea_t, old_value: int) -> None:
        r"""A byte has been patched. 
                  
        :param ea: (::ea_t)
        :param old_value: (uint32)
        """
        ...
    def callee_addr_changed(self, ea: ida_idaapi.ea_t, callee: ida_idaapi.ea_t) -> None:
        r"""Callee address has been updated by the user. 
                  
        :param ea: (::ea_t)
        :param callee: (::ea_t)
        """
        ...
    def changing_cmt(self, ea: ida_idaapi.ea_t, repeatable_cmt: bool, newcmt: str) -> None:
        r"""An item comment is to be changed. 
                  
        :param ea: (::ea_t)
        :param repeatable_cmt: (bool)
        :param newcmt: (const char *)
        """
        ...
    def changing_op_ti(self, ea: ida_idaapi.ea_t, n: int, new_type: type_t, new_fnames: p_list) -> None:
        r"""An operand typestring (c/c++ prototype) is to be changed. 
                  
        :param ea: (::ea_t)
        :param n: (int)
        :param new_type: (const type_t *)
        :param new_fnames: (const p_list *)
        """
        ...
    def changing_op_type(self, ea: ida_idaapi.ea_t, n: int, opinfo: opinfo_t) -> None:
        r"""An operand type (offset, hex, etc...) is to be changed. 
                  
        :param ea: (::ea_t)
        :param n: (int) eventually or'ed with OPND_OUTER or OPND_ALL
        :param opinfo: (const opinfo_t *) additional operand info
        """
        ...
    def changing_range_cmt(self, kind: range_kind_t, a: range_t, cmt: str, repeatable: bool) -> None:
        r"""Range comment is to be changed. 
                  
        :param kind: (range_kind_t)
        :param a: (const range_t *)
        :param cmt: (const char *)
        :param repeatable: (bool)
        """
        ...
    def changing_segm_class(self, s: segment_t) -> None:
        r"""Segment class is being changed. 
                  
        :param s: (segment_t *)
        """
        ...
    def changing_segm_end(self, s: segment_t, new_end: ida_idaapi.ea_t, segmod_flags: int) -> None:
        r"""Segment end address is to be changed. 
                  
        :param s: (segment_t *)
        :param new_end: (::ea_t)
        :param segmod_flags: (int)
        """
        ...
    def changing_segm_name(self, s: segment_t, oldname: str) -> None:
        r"""Segment name is being changed. 
                  
        :param s: (segment_t *)
        :param oldname: (const char *)
        """
        ...
    def changing_segm_start(self, s: segment_t, new_start: ida_idaapi.ea_t, segmod_flags: int) -> None:
        r"""Segment start address is to be changed. 
                  
        :param s: (segment_t *)
        :param new_start: (::ea_t)
        :param segmod_flags: (int)
        """
        ...
    def changing_ti(self, ea: ida_idaapi.ea_t, new_type: type_t, new_fnames: p_list) -> None:
        r"""An item typestring (c/c++ prototype) is to be changed. 
                  
        :param ea: (::ea_t)
        :param new_type: (const type_t *)
        :param new_fnames: (const p_list *)
        """
        ...
    def closebase(self) -> None:
        r"""The database will be closed now.
        
        """
        ...
    def cmt_changed(self, ea: ida_idaapi.ea_t, repeatable_cmt: bool) -> None:
        r"""An item comment has been changed. 
                  
        :param ea: (::ea_t)
        :param repeatable_cmt: (bool)
        """
        ...
    def compiler_changed(self, adjust_inf_fields: bool) -> None:
        r"""The kernel has changed the compiler information. ( idainfo::cc structure; get_abi_name) 
                  
        :param adjust_inf_fields: (::bool) may change inf fields?
        """
        ...
    def deleting_func(self, pfn: func_t) -> None:
        r"""The kernel is about to delete a function. 
                  
        :param pfn: (func_t *)
        """
        ...
    def deleting_func_tail(self, pfn: func_t, tail: range_t) -> None:
        r"""A function tail chunk is to be removed. 
                  
        :param pfn: (func_t *)
        :param tail: (const range_t *)
        """
        ...
    def deleting_segm(self, start_ea: ida_idaapi.ea_t) -> None:
        r"""A segment is to be deleted. 
                  
        :param start_ea: (::ea_t)
        """
        ...
    def deleting_tryblks(self, range: range_t) -> None:
        r"""About to delete tryblk information in given range 
                  
        :param range: (const range_t *)
        """
        ...
    def destroyed_items(self, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, will_disable_range: bool) -> None:
        r"""Instructions/data have been destroyed in [ea1,ea2). 
                  
        :param ea1: (::ea_t)
        :param ea2: (::ea_t)
        :param will_disable_range: (bool)
        """
        ...
    def determined_main(self, main: ida_idaapi.ea_t) -> None:
        r"""The main() function has been determined. 
                  
        :param main: (::ea_t) address of the main() function
        """
        ...
    def dirtree_link(self, dt: dirtree_t, path: str, link: bool) -> None:
        r"""Dirtree: an item has been linked/unlinked. 
                  
        :param dt: (dirtree_t *)
        :param path: (::const char *)
        :param link: (::bool)
        """
        ...
    def dirtree_mkdir(self, dt: dirtree_t, path: str) -> None:
        r"""Dirtree: a directory has been created. 
                  
        :param dt: (dirtree_t *)
        :param path: (::const char *)
        """
        ...
    def dirtree_move(self, dt: dirtree_t, _from: str, to: str) -> None:
        r"""Dirtree: a directory or item has been moved. 
                  
        :param dt: (dirtree_t *)
        :param to: (::const char *)
        """
        ...
    def dirtree_rank(self, dt: dirtree_t, path: str, rank: size_t) -> None:
        r"""Dirtree: a directory or item rank has been changed. 
                  
        :param dt: (dirtree_t *)
        :param path: (::const char *)
        :param rank: (::size_t)
        """
        ...
    def dirtree_rmdir(self, dt: dirtree_t, path: str) -> None:
        r"""Dirtree: a directory has been deleted. 
                  
        :param dt: (dirtree_t *)
        :param path: (::const char *)
        """
        ...
    def dirtree_rminode(self, dt: dirtree_t, inode: inode_t) -> None:
        r"""Dirtree: an inode became unavailable. 
                  
        :param dt: (dirtree_t *)
        :param inode: (inode_t)
        """
        ...
    def dirtree_segm_moved(self, dt: dirtree_t) -> None:
        r"""Dirtree: inodes were changed due to a segment movement or a program rebasing 
                  
        :param dt: (dirtree_t *)
        """
        ...
    def extlang_changed(self, kind: int, el: extlang_t, idx: int) -> None:
        r"""The list of extlangs or the default extlang was changed. 
                  
        :param kind: (int) 0: extlang installed 1: extlang removed 2: default extlang changed
        :param el: (extlang_t *) pointer to the extlang affected
        :param idx: (int) extlang index
        """
        ...
    def extra_cmt_changed(self, ea: ida_idaapi.ea_t, line_idx: int, cmt: str) -> None:
        r"""An extra comment has been changed. 
                  
        :param ea: (::ea_t)
        :param line_idx: (int)
        :param cmt: (const char *)
        """
        ...
    def flow_chart_created(self, fc: qflow_chart_t) -> None:
        r"""Gui has retrieved a function flow chart. Plugins may modify the flow chart in this callback. 
                  
        :param fc: (qflow_chart_t *)
        """
        ...
    def frame_created(self, func_ea: ida_idaapi.ea_t) -> None:
        r"""A function frame has been created. 
                  
        :param func_ea: (::ea_t) idb_event::frame_deleted
        """
        ...
    def frame_deleted(self, pfn: func_t) -> None:
        r"""The kernel has deleted a function frame. 
                  
        :param pfn: (func_t *) idb_event::frame_created
        """
        ...
    def frame_expanded(self, func_ea: ida_idaapi.ea_t, udm_tid: tid_t, delta: adiff_t) -> None:
        r"""A frame type has been expanded/shrank. 
                  
        :param func_ea: (::ea_t)
        :param udm_tid: (tid_t) the gap was added/removed before this member
        :param delta: (::adiff_t) number of added/removed bytes
        """
        ...
    def frame_udm_changed(self, func_ea: ida_idaapi.ea_t, udm_tid: tid_t, udmold: udm_t, udmnew: udm_t) -> None:
        r"""Frame member has been changed. 
                  
        :param func_ea: (::ea_t)
        :param udm_tid: (tid_t)
        :param udmold: (::const udm_t *)
        :param udmnew: (::const udm_t *)
        """
        ...
    def frame_udm_created(self, func_ea: ida_idaapi.ea_t, udm: udm_t) -> None:
        r"""Frame member has been added. 
                  
        :param func_ea: (::ea_t)
        :param udm: (::const udm_t *)
        """
        ...
    def frame_udm_deleted(self, func_ea: ida_idaapi.ea_t, udm_tid: tid_t, udm: udm_t) -> None:
        r"""Frame member has been deleted. 
                  
        :param func_ea: (::ea_t)
        :param udm_tid: (tid_t)
        :param udm: (::const udm_t *)
        """
        ...
    def frame_udm_renamed(self, func_ea: ida_idaapi.ea_t, udm: udm_t, oldname: str) -> None:
        r"""Frame member has been renamed. 
                  
        :param func_ea: (::ea_t)
        :param udm: (::const udm_t *)
        :param oldname: (::const char *)
        """
        ...
    def func_added(self, pfn: func_t) -> None:
        r"""The kernel has added a function. 
                  
        :param pfn: (func_t *)
        """
        ...
    def func_deleted(self, func_ea: ida_idaapi.ea_t) -> None:
        r"""A function has been deleted. 
                  
        :param func_ea: (::ea_t)
        """
        ...
    def func_noret_changed(self, pfn: func_t) -> None:
        r"""FUNC_NORET bit has been changed. 
                  
        :param pfn: (func_t *)
        """
        ...
    def func_tail_appended(self, pfn: func_t, tail: func_t) -> None:
        r"""A function tail chunk has been appended. 
                  
        :param pfn: (func_t *)
        :param tail: (func_t *)
        """
        ...
    def func_tail_deleted(self, pfn: func_t, tail_ea: ida_idaapi.ea_t) -> None:
        r"""A function tail chunk has been removed. 
                  
        :param pfn: (func_t *)
        :param tail_ea: (::ea_t)
        """
        ...
    def func_updated(self, pfn: func_t) -> None:
        r"""The kernel has updated a function. 
                  
        :param pfn: (func_t *)
        """
        ...
    def hook(self) -> bool:
        ...
    def idasgn_loaded(self, short_sig_name: str) -> None:
        r"""FLIRT signature has been loaded for normal processing (not for recognition of startup sequences). 
                  
        :param short_sig_name: (const char *)
        """
        ...
    def idasgn_matched_ea(self, ea: ida_idaapi.ea_t, name: str, lib_name: str) -> None:
        r"""A FLIRT match has been found 
                  
        :param ea: (::ea_t) the matching address
        :param name: (::const char *) the matched name
        :param lib_name: (::const char *) library name extracted from signature file
        """
        ...
    def item_color_changed(self, ea: ida_idaapi.ea_t, color: bgcolor_t) -> None:
        r"""An item color has been changed. 
                  
        :param ea: (::ea_t)
        :param color: (bgcolor_t) if color==DEFCOLOR, the the color is deleted.
        """
        ...
    def kernel_config_loaded(self, pass_number: int) -> None:
        r"""This event is issued when ida.cfg is parsed. 
                  
        :param pass_number: (int)
        """
        ...
    def loader_finished(self, li: linput_t, neflags: uint16, filetypename: str) -> None:
        r"""External file loader finished its work. Use this event to augment the existing loader functionality. 
                  
        :param li: (linput_t *)
        :param neflags: (uint16) Load file flags
        :param filetypename: (const char *)
        """
        ...
    def local_type_renamed(self, ordinal: int, oldname: str, newname: str) -> None:
        r"""Local type has been renamed 
                  
        :param ordinal: (uint32) 0 means ordinal is unknown
        :param oldname: (const char *) nullptr means name is unknown
        :param newname: (const char *) nullptr means name is unknown
        """
        ...
    def local_types_changed(self, ltc: local_type_change_t, ordinal: int, name: str) -> None:
        r"""Local types have been changed 
                  
        :param ltc: (local_type_change_t)
        :param ordinal: (uint32) 0 means ordinal is unknown
        :param name: (const char *) nullptr means name is unknown
        """
        ...
    def lt_edm_changed(self, enumname: str, edm_tid: tid_t, edmold: edm_t, edmnew: edm_t) -> None:
        r"""local type enum member has been changed 
                  
        :param enumname: (::const char *)
        :param edm_tid: (tid_t)
        :param edmold: (::const edm_t *)
        :param edmnew: (::const edm_t *)
        """
        ...
    def lt_edm_created(self, enumname: str, edm: edm_t) -> None:
        r"""local type enum member has been added 
                  
        :param enumname: (::const char *)
        :param edm: (::const edm_t *)
        """
        ...
    def lt_edm_deleted(self, enumname: str, edm_tid: tid_t, edm: edm_t) -> None:
        r"""local type enum member has been deleted 
                  
        :param enumname: (::const char *)
        :param edm_tid: (tid_t)
        :param edm: (::const edm_t *)
        """
        ...
    def lt_edm_renamed(self, enumname: str, edm: edm_t, oldname: str) -> None:
        r"""local type enum member has been renamed 
                  
        :param enumname: (::const char *)
        :param edm: (::const edm_t *)
        :param oldname: (::const char *)
        """
        ...
    def lt_udm_changed(self, udtname: str, udm_tid: tid_t, udmold: udm_t, udmnew: udm_t) -> None:
        r"""local type udt member has been changed 
                  
        :param udtname: (::const char *)
        :param udm_tid: (tid_t)
        :param udmold: (::const udm_t *)
        :param udmnew: (::const udm_t *)
        """
        ...
    def lt_udm_created(self, udtname: str, udm: udm_t) -> None:
        r"""local type udt member has been added 
                  
        :param udtname: (::const char *)
        :param udm: (::const udm_t *)
        """
        ...
    def lt_udm_deleted(self, udtname: str, udm_tid: tid_t, udm: udm_t) -> None:
        r"""local type udt member has been deleted 
                  
        :param udtname: (::const char *)
        :param udm_tid: (tid_t)
        :param udm: (::const udm_t *)
        """
        ...
    def lt_udm_renamed(self, udtname: str, udm: udm_t, oldname: str) -> None:
        r"""local type udt member has been renamed 
                  
        :param udtname: (::const char *)
        :param udm: (::const udm_t *)
        :param oldname: (::const char *)
        """
        ...
    def lt_udt_expanded(self, udtname: str, udm_tid: tid_t, delta: adiff_t) -> None:
        r"""A structure type has been expanded/shrank. 
                  
        :param udtname: (::const char *)
        :param udm_tid: (tid_t) the gap was added/removed before this member
        :param delta: (::adiff_t) number of added/removed bytes
        """
        ...
    def make_code(self, insn: insn_t) -> None:
        r"""An instruction is being created. 
                  
        :param insn: (const insn_t*)
        """
        ...
    def make_data(self, ea: ida_idaapi.ea_t, flags: flags64_t, tid: tid_t, len: asize_t) -> None:
        r"""A data item is being created. 
                  
        :param ea: (::ea_t)
        :param flags: (flags64_t)
        :param tid: (tid_t)
        :param len: (::asize_t)
        """
        ...
    def op_ti_changed(self, ea: ida_idaapi.ea_t, n: int, type: type_t, fnames: p_list) -> None:
        r"""An operand typestring (c/c++ prototype) has been changed. 
                  
        :param ea: (::ea_t)
        :param n: (int)
        :param type: (const type_t *)
        :param fnames: (const p_list *)
        """
        ...
    def op_type_changed(self, ea: ida_idaapi.ea_t, n: int) -> None:
        r"""An operand type (offset, hex, etc...) has been set or deleted. 
                  
        :param ea: (::ea_t)
        :param n: (int) eventually or'ed with OPND_OUTER or OPND_ALL
        """
        ...
    def range_cmt_changed(self, kind: range_kind_t, a: range_t, cmt: str, repeatable: bool) -> None:
        r"""Range comment has been changed. 
                  
        :param kind: (range_kind_t)
        :param a: (const range_t *)
        :param cmt: (const char *)
        :param repeatable: (bool)
        """
        ...
    def renamed(self, ea: ida_idaapi.ea_t, new_name: str, local_name: bool, old_name: str) -> None:
        r"""The kernel has renamed a byte. See also the rename event 
                  
        :param ea: (::ea_t)
        :param new_name: (const char *) can be nullptr
        :param local_name: (bool)
        :param old_name: (const char *) can be nullptr
        """
        ...
    def savebase(self) -> None:
        r"""The database is being saved.
        
        """
        ...
    def segm_added(self, s: segment_t) -> None:
        r"""A new segment has been created. 
                  
        :param s: (segment_t *) See also adding_segm
        """
        ...
    def segm_attrs_updated(self, s: segment_t) -> None:
        r"""Segment attributes has been changed. 
                  
        :param s: (segment_t *) This event is generated for secondary segment attributes (examples: color, permissions, etc)
        """
        ...
    def segm_class_changed(self, s: segment_t, sclass: str) -> None:
        r"""Segment class has been changed. 
                  
        :param s: (segment_t *)
        :param sclass: (const char *)
        """
        ...
    def segm_deleted(self, start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, flags: int) -> None:
        r"""A segment has been deleted. 
                  
        :param start_ea: (::ea_t)
        :param end_ea: (::ea_t)
        :param flags: (int)
        """
        ...
    def segm_end_changed(self, s: segment_t, oldend: ida_idaapi.ea_t) -> None:
        r"""Segment end address has been changed. 
                  
        :param s: (segment_t *)
        :param oldend: (::ea_t)
        """
        ...
    def segm_moved(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, size: asize_t, changed_netmap: bool) -> None:
        r"""Segment has been moved. 
                  
        :param to: (::ea_t)
        :param size: (::asize_t)
        :param changed_netmap: (bool) See also idb_event::allsegs_moved
        """
        ...
    def segm_name_changed(self, s: segment_t, name: str) -> None:
        r"""Segment name has been changed. 
                  
        :param s: (segment_t *)
        :param name: (const char *)
        """
        ...
    def segm_start_changed(self, s: segment_t, oldstart: ida_idaapi.ea_t) -> None:
        r"""Segment start address has been changed. 
                  
        :param s: (segment_t *)
        :param oldstart: (::ea_t)
        """
        ...
    def set_func_end(self, pfn: func_t, new_end: ida_idaapi.ea_t) -> None:
        r"""Function chunk end address will be changed. 
                  
        :param pfn: (func_t *)
        :param new_end: (::ea_t)
        """
        ...
    def set_func_start(self, pfn: func_t, new_start: ida_idaapi.ea_t) -> None:
        r"""Function chunk start address will be changed. 
                  
        :param pfn: (func_t *)
        :param new_start: (::ea_t)
        """
        ...
    def sgr_changed(self, start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, regnum: int, value: sel_t, old_value: sel_t, tag: uchar) -> None:
        r"""The kernel has changed a segment register value. 
                  
        :param start_ea: (::ea_t)
        :param end_ea: (::ea_t)
        :param regnum: (int)
        :param value: (::sel_t)
        :param old_value: (::sel_t)
        :param tag: (uchar) Segment register range tags
        """
        ...
    def sgr_deleted(self, start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, regnum: int) -> None:
        r"""The kernel has deleted a segment register value. 
                  
        :param start_ea: (::ea_t)
        :param end_ea: (::ea_t)
        :param regnum: (int)
        """
        ...
    def stkpnts_changed(self, pfn: func_t) -> None:
        r"""Stack change points have been modified. 
                  
        :param pfn: (func_t *)
        """
        ...
    def tail_owner_changed(self, tail: func_t, owner_func: ida_idaapi.ea_t, old_owner: ida_idaapi.ea_t) -> None:
        r"""A tail chunk owner has been changed. 
                  
        :param tail: (func_t *)
        :param owner_func: (::ea_t)
        :param old_owner: (::ea_t)
        """
        ...
    def thunk_func_created(self, pfn: func_t) -> None:
        r"""A thunk bit has been set for a function. 
                  
        :param pfn: (func_t *)
        """
        ...
    def ti_changed(self, ea: ida_idaapi.ea_t, type: type_t, fnames: p_list) -> None:
        r"""An item typestring (c/c++ prototype) has been changed. 
                  
        :param ea: (::ea_t)
        :param type: (const type_t *)
        :param fnames: (const p_list *)
        """
        ...
    def tryblks_updated(self, tbv: tryblks_t) -> None:
        r"""Updated tryblk information 
                  
        :param tbv: (const ::tryblks_t *)
        """
        ...
    def unhook(self) -> bool:
        ...
    def updating_tryblks(self, tbv: tryblks_t) -> None:
        r"""About to update tryblk information 
                  
        :param tbv: (const ::tryblks_t *)
        """
        ...
    def upgraded(self, _from: int) -> None:
        r"""The database has been upgraded and the receiver can upgrade its info as well 
                  
        """
        ...

class IDP_Hooks:
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
    def ev_add_cref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: cref_t) -> int:
        r"""A code reference is being created. 
                  
        :param to: (::ea_t)
        :param type: (cref_t)
        :returns: <0: cancel cref creation
        :returns: 0: not implemented or continue
        """
        ...
    def ev_add_dref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: dref_t) -> int:
        r"""A data reference is being created. 
                  
        :param to: (::ea_t)
        :param type: (dref_t)
        :returns: <0: cancel dref creation
        :returns: 0: not implemented or continue
        """
        ...
    def ev_adjust_argloc(self, argloc: argloc_t, optional_type: tinfo_t, size: int) -> int:
        r"""Adjust argloc according to its type/size and platform endianess 
                  
        :param argloc: (argloc_t *), inout
        :param size: (int) 'size' makes no sense if type != nullptr (type->get_size() should be used instead)
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error
        """
        ...
    def ev_adjust_libfunc_ea(self, sig: idasgn_t, libfun: libfunc_t, ea: ea_t) -> int:
        r"""Called when a signature module has been matched against bytes in the database. This is used to compute the offset at which a particular module's libfunc should be applied. 
                  
        :param sig: (const idasgn_t *)
        :param libfun: (const libfunc_t *)
        :param ea: (::ea_t *)
        :returns: 1: the ea_t pointed to by the third argument was modified.
        :returns: <=0: not modified. use default algorithm.
        """
        ...
    def ev_adjust_refinfo(self, ri: refinfo_t, ea: ida_idaapi.ea_t, n: int, fd: fixup_data_t) -> int:
        r"""Called from apply_fixup before converting operand to reference. Can be used for changing the reference info. (e.g. the PPC module adds REFINFO_NOBASE for some references) 
                  
        :param ri: (refinfo_t *)
        :param ea: (::ea_t) instruction address
        :param n: (int) operand number
        :param fd: (const fixup_data_t *)
        :returns: <0: do not create an offset
        :returns: 0: not implemented or refinfo adjusted
        """
        ...
    def ev_ana_insn(self, out: insn_t) -> bool:
        r"""Analyze one instruction and fill 'out' structure. This function shouldn't change the database, flags or anything else. All these actions should be performed only by emu_insn() function. insn_t::ea contains address of instruction to analyze. 
                  
        :param out: (insn_t *)
        :returns: length of the instruction in bytes, 0 if instruction can't be decoded.
        :returns: 0: if instruction can't be decoded.
        """
        ...
    def ev_analyze_prolog(self, ea: ida_idaapi.ea_t) -> int:
        r"""Analyzes function prolog, epilog, and updates purge, and function attributes 
                  
        :param ea: (::ea_t) start of function
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_arch_changed(self) -> int:
        r"""The loader is done parsing arch-related information, which the processor module might want to use to finish its initialization. 
                  
        :returns: 1: if success
        :returns: 0: not implemented or failed
        """
        ...
    def ev_arg_addrs_ready(self, caller: ida_idaapi.ea_t, n: int, tif: tinfo_t, addrs: ea_t) -> int:
        r"""Argument address info is ready. 
                  
        :param caller: (::ea_t)
        :param n: (int) number of formal arguments
        :param tif: (tinfo_t *) call prototype
        :param addrs: (::ea_t *) argument intilization addresses
        :returns: <0: do not save into idb; other values mean "ok to save"
        """
        ...
    def ev_asm_installed(self, asmnum: int) -> int:
        r"""After setting a new assembler 
                  
        :param asmnum: (int) See also ev_newasm
        """
        ...
    def ev_assemble(self, ea: ida_idaapi.ea_t, cs: ida_idaapi.ea_t, ip: ida_idaapi.ea_t, use32: bool, line: str) -> Any:
        r"""Assemble an instruction. (display a warning if an error is found). 
                  
        :param ea: (::ea_t) linear address of instruction
        :param cs: (::ea_t) cs of instruction
        :param ip: (::ea_t) ip of instruction
        :param use32: (bool) is 32bit segment?
        :param line: (const char *) line to assemble
        :returns: size of the instruction in bytes
        """
        ...
    def ev_auto_queue_empty(self, type: atype_t) -> int:
        r"""One analysis queue is empty. 
                  
        :param type: (atype_t)
        :returns: void: see also idb_event::auto_empty_finally
        """
        ...
    def ev_calc_arglocs(self, fti: func_type_data_t) -> int:
        r"""Calculate function argument locations. This callback should fill retloc, all arglocs, and stkargs. This callback is never called for CM_CC_SPECIAL functions. 
                  
        :param fti: (func_type_data_t *) points to the func type info
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error
        """
        ...
    def ev_calc_cdecl_purged_bytes(self, ea: ida_idaapi.ea_t) -> int:
        r"""Calculate number of purged bytes after call. 
                  
        :param ea: (::ea_t) address of the call instruction
        :returns: number of purged bytes (usually add sp, N)
        """
        ...
    def ev_calc_next_eas(self, res: eavec_t, insn: insn_t, over: bool) -> int:
        r"""Calculate list of addresses the instruction in 'insn' may pass control to. This callback is required for source level debugging. 
                  
        :param res: (eavec_t *), out: array for the results.
        :param insn: (const insn_t*) the instruction
        :param over: (bool) calculate for step over (ignore call targets)
        :returns: <0: incalculable (indirect jumps, for example)
        :returns: >=0: number of addresses of called functions in the array. They must be put at the beginning of the array (0 if over=true)
        """
        ...
    def ev_calc_purged_bytes(self, p_purged_bytes: int, fti: func_type_data_t) -> int:
        r"""Calculate number of purged bytes by the given function type. 
                  
        :param p_purged_bytes: (int *) ptr to output
        :param fti: (const func_type_data_t *) func type details
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_calc_retloc(self, retloc: argloc_t, rettype: tinfo_t, cc: callcnv_t) -> int:
        r"""Calculate return value location. 
                  
        :param retloc: (argloc_t *)
        :param rettype: (const tinfo_t *)
        :param cc: (::callcnv_t)
        :returns: 0: not implemented
        :returns: 1: ok,
        :returns: -1: error
        """
        ...
    def ev_calc_spdelta(self, spdelta: sval_t, insn: insn_t) -> int:
        r"""Calculate amount of change to sp for the given insn. This event is required to decompile code snippets. 
                  
        :param spdelta: (sval_t *)
        :param insn: (const insn_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_calc_step_over(self, target: ea_t, ip: ida_idaapi.ea_t) -> int:
        r"""Calculate the address of the instruction which will be executed after "step over". The kernel will put a breakpoint there. If the step over is equal to step into or we cannot calculate the address, return BADADDR. 
                  
        :param target: (::ea_t *) pointer to the answer
        :param ip: (::ea_t) instruction address
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_calc_switch_cases(self, casevec: casevec_t, targets: eavec_t, insn_ea: ida_idaapi.ea_t, si: switch_info_t) -> int:
        r"""Calculate case values and targets for a custom jump table. 
                  
        :param casevec: (::casevec_t *) vector of case values (may be nullptr)
        :param targets: (eavec_t *) corresponding target addresses (my be nullptr)
        :param insn_ea: (::ea_t) address of the 'indirect jump' instruction
        :param si: (switch_info_t *) switch information
        :returns: 1: ok
        :returns: <=0: failed
        """
        ...
    def ev_calc_varglocs(self, ftd: func_type_data_t, aux_regs: regobjs_t, aux_stkargs: relobj_t, nfixed: int) -> int:
        r"""Calculate locations of the arguments that correspond to '...'. 
                  
        :param ftd: (func_type_data_t *), inout: info about all arguments (including varargs)
        :param aux_regs: (regobjs_t *) buffer for hidden register arguments, may be nullptr
        :param aux_stkargs: (relobj_t *) buffer for hidden stack arguments, may be nullptr
        :param nfixed: (int) number of fixed arguments
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error On some platforms variadic calls require passing additional information: for example, number of floating variadic arguments must be passed in rax on gcc-x64. The locations and values that constitute this additional information are returned in the buffers pointed by aux_regs and aux_stkargs
        """
        ...
    def ev_calcrel(self) -> int:
        r"""Reserved.
        
        """
        ...
    def ev_can_have_type(self, op: op_t) -> int:
        r"""Can the operand have a type as offset, segment, decimal, etc? (for example, a register AX can't have a type, meaning that the user can't change its representation. see bytes.hpp for information about types and flags) 
                  
        :param op: (const op_t *)
        :returns: 0: unknown
        :returns: <0: no
        :returns: 1: yes
        """
        ...
    def ev_clean_tbit(self, ea: ida_idaapi.ea_t, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Clear the TF bit after an insn like pushf stored it in memory. 
                  
        :param ea: (::ea_t) instruction address
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 1: ok
        :returns: 0: failed
        """
        ...
    def ev_cmp_operands(self, op1: op_t, op2: op_t) -> int:
        r"""Compare instruction operands 
                  
        :param op1: (const op_t*)
        :param op2: (const op_t*)
        :returns: 1: equal
        :returns: -1: not equal
        :returns: 0: not implemented
        """
        ...
    def ev_coagulate(self, start_ea: ida_idaapi.ea_t) -> int:
        r"""Try to define some unexplored bytes. This notification will be called if the kernel tried all possibilities and could not find anything more useful than to convert to array of bytes. The module can help the kernel and convert the bytes into something more useful. 
                  
        :param start_ea: (::ea_t)
        :returns: number of converted bytes
        """
        ...
    def ev_coagulate_dref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, may_define: bool, code_ea: ea_t) -> int:
        r"""Data reference is being analyzed. plugin may correct 'code_ea' (e.g. for thumb mode refs, we clear the last bit) 
                  
        :param to: (::ea_t)
        :param may_define: (bool)
        :param code_ea: (::ea_t *)
        :returns: <0: failed dref analysis, >0 done dref analysis
        :returns: 0: not implemented or continue
        """
        ...
    def ev_create_flat_group(self, image_base: ida_idaapi.ea_t, bitness: int, dataseg_sel: sel_t) -> int:
        r"""Create special segment representing the flat group. 
                  
        :param image_base: (::ea_t)
        :param bitness: (int)
        :param dataseg_sel: (::sel_t) return value is ignored
        """
        ...
    def ev_create_func_frame(self, pfn: func_t) -> int:
        r"""Create a function frame for a newly created function Set up frame size, its attributes etc 
                  
        :param pfn: (func_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_create_merge_handlers(self, md: merge_data_t) -> int:
        r"""Create merge handlers, if needed 
                  
        :param md: (merge_data_t *) This event is generated immediately after opening idbs.
        :returns: must be 0
        """
        ...
    def ev_create_switch_xrefs(self, jumpea: ida_idaapi.ea_t, si: switch_info_t) -> int:
        r"""Create xrefs for a custom jump table. 
                  
        :param jumpea: (::ea_t) address of the jump insn
        :param si: (const switch_info_t *) switch information
        :returns: must return 1 Must be implemented if module uses custom jump tables, SWI_CUSTOM
        """
        ...
    def ev_creating_segm(self, seg: segment_t) -> int:
        r"""A new segment is about to be created. 
                  
        :param seg: (segment_t *)
        :returns: 1: ok
        :returns: <0: segment should not be created
        """
        ...
    def ev_cvt64_hashval(self, node: nodeidx_t, tag: uchar, name: str, data: uchar) -> int:
        r"""perform 32-64 conversion for a hash value 
                  
        :param node: (::nodeidx_t)
        :param tag: (uchar)
        :param name: (const ::char *)
        :param data: (const uchar *)
        :returns: 0: nothing was done
        :returns: 1: converted successfully
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_cvt64_supval(self, node: nodeidx_t, tag: uchar, idx: nodeidx_t, data: uchar) -> int:
        r"""perform 32-64 conversion for a netnode array element 
                  
        :param node: (::nodeidx_t)
        :param tag: (uchar)
        :param idx: (::nodeidx_t)
        :param data: (const uchar *)
        :returns: 0: nothing was done
        :returns: 1: converted successfully
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_decorate_name(self, name: str, mangle: bool, cc: int, optional_type: tinfo_t) -> Any:
        r"""Decorate/undecorate a C symbol name. 
                  
        :param name: (const char *) name of symbol
        :param mangle: (bool) true-mangle, false-unmangle
        :param cc: (::callcnv_t) calling convention
        :returns: 1: if success
        :returns: 0: not implemented or failed
        """
        ...
    def ev_del_cref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, expand: bool) -> int:
        r"""A code reference is being deleted. 
                  
        :param to: (::ea_t)
        :param expand: (bool)
        :returns: <0: cancel cref deletion
        :returns: 0: not implemented or continue
        """
        ...
    def ev_del_dref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t) -> int:
        r"""A data reference is being deleted. 
                  
        :param to: (::ea_t)
        :returns: <0: cancel dref deletion
        :returns: 0: not implemented or continue
        """
        ...
    def ev_delay_slot_insn(self, ea: ida_idaapi.ea_t, bexec: bool, fexec: bool) -> Any:
        r"""Get delay slot instruction 
                  
        :param ea: (::ea_t *) in: instruction address in question, out: (if the answer is positive) if the delay slot contains valid insn: the address of the delay slot insn else: BADADDR (invalid insn, e.g. a branch)
        :param bexec: (bool *) execute slot if jumping, initially set to 'true'
        :param fexec: (bool *) execute slot if not jumping, initally set to 'true'
        :returns: 1: positive answer
        :returns: <=0: ordinary insn
        """
        ...
    def ev_demangle_name(self, name: str, disable_mask: int, demreq: int) -> Any:
        r"""Demangle a C++ (or another language) name into a user-readable string. This event is called by demangle_name() 
                  
        :param name: (const char *) mangled name
        :param disable_mask: (uint32) flags to inhibit parts of output or compiler info/other (see MNG_)
        :param demreq: (demreq_type_t) operation to perform
        :returns: 1: if success
        :returns: 0: not implemented
        """
        ...
    def ev_emu_insn(self, insn: insn_t) -> bool:
        r"""Emulate instruction, create cross-references, plan to analyze subsequent instructions, modify flags etc. Upon entrance to this function, all information about the instruction is in 'insn' structure. 
                  
        :param insn: (const insn_t *)
        :returns: 1: ok
        :returns: -1: the kernel will delete the instruction
        """
        ...
    def ev_endbinary(self, ok: bool) -> int:
        r"""IDA has loaded a binary file. 
                  
        :param ok: (bool) file loaded successfully?
        """
        ...
    def ev_ending_undo(self, action_name: str, is_undo: bool) -> int:
        r"""Ended undoing/redoing an action 
                  
        :param action_name: (const char *) action that we finished undoing/redoing. is not nullptr.
        :param is_undo: (bool) true if performing undo, false if performing redo
        """
        ...
    def ev_equal_reglocs(self, a1: argloc_t, a2: argloc_t) -> int:
        r"""Are 2 register arglocs the same?. We need this callback for the pc module. 
                  
        :param a1: (argloc_t *)
        :param a2: (argloc_t *)
        :returns: 1: yes
        :returns: -1: no
        :returns: 0: not implemented
        """
        ...
    def ev_extract_address(self, out_ea: ea_t, screen_ea: ida_idaapi.ea_t, string: str, position: size_t) -> int:
        r"""Extract address from a string. 
                  
        :param out_ea: (ea_t *), out
        :param screen_ea: (ea_t)
        :param string: (const char *)
        :param position: (size_t)
        :returns: 1: ok
        :returns: 0: kernel should use the standard algorithm
        :returns: -1: error
        """
        ...
    def ev_find_op_value(self, pinsn: insn_t, opn: int) -> Any:
        r"""Find operand value via a register tracker. The returned value in 'out' is valid before executing the instruction. 
                  
        :param pinsn: (const insn_t *) instruction
        :param opn: (int) operand index
        :returns: 1: if implemented, and value was found
        :returns: 0: not implemented, -1 decoding failed, or no value found
        """
        ...
    def ev_find_reg_value(self, pinsn: insn_t, reg: int) -> Any:
        r"""Find register value via a register tracker. The returned value in 'out' is valid before executing the instruction. 
                  
        :param pinsn: (const insn_t *) instruction
        :param reg: (int) register index
        :returns: 1: if implemented, and value was found
        :returns: 0: not implemented, -1 decoding failed, or no value found
        """
        ...
    def ev_func_bounds(self, possible_return_code: int, pfn: func_t, max_func_end_ea: ida_idaapi.ea_t) -> int:
        r"""find_func_bounds() finished its work. The module may fine tune the function bounds 
                  
        :param possible_return_code: (int *), in/out
        :param pfn: (func_t *)
        :param max_func_end_ea: (::ea_t) (from the kernel's point of view)
        :returns: void: 
        """
        ...
    def ev_gen_asm_or_lst(self, starting: bool, fp: FILE, is_asm: bool, flags: int, outline: html_line_cb_t) -> int:
        r"""Callback: generating asm or lst file. The kernel calls this callback twice, at the beginning and at the end of listing generation. The processor module can intercept this event and adjust its output 
                  
        :param starting: (bool) beginning listing generation
        :param fp: (FILE *) output file
        :param is_asm: (bool) true:assembler, false:listing
        :param flags: (int) flags passed to gen_file()
        :param outline: (html_line_cb_t **) ptr to ptr to outline callback. if this callback is defined for this code, it will be used by the kernel to output the generated lines
        :returns: void: 
        """
        ...
    def ev_gen_map_file(self, nlines: int, fp: FILE) -> int:
        r"""Generate map file. If not implemented the kernel itself will create the map file. 
                  
        :param nlines: (int *) number of lines in map file (-1 means write error)
        :param fp: (FILE *) output file
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: write error
        """
        ...
    def ev_gen_regvar_def(self, outctx: outctx_t, v: regvar_t) -> int:
        r"""Generate register variable definition line. 
                  
        :param outctx: (outctx_t *)
        :param v: (regvar_t *)
        :returns: >0: ok, generated the definition text
        :returns: 0: not implemented
        """
        ...
    def ev_gen_src_file_lnnum(self, outctx: outctx_t, file: str, lnnum: size_t) -> int:
        r"""Callback: generate analog of: 
             #line  123
            
        
        
                  
        :param outctx: (outctx_t *) output context
        :param file: (const char *) source file (may be nullptr)
        :param lnnum: (size_t) line number
        :returns: 1: directive has been generated
        :returns: 0: not implemented
        """
        ...
    def ev_gen_stkvar_def(self, outctx: outctx_t, stkvar: udm_t, v: int, tid: tid_t) -> int:
        r"""Generate stack variable definition line Default line is varname = type ptr value, where 'type' is one of byte,word,dword,qword,tbyte 
                  
        :param outctx: (outctx_t *)
        :param stkvar: (const udm_t *)
        :param v: (sval_t)
        :param tid: (tid_t) stkvar TID
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_get_abi_info(self, comp: comp_t) -> int:
        r"""Get all possible ABI names and optional extensions for given compiler abiname/option is a string entirely consisting of letters, digits and underscore 
                  
        :param comp: (comp_t) - compiler ID
        :returns: 0: not implemented
        :returns: 1: ok
        """
        ...
    def ev_get_autocmt(self, insn: insn_t) -> Any:
        r"""Callback: get dynamic auto comment. Will be called if the autocomments are enabled and the comment retrieved from ida.int starts with '$!'. 'insn' contains valid info. 
                  
        :param insn: (const insn_t*) the instruction
        :returns: 1: new comment has been generated
        :returns: 0: callback has not been handled. the buffer must not be changed in this case
        """
        ...
    def ev_get_bg_color(self, color: bgcolor_t, ea: ida_idaapi.ea_t) -> int:
        r"""Get item background color. Plugins can hook this callback to color disassembly lines dynamically 
                  
        :param color: (bgcolor_t *), out
        :param ea: (::ea_t)
        :returns: 0: not implemented
        :returns: 1: color set
        """
        ...
    def ev_get_cc_regs(self, regs: callregs_t, cc: callcnv_t) -> int:
        r"""Get register allocation convention for given calling convention 
                  
        :param regs: (callregs_t *), out
        :param cc: (::callcnv_t)
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_get_code16_mode(self, ea: ida_idaapi.ea_t) -> int:
        r"""Get ISA 16-bit mode 
                  
        :param ea: (ea_t) address to get the ISA mode
        :returns: 1: 16-bit mode
        :returns: 0: not implemented or 32-bit mode
        """
        ...
    def ev_get_dbr_opnum(self, opnum: int, insn: insn_t) -> int:
        r"""Get the number of the operand to be displayed in the debugger reference view (text mode). 
                  
        :param opnum: (int *) operand number (out, -1 means no such operand)
        :param insn: (const insn_t*) the instruction
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_get_default_enum_size(self) -> int:
        r"""Get default enum size. Not generated anymore. inf_get_cc_size_e() is used instead 
                  
        """
        ...
    def ev_get_frame_retsize(self, frsize: int, pfn: func_t) -> int:
        r"""Get size of function return address in bytes If this event is not implemented, the kernel will assume
        * 8 bytes for 64-bit function
        * 4 bytes for 32-bit function
        * 2 bytes otherwise
        
        
        
        :param frsize: (int *) frame size (out)
        :param pfn: (const func_t *), can't be nullptr
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_get_macro_insn_head(self, head: ea_t, ip: ida_idaapi.ea_t) -> int:
        r"""Calculate the start of a macro instruction. This notification is called if IP points to the middle of an instruction 
                  
        :param head: (::ea_t *), out: answer, BADADDR means normal instruction
        :param ip: (::ea_t) instruction address
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_get_operand_string(self, insn: insn_t, opnum: int) -> Any:
        r"""Request text string for operand (cli, java, ...). 
                  
        :param insn: (const insn_t*) the instruction
        :param opnum: (int) operand number, -1 means any string operand
        :returns: 0: no string (or empty string)
        :returns: >0: original string length without terminating zero
        """
        ...
    def ev_get_procmod(self) -> int:
        r"""Get pointer to the processor module object. All processor modules must implement this. The pointer is returned as size_t. 
                  
        """
        ...
    def ev_get_reg_accesses(self, accvec: reg_accesses_t, insn: insn_t, flags: int) -> int:
        r"""Get info about the registers that are used/changed by an instruction. 
                  
        :param accvec: (reg_accesses_t*) out: info about accessed registers
        :param insn: (const insn_t *) instruction in question
        :param flags: (int) reserved, must be 0
        :returns: -1: if accvec is nullptr
        :returns: 1: found the requested access (and filled accvec)
        :returns: 0: not implemented
        """
        ...
    def ev_get_reg_info(self, main_regname: char, bitrange: bitrange_t, regname: str) -> int:
        r"""Get register information by its name. example: "ah" returns:
        * main_regname="eax"
        * bitrange_t = { offset==8, nbits==8 }
        
        
        This callback may be unimplemented if the register names are all present in processor_t::reg_names and they all have the same size 
                  
        :param main_regname: (const char **), out
        :param bitrange: (bitrange_t *), out: position and size of the value within 'main_regname' (empty bitrange == whole register)
        :param regname: (const char *)
        :returns: 1: ok
        :returns: -1: failed (not found)
        :returns: 0: unimplemented
        """
        ...
    def ev_get_reg_name(self, reg: int, width: size_t, reghi: int) -> Any:
        r"""Generate text representation of a register. Most processor modules do not need to implement this callback. It is useful only if processor_t::reg_names[reg] does not provide the correct register name. 
                  
        :param reg: (int) internal register number as defined in the processor module
        :param width: (size_t) register width in bytes
        :param reghi: (int) if not -1 then this function will return the register pair
        :returns: -1: if error
        :returns: strlen(buf): if success
        """
        ...
    def ev_get_simd_types(self, out: simd_info_vec_t, simd_attrs: simd_info_t, argloc: argloc_t, create_tifs: bool) -> int:
        r"""Get SIMD-related types according to given attributes ant/or argument location 
                  
        :param out: (::simd_info_vec_t *)
        :param simd_attrs: (const simd_info_t *), may be nullptr
        :param argloc: (const argloc_t *), may be nullptr
        :param create_tifs: (bool) return valid tinfo_t objects, create if neccessary
        :returns: number: of found types
        :returns: -1: error If name==nullptr, initialize all SIMD types
        """
        ...
    def ev_get_stkarg_area_info(self, out: stkarg_area_info_t, cc: callcnv_t) -> int:
        r"""Get some metrics of the stack argument area. 
                  
        :param out: (stkarg_area_info_t *) ptr to stkarg_area_info_t
        :param cc: (::callcnv_t) calling convention
        :returns: 1: if success
        :returns: 0: not implemented
        """
        ...
    def ev_get_stkvar_scale_factor(self) -> int:
        r"""Should stack variable references be multiplied by a coefficient before being used in the stack frame?. Currently used by TMS320C55 because the references into the stack should be multiplied by 2 
                  
        :returns: scaling factor
        :returns: 0: not implemented
        """
        ...
    def ev_getreg(self, regval: uval_t, regnum: int) -> int:
        r"""IBM PC only internal request, should never be used for other purpose Get register value by internal index 
                  
        :param regval: (uval_t *), out
        :param regnum: (int)
        :returns: 1: ok
        :returns: 0: not implemented
        :returns: -1: failed (undefined value or bad regnum)
        """
        ...
    def ev_init(self, idp_modname: str) -> int:
        r"""The IDP module is just loaded. 
                  
        :param idp_modname: (const char *) processor module name
        :returns: <0: on failure
        """
        ...
    def ev_insn_reads_tbit(self, insn: insn_t, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Check if insn will read the TF bit. 
                  
        :param insn: (const insn_t*) the instruction
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 2: yes, will generate 'step' exception
        :returns: 1: yes, will store the TF bit in memory
        :returns: 0: no
        """
        ...
    def ev_is_addr_insn(self, type: int, insn: insn_t) -> int:
        r"""Does the instruction calculate some address using an immediate operand? e.g. in PC such operand may be o_displ: 'lea eax, [esi+4]' 
                  
        :param type: (int *) pointer to the returned instruction type:
        * 0 the "add" instruction (the immediate operand is a relative value)
        * 1 the "move" instruction (the immediate operand is an absolute value)
        * 2 the "sub" instruction (the immediate operand is a relative value)
        :param insn: (const insn_t *) instruction
        :returns: >0 the operand number+1
        :returns: 0: not implemented
        """
        ...
    def ev_is_align_insn(self, ea: ida_idaapi.ea_t) -> int:
        r"""Is the instruction created only for alignment purposes?. Do not directly call this function, use is_align_insn() 
                  
        :param ea: (ea_t) - instruction address
        :returns: number: of bytes in the instruction
        """
        ...
    def ev_is_alloca_probe(self, ea: ida_idaapi.ea_t) -> int:
        r"""Does the function at 'ea' behave as __alloca_probe? 
                  
        :param ea: (::ea_t)
        :returns: 1: yes
        :returns: 0: no
        """
        ...
    def ev_is_basic_block_end(self, insn: insn_t, call_insn_stops_block: bool) -> int:
        r"""Is the current instruction end of a basic block?. This function should be defined for processors with delayed jump slots. 
                  
        :param insn: (const insn_t*) the instruction
        :param call_insn_stops_block: (bool)
        :returns: 0: unknown
        :returns: <0: no
        :returns: 1: yes
        """
        ...
    def ev_is_call_insn(self, insn: insn_t) -> int:
        r"""Is the instruction a "call"? 
                  
        :param insn: (const insn_t *) instruction
        :returns: 0: unknown
        :returns: <0: no
        :returns: 1: yes
        """
        ...
    def ev_is_cond_insn(self, insn: insn_t) -> int:
        r"""Is conditional instruction? 
                  
        :param insn: (const insn_t *) instruction address
        :returns: 1: yes
        :returns: -1: no
        :returns: 0: not implemented or not instruction
        """
        ...
    def ev_is_control_flow_guard(self, p_reg: int, insn: insn_t) -> int:
        r"""Detect if an instruction is a "thunk call" to a flow guard function (equivalent to call reg/return/nop) 
                  
        :param p_reg: (int *) indirect register number, may be -1
        :param insn: (const insn_t *) call/jump instruction
        :returns: -1: no thunk detected
        :returns: 1: indirect call
        :returns: 2: security check routine call (NOP)
        :returns: 3: return thunk
        :returns: 0: not implemented
        """
        ...
    def ev_is_far_jump(self, icode: int) -> int:
        r"""is indirect far jump or call instruction? meaningful only if the processor has 'near' and 'far' reference types 
                  
        :param icode: (int)
        :returns: 0: not implemented
        :returns: 1: yes
        :returns: -1: no
        """
        ...
    def ev_is_indirect_jump(self, insn: insn_t) -> int:
        r"""Determine if instruction is an indirect jump. If CF_JUMP bit cannot describe all jump types jumps, please define this callback. 
                  
        :param insn: (const insn_t*) the instruction
        :returns: 0: use CF_JUMP
        :returns: 1: no
        :returns: 2: yes
        """
        ...
    def ev_is_insn_table_jump(self) -> int:
        r"""Reserved.
        
        """
        ...
    def ev_is_jump_func(self, pfn: func_t, jump_target: ea_t, func_pointer: ea_t) -> int:
        r"""Is the function a trivial "jump" function?. 
                  
        :param pfn: (func_t *)
        :param jump_target: (::ea_t *)
        :param func_pointer: (::ea_t *)
        :returns: <0: no
        :returns: 0: don't know
        :returns: 1: yes, see 'jump_target' and 'func_pointer'
        """
        ...
    def ev_is_ret_insn(self, insn: insn_t, flags: uchar) -> int:
        r"""Is the instruction a "return"? 
                  
        :param insn: (const insn_t *) instruction
        :param flags: (uchar), combination of IRI_... flags (see above)
        :returns: 0: unknown
        :returns: <0: no
        :returns: 1: yes
        """
        ...
    def ev_is_sane_insn(self, insn: insn_t, no_crefs: int) -> int:
        r"""Is the instruction sane for the current file type?. 
                  
        :param insn: (const insn_t*) the instruction
        :param no_crefs: (int) 1: the instruction has no code refs to it. ida just tries to convert unexplored bytes to an instruction (but there is no other reason to convert them into an instruction) 0: the instruction is created because of some coderef, user request or another weighty reason.
        :returns: >=0: ok
        :returns: <0: no, the instruction isn't likely to appear in the program
        """
        ...
    def ev_is_sp_based(self, mode: int, insn: insn_t, op: op_t) -> int:
        r"""Check whether the operand is relative to stack pointer or frame pointer This event is used to determine how to output a stack variable If not implemented, then all operands are sp based by default. Implement this event only if some stack references use frame pointer instead of stack pointer. 
                  
        :param mode: (int *) out, combination of SP/FP operand flags
        :param insn: (const insn_t *)
        :param op: (const op_t *)
        :returns: 0: not implemented
        :returns: 1: ok
        """
        ...
    def ev_is_switch(self, si: switch_info_t, insn: insn_t) -> int:
        r"""Find 'switch' idiom or override processor module's decision. It will be called for instructions marked with CF_JUMP. 
                  
        :param si: (switch_info_t *), out
        :param insn: (const insn_t *) instruction possibly belonging to a switch
        :returns: 1: switch is found, 'si' is filled. IDA will create the switch using the filled 'si'
        :returns: -1: no switch found. This value forbids switch creation by the processor module
        :returns: 0: not implemented
        """
        ...
    def ev_last_cb_before_loader(self) -> int:
        ...
    def ev_loader(self) -> int:
        r"""This code and higher ones are reserved for the loaders. The arguments and the return values are defined by the loaders 
                  
        """
        ...
    def ev_lower_func_type(self, argnums: intvec_t, fti: func_type_data_t) -> int:
        r"""Get function arguments which should be converted to pointers when lowering function prototype. The processor module can also modify 'fti' in order to make non-standard conversion of some arguments. 
                  
        :param argnums: (intvec_t *), out - numbers of arguments to be converted to pointers in acsending order
        :param fti: (func_type_data_t *), inout func type details
        :returns: 0: not implemented
        :returns: 1: argnums was filled
        :returns: 2: argnums was filled and made substantial changes to fti argnums[0] can contain a special negative value indicating that the return value should be passed as a hidden 'retstr' argument: -1 this argument is passed as the first one and the function returns a pointer to the argument, -2 this argument is passed as the last one and the function returns a pointer to the argument, -3 this argument is passed as the first one and the function returns 'void'.
        """
        ...
    def ev_max_ptr_size(self) -> int:
        r"""Get maximal size of a pointer in bytes. 
                  
        :returns: max possible size of a pointer
        """
        ...
    def ev_may_be_func(self, insn: insn_t, state: int) -> int:
        r"""Can a function start here? 
                  
        :param insn: (const insn_t*) the instruction
        :param state: (int) autoanalysis phase 0: creating functions 1: creating chunks
        :returns: probability 1..100
        """
        ...
    def ev_may_show_sreg(self, current_ea: ida_idaapi.ea_t) -> int:
        r"""The kernel wants to display the segment registers in the messages window. 
                  
        :param current_ea: (::ea_t)
        :returns: <0: if the kernel should not show the segment registers. (assuming that the module has done it)
        :returns: 0: not implemented
        """
        ...
    def ev_moving_segm(self, seg: segment_t, to: ida_idaapi.ea_t, flags: int) -> int:
        r"""May the kernel move the segment? 
                  
        :param seg: (segment_t *) segment to move
        :param to: (::ea_t) new segment start address
        :param flags: (int) combination of Move segment flags
        :returns: 0: yes
        :returns: <0: the kernel should stop
        """
        ...
    def ev_newasm(self, asmnum: int) -> int:
        r"""Before setting a new assembler. 
                  
        :param asmnum: (int) See also ev_asm_installed
        """
        ...
    def ev_newbinary(self, filename: char, fileoff: qoff64_t, basepara: ida_idaapi.ea_t, binoff: ida_idaapi.ea_t, nbytes: uint64) -> int:
        r"""IDA is about to load a binary file. 
                  
        :param filename: (char *) binary file name
        :param fileoff: (qoff64_t) offset in the file
        :param basepara: (::ea_t) base loading paragraph
        :param binoff: (::ea_t) loader offset
        :param nbytes: (::uint64) number of bytes to load
        """
        ...
    def ev_newfile(self, fname: char) -> int:
        r"""A new file has been loaded. 
                  
        :param fname: (char *) input file name
        """
        ...
    def ev_newprc(self, pnum: int, keep_cfg: bool) -> int:
        r"""Before changing processor type. 
                  
        :param pnum: (int) processor number in the array of processor names
        :param keep_cfg: (bool) true: do not modify kernel configuration
        :returns: 1: ok
        :returns: <0: prohibit
        """
        ...
    def ev_next_exec_insn(self, target: ea_t, ea: ida_idaapi.ea_t, tid: int, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Get next address to be executed This function must return the next address to be executed. If the instruction following the current one is executed, then it must return BADADDR Usually the instructions to consider are: jumps, branches, calls, returns. This function is essential if the 'single step' is not supported in hardware. 
                  
        :param target: (::ea_t *), out: pointer to the answer
        :param ea: (::ea_t) instruction address
        :param tid: (int) current therad id
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_oldfile(self, fname: char) -> int:
        r"""An old file has been loaded. 
                  
        :param fname: (char *) input file name
        """
        ...
    def ev_out_assumes(self, outctx: outctx_t) -> int:
        r"""Function to produce assume directives when segment register value changes. 
                  
        :param outctx: (outctx_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_out_data(self, outctx: outctx_t, analyze_only: bool) -> int:
        r"""Generate text representation of data items This function may change the database and create cross-references if analyze_only is set 
                  
        :param outctx: (outctx_t *)
        :param analyze_only: (bool)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_out_footer(self, outctx: outctx_t) -> int:
        r"""Function to produce end of disassembled text 
                  
        :param outctx: (outctx_t *)
        :returns: void: 
        """
        ...
    def ev_out_header(self, outctx: outctx_t) -> int:
        r"""Function to produce start of disassembled text 
                  
        :param outctx: (outctx_t *)
        :returns: void: 
        """
        ...
    def ev_out_insn(self, outctx: outctx_t) -> bool:
        r"""Generate text representation of an instruction in 'ctx.insn' outctx_t provides functions to output the generated text. This function shouldn't change the database, flags or anything else. All these actions should be performed only by emu_insn() function. 
                  
        :param outctx: (outctx_t *)
        :returns: void: 
        """
        ...
    def ev_out_label(self, outctx: outctx_t, colored_name: str) -> int:
        r"""The kernel is going to generate an instruction label line or a function header. 
                  
        :param outctx: (outctx_t *)
        :param colored_name: (const char *)
        :returns: <0: if the kernel should not generate the label
        :returns: 0: not implemented or continue
        """
        ...
    def ev_out_mnem(self, outctx: outctx_t) -> int:
        r"""Generate instruction mnemonics. This callback should append the colored mnemonics to ctx.outbuf Optional notification, if absent, out_mnem will be called. 
                  
        :param outctx: (outctx_t *)
        :returns: 1: if appended the mnemonics
        :returns: 0: not implemented
        """
        ...
    def ev_out_operand(self, outctx: outctx_t, op: op_t) -> bool:
        r"""Generate text representation of an instruction operand outctx_t provides functions to output the generated text. All these actions should be performed only by emu_insn() function. 
                  
        :param outctx: (outctx_t *)
        :param op: (const op_t *)
        :returns: 1: ok
        :returns: -1: operand is hidden
        """
        ...
    def ev_out_segend(self, outctx: outctx_t, seg: segment_t) -> int:
        r"""Function to produce end of segment 
                  
        :param outctx: (outctx_t *)
        :param seg: (segment_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_out_segstart(self, outctx: outctx_t, seg: segment_t) -> int:
        r"""Function to produce start of segment 
                  
        :param outctx: (outctx_t *)
        :param seg: (segment_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_out_special_item(self, outctx: outctx_t, segtype: uchar) -> int:
        r"""Generate text representation of an item in a special segment i.e. absolute symbols, externs, communal definitions etc 
                  
        :param outctx: (outctx_t *)
        :param segtype: (uchar)
        :returns: 1: ok
        :returns: 0: not implemented
        :returns: -1: overflow
        """
        ...
    def ev_privrange_changed(self, old_privrange: range_t, delta: adiff_t) -> int:
        r"""Privrange interval has been moved to a new location. Most common actions to be done by module in this case: fix indices of netnodes used by module 
                  
        :param old_privrange: (const range_t *) - old privrange interval
        :param delta: (::adiff_t)
        :returns: 0: Ok
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_realcvt(self, m: void, e: fpvalue_t, swt: uint16) -> int:
        r"""Floating point -> IEEE conversion 
                  
        :param m: (void *) ptr to processor-specific floating point value
        :param e: (fpvalue_t *) IDA representation of a floating point value
        :param swt: (uint16) operation (see realcvt() in ieee.h)
        :returns: 0: not implemented
        """
        ...
    def ev_rename(self, ea: ida_idaapi.ea_t, new_name: str) -> int:
        r"""The kernel is going to rename a byte. 
                  
        :param ea: (::ea_t)
        :param new_name: (const char *)
        :returns: <0: if the kernel should not rename it.
        :returns: 2: to inhibit the notification. I.e., the kernel should not rename, but 'set_name()' should return 'true'. also see renamed the return value is ignored when kernel is going to delete name
        """
        ...
    def ev_replaying_undo(self, action_name: str, vec: undo_records_t, is_undo: bool) -> int:
        r"""Replaying an undo/redo buffer 
                  
        :param action_name: (const char *) action that we perform undo/redo for. may be nullptr for intermediary buffers.
        :param vec: (const undo_records_t *)
        :param is_undo: (bool) true if performing undo, false if performing redo This event may be generated multiple times per undo/redo
        """
        ...
    def ev_set_code16_mode(self, ea: ida_idaapi.ea_t, code16: bool) -> int:
        r"""Some processors have ISA 16-bit mode e.g. ARM Thumb mode, PPC VLE, MIPS16 Set ISA 16-bit mode 
                  
        :param ea: (ea_t) address to set new ISA mode
        :param code16: (bool) true for 16-bit mode, false for 32-bit mode
        """
        ...
    def ev_set_idp_options(self, keyword: str, value_type: int, value: void, idb_loaded: bool) -> int:
        r"""Set IDP-specific configuration option Also see set_options_t in config.hpp 
                  
        :param keyword: (const char *)
        :param value_type: (int)
        :param value: (const void *)
        :param idb_loaded: (bool) true if the ev_oldfile/ev_newfile events have been generated
        :returns: 1: ok
        :returns: 0: not implemented
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_set_proc_options(self, options: str, confidence: int) -> int:
        r"""Called if the user specified an option string in the command line: -p<processor name>:<options>. Can be used for setting a processor subtype. Also called if option string is passed to set_processor_type() and IDC's SetProcessorType(). 
                  
        :param options: (const char *)
        :param confidence: (int) 0: loader's suggestion 1: user's decision
        :returns: <0: if bad option string
        """
        ...
    def ev_setup_til(self) -> int:
        r"""Setup default type libraries. (called after loading a new file into the database). The processor module may load tils, setup memory model and perform other actions required to set up the type system. This is an optional callback. 
                  
        :returns: void: 
        """
        ...
    def ev_str2reg(self, regname: str) -> int:
        r"""Convert a register name to a register number. The register number is the register index in the processor_t::reg_names array Most processor modules do not need to implement this callback It is useful only if processor_t::reg_names[reg] does not provide the correct register names 
                  
        :param regname: (const char *)
        :returns: register: number + 1
        :returns: 0: not implemented or could not be decoded
        """
        ...
    def ev_term(self) -> int:
        r"""The IDP module is being unloaded.
        
        """
        ...
    def ev_treat_hindering_item(self, hindering_item_ea: ida_idaapi.ea_t, new_item_flags: flags64_t, new_item_ea: ida_idaapi.ea_t, new_item_length: asize_t) -> int:
        r"""An item hinders creation of another item. 
                  
        :param hindering_item_ea: (::ea_t)
        :param new_item_flags: (flags64_t) (0 for code)
        :param new_item_ea: (::ea_t)
        :param new_item_length: (::asize_t)
        :returns: 0: no reaction
        :returns: !=0: the kernel may delete the hindering item
        """
        ...
    def ev_undefine(self, ea: ida_idaapi.ea_t) -> int:
        r"""An item in the database (insn or data) is being deleted. 
                  
        :param ea: (ea_t)
        :returns: 1: do not delete srranges at the item end
        :returns: 0: srranges can be deleted
        """
        ...
    def ev_update_call_stack(self, stack: call_stack_t, tid: int, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Calculate the call stack trace for the given thread. This callback is invoked when the process is suspended and should fill the 'trace' object with the information about the current call stack. Note that this callback is NOT invoked if the current debugger backend implements stack tracing via debugger_t::event_t::ev_update_call_stack. The debugger-specific algorithm takes priority. Implementing this callback in the processor module is useful when multiple debugging platforms follow similar patterns, and thus the same processor-specific algorithm can be used for different platforms. 
                  
        :param stack: (call_stack_t *) result
        :param tid: (int) thread id
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 1: ok
        :returns: -1: failed
        :returns: 0: unimplemented
        """
        ...
    def ev_use_arg_types(self, ea: ida_idaapi.ea_t, fti: func_type_data_t, rargs: funcargvec_t) -> int:
        r"""Use information about callee arguments. 
                  
        :param ea: (::ea_t) address of the call instruction
        :param fti: (func_type_data_t *) info about function type
        :param rargs: (funcargvec_t *) array of register arguments
        :returns: 1: (and removes handled arguments from fti and rargs)
        :returns: 0: not implemented
        """
        ...
    def ev_use_regarg_type(self, ea: ida_idaapi.ea_t, rargs: funcargvec_t) -> Any:
        r"""Use information about register argument. 
                  
        :param ea: (::ea_t) address of the instruction
        :param rargs: (const funcargvec_t *) vector of register arguments (including regs extracted from scattered arguments)
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_use_stkarg_type(self, ea: ida_idaapi.ea_t, arg: funcarg_t) -> int:
        r"""Use information about a stack argument. 
                  
        :param ea: (::ea_t) address of the push instruction which pushes the function argument into the stack
        :param arg: (const funcarg_t *) argument info
        :returns: 1: ok
        :returns: <=0: failed, the kernel will create a comment with the argument name or type for the instruction
        """
        ...
    def ev_validate_flirt_func(self, start_ea: ida_idaapi.ea_t, funcname: str) -> int:
        r"""Flirt has recognized a library function. This callback can be used by a plugin or proc module to intercept it and validate such a function. 
                  
        :param start_ea: (::ea_t)
        :param funcname: (const char *)
        :returns: -1: do not create a function,
        :returns: 0: function is validated
        """
        ...
    def ev_verify_noreturn(self, pfn: func_t) -> int:
        r"""The kernel wants to set 'noreturn' flags for a function. 
                  
        :param pfn: (func_t *)
        :returns: 0: ok. any other value: do not set 'noreturn' flag
        """
        ...
    def ev_verify_sp(self, pfn: func_t) -> int:
        r"""All function instructions have been analyzed. Now the processor module can analyze the stack pointer for the whole function 
                  
        :param pfn: (func_t *)
        :returns: 0: ok
        :returns: <0: bad stack pointer
        """
        ...
    def hook(self) -> bool:
        ...
    def unhook(self) -> bool:
        ...

class asm_t:
    @property
    def a_align(self) -> Any: ...
    @property
    def a_ascii(self) -> Any: ...
    @property
    def a_band(self) -> Any: ...
    @property
    def a_bnot(self) -> Any: ...
    @property
    def a_bor(self) -> Any: ...
    @property
    def a_bss(self) -> Any: ...
    @property
    def a_byte(self) -> Any: ...
    @property
    def a_comdef(self) -> Any: ...
    @property
    def a_curip(self) -> Any: ...
    @property
    def a_double(self) -> Any: ...
    @property
    def a_dups(self) -> Any: ...
    @property
    def a_dword(self) -> Any: ...
    @property
    def a_equ(self) -> Any: ...
    @property
    def a_extrn(self) -> Any: ...
    @property
    def a_float(self) -> Any: ...
    @property
    def a_include_fmt(self) -> Any: ...
    @property
    def a_mod(self) -> Any: ...
    @property
    def a_oword(self) -> Any: ...
    @property
    def a_packreal(self) -> Any: ...
    @property
    def a_public(self) -> Any: ...
    @property
    def a_qword(self) -> Any: ...
    @property
    def a_rva(self) -> Any: ...
    @property
    def a_seg(self) -> Any: ...
    @property
    def a_shl(self) -> Any: ...
    @property
    def a_shr(self) -> Any: ...
    @property
    def a_sizeof_fmt(self) -> Any: ...
    @property
    def a_tbyte(self) -> Any: ...
    @property
    def a_vstruc_fmt(self) -> Any: ...
    @property
    def a_weak(self) -> Any: ...
    @property
    def a_word(self) -> Any: ...
    @property
    def a_xor(self) -> Any: ...
    @property
    def a_yword(self) -> Any: ...
    @property
    def a_zword(self) -> Any: ...
    @property
    def accsep(self) -> Any: ...
    @property
    def ascsep(self) -> Any: ...
    @property
    def cmnt(self) -> Any: ...
    @property
    def cmnt2(self) -> Any: ...
    @property
    def end(self) -> Any: ...
    @property
    def esccodes(self) -> Any: ...
    @property
    def flag(self) -> Any: ...
    @property
    def flag2(self) -> Any: ...
    @property
    def header(self) -> Any: ...
    @property
    def help(self) -> Any: ...
    @property
    def high16(self) -> Any: ...
    @property
    def high8(self) -> Any: ...
    @property
    def lbrace(self) -> Any: ...
    @property
    def low16(self) -> Any: ...
    @property
    def low8(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def origin(self) -> Any: ...
    @property
    def rbrace(self) -> Any: ...
    @property
    def uflag(self) -> Any: ...
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

class num_range_t:
    @property
    def maxval(self) -> Any: ...
    @property
    def minval(self) -> Any: ...
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
    def __init__(self, _min: int64, _max: int64) -> Any:
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

class params_t:
    @property
    def p1(self) -> Any: ...
    @property
    def p2(self) -> Any: ...
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
    def __init__(self, _p1: int64, _p2: int64) -> Any:
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

class processor_t(IDP_Hooks):
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
    def auto_empty(self, args: Any) -> None:
        ...
    def auto_empty_finally(self, args: Any) -> None:
        ...
    def closebase(self, args: Any) -> None:
        ...
    def compiler_changed(self, args: Any) -> None:
        ...
    def deleting_func(self, pfn: Any) -> None:
        ...
    def determined_main(self, args: Any) -> None:
        ...
    def ev_add_cref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: cref_t) -> int:
        r"""A code reference is being created. 
                  
        :param to: (::ea_t)
        :param type: (cref_t)
        :returns: <0: cancel cref creation
        :returns: 0: not implemented or continue
        """
        ...
    def ev_add_dref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: dref_t) -> int:
        r"""A data reference is being created. 
                  
        :param to: (::ea_t)
        :param type: (dref_t)
        :returns: <0: cancel dref creation
        :returns: 0: not implemented or continue
        """
        ...
    def ev_adjust_argloc(self, argloc: argloc_t, optional_type: tinfo_t, size: int) -> int:
        r"""Adjust argloc according to its type/size and platform endianess 
                  
        :param argloc: (argloc_t *), inout
        :param size: (int) 'size' makes no sense if type != nullptr (type->get_size() should be used instead)
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error
        """
        ...
    def ev_adjust_libfunc_ea(self, sig: idasgn_t, libfun: libfunc_t, ea: ea_t) -> int:
        r"""Called when a signature module has been matched against bytes in the database. This is used to compute the offset at which a particular module's libfunc should be applied. 
                  
        :param sig: (const idasgn_t *)
        :param libfun: (const libfunc_t *)
        :param ea: (::ea_t *)
        :returns: 1: the ea_t pointed to by the third argument was modified.
        :returns: <=0: not modified. use default algorithm.
        """
        ...
    def ev_adjust_refinfo(self, ri: refinfo_t, ea: ida_idaapi.ea_t, n: int, fd: fixup_data_t) -> int:
        r"""Called from apply_fixup before converting operand to reference. Can be used for changing the reference info. (e.g. the PPC module adds REFINFO_NOBASE for some references) 
                  
        :param ri: (refinfo_t *)
        :param ea: (::ea_t) instruction address
        :param n: (int) operand number
        :param fd: (const fixup_data_t *)
        :returns: <0: do not create an offset
        :returns: 0: not implemented or refinfo adjusted
        """
        ...
    def ev_ana_insn(self, args: Any) -> bool:
        ...
    def ev_analyze_prolog(self, ea: ida_idaapi.ea_t) -> int:
        r"""Analyzes function prolog, epilog, and updates purge, and function attributes 
                  
        :param ea: (::ea_t) start of function
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_arch_changed(self) -> int:
        r"""The loader is done parsing arch-related information, which the processor module might want to use to finish its initialization. 
                  
        :returns: 1: if success
        :returns: 0: not implemented or failed
        """
        ...
    def ev_arg_addrs_ready(self, caller: ida_idaapi.ea_t, n: int, tif: tinfo_t, addrs: ea_t) -> int:
        r"""Argument address info is ready. 
                  
        :param caller: (::ea_t)
        :param n: (int) number of formal arguments
        :param tif: (tinfo_t *) call prototype
        :param addrs: (::ea_t *) argument intilization addresses
        :returns: <0: do not save into idb; other values mean "ok to save"
        """
        ...
    def ev_asm_installed(self, asmnum: int) -> int:
        r"""After setting a new assembler 
                  
        :param asmnum: (int) See also ev_newasm
        """
        ...
    def ev_assemble(self, args: Any) -> Any:
        ...
    def ev_auto_queue_empty(self, args: Any) -> int:
        ...
    def ev_calc_arglocs(self, fti: func_type_data_t) -> int:
        r"""Calculate function argument locations. This callback should fill retloc, all arglocs, and stkargs. This callback is never called for CM_CC_SPECIAL functions. 
                  
        :param fti: (func_type_data_t *) points to the func type info
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error
        """
        ...
    def ev_calc_cdecl_purged_bytes(self, ea: ida_idaapi.ea_t) -> int:
        r"""Calculate number of purged bytes after call. 
                  
        :param ea: (::ea_t) address of the call instruction
        :returns: number of purged bytes (usually add sp, N)
        """
        ...
    def ev_calc_next_eas(self, res: eavec_t, insn: insn_t, over: bool) -> int:
        r"""Calculate list of addresses the instruction in 'insn' may pass control to. This callback is required for source level debugging. 
                  
        :param res: (eavec_t *), out: array for the results.
        :param insn: (const insn_t*) the instruction
        :param over: (bool) calculate for step over (ignore call targets)
        :returns: <0: incalculable (indirect jumps, for example)
        :returns: >=0: number of addresses of called functions in the array. They must be put at the beginning of the array (0 if over=true)
        """
        ...
    def ev_calc_purged_bytes(self, p_purged_bytes: int, fti: func_type_data_t) -> int:
        r"""Calculate number of purged bytes by the given function type. 
                  
        :param p_purged_bytes: (int *) ptr to output
        :param fti: (const func_type_data_t *) func type details
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_calc_retloc(self, retloc: argloc_t, rettype: tinfo_t, cc: callcnv_t) -> int:
        r"""Calculate return value location. 
                  
        :param retloc: (argloc_t *)
        :param rettype: (const tinfo_t *)
        :param cc: (::callcnv_t)
        :returns: 0: not implemented
        :returns: 1: ok,
        :returns: -1: error
        """
        ...
    def ev_calc_spdelta(self, spdelta: sval_t, insn: insn_t) -> int:
        r"""Calculate amount of change to sp for the given insn. This event is required to decompile code snippets. 
                  
        :param spdelta: (sval_t *)
        :param insn: (const insn_t *)
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_calc_step_over(self, target: Any, ip: Any) -> int:
        ...
    def ev_calc_switch_cases(self, casevec: casevec_t, targets: eavec_t, insn_ea: ida_idaapi.ea_t, si: switch_info_t) -> int:
        r"""Calculate case values and targets for a custom jump table. 
                  
        :param casevec: (::casevec_t *) vector of case values (may be nullptr)
        :param targets: (eavec_t *) corresponding target addresses (my be nullptr)
        :param insn_ea: (::ea_t) address of the 'indirect jump' instruction
        :param si: (switch_info_t *) switch information
        :returns: 1: ok
        :returns: <=0: failed
        """
        ...
    def ev_calc_varglocs(self, ftd: func_type_data_t, aux_regs: regobjs_t, aux_stkargs: relobj_t, nfixed: int) -> int:
        r"""Calculate locations of the arguments that correspond to '...'. 
                  
        :param ftd: (func_type_data_t *), inout: info about all arguments (including varargs)
        :param aux_regs: (regobjs_t *) buffer for hidden register arguments, may be nullptr
        :param aux_stkargs: (relobj_t *) buffer for hidden stack arguments, may be nullptr
        :param nfixed: (int) number of fixed arguments
        :returns: 0: not implemented
        :returns: 1: ok
        :returns: -1: error On some platforms variadic calls require passing additional information: for example, number of floating variadic arguments must be passed in rax on gcc-x64. The locations and values that constitute this additional information are returned in the buffers pointed by aux_regs and aux_stkargs
        """
        ...
    def ev_calcrel(self) -> int:
        r"""Reserved.
        
        """
        ...
    def ev_can_have_type(self, args: Any) -> int:
        ...
    def ev_clean_tbit(self, ea: ida_idaapi.ea_t, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Clear the TF bit after an insn like pushf stored it in memory. 
                  
        :param ea: (::ea_t) instruction address
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 1: ok
        :returns: 0: failed
        """
        ...
    def ev_cmp_operands(self, args: Any) -> int:
        ...
    def ev_coagulate(self, args: Any) -> int:
        ...
    def ev_coagulate_dref(self, from_ea: Any, to_ea: Any, may_define: Any, _code_ea: Any) -> int:
        ...
    def ev_create_flat_group(self, image_base: ida_idaapi.ea_t, bitness: int, dataseg_sel: sel_t) -> int:
        r"""Create special segment representing the flat group. 
                  
        :param image_base: (::ea_t)
        :param bitness: (int)
        :param dataseg_sel: (::sel_t) return value is ignored
        """
        ...
    def ev_create_func_frame(self, pfn: Any) -> int:
        ...
    def ev_create_merge_handlers(self, md: merge_data_t) -> int:
        r"""Create merge handlers, if needed 
                  
        :param md: (merge_data_t *) This event is generated immediately after opening idbs.
        :returns: must be 0
        """
        ...
    def ev_create_switch_xrefs(self, args: Any) -> int:
        ...
    def ev_creating_segm(self, s: Any) -> int:
        ...
    def ev_cvt64_hashval(self, node: nodeidx_t, tag: uchar, name: str, data: uchar) -> int:
        r"""perform 32-64 conversion for a hash value 
                  
        :param node: (::nodeidx_t)
        :param tag: (uchar)
        :param name: (const ::char *)
        :param data: (const uchar *)
        :returns: 0: nothing was done
        :returns: 1: converted successfully
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_cvt64_supval(self, node: nodeidx_t, tag: uchar, idx: nodeidx_t, data: uchar) -> int:
        r"""perform 32-64 conversion for a netnode array element 
                  
        :param node: (::nodeidx_t)
        :param tag: (uchar)
        :param idx: (::nodeidx_t)
        :param data: (const uchar *)
        :returns: 0: nothing was done
        :returns: 1: converted successfully
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_decorate_name(self, name: str, mangle: bool, cc: int, optional_type: tinfo_t) -> Any:
        r"""Decorate/undecorate a C symbol name. 
                  
        :param name: (const char *) name of symbol
        :param mangle: (bool) true-mangle, false-unmangle
        :param cc: (::callcnv_t) calling convention
        :returns: 1: if success
        :returns: 0: not implemented or failed
        """
        ...
    def ev_del_cref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, expand: bool) -> int:
        r"""A code reference is being deleted. 
                  
        :param to: (::ea_t)
        :param expand: (bool)
        :returns: <0: cancel cref deletion
        :returns: 0: not implemented or continue
        """
        ...
    def ev_del_dref(self, _from: ida_idaapi.ea_t, to: ida_idaapi.ea_t) -> int:
        r"""A data reference is being deleted. 
                  
        :param to: (::ea_t)
        :returns: <0: cancel dref deletion
        :returns: 0: not implemented or continue
        """
        ...
    def ev_delay_slot_insn(self, ea: ida_idaapi.ea_t, bexec: bool, fexec: bool) -> Any:
        r"""Get delay slot instruction 
                  
        :param ea: (::ea_t *) in: instruction address in question, out: (if the answer is positive) if the delay slot contains valid insn: the address of the delay slot insn else: BADADDR (invalid insn, e.g. a branch)
        :param bexec: (bool *) execute slot if jumping, initially set to 'true'
        :param fexec: (bool *) execute slot if not jumping, initally set to 'true'
        :returns: 1: positive answer
        :returns: <=0: ordinary insn
        """
        ...
    def ev_demangle_name(self, name: str, disable_mask: int, demreq: int) -> Any:
        r"""Demangle a C++ (or another language) name into a user-readable string. This event is called by demangle_name() 
                  
        :param name: (const char *) mangled name
        :param disable_mask: (uint32) flags to inhibit parts of output or compiler info/other (see MNG_)
        :param demreq: (demreq_type_t) operation to perform
        :returns: 1: if success
        :returns: 0: not implemented
        """
        ...
    def ev_emu_insn(self, args: Any) -> bool:
        ...
    def ev_endbinary(self, args: Any) -> int:
        ...
    def ev_ending_undo(self, action_name: str, is_undo: bool) -> int:
        r"""Ended undoing/redoing an action 
                  
        :param action_name: (const char *) action that we finished undoing/redoing. is not nullptr.
        :param is_undo: (bool) true if performing undo, false if performing redo
        """
        ...
    def ev_equal_reglocs(self, a1: argloc_t, a2: argloc_t) -> int:
        r"""Are 2 register arglocs the same?. We need this callback for the pc module. 
                  
        :param a1: (argloc_t *)
        :param a2: (argloc_t *)
        :returns: 1: yes
        :returns: -1: no
        :returns: 0: not implemented
        """
        ...
    def ev_extract_address(self, out_ea: ea_t, screen_ea: ida_idaapi.ea_t, string: str, position: size_t) -> int:
        r"""Extract address from a string. 
                  
        :param out_ea: (ea_t *), out
        :param screen_ea: (ea_t)
        :param string: (const char *)
        :param position: (size_t)
        :returns: 1: ok
        :returns: 0: kernel should use the standard algorithm
        :returns: -1: error
        """
        ...
    def ev_find_op_value(self, pinsn: insn_t, opn: int) -> Any:
        r"""Find operand value via a register tracker. The returned value in 'out' is valid before executing the instruction. 
                  
        :param pinsn: (const insn_t *) instruction
        :param opn: (int) operand index
        :returns: 1: if implemented, and value was found
        :returns: 0: not implemented, -1 decoding failed, or no value found
        """
        ...
    def ev_find_reg_value(self, pinsn: insn_t, reg: int) -> Any:
        r"""Find register value via a register tracker. The returned value in 'out' is valid before executing the instruction. 
                  
        :param pinsn: (const insn_t *) instruction
        :param reg: (int) register index
        :returns: 1: if implemented, and value was found
        :returns: 0: not implemented, -1 decoding failed, or no value found
        """
        ...
    def ev_func_bounds(self, _possible_return_code: Any, pfn: Any, max_func_end_ea: Any) -> int:
        ...
    def ev_gen_asm_or_lst(self, starting: bool, fp: FILE, is_asm: bool, flags: int, outline: html_line_cb_t) -> int:
        r"""Callback: generating asm or lst file. The kernel calls this callback twice, at the beginning and at the end of listing generation. The processor module can intercept this event and adjust its output 
                  
        :param starting: (bool) beginning listing generation
        :param fp: (FILE *) output file
        :param is_asm: (bool) true:assembler, false:listing
        :param flags: (int) flags passed to gen_file()
        :param outline: (html_line_cb_t **) ptr to ptr to outline callback. if this callback is defined for this code, it will be used by the kernel to output the generated lines
        :returns: void: 
        """
        ...
    def ev_gen_map_file(self, nlines: Any, fp: Any) -> int:
        ...
    def ev_gen_regvar_def(self, ctx: Any, v: Any) -> int:
        ...
    def ev_gen_src_file_lnnum(self, args: Any) -> int:
        ...
    def ev_gen_stkvar_def(self, outctx: outctx_t, stkvar: udm_t, v: int, tid: tid_t) -> int:
        r"""Generate stack variable definition line Default line is varname = type ptr value, where 'type' is one of byte,word,dword,qword,tbyte 
                  
        :param outctx: (outctx_t *)
        :param stkvar: (const udm_t *)
        :param v: (sval_t)
        :param tid: (tid_t) stkvar TID
        :returns: 1: ok
        :returns: 0: not implemented
        """
        ...
    def ev_get_abi_info(self, comp: comp_t) -> int:
        r"""Get all possible ABI names and optional extensions for given compiler abiname/option is a string entirely consisting of letters, digits and underscore 
                  
        :param comp: (comp_t) - compiler ID
        :returns: 0: not implemented
        :returns: 1: ok
        """
        ...
    def ev_get_autocmt(self, args: Any) -> Any:
        ...
    def ev_get_bg_color(self, color: bgcolor_t, ea: ida_idaapi.ea_t) -> int:
        r"""Get item background color. Plugins can hook this callback to color disassembly lines dynamically 
                  
        :param color: (bgcolor_t *), out
        :param ea: (::ea_t)
        :returns: 0: not implemented
        :returns: 1: color set
        """
        ...
    def ev_get_cc_regs(self, regs: callregs_t, cc: callcnv_t) -> int:
        r"""Get register allocation convention for given calling convention 
                  
        :param regs: (callregs_t *), out
        :param cc: (::callcnv_t)
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_get_code16_mode(self, ea: ida_idaapi.ea_t) -> int:
        r"""Get ISA 16-bit mode 
                  
        :param ea: (ea_t) address to get the ISA mode
        :returns: 1: 16-bit mode
        :returns: 0: not implemented or 32-bit mode
        """
        ...
    def ev_get_dbr_opnum(self, opnum: int, insn: insn_t) -> int:
        r"""Get the number of the operand to be displayed in the debugger reference view (text mode). 
                  
        :param opnum: (int *) operand number (out, -1 means no such operand)
        :param insn: (const insn_t*) the instruction
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_get_default_enum_size(self) -> int:
        r"""Get default enum size. Not generated anymore. inf_get_cc_size_e() is used instead 
                  
        """
        ...
    def ev_get_frame_retsize(self, frsize: Any, pfn: Any) -> int:
        ...
    def ev_get_macro_insn_head(self, head: ea_t, ip: ida_idaapi.ea_t) -> int:
        r"""Calculate the start of a macro instruction. This notification is called if IP points to the middle of an instruction 
                  
        :param head: (::ea_t *), out: answer, BADADDR means normal instruction
        :param ip: (::ea_t) instruction address
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_get_operand_string(self, buf: Any, insn: Any, opnum: Any) -> Any:
        ...
    def ev_get_procmod(self) -> int:
        r"""Get pointer to the processor module object. All processor modules must implement this. The pointer is returned as size_t. 
                  
        """
        ...
    def ev_get_reg_accesses(self, accvec: reg_accesses_t, insn: insn_t, flags: int) -> int:
        r"""Get info about the registers that are used/changed by an instruction. 
                  
        :param accvec: (reg_accesses_t*) out: info about accessed registers
        :param insn: (const insn_t *) instruction in question
        :param flags: (int) reserved, must be 0
        :returns: -1: if accvec is nullptr
        :returns: 1: found the requested access (and filled accvec)
        :returns: 0: not implemented
        """
        ...
    def ev_get_reg_info(self, main_regname: char, bitrange: bitrange_t, regname: str) -> int:
        r"""Get register information by its name. example: "ah" returns:
        * main_regname="eax"
        * bitrange_t = { offset==8, nbits==8 }
        
        
        This callback may be unimplemented if the register names are all present in processor_t::reg_names and they all have the same size 
                  
        :param main_regname: (const char **), out
        :param bitrange: (bitrange_t *), out: position and size of the value within 'main_regname' (empty bitrange == whole register)
        :param regname: (const char *)
        :returns: 1: ok
        :returns: -1: failed (not found)
        :returns: 0: unimplemented
        """
        ...
    def ev_get_reg_name(self, reg: int, width: size_t, reghi: int) -> Any:
        r"""Generate text representation of a register. Most processor modules do not need to implement this callback. It is useful only if processor_t::reg_names[reg] does not provide the correct register name. 
                  
        :param reg: (int) internal register number as defined in the processor module
        :param width: (size_t) register width in bytes
        :param reghi: (int) if not -1 then this function will return the register pair
        :returns: -1: if error
        :returns: strlen(buf): if success
        """
        ...
    def ev_get_simd_types(self, out: simd_info_vec_t, simd_attrs: simd_info_t, argloc: argloc_t, create_tifs: bool) -> int:
        r"""Get SIMD-related types according to given attributes ant/or argument location 
                  
        :param out: (::simd_info_vec_t *)
        :param simd_attrs: (const simd_info_t *), may be nullptr
        :param argloc: (const argloc_t *), may be nullptr
        :param create_tifs: (bool) return valid tinfo_t objects, create if neccessary
        :returns: number: of found types
        :returns: -1: error If name==nullptr, initialize all SIMD types
        """
        ...
    def ev_get_stkarg_area_info(self, out: stkarg_area_info_t, cc: callcnv_t) -> int:
        r"""Get some metrics of the stack argument area. 
                  
        :param out: (stkarg_area_info_t *) ptr to stkarg_area_info_t
        :param cc: (::callcnv_t) calling convention
        :returns: 1: if success
        :returns: 0: not implemented
        """
        ...
    def ev_get_stkvar_scale_factor(self) -> int:
        r"""Should stack variable references be multiplied by a coefficient before being used in the stack frame?. Currently used by TMS320C55 because the references into the stack should be multiplied by 2 
                  
        :returns: scaling factor
        :returns: 0: not implemented
        """
        ...
    def ev_getreg(self, regval: uval_t, regnum: int) -> int:
        r"""IBM PC only internal request, should never be used for other purpose Get register value by internal index 
                  
        :param regval: (uval_t *), out
        :param regnum: (int)
        :returns: 1: ok
        :returns: 0: not implemented
        :returns: -1: failed (undefined value or bad regnum)
        """
        ...
    def ev_init(self, idp_modname: str) -> int:
        r"""The IDP module is just loaded. 
                  
        :param idp_modname: (const char *) processor module name
        :returns: <0: on failure
        """
        ...
    def ev_insn_reads_tbit(self, insn: insn_t, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Check if insn will read the TF bit. 
                  
        :param insn: (const insn_t*) the instruction
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 2: yes, will generate 'step' exception
        :returns: 1: yes, will store the TF bit in memory
        :returns: 0: no
        """
        ...
    def ev_is_addr_insn(self, type: int, insn: insn_t) -> int:
        r"""Does the instruction calculate some address using an immediate operand? e.g. in PC such operand may be o_displ: 'lea eax, [esi+4]' 
                  
        :param type: (int *) pointer to the returned instruction type:
        * 0 the "add" instruction (the immediate operand is a relative value)
        * 1 the "move" instruction (the immediate operand is an absolute value)
        * 2 the "sub" instruction (the immediate operand is a relative value)
        :param insn: (const insn_t *) instruction
        :returns: >0 the operand number+1
        :returns: 0: not implemented
        """
        ...
    def ev_is_align_insn(self, args: Any) -> int:
        ...
    def ev_is_alloca_probe(self, args: Any) -> int:
        ...
    def ev_is_basic_block_end(self, args: Any) -> int:
        ...
    def ev_is_call_insn(self, args: Any) -> int:
        ...
    def ev_is_cond_insn(self, insn: insn_t) -> int:
        r"""Is conditional instruction? 
                  
        :param insn: (const insn_t *) instruction address
        :returns: 1: yes
        :returns: -1: no
        :returns: 0: not implemented or not instruction
        """
        ...
    def ev_is_control_flow_guard(self, p_reg: int, insn: insn_t) -> int:
        r"""Detect if an instruction is a "thunk call" to a flow guard function (equivalent to call reg/return/nop) 
                  
        :param p_reg: (int *) indirect register number, may be -1
        :param insn: (const insn_t *) call/jump instruction
        :returns: -1: no thunk detected
        :returns: 1: indirect call
        :returns: 2: security check routine call (NOP)
        :returns: 3: return thunk
        :returns: 0: not implemented
        """
        ...
    def ev_is_far_jump(self, args: Any) -> int:
        ...
    def ev_is_indirect_jump(self, args: Any) -> int:
        ...
    def ev_is_insn_table_jump(self, args: Any) -> int:
        ...
    def ev_is_jump_func(self, pfn: func_t, jump_target: ea_t, func_pointer: ea_t) -> int:
        r"""Is the function a trivial "jump" function?. 
                  
        :param pfn: (func_t *)
        :param jump_target: (::ea_t *)
        :param func_pointer: (::ea_t *)
        :returns: <0: no
        :returns: 0: don't know
        :returns: 1: yes, see 'jump_target' and 'func_pointer'
        """
        ...
    def ev_is_ret_insn(self, args: Any) -> int:
        ...
    def ev_is_sane_insn(self, args: Any) -> int:
        ...
    def ev_is_sp_based(self, mode: Any, insn: Any, op: Any) -> int:
        ...
    def ev_is_switch(self, args: Any) -> int:
        ...
    def ev_last_cb_before_loader(self) -> int:
        ...
    def ev_loader(self) -> int:
        r"""This code and higher ones are reserved for the loaders. The arguments and the return values are defined by the loaders 
                  
        """
        ...
    def ev_lower_func_type(self, argnums: intvec_t, fti: func_type_data_t) -> int:
        r"""Get function arguments which should be converted to pointers when lowering function prototype. The processor module can also modify 'fti' in order to make non-standard conversion of some arguments. 
                  
        :param argnums: (intvec_t *), out - numbers of arguments to be converted to pointers in acsending order
        :param fti: (func_type_data_t *), inout func type details
        :returns: 0: not implemented
        :returns: 1: argnums was filled
        :returns: 2: argnums was filled and made substantial changes to fti argnums[0] can contain a special negative value indicating that the return value should be passed as a hidden 'retstr' argument: -1 this argument is passed as the first one and the function returns a pointer to the argument, -2 this argument is passed as the last one and the function returns a pointer to the argument, -3 this argument is passed as the first one and the function returns 'void'.
        """
        ...
    def ev_max_ptr_size(self) -> int:
        r"""Get maximal size of a pointer in bytes. 
                  
        :returns: max possible size of a pointer
        """
        ...
    def ev_may_be_func(self, args: Any) -> int:
        ...
    def ev_may_show_sreg(self, args: Any) -> int:
        ...
    def ev_moving_segm(self, s: Any, to_ea: Any, flags: Any) -> int:
        ...
    def ev_newasm(self, asmnum: int) -> int:
        r"""Before setting a new assembler. 
                  
        :param asmnum: (int) See also ev_asm_installed
        """
        ...
    def ev_newbinary(self, args: Any) -> int:
        ...
    def ev_newfile(self, args: Any) -> int:
        ...
    def ev_newprc(self, args: Any) -> int:
        ...
    def ev_next_exec_insn(self, target: ea_t, ea: ida_idaapi.ea_t, tid: int, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Get next address to be executed This function must return the next address to be executed. If the instruction following the current one is executed, then it must return BADADDR Usually the instructions to consider are: jumps, branches, calls, returns. This function is essential if the 'single step' is not supported in hardware. 
                  
        :param target: (::ea_t *), out: pointer to the answer
        :param ea: (::ea_t) instruction address
        :param tid: (int) current therad id
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 0: unimplemented
        :returns: 1: implemented
        """
        ...
    def ev_oldfile(self, args: Any) -> int:
        ...
    def ev_out_assumes(self, args: Any) -> int:
        ...
    def ev_out_data(self, args: Any) -> int:
        ...
    def ev_out_footer(self, args: Any) -> int:
        ...
    def ev_out_header(self, args: Any) -> int:
        ...
    def ev_out_insn(self, args: Any) -> bool:
        ...
    def ev_out_label(self, args: Any) -> int:
        ...
    def ev_out_mnem(self, args: Any) -> int:
        ...
    def ev_out_operand(self, args: Any) -> bool:
        ...
    def ev_out_segend(self, ctx: Any, s: Any) -> int:
        ...
    def ev_out_segstart(self, ctx: Any, s: Any) -> int:
        ...
    def ev_out_special_item(self, args: Any) -> int:
        ...
    def ev_privrange_changed(self, old_privrange: range_t, delta: adiff_t) -> int:
        r"""Privrange interval has been moved to a new location. Most common actions to be done by module in this case: fix indices of netnodes used by module 
                  
        :param old_privrange: (const range_t *) - old privrange interval
        :param delta: (::adiff_t)
        :returns: 0: Ok
        :returns: -1: error (and message in errbuf)
        """
        ...
    def ev_realcvt(self, m: void, e: fpvalue_t, swt: uint16) -> int:
        r"""Floating point -> IEEE conversion 
                  
        :param m: (void *) ptr to processor-specific floating point value
        :param e: (fpvalue_t *) IDA representation of a floating point value
        :param swt: (uint16) operation (see realcvt() in ieee.h)
        :returns: 0: not implemented
        """
        ...
    def ev_rename(self, args: Any) -> int:
        ...
    def ev_replaying_undo(self, action_name: str, vec: undo_records_t, is_undo: bool) -> int:
        r"""Replaying an undo/redo buffer 
                  
        :param action_name: (const char *) action that we perform undo/redo for. may be nullptr for intermediary buffers.
        :param vec: (const undo_records_t *)
        :param is_undo: (bool) true if performing undo, false if performing redo This event may be generated multiple times per undo/redo
        """
        ...
    def ev_set_code16_mode(self, ea: ida_idaapi.ea_t, code16: bool) -> int:
        r"""Some processors have ISA 16-bit mode e.g. ARM Thumb mode, PPC VLE, MIPS16 Set ISA 16-bit mode 
                  
        :param ea: (ea_t) address to set new ISA mode
        :param code16: (bool) true for 16-bit mode, false for 32-bit mode
        """
        ...
    def ev_set_idp_options(self, keyword: Any, value_type: Any, value: Any, idb_loaded: Any) -> int:
        ...
    def ev_set_proc_options(self, args: Any) -> int:
        ...
    def ev_setup_til(self) -> int:
        r"""Setup default type libraries. (called after loading a new file into the database). The processor module may load tils, setup memory model and perform other actions required to set up the type system. This is an optional callback. 
                  
        :returns: void: 
        """
        ...
    def ev_str2reg(self, args: Any) -> int:
        ...
    def ev_term(self) -> int:
        r"""The IDP module is being unloaded.
        
        """
        ...
    def ev_treat_hindering_item(self, args: Any) -> int:
        ...
    def ev_undefine(self, args: Any) -> int:
        ...
    def ev_update_call_stack(self, stack: call_stack_t, tid: int, getreg: regval_getter_t, regvalues: regval_t) -> int:
        r"""Calculate the call stack trace for the given thread. This callback is invoked when the process is suspended and should fill the 'trace' object with the information about the current call stack. Note that this callback is NOT invoked if the current debugger backend implements stack tracing via debugger_t::event_t::ev_update_call_stack. The debugger-specific algorithm takes priority. Implementing this callback in the processor module is useful when multiple debugging platforms follow similar patterns, and thus the same processor-specific algorithm can be used for different platforms. 
                  
        :param stack: (call_stack_t *) result
        :param tid: (int) thread id
        :param getreg: (::processor_t::regval_getter_t *) function to get register values
        :param regvalues: (const regval_t *) register values array
        :returns: 1: ok
        :returns: -1: failed
        :returns: 0: unimplemented
        """
        ...
    def ev_use_arg_types(self, ea: ida_idaapi.ea_t, fti: func_type_data_t, rargs: funcargvec_t) -> int:
        r"""Use information about callee arguments. 
                  
        :param ea: (::ea_t) address of the call instruction
        :param fti: (func_type_data_t *) info about function type
        :param rargs: (funcargvec_t *) array of register arguments
        :returns: 1: (and removes handled arguments from fti and rargs)
        :returns: 0: not implemented
        """
        ...
    def ev_use_regarg_type(self, ea: ida_idaapi.ea_t, rargs: funcargvec_t) -> Any:
        r"""Use information about register argument. 
                  
        :param ea: (::ea_t) address of the instruction
        :param rargs: (const funcargvec_t *) vector of register arguments (including regs extracted from scattered arguments)
        :returns: 1: 
        :returns: 0: not implemented
        """
        ...
    def ev_use_stkarg_type(self, ea: ida_idaapi.ea_t, arg: funcarg_t) -> int:
        r"""Use information about a stack argument. 
                  
        :param ea: (::ea_t) address of the push instruction which pushes the function argument into the stack
        :param arg: (const funcarg_t *) argument info
        :returns: 1: ok
        :returns: <=0: failed, the kernel will create a comment with the argument name or type for the instruction
        """
        ...
    def ev_validate_flirt_func(self, args: Any) -> int:
        ...
    def ev_verify_noreturn(self, pfn: Any) -> int:
        ...
    def ev_verify_sp(self, pfn: Any) -> int:
        ...
    def func_added(self, pfn: Any) -> None:
        ...
    def get_auxpref(self, insn: Any) -> Any:
        r"""This function returns insn.auxpref value"""
        ...
    def get_idpdesc(self) -> Any:
        r"""
                This function must be present and should return the list of
                short processor names similar to the one in ph.psnames.
                This method can be overridden to return to the kernel a different IDP description.
                
        """
        ...
    def hook(self) -> bool:
        ...
    def idasgn_loaded(self, args: Any) -> None:
        ...
    def kernel_config_loaded(self, args: Any) -> None:
        ...
    def make_code(self, args: Any) -> None:
        ...
    def make_data(self, args: Any) -> None:
        ...
    def renamed(self, args: Any) -> None:
        ...
    def savebase(self, args: Any) -> None:
        ...
    def segm_moved(self, from_ea: Any, to_ea: Any, size: Any, changed_netmap: Any) -> None:
        ...
    def set_func_end(self, args: Any) -> None:
        ...
    def set_func_start(self, args: Any) -> None:
        ...
    def sgr_changed(self, args: Any) -> None:
        ...
    def unhook(self) -> bool:
        ...

class reg_access_t:
    @property
    def access_type(self) -> Any: ...
    @property
    def opnum(self) -> Any: ...
    @property
    def range(self) -> Any: ...
    @property
    def regnum(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reg_access_t) -> bool:
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
    def __ne__(self, r: reg_access_t) -> bool:
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
    def have_common_bits(self, r: reg_access_t) -> bool:
        ...

class reg_access_vec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reg_access_vec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> reg_access_t:
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
    def __ne__(self, r: reg_access_vec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: reg_access_t) -> None:
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
    def add_unique(self, x: reg_access_t) -> bool:
        ...
    def append(self, x: reg_access_t) -> None:
        ...
    def at(self, _idx: size_t) -> reg_access_t:
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
    def extend(self, x: reg_access_vec_t) -> None:
        ...
    def extract(self) -> reg_access_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: reg_access_t) -> bool:
        ...
    def inject(self, s: reg_access_t, len: size_t) -> None:
        ...
    def insert(self, it: reg_access_t, x: reg_access_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> reg_access_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: reg_access_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class reg_accesses_t(reg_access_vec_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reg_access_vec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> reg_access_t:
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
    def __ne__(self, r: reg_access_vec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: reg_access_t) -> None:
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
    def add_unique(self, x: reg_access_t) -> bool:
        ...
    def append(self, x: reg_access_t) -> None:
        ...
    def at(self, _idx: size_t) -> reg_access_t:
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
    def extend(self, x: reg_access_vec_t) -> None:
        ...
    def extract(self) -> reg_access_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: reg_access_t) -> bool:
        ...
    def inject(self, s: reg_access_t, len: size_t) -> None:
        ...
    def insert(self, it: reg_access_t, x: reg_access_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> reg_access_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: reg_access_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class reg_info_t:
    @property
    def reg(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reg_info_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: reg_info_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: reg_info_t) -> bool:
        ...
    def __init__(self) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: reg_info_t) -> bool:
        ...
    def __lt__(self, r: reg_info_t) -> bool:
        ...
    def __ne__(self, r: reg_info_t) -> bool:
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
    def compare(self, r: reg_info_t) -> int:
        ...

def AssembleLine(ea: Any, cs: Any, ip: Any, use32: Any, line: Any) -> Any:
    r"""Assemble an instruction to a string (display a warning if an error is found)
    
    :param ea: linear address of instruction
    :param cs:  cs of instruction
    :param ip:  ip of instruction
    :param use32: is 32bit segment
    :param line: line to assemble
    :returns: a string containing the assembled instruction, or None on failure
    """
    ...

def assemble(ea: Any, cs: Any, ip: Any, use32: Any, line: Any) -> ssize_t:
    r"""Assemble an instruction into the database (display a warning if an error is found)
    
    :param ea: linear address of instruction
    :param cs: cs of instruction
    :param ip: ip of instruction
    :param use32: is 32bit segment?
    :param line: line to assemble
    
    :returns: Boolean. True on success.
    """
    ...

def cfg_get_cc_header_path(compid: comp_t) -> str:
    ...

def cfg_get_cc_parm(compid: comp_t, name: str) -> str:
    ...

def cfg_get_cc_predefined_macros(compid: comp_t) -> str:
    ...

def delay_slot_insn(ea: ea_t, bexec: bool, fexec: bool) -> bool:
    ...

def gen_idb_event(args: Any) -> None:
    r"""the kernel will use this function to generate idb_events
    
    """
    ...

def get_ash() -> asm_t:
    ...

def get_config_value(key: str) -> jvalue_t:
    ...

def get_idb_notifier_addr(arg1: Any) -> Any:
    ...

def get_idb_notifier_ud_addr(hooks: IDB_Hooks) -> Any:
    ...

def get_idp_name() -> str:
    r"""Get name of the current processor module. The name is derived from the file name. For example, for IBM PC the module is named "pc.w32" (windows version), then the module name is "PC" (uppercase). If no processor module is loaded, this function will return nullptr 
            
    """
    ...

def get_idp_notifier_addr(arg1: Any) -> Any:
    ...

def get_idp_notifier_ud_addr(hooks: IDP_Hooks) -> Any:
    ...

def get_ph() -> processor_t:
    ...

def get_reg_info(regname: str, bitrange: bitrange_t) -> str:
    ...

def get_reg_name(reg: int, width: size_t, reghi: int = -1) -> str:
    r"""Get text representation of a register. For most processors this function will just return processor_t::reg_names[reg]. If the processor module has implemented processor_t::get_reg_name, it will be used instead 
            
    :param reg: internal register number as defined in the processor module
    :param width: register width in bytes
    :param reghi: if specified, then this function will return the register pair
    :returns: length of register name in bytes or -1 if failure
    """
    ...

def has_cf_chg(feature: int, opnum: uint) -> bool:
    r"""Does an instruction with the specified feature modify the i-th operand?
    
    """
    ...

def has_cf_use(feature: int, opnum: uint) -> bool:
    r"""Does an instruction with the specified feature use a value of the i-th operand?
    
    """
    ...

def has_insn_feature(icode: uint16, bit: int) -> bool:
    r"""Does the specified instruction have the specified feature?
    
    """
    ...

def is_align_insn(ea: ida_idaapi.ea_t) -> int:
    r"""If the instruction at 'ea' looks like an alignment instruction, return its length in bytes. Otherwise return 0. 
            
    """
    ...

def is_basic_block_end(insn: insn_t, call_insn_stops_block: bool) -> ssize_t:
    r"""Is the instruction the end of a basic block?
    
    """
    ...

def is_call_insn(insn: insn_t) -> ssize_t:
    r"""Is the instruction a "call"?
    
    """
    ...

def is_indirect_jump_insn(insn: insn_t) -> bool:
    r"""Is the instruction an indirect jump?
    
    """
    ...

def is_ret_insn(args: Any) -> ssize_t:
    ...

def parse_reg_name(ri: reg_info_t, regname: str) -> bool:
    r"""Get register info by name. 
            
    :param ri: result
    :param regname: name of register
    :returns: success
    """
    ...

def ph_calcrel(ea: ida_idaapi.ea_t) -> Any:
    ...

def ph_find_op_value(insn: insn_t, op: int) -> uint64:
    ...

def ph_find_reg_value(insn: insn_t, reg: int) -> uint64:
    ...

def ph_get_abi_info(comp: comp_t) -> Any:
    ...

def ph_get_cnbits() -> Any:
    r"""Returns the 'ph.cnbits'"""
    ...

def ph_get_dnbits() -> Any:
    r"""Returns the 'ph.dnbits'"""
    ...

def ph_get_flag() -> Any:
    r"""Returns the 'ph.flag'"""
    ...

def ph_get_icode_return() -> Any:
    r"""Returns the 'ph.icode_return'"""
    ...

def ph_get_id() -> Any:
    r"""Returns the 'ph.id' field"""
    ...

def ph_get_instruc() -> Any:
    r"""Returns a list of tuples (instruction_name, instruction_feature) containing the
    instructions list as defined in he processor module
    """
    ...

def ph_get_instruc_end() -> Any:
    r"""Returns the 'ph.instruc_end'"""
    ...

def ph_get_instruc_start() -> Any:
    r"""Returns the 'ph.instruc_start'"""
    ...

def ph_get_operand_info(ea: ida_idaapi.ea_t, n: int) -> Any:
    r"""Returns the operand information given an ea and operand number.
    
    :param ea: address
    :param n: operand number
    
    :returns: Returns an idd_opinfo_t as a tuple: (modified, ea, reg_ival, regidx, value_size).
              Please refer to idd_opinfo_t structure in the SDK.
    """
    ...

def ph_get_reg_accesses(accvec: reg_accesses_t, insn: insn_t, flags: int) -> ssize_t:
    ...

def ph_get_reg_code_sreg() -> Any:
    r"""Returns the 'ph.reg_code_sreg'"""
    ...

def ph_get_reg_data_sreg() -> Any:
    r"""Returns the 'ph.reg_data_sreg'"""
    ...

def ph_get_reg_first_sreg() -> Any:
    r"""Returns the 'ph.reg_first_sreg'"""
    ...

def ph_get_reg_last_sreg() -> Any:
    r"""Returns the 'ph.reg_last_sreg'"""
    ...

def ph_get_regnames() -> Any:
    r"""Returns the list of register names as defined in the processor module"""
    ...

def ph_get_segreg_size() -> Any:
    r"""Returns the 'ph.segreg_size'"""
    ...

def ph_get_tbyte_size() -> Any:
    r"""Returns the 'ph.tbyte_size' field as defined in he processor module"""
    ...

def ph_get_version() -> Any:
    r"""Returns the 'ph.version'"""
    ...

def process_config_directive(directive: str, priority: int = 2) -> None:
    ...

def register_cfgopts(opts: Any, nopts: size_t, cb: config_changed_cb_t = None, obj: void = None) -> bool:
    ...

def set_processor_type(procname: str, level: setproc_level_t) -> bool:
    r"""Set target processor type. Once a processor module is loaded, it cannot be replaced until we close the idb. 
            
    :param procname: name of processor type (one of names present in processor_t::psnames)
    :param level: SETPROC_
    :returns: success
    """
    ...

def set_target_assembler(asmnum: int) -> bool:
    r"""Set target assembler. 
            
    :param asmnum: number of assembler in the current processor module
    :returns: success
    """
    ...

def sizeof_ldbl() -> int:
    ...

def str2reg(p: str) -> int:
    r"""Get any register number (-1 on error)
    
    """
    ...

def str2sreg(name: str) -> Any:
    r"""get segment register number from its name or -1"""
    ...

AS2_BRACE: int  # 1
AS2_BYTE1CHAR: int  # 4
AS2_COLONSUF: int  # 32
AS2_IDEALDSCR: int  # 8
AS2_STRINV: int  # 2
AS2_TERSESTR: int  # 16
AS2_YWORD: int  # 64
AS2_ZWORD: int  # 128
ASB_BINF0: int  # 0
ASB_BINF1: int  # 131072
ASB_BINF2: int  # 262144
ASB_BINF3: int  # 393216
ASB_BINF4: int  # 524288
ASB_BINF5: int  # 655360
ASD_DECF0: int  # 0
ASD_DECF1: int  # 4096
ASD_DECF2: int  # 8192
ASD_DECF3: int  # 12288
ASH_HEXF0: int  # 0
ASH_HEXF1: int  # 512
ASH_HEXF2: int  # 1024
ASH_HEXF3: int  # 1536
ASH_HEXF4: int  # 2048
ASH_HEXF5: int  # 2560
ASO_OCTF0: int  # 0
ASO_OCTF1: int  # 16384
ASO_OCTF2: int  # 32768
ASO_OCTF3: int  # 49152
ASO_OCTF4: int  # 65536
ASO_OCTF5: int  # 81920
ASO_OCTF6: int  # 98304
ASO_OCTF7: int  # 114688
AS_1TEXT: int  # 64
AS_2CHRE: int  # 8
AS_ALIGN2: int  # 536870912
AS_ASCIIC: int  # 1073741824
AS_ASCIIZ: int  # -2147483648
AS_BINFM: int  # 917504
AS_COLON: int  # 2
AS_DECFM: int  # 12288
AS_HEXFM: int  # 3584
AS_LALIGN: int  # 33554432
AS_N2CHR: int  # 32
AS_NCHRE: int  # 16
AS_NCMAS: int  # 256
AS_NHIAS: int  # 128
AS_NOCODECLN: int  # 67108864
AS_NOSPACE: int  # 268435456
AS_NOXRF: int  # 4194304
AS_OCTFM: int  # 114688
AS_OFFST: int  # 1
AS_ONEDUP: int  # 2097152
AS_RELSUP: int  # 16777216
AS_UDATA: int  # 4
AS_UNEQU: int  # 1048576
AS_XTRNTYPE: int  # 8388608
CF_CALL: int  # 2
CF_CHG1: int  # 4
CF_CHG2: int  # 8
CF_CHG3: int  # 16
CF_CHG4: int  # 32
CF_CHG5: int  # 64
CF_CHG6: int  # 128
CF_CHG7: int  # 131072
CF_CHG8: int  # 262144
CF_HLL: int  # 65536
CF_JUMP: int  # 16384
CF_SHFT: int  # 32768
CF_STOP: int  # 1
CF_USE1: int  # 256
CF_USE2: int  # 512
CF_USE3: int  # 1024
CF_USE4: int  # 2048
CF_USE5: int  # 4096
CF_USE6: int  # 8192
CF_USE7: int  # 524288
CF_USE8: int  # 1048576
CUSTOM_INSN_ITYPE: int  # 32768
HKCB_GLOBAL: int  # 1
IDPOPT_BADKEY: int  # 1
IDPOPT_BADTYPE: int  # 2
IDPOPT_BADVALUE: int  # 3
IDPOPT_BIT: int  # 3
IDPOPT_BIT_BOOL: int  # 50331648
IDPOPT_BIT_UCHAR: int  # 16777216
IDPOPT_BIT_UINT: int  # 0
IDPOPT_BIT_USHORT: int  # 33554432
IDPOPT_CST: int  # 6
IDPOPT_CST_PARAMS: int  # 16777216
IDPOPT_FLT: int  # 4
IDPOPT_I64: int  # 5
IDPOPT_I64_RANGE: int  # 16777216
IDPOPT_I64_UNS: int  # 33554432
IDPOPT_JVL: int  # 7
IDPOPT_MBROFF: int  # 262144
IDPOPT_NUM: int  # 2
IDPOPT_NUM_CHAR: int  # 16777216
IDPOPT_NUM_INT: int  # 0
IDPOPT_NUM_RANGE: int  # 67108864
IDPOPT_NUM_SHORT: int  # 33554432
IDPOPT_NUM_UNS: int  # 134217728
IDPOPT_OK: int  # 0
IDPOPT_PRI_DEFAULT: int  # 1
IDPOPT_PRI_HIGH: int  # 2
IDPOPT_STR: int  # 1
IDPOPT_STR_LONG: int  # 33554432
IDPOPT_STR_QSTRING: int  # 16777216
IDP_INTERFACE_VERSION: int  # 900
IRI_EXTENDED: int  # 0
IRI_RET_LITERALLY: int  # 1
IRI_SKIP_RETTARGET: int  # 2
IRI_STRICT: int  # 3
LTC_ADDED: int  # 1
LTC_ALIASED: int  # 4
LTC_COMPILER: int  # 5
LTC_DELETED: int  # 2
LTC_EDITED: int  # 3
LTC_NONE: int  # 0
LTC_TIL_COMPACTED: int  # 8
LTC_TIL_LOADED: int  # 6
LTC_TIL_UNLOADED: int  # 7
NO_ACCESS: int  # 0
OP_FP_BASED: int  # 0
OP_SP_ADD: int  # 0
OP_SP_BASED: int  # 1
OP_SP_SUB: int  # 2
PLFM_386: int  # 0
PLFM_6502: int  # 5
PLFM_65C816: int  # 61
PLFM_6800: int  # 9
PLFM_68K: int  # 7
PLFM_80196: int  # 16
PLFM_8051: int  # 3
PLFM_AD2106X: int  # 68
PLFM_AD218X: int  # 48
PLFM_ALPHA: int  # 24
PLFM_ARC: int  # 63
PLFM_ARM: int  # 13
PLFM_AVR: int  # 20
PLFM_C166: int  # 29
PLFM_C39: int  # 51
PLFM_CR16: int  # 52
PLFM_DALVIK: int  # 60
PLFM_DSP56K: int  # 28
PLFM_DSP96K: int  # 66
PLFM_EBC: int  # 57
PLFM_F2MC: int  # 33
PLFM_FR: int  # 43
PLFM_H8: int  # 21
PLFM_H8500: int  # 26
PLFM_HPPA: int  # 25
PLFM_I860: int  # 2
PLFM_I960: int  # 32
PLFM_IA64: int  # 31
PLFM_JAVA: int  # 8
PLFM_KR1878: int  # 47
PLFM_M16C: int  # 62
PLFM_M32R: int  # 37
PLFM_M740: int  # 40
PLFM_M7700: int  # 41
PLFM_M7900: int  # 45
PLFM_MC6812: int  # 11
PLFM_MC6816: int  # 44
PLFM_MIPS: int  # 12
PLFM_MN102L00: int  # 53
PLFM_MSP430: int  # 58
PLFM_NEC_78K0: int  # 38
PLFM_NEC_78K0S: int  # 39
PLFM_NEC_V850X: int  # 55
PLFM_NET: int  # 19
PLFM_OAKDSP: int  # 49
PLFM_PDP: int  # 6
PLFM_PIC: int  # 22
PLFM_PIC16: int  # 69
PLFM_PPC: int  # 15
PLFM_RISCV: int  # 72
PLFM_RL78: int  # 73
PLFM_RX: int  # 74
PLFM_S390: int  # 70
PLFM_SCR_ADPT: int  # 56
PLFM_SH: int  # 18
PLFM_SPARC: int  # 23
PLFM_SPC700: int  # 67
PLFM_SPU: int  # 59
PLFM_ST20: int  # 30
PLFM_ST7: int  # 10
PLFM_ST9: int  # 42
PLFM_TLCS900: int  # 50
PLFM_TMS: int  # 4
PLFM_TMS320C1X: int  # 54
PLFM_TMS320C28: int  # 65
PLFM_TMS320C3: int  # 46
PLFM_TMS320C54: int  # 34
PLFM_TMS320C55: int  # 35
PLFM_TMSC6: int  # 14
PLFM_TRICORE: int  # 27
PLFM_TRIMEDIA: int  # 36
PLFM_UNSP: int  # 64
PLFM_WASM: int  # 75
PLFM_XTENSA: int  # 71
PLFM_Z8: int  # 17
PLFM_Z80: int  # 1
PR2_CODE16_BIT: int  # 8
PR2_FORCE_16BIT: int  # 128
PR2_IDP_OPTS: int  # 2
PR2_MACRO: int  # 16
PR2_MAPPINGS: int  # 1
PR2_REL_BITS: int  # 64
PR2_USE_CALCREL: int  # 32
PRN_BIN: int  # 192
PRN_DEC: int  # 128
PRN_HEX: int  # 0
PRN_OCT: int  # 64
PR_ADJSEGS: int  # 32
PR_ALIGN: int  # 2048
PR_ALIGN_INSN: int  # 16777216
PR_ASSEMBLE: int  # 1024
PR_BINMEM: int  # 65536
PR_CHK_XREF: int  # 262144
PR_CNDINSNS: int  # 67108864
PR_DEFNUM: int  # 192
PR_DEFSEG32: int  # 4
PR_DEFSEG64: int  # 268435456
PR_DELAYED: int  # 8388608
PR_NOCHANGE: int  # 512
PR_NO_SEGMOVE: int  # 524288
PR_OUTER: int  # 536870912
PR_PURGING: int  # 33554432
PR_RNAMESOK: int  # 8
PR_SCALE_STKVARS: int  # 4194304
PR_SEGS: int  # 1
PR_SEGTRANS: int  # 131072
PR_SGROTHER: int  # 16384
PR_STACK_UP: int  # 32768
PR_TYPEINFO: int  # 4096
PR_USE32: int  # 2
PR_USE64: int  # 8192
PR_USE_ARG_TYPES: int  # 2097152
PR_USE_TBYTE: int  # 134217728
PR_WORD_INS: int  # 256
READ_ACCESS: int  # 2
REAL_ERROR_BADDATA: int  # -3
REAL_ERROR_FORMAT: int  # -1
REAL_ERROR_RANGE: int  # -2
REG_SPOIL: int  # -2147483648
RW_ACCESS: int  # 3
SETPROC_IDB: int  # 0
SETPROC_LOADER: int  # 1
SETPROC_LOADER_NON_FATAL: int  # 2
SETPROC_USER: int  # 3
SWIG_PYTHON_LEGACY_BOOL: int  # 1
WRITE_ACCESS: int  # 1
adding_segm: int  # 63
allsegs_moved: int  # 31
annotations: _Feature
auto_empty: int  # 3
auto_empty_finally: int  # 4
bookmark_changed: int  # 61
byte_patched: int  # 53
callee_addr_changed: int  # 60
changing_cmt: int  # 54
changing_op_ti: int  # 14
changing_op_type: int  # 16
changing_range_cmt: int  # 56
changing_segm_class: int  # 27
changing_segm_end: int  # 23
changing_segm_name: int  # 25
changing_segm_start: int  # 21
changing_ti: int  # 12
cik_filename: int  # 1
cik_path: int  # 2
cik_string: int  # 0
closebase: int  # 0
cmt_changed: int  # 55
compiler_changed: int  # 11
deleting_func: int  # 36
deleting_func_tail: int  # 40
deleting_segm: int  # 19
deleting_tryblks: int  # 47
destroyed_items: int  # 51
determined_main: int  # 5
dirtree_link: int  # 67
dirtree_mkdir: int  # 65
dirtree_move: int  # 68
dirtree_rank: int  # 69
dirtree_rmdir: int  # 66
dirtree_rminode: int  # 70
dirtree_segm_moved: int  # 71
extlang_changed: int  # 6
extra_cmt_changed: int  # 58
flow_chart_created: int  # 10
frame_created: int  # 78
frame_deleted: int  # 37
frame_expanded: int  # 83
frame_udm_changed: int  # 82
frame_udm_created: int  # 79
frame_udm_deleted: int  # 80
frame_udm_renamed: int  # 81
func_added: int  # 32
func_deleted: int  # 64
func_noret_changed: int  # 43
func_tail_appended: int  # 39
func_tail_deleted: int  # 41
func_updated: int  # 33
ida_funcs: module
ida_idaapi: module
ida_pro: module
ida_segment: module
ida_ua: module
idasgn_loaded: int  # 7
idasgn_matched_ea: int  # 84
item_color_changed: int  # 59
kernel_config_loaded: int  # 8
loader_finished: int  # 9
local_type_renamed: int  # 89
local_types_changed: int  # 72
lt_edm_changed: int  # 88
lt_edm_created: int  # 85
lt_edm_deleted: int  # 86
lt_edm_renamed: int  # 87
lt_udm_changed: int  # 76
lt_udm_created: int  # 73
lt_udm_deleted: int  # 74
lt_udm_renamed: int  # 75
lt_udt_expanded: int  # 77
make_code: int  # 49
make_data: int  # 50
op_ti_changed: int  # 15
op_type_changed: int  # 17
ph: __ph  # <ida_idp.__ph object at 0x71772a18dfc0>
range_cmt_changed: int  # 57
renamed: int  # 52
savebase: int  # 1
segm_added: int  # 18
segm_attrs_updated: int  # 29
segm_class_changed: int  # 28
segm_deleted: int  # 20
segm_end_changed: int  # 24
segm_moved: int  # 30
segm_name_changed: int  # 26
segm_start_changed: int  # 22
set_func_end: int  # 35
set_func_start: int  # 34
sgr_changed: int  # 48
sgr_deleted: int  # 62
stkpnts_changed: int  # 44
tail_owner_changed: int  # 42
thunk_func_created: int  # 38
ti_changed: int  # 13
tryblks_updated: int  # 46
updating_tryblks: int  # 45
upgraded: int  # 2
weakref: module