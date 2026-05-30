from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Merge functionality.

NOTE: this functionality is available in IDA Teams (not IDA Pro)
There are 3 databases involved in merging: base_idb, local_db, and remote_idb.
* base_idb: the common base ancestor of 'local_db' and 'remote_db'. in the UI this database is located in the middle.
* local_idb: local database that will contain the result of the merging. in the UI this database is located on the left.
* remote_idb: remote database that will merge into local_idb. It may reside locally on the current computer, despite its name. in the UI this database is located on the right. base_idb and remote_idb are opened for reading only. base_idb may be absent, in this case a 2-way merging is performed.


Conflicts can be resolved automatically or interactively. The automatic resolving scores the conflicting blocks and takes the better one. The interactive resolving displays the full rendered contents side by side, and expects the user to select the better side for each conflict.
Since IDB files contain various kinds of information, there are many merging phases. The entire list can be found in merge.cpp. Below are just some selected examples:
* merge global database settings (inf and other global vars)
* merge segmentation and changes to the database bytes
* merge various lists: exports, imports, loaded tils, etc
* merge names, functions, function frames
* merge debugger settings, breakpoints
* merge struct/enum views
* merge local type libraries
* merge the disassembly items (i.e. the segment contents) this includes operand types, code/data separation, etc
* merge plugin specific info like decompiler types, dwarf mappings, etc


To unify UI elements of each merge phase, we use merger views:
* A view that consists of 2 or 3 panes: left (local_idb) and right (remote_idb). The common base is in the middle, if present.
* Rendering of the panes depends on the phase, different phases show different contents.
* The conflicts are highlighted by a colored background. Also, the detail pane can be consulted for additional info.
* The user can select a conflict (or a bunch of conflicts) and say "use this block".
* The user can browse the panes as he wishes. He will not be forced to handle conflicts in any particular order. However, once he finishes working with a merge handler and proceeds to the next one, he cannot go back.
* Scrolling the left pane will synchronously scroll the right pane and vice versa.
* There are the navigation commands like "go to the prev/next conflict"
* The number of remaining conflicts to resolve is printed in the "Progress" chooser.
* The user may manually modify local database inside the merger view. For that he may use the regular hotkeys. However, editing the database may lead to new conflicts, so we better restrict the available actions to some reasonable minimum. Currently, this is not implemented.


IDA works in a new "merge" mode during merging. In this mode most events are not generated. We forbid them to reduce the risk that a rogue third-party plugin that is not aware of the "merge" mode would spoil something.
For example, normally renaming a function causes a cascade of events and may lead to other database modifications. Some of them may be desired, some - not. Since there are some undesired events, it is better to stop generating them. However, some events are required to render the disassembly listing. For example, ev_ana_insn, av_out_insn. This is why some events are still generated in the "merge" mode.
To let processor modules and plugins merge their data, we introduce a new event: ev_create_merge_handlers. It is generated immediately after opening all three idbs. The interested modules should react to this event by creating new merge handlers, if they need them.
While the kernel can create arbitrary merge handlers, modules can create only the standard ones returned by:
create_nodeval_merge_handler() create_nodeval_merge_handlers() create_std_modmerge_handlers()
We do not document merge_handler_t because once a merge handler is created, it is used exclusively by the kernel.
See mergemod.hpp for more information about the merge mode for modules. 
    
"""

class item_block_locator_t:
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
    def get_block_head(self, md: merge_data_t, idx: diff_source_idx_t, item_head: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        ...
    def setup_blocks(self, md: merge_data_t, _from: diff_source_idx_t, to: diff_source_idx_t, region: diff_range_t) -> bool:
        ...

class merge_data_t:
    @property
    def dbctx_ids(self) -> Any: ...
    @property
    def ev_handlers(self) -> Any: ...
    @property
    def item_block_locator(self) -> Any: ...
    @property
    def last_udt_related_merger(self) -> Any: ...
    @property
    def nbases(self) -> Any: ...
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
    def add_event_handler(self, handler: merge_handler_t) -> None:
        ...
    def base_id(self) -> int:
        ...
    def compare_merging_tifs(self, tif1: tinfo_t, diffidx1: diff_source_idx_t, tif2: tinfo_t, diffidx2: diff_source_idx_t) -> int:
        r"""compare types from two databases 
                
        :param tif1: type
        :param diffidx1: database index, diff_source_idx_t
        :param tif2: type
        :param diffidx2: database index, diff_source_idx_t
        :returns: -1, 0, 1
        """
        ...
    def get_block_head(self, idx: diff_source_idx_t, item_head: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        ...
    def has_existing_node(self, nodename: str) -> bool:
        r"""check that node exists in any of databases
        
        """
        ...
    def local_id(self) -> int:
        ...
    def map_privrange_id(self, tid: tid_t, ea: ida_idaapi.ea_t, _from: diff_source_idx_t, to: diff_source_idx_t, strict: bool = True) -> bool:
        r"""map IDs of structures, enumerations and their members 
                
        :param tid: item ID in TO database
        :param ea: item ID to find counterpart
        :param to: destination database index, diff_source_idx_t
        :param strict: raise interr if could not map
        :returns: success
        """
        ...
    def map_tinfo(self, tif: tinfo_t, _from: diff_source_idx_t, to: diff_source_idx_t, strict: bool = True) -> bool:
        r"""migrate type, replaces type references into FROM database to references into TO database 
                
        :param tif: type to migrate, will be cleared in case of fail
        :param to: destination database index, diff_source_idx_t
        :param strict: raise interr if could not map
        :returns: success
        """
        ...
    def remote_id(self) -> int:
        ...
    def remove_event_handler(self, handler: merge_handler_t) -> None:
        ...
    def set_dbctx_ids(self, local: int, remote: int, base: int) -> None:
        ...
    def setup_blocks(self, dst_idx: diff_source_idx_t, src_idx: diff_source_idx_t, region: diff_range_t) -> bool:
        ...

class merge_handler_params_t:
    @property
    def insert_after(self) -> Any: ...
    @property
    def kind(self) -> Any: ...
    @property
    def label(self) -> Any: ...
    @property
    def md(self) -> Any: ...
    @property
    def mh_flags(self) -> Any: ...
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
    def __init__(self, _md: merge_data_t, _label: str, _kind: merge_kind_t, _insert_after: merge_kind_t, _mh_flags: int) -> Any:
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
    def ui_complex_details(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_complex_details() -> bool
            1. ui_complex_details(_mh_flags: int) -> bool
        
        # 0: ui_complex_details() -> bool
        
        
        # 1: ui_complex_details(_mh_flags: int) -> bool
        
        Do not display the diffpos details in the chooser. For example, the MERGE_KIND_SCRIPTS handler puts the script body as the diffpos detail. It would not be great to show them as part of the chooser. 
                
        
        """
        ...
    def ui_complex_name(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_complex_name() -> bool
            1. ui_complex_name(_mh_flags: int) -> bool
        
        # 0: ui_complex_name() -> bool
        
        
        # 1: ui_complex_name(_mh_flags: int) -> bool
        
        It customary to create long diffpos names having many components that are separated by any 7-bit ASCII character (besides of '\0'). In this case it is possible to instruct IDA to use this separator to create a multi-column chooser. For example the MERGE_KIND_ENUMS handler has the following diffpos name: enum_1,enum_2 If MH_UI_COMMANAME is specified, IDA will create 2 columns for these names. 
                
        
        """
        ...
    def ui_dp_shortname(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_dp_shortname() -> bool
            1. ui_dp_shortname(_mh_flags: int) -> bool
        
        # 0: ui_dp_shortname() -> bool
        
        
        # 1: ui_dp_shortname(_mh_flags: int) -> bool
        
        The detail pane shows the diffpos details for the current diffpos range as a tree-like view. In this pane the diffpos names are used as tree node names and the diffpos details as their children. Sometimes, for complex diffpos names, the first part of the name looks better than the entire name. For example, the MERGE_KIND_SEGMENTS handler has the following diffpos name: <range>,<segm1>,<segm2>,<segm3> if MH_UI_DP_SHORTNAME is specified, IDA will use <range> as a tree node name 
                
        
        """
        ...
    def ui_has_details(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_has_details() -> bool
            1. ui_has_details(_mh_flags: int) -> bool
        
        # 0: ui_has_details() -> bool
        
        
        # 1: ui_has_details(_mh_flags: int) -> bool
        
        Should IDA display the diffpos detail pane?
        
        
        """
        ...
    def ui_indent(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_indent() -> bool
            1. ui_indent(_mh_flags: int) -> bool
        
        # 0: ui_indent() -> bool
        
        
        # 1: ui_indent(_mh_flags: int) -> bool
        
        In the ordinary situation the spaces from the both sides of diffpos name are trimmed. Use this UI hint to preserve the leading spaces. 
                
        
        """
        ...
    def ui_linediff(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. ui_linediff() -> bool
            1. ui_linediff(_mh_flags: int) -> bool
        
        # 0: ui_linediff() -> bool
        
        
        # 1: ui_linediff(_mh_flags: int) -> bool
        
        In detail pane IDA shows difference between diffpos details. IDA marks added or deleted detail by color. In the modified detail the changes are marked. Use this UI hint if you do not want to show the differences inside detail. 
                
        
        """
        ...
    def ui_split_char(self, args: Any) -> char:
        r"""This function has the following signatures:
        
            0. ui_split_char() -> char
            1. ui_split_char(_mh_flags: int) -> char
        
        # 0: ui_split_char() -> char
        
        
        # 1: ui_split_char(_mh_flags: int) -> char
        
        
        """
        ...
    def ui_split_str(self, args: Any) -> str:
        r"""This function has the following signatures:
        
            0. ui_split_str() -> str
            1. ui_split_str(_mh_flags: int) -> str
        
        # 0: ui_split_str() -> str
        
        
        # 1: ui_split_str(_mh_flags: int) -> str
        
        
        """
        ...

class merge_node_helper_t:
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
    def append_eavec(self, s: str, prefix: str, eas: eavec_t) -> None:
        r"""can be used by derived classes
        
        """
        ...
    def get_column_headers(self, arg0: qstrvec_t, arg1: uchar, arg2: void) -> None:
        r"""get column headers for chooser (to be used in linear_diff_source_t::get_column_headers) 
                
        """
        ...
    def get_netnode(self) -> netnode:
        r"""return netnode to be used as source. If this function returns BADNODE netnode will be created using netnode name passed to create_nodeval_diff_source 
                
        """
        ...
    def is_mergeable(self, arg0: uchar, arg1: nodeidx_t) -> bool:
        r"""filter: check if we should perform merging for given record
        
        """
        ...
    def map_scalar(self, arg0: nodeidx_t, arg1: void, arg2: diff_source_idx_t, arg3: diff_source_idx_t) -> None:
        r"""map scalar/string/buffered value
        
        """
        ...
    def map_string(self, arg0: str, arg1: void, arg2: diff_source_idx_t, arg3: diff_source_idx_t) -> None:
        ...
    def print_entry_details(self, arg0: qstrvec_t, arg1: uchar, arg2: nodeidx_t, arg3: void) -> None:
        r"""print the details of the specified entry usually contains multiple lines, one for each attribute or detail. (to be used in print_diffpos_details) 
                
        """
        ...
    def print_entry_name(self, arg0: uchar, arg1: nodeidx_t, arg2: void) -> str:
        r"""print the name of the specified entry (to be used in print_diffpos_name) 
                
        """
        ...
    def refresh(self, arg0: uchar, arg1: void) -> None:
        r"""notify helper that some data was changed in the database and internal structures (e.g. caches) should be refreshed 
                
        """
        ...

class merge_node_info_t:
    @property
    def name(self) -> Any: ...
    @property
    def nds_flags(self) -> Any: ...
    @property
    def node_helper(self) -> Any: ...
    @property
    def tag(self) -> Any: ...
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
    def __init__(self, name: str, tag: uchar, nds_flags: int, node_helper: merge_node_helper_t = None) -> Any:
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

class moddata_diff_helper_t:
    @property
    def additional_mh_flags(self) -> Any: ...
    @property
    def fields(self) -> Any: ...
    @property
    def module_name(self) -> Any: ...
    @property
    def netnode_name(self) -> Any: ...
    @property
    def nfields(self) -> Any: ...
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
    def __init__(self, _module_name: str, _netnode_name: str, _fields: idbattr_info_t) -> Any:
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
    def get_struc_ptr(self, arg0: merge_data_t, arg1: diff_source_idx_t, arg2: idbattr_info_t) -> None:
        ...
    def merge_ending(self, arg0: diff_source_idx_t, arg1: void) -> None:
        ...
    def merge_starting(self, arg0: diff_source_idx_t, arg1: void) -> None:
        ...
    def print_diffpos_details(self, arg0: qstrvec_t, arg1: idbattr_info_t) -> None:
        ...
    def str2val(self, arg0: uint64, arg1: idbattr_info_t, arg2: str) -> bool:
        ...
    def val2str(self, arg0: str, arg1: idbattr_info_t, arg2: uint64) -> bool:
        ...

def create_nodeval_merge_handler(mhp: merge_handler_params_t, label: str, nodename: str, tag: uchar, nds_flags: int, node_helper: merge_node_helper_t = None, skip_empty_nodes: bool = True) -> merge_handler_t:
    r"""Create a merge handler for netnode scalar/string values 
            
    :param mhp: merging parameters
    :param label: handler short name (to be be appended to mhp.label)
    :param nodename: netnode name
    :param tag: a tag used to access values in the netnode
    :param nds_flags: netnode value attributes (a combination of nds_flags_t)
    :param skip_empty_nodes: do not create handler in case of empty netnode
    :returns: diff source object (normally should be attahced to a merge handler)
    """
    ...

def create_nodeval_merge_handlers(out: merge_handlers_t, mhp: merge_handler_params_t, nodename: str, valdesc: merge_node_info_t, skip_empty_nodes: bool = True) -> None:
    r"""Create a serie of merge handlers for netnode scalar/string values (call create_nodeval_merge_handler() for each member of VALDESC) 
            
    :param out: [out] created handlers will be placed here
    :param mhp: merging parameters
    :param nodename: netnode name
    :param valdesc: array of handler descriptions
    :param skip_empty_nodes: do not create handlers for empty netnodes
    :returns: diff source object (normally should be attahced to a merge handler)
    """
    ...

def destroy_moddata_merge_handlers(data_id: int) -> None:
    ...

def get_ea_diffpos_name(ea: ida_idaapi.ea_t) -> str:
    r"""Get nice name for EA diffpos 
            
    :param ea: diffpos
    """
    ...

def is_diff_merge_mode() -> bool:
    r"""Return TRUE if IDA is running in diff mode (MERGE_POLICY_MDIFF/MERGE_POLICY_VDIFF)
    
    """
    ...

MERGE_KIND_AFLAGS_EA: int  # 31
MERGE_KIND_AUTOQ: int  # 1
MERGE_KIND_BOOKMARKS: int  # 45
MERGE_KIND_BPTS: int  # 43
MERGE_KIND_BYTEVAL: int  # 23
MERGE_KIND_CREFS: int  # 41
MERGE_KIND_CUSTCNV: int  # 8
MERGE_KIND_CUSTDATA: int  # 7
MERGE_KIND_DBG_MEMREGS: int  # 56
MERGE_KIND_DEBUGGER: int  # 55
MERGE_KIND_DEKSTOPS: int  # 52
MERGE_KIND_DIRTREE: int  # 47
MERGE_KIND_DREFS: int  # 42
MERGE_KIND_ENCODINGS: int  # 3
MERGE_KIND_ENCODINGS2: int  # 4
MERGE_KIND_END: int  # -2
MERGE_KIND_ENUMS: int  # 9
MERGE_KIND_EXPORTS: int  # 26
MERGE_KIND_EXTRACMT: int  # 30
MERGE_KIND_FILEREGIONS: int  # 33
MERGE_KIND_FIXUPS: int  # 24
MERGE_KIND_FLAGS: int  # 29
MERGE_KIND_FLOWS: int  # 40
MERGE_KIND_FRAME: int  # 38
MERGE_KIND_FRAMEMGR: int  # 37
MERGE_KIND_FUNC: int  # 36
MERGE_KIND_GHSTRCMT: int  # 15
MERGE_KIND_HIDDENRANGES: int  # 34
MERGE_KIND_IGNOREMICRO: int  # 32
MERGE_KIND_IMPORTS: int  # 27
MERGE_KIND_INF: int  # 2
MERGE_KIND_LAST: int  # 58
MERGE_KIND_LOADER: int  # 54
MERGE_KIND_LUMINA: int  # 57
MERGE_KIND_MAPPING: int  # 25
MERGE_KIND_NETNODE: int  # 0
MERGE_KIND_NONE: int  # -1
MERGE_KIND_NOTEPAD: int  # 53
MERGE_KIND_ORPHANS: int  # 22
MERGE_KIND_PATCHES: int  # 28
MERGE_KIND_PROBLEMS: int  # 50
MERGE_KIND_SCRIPTS: int  # 6
MERGE_KIND_SCRIPTS2: int  # 5
MERGE_KIND_SEGGRPS: int  # 20
MERGE_KIND_SEGMENTS: int  # 19
MERGE_KIND_SEGREGS: int  # 21
MERGE_KIND_SELECTORS: int  # 17
MERGE_KIND_SIGNATURES: int  # 49
MERGE_KIND_SOURCEFILES: int  # 35
MERGE_KIND_STKPNTS: int  # 39
MERGE_KIND_STRMEM: int  # 13
MERGE_KIND_STRMEMCMT: int  # 16
MERGE_KIND_STRUCTS: int  # 10
MERGE_KIND_STT: int  # 18
MERGE_KIND_TILS: int  # 11
MERGE_KIND_TINFO: int  # 12
MERGE_KIND_TRYBLKS: int  # 46
MERGE_KIND_UDTMEM: int  # 14
MERGE_KIND_UI: int  # 51
MERGE_KIND_VFTABLES: int  # 48
MERGE_KIND_WATCHPOINTS: int  # 44
MH_LISTEN: int  # 1
MH_TERSE: int  # 2
MH_UI_CHAR_MASK: int  # 8323072
MH_UI_COLONNAME: int  # 12189696
MH_UI_COMMANAME: int  # 11272192
MH_UI_COMPLEX: int  # 512
MH_UI_DP_NOLINEDIFF: int  # 1024
MH_UI_DP_SHORTNAME: int  # 2048
MH_UI_INDENT: int  # 4096
MH_UI_NODETAILS: int  # 256
MH_UI_SPLITNAME: int  # 8388608
NDS_BLOB: int  # 32
NDS_EV_FUNC: int  # 128
NDS_EV_RANGE: int  # 64
NDS_INC: int  # 8192
NDS_IS_BOOL: int  # 1
NDS_IS_EA: int  # 2
NDS_IS_RELATIVE: int  # 4
NDS_IS_STR: int  # 8
NDS_MAP_IDX: int  # 256
NDS_MAP_VAL: int  # 512
NDS_SUPVAL: int  # 16
NDS_UI_ND: int  # 16384
NDS_VAL8: int  # 4096
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module