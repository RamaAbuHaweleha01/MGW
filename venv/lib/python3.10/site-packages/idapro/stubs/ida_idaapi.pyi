from typing import Any, Optional, List, Dict, Tuple, Callable, Union

class CustomIDAMemo:
    def CreateGroups(self, groups_infos: Any) -> Any:
        r"""
                Send a request to modify the graph by creating a
                (set of) group(s), and perform an animation.
        
                Each object in the 'groups_infos' list must be of the format:
                {
                  "nodes" : [<int>, <int>, <int>, ...] # The list of nodes to group
                  "text" : <string>                    # The synthetic text for that group
                }
        
                :param groups_infos: A list of objects that describe those groups.
                :returns: A [<int>, <int>, ...] list of group nodes, or None (failure).
                
        """
        ...
    def DelNodesInfos(self, nodes: Any) -> Any:
        r"""
                Delete the properties for the given node(s).
        
                :param nodes: A list of node IDs
                
        """
        ...
    def DeleteGroups(self, groups: Any, new_current: Any = -1) -> Any:
        r"""
                Send a request to delete the specified groups in the graph,
                and perform an animation.
        
                :param groups: A list of group node numbers.
                :param new_current: A node to focus on after the groups have been deleted
                :returns: True on success, False otherwise.
                
        """
        ...
    def GetCurrentRendererType(self) -> Any:
        ...
    def GetNodeInfo(self, args: Any) -> Any:
        r"""
                Get the properties for the given node.
        
                :param ni: A node_info_t instance
                :param node: The index of the node.
                :returns: success
                
        """
        ...
    def GetWidget(self) -> Any:
        r"""
                Return the TWidget underlying this view.
        
                :returns: The TWidget underlying this view, or None.
                
        """
        ...
    def GetWidgetAsGraphViewer(self) -> Any:
        r"""
                Return the graph_viewer_t underlying this view.
        
                :returns: The graph_viewer_t underlying this view, or None.
                
        """
        ...
    def Refresh(self) -> Any:
        r"""
                Refreshes the view. This causes the OnRefresh() to be called
                
        """
        ...
    def SetCurrentRendererType(self, rtype: Any) -> Any:
        r"""
                Set the current view's renderer.
        
                :param rtype: The renderer type. Should be one of the idaapi.TCCRT_* values.
                
        """
        ...
    def SetGroupsVisibility(self, groups: Any, expand: Any, new_current: Any = -1) -> Any:
        r"""
                Send a request to expand/collapse the specified groups in the graph,
                and perform an animation.
        
                :param groups: A list of group node numbers.
                :param expand: True to expand the group, False otherwise.
                :param new_current: A node to focus on after the groups have been expanded/collapsed.
                :returns: True on success, False otherwise.
                
        """
        ...
    def SetNodeInfo(self, node_index: Any, node_info: Any, flags: Any) -> Any:
        r"""
                Set the properties for the given node.
        
                Example usage (set second nodes's bg color to red):
                  inst = ...
                  p = idaapi.node_info_t()
                  p.bg_color = 0x00ff0000
                  inst.SetNodeInfo(1, p, idaapi.NIF_BG_COLOR)
        
                :param node_index: The node index.
                :param node_info: An idaapi.node_info_t instance.
                :param flags: An OR'ed value of NIF_* values.
                
        """
        ...
    def SetNodesInfos(self, values: Any) -> Any:
        r"""
                Set the properties for the given nodes.
        
                Example usage (set first three nodes's bg color to purple):
                  inst = ...
                  p = idaapi.node_info_t()
                  p.bg_color = 0x00ff00ff
                  inst.SetNodesInfos({0 : p, 1 : p, 2 : p})
        
                :param values: A dictionary of 'int -> node_info_t' objects.
                
        """
        ...
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
    def hook(self) -> bool:
        ...
    def unhook(self) -> bool:
        ...
    def view_activated(self, view: Any) -> Any:
        ...
    def view_click(self, view: Any, ve: Any) -> Any:
        ...
    def view_close(self, view: Any, args: Any) -> Any:
        ...
    def view_created(self, view: TWidget) -> None:
        r"""A view is being created. 
                  
        :param view: (TWidget *)
        """
        ...
    def view_curpos(self, view: Any, args: Any) -> Any:
        ...
    def view_dblclick(self, view: Any, ve: Any) -> Any:
        ...
    def view_deactivated(self, view: Any) -> Any:
        ...
    def view_keydown(self, view: Any, key: Any, state: Any) -> Any:
        ...
    def view_loc_changed(self, view: Any, now: Any, was: Any) -> Any:
        ...
    def view_mouse_moved(self, view: Any, ve: Any) -> Any:
        ...
    def view_mouse_over(self, view: Any, ve: Any) -> Any:
        ...
    def view_switched(self, view: Any, rt: Any) -> Any:
        ...

class IDAPython_displayhook:
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
        r"""Return repr(self)."""
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
    def displayhook(self, item: Any) -> Any:
        ...
    def displayhook_format(self, item: Any) -> Any:
        ...
    def format_item(self, num_printer: Any, storage: Any, item: Any) -> Any:
        ...
    def format_seq(self, num_printer: Any, storage: Any, item: Any, opn: Any, cls: Any) -> Any:
        ...

class PyIdc_cvt_int64__(pyidc_cvt_helper__):
    r"""Helper class for explicitly representing VT_INT64 values"""
    @property
    def value(self) -> Any: ...
    def __add__(self, other: Any) -> Any:
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __div__(self, other: Any) -> Any:
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
    def __init__(self, v: Any) -> Any:
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
    def __mul__(self, other: Any) -> Any:
        ...
    def __ne__(self, value: Any) -> Any:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __radd__(self, other: Any) -> Any:
        ...
    def __rdiv__(self, other: Any) -> Any:
        ...
    def __reduce__(self) -> Any:
        r"""Helper for pickle."""
        ...
    def __reduce_ex__(self, protocol: Any) -> Any:
        r"""Helper for pickle."""
        ...
    def __repr__(self) -> Any:
        r"""Return repr(self)."""
        ...
    def __rmul__(self, other: Any) -> Any:
        ...
    def __rsub__(self, other: Any) -> Any:
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
    def __sub__(self, other: Any) -> Any:
        ...
    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Abstract classes can override this to customize issubclass().
        
        This is invoked early on by abc.ABCMeta.__subclasscheck__().
        It should return True, False or NotImplemented.  If it returns
        NotImplemented, the normal algorithm is used.  Otherwise, it
        overrides the normal algorithm (and the outcome is cached).
        
        """
        ...

class PyIdc_cvt_refclass__(pyidc_cvt_helper__):
    r"""Helper class for representing references to immutable objects"""
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
    def __init__(self, v: Any) -> Any:
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
        r"""Return repr(self)."""
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
    def cstr(self) -> Any:
        r"""Returns the string as a C string (up to the zero termination)"""
        ...

class loader_input_t:
    r"""A helper class to work with linput_t related functions.
    This class is also used by file loaders scripts.
    """
    @property
    def __idc_cvt_id__(self) -> Any: ...
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
    def __init__(self, pycapsule: Any = None) -> Any:
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
    def close(self) -> Any:
        r"""Closes the file"""
        ...
    def file2base(self, pos: int, ea1: ea_t, ea2: ea_t, patchable: bool) -> Any:
        r"""Load portion of file into the database
        This function will include (ea1..ea2) into the addressing space of the
        program (make it enabled)
        
        :param li: pointer ot input source
        :param pos: position in the file
        :param ea1: start of range of destination linear addresses
        :param ea2: end of range of destination linear addresses
        :param patchable: should the kernel remember correspondance of
                          file offsets to linear addresses.
        :returns: 1-ok,0-read error, a warning is displayed
        """
        ...
    def filename(self) -> Any:
        ...
    def from_capsule(self, pycapsule: Any) -> loader_input_t:
        ...
    def from_fp(self, fp: Any) -> Any:
        r"""A static method to construct an instance from a FILE*
        
        :param fp: a FILE pointer
        :returns: a new instance, or None
        """
        ...
    def from_linput(self, linput: linput_t) -> loader_input_t:
        ...
    def get_byte(self) -> Any:
        r"""Reads a single byte from the file. Returns None if EOF or the read byte"""
        ...
    def get_linput(self) -> linput_t:
        ...
    def gets(self, len: int) -> Any:
        r"""Reads a line from the input file. Returns the read line or None
        
        :param len: the maximum line length
        :returns: a str, or None
        """
        ...
    def getz(self, size: int, fpos: int = -1) -> Any:
        r"""Returns a zero terminated string at the given position
        
        :param size: maximum size of the string
        :param fpos: if != -1 then seek will be performed before reading
        :returns: The string or None on failure.
        """
        ...
    def open(self, filename: Any, remote: Any = False) -> Any:
        r"""Opens a file (or a remote file)
        
        :param filename: the file name
        :param remote: whether the file is local, or remote
        :returns: Boolean
        """
        ...
    def open_memory(self, start: ea_t, size: int) -> Any:
        r"""Create a linput for process memory (By internally calling idaapi.create_memory_linput())
        This linput will use dbg->read_memory() to read data
        
        :param start: starting address of the input
        :param size: size of the memory range to represent as linput
                    if unknown, may be passed as 0
        """
        ...
    def opened(self) -> Any:
        r"""Checks if the file is opened or not"""
        ...
    def read(self, size: int = -1) -> Any:
        r"""Read up to size bytes (all data if size is negative). Return an empty bytes object on EOF.
        
        :param size: the maximum number of bytes to read
        :returns: a bytes object
        """
        ...
    def readbytes(self, size: int, big_endian: bool) -> Any:
        r"""Similar to read() but it respect the endianness
        
        :param size: the maximum number of bytes to read
        :param big_endian: endianness
        :returns: a str, or None
        """
        ...
    def seek(self, offset: int, whence: Any = 0) -> Any:
        r"""Set input source position
        
        :param offset: the seek offset
        :param whence: the position to seek from
        :returns: the new position (not 0 as fseek!)
        """
        ...
    def set_linput(self, linput: Any) -> Any:
        r"""Links the current loader_input_t instance to a linput_t instance
        
        :param linput: the linput_t to link to
        """
        ...
    def size(self) -> int64:
        ...
    def tell(self) -> Any:
        r"""Returns the current position"""
        ...

class object_t:
    r"""Helper class used to initialize empty objects"""
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
    def __getitem__(self, idx: Any) -> Any:
        r"""Allow access to object attributes by index (like dictionaries)"""
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, kwds: Any) -> Any:
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
        r"""Return repr(self)."""
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

class plugin_t(pyidc_opaque_object_t):
    r"""Base class for all scripted plugins."""
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
        r"""Initialize self.  See help(type(self)) for accurate signature."""
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
        r"""Return repr(self)."""
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
    def run(self, arg: Any) -> Any:
        ...
    def term(self) -> Any:
        ...

class plugmod_t(pyidc_opaque_object_t):
    r"""Base class for all scripted multi-plugins."""
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
        r"""Initialize self.  See help(type(self)) for accurate signature."""
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
        r"""Return repr(self)."""
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

class py_clinked_object_t(pyidc_opaque_object_t):
    r"""
        This is a utility and base class for C linked objects
        
    """
    @property
    def clink(self) -> Any: ...
    @property
    def clink_ptr(self) -> Any: ...
    def __del__(self) -> Any:
        r"""Delete the link upon object destruction (only if not static)"""
        ...
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
    def __init__(self, lnk: Any = None) -> Any:
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
        r"""Return repr(self)."""
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
    def assign(self, other: Any) -> Any:
        r"""
                Overwrite me.
                This method allows you to assign an instance contents to anothers
                :returns: Boolean
                
        """
        ...
    def copy(self) -> Any:
        r"""Returns a new copy of this class"""
        ...

class pyidc_cvt_helper__:
    r"""
        This is a special helper object that helps detect which kind
        of object is this python object wrapping and how to convert it
        back and from IDC.
        This object is characterized by its special attribute and its value
        
    """
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
    def __init__(self, cvt_id: Any, value: Any) -> Any:
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
        r"""Return repr(self)."""
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

class pyidc_opaque_object_t:
    r"""This is the base class for all Python<->IDC opaque objects"""
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
        r"""Initialize self.  See help(type(self)) for accurate signature."""
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
        r"""Return repr(self)."""
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

def IDAPython_Completion(line: Any, x: Any) -> Any:
    r"""Internal utility class for auto-completion support"""
    ...

def IDAPython_ExecScript(path: Any, g: Any, print_error: Any = True) -> Any:
    r"""
        Run the specified script.
    
        This function is used by the low-level plugin code.
        
    """
    ...

def IDAPython_ExecSystem(cmd: Any) -> Any:
    r"""
        Executes a command with popen().
        
    """
    ...

def IDAPython_FormatExc(etype: Any, value: Any = None, tb: Any = None, limit: Any = None) -> Any:
    r"""
        This function is used to format an exception given the
        values returned by a PyErr_Fetch()
        
    """
    ...

def IDAPython_GetDocstrings(obj: Any) -> Any:
    ...

def IDAPython_LoadProcMod(path: Any, g: Any, print_error: Any = True) -> Any:
    r"""
        Load processor module.
        
    """
    ...

def IDAPython_UnLoadProcMod(script: Any, g: Any, print_error: Any = True) -> Any:
    r"""
        Unload processor module.
        
    """
    ...

def TRUNC(ea: Any) -> Any:
    r""" Truncate EA for the current application bitness"""
    ...

def as_UTF16(s: Any) -> Any:
    r"""Convenience function to convert a string into appropriate unicode format"""
    ...

def as_cstr(val: Any) -> Any:
    r"""
        Returns a C str from the passed value. The passed value can be of type refclass (returned by a call to buffer() or byref())
        It scans for the first \x00 and returns the string value up to that point.
        
    """
    ...

def as_int32(v: Any) -> Any:
    r"""Returns a number as a signed int32 number"""
    ...

def as_signed(v: Any, nbits: Any = 32) -> Any:
    r"""
        Returns a number as signed. The number of bits are specified by the user.
        The MSB holds the sign.
        
    """
    ...

def as_uint32(v: Any) -> Any:
    r"""Returns a number as an unsigned int32 number"""
    ...

def as_unicode(s: Any) -> Any:
    r"""Convenience function to convert a string into appropriate unicode format"""
    ...

def copy_bits(v: Any, s: Any, e: Any = -1) -> Any:
    r"""
        Copy bits from a value
        :param v: the value
        :param s: starting bit (0-based)
        :param e: ending bit
        
    """
    ...

def disable_script_timeout() -> Any:
    r"""Disables the script timeout and hides the script wait box.
    Calling L{set_script_timeout} will not have any effects until the script is compiled and executed again
    
    :returns: None
    """
    ...

def enable_extlang_python(enable: Any) -> Any:
    r"""Enables or disables Python extlang.
    When enabled, all expressions will be evaluated by Python.
    
    :param enable: Set to True to enable, False otherwise
    """
    ...

def enable_python_cli(enable: bool) -> None:
    ...

def format_basestring(_in: Any) -> str:
    ...

def notify_when(when: Any, callback: Any) -> Any:
    r"""
        Register a callback that will be called when an event happens.
        :param when: one of NW_XXXX constants
        :param callback: This callback prototype varies depending on the 'when' parameter:
                         The general callback format:
                             def notify_when_callback(nw_code)
                         In the case of NW_OPENIDB:
                             def notify_when_callback(nw_code, is_old_database)
        :returns: Boolean
        
    """
    ...

def parse_command_line3(cmdline: str) -> Any:
    ...

def pycim_get_widget(_self: Any) -> TWidget:
    ...

def pycim_view_close(_self: Any) -> None:
    ...

def pygc_create_groups(_self: Any, groups_infos: Any) -> Any:
    ...

def pygc_delete_groups(_self: Any, groups: Any, new_current: Any) -> Any:
    ...

def pygc_refresh(_self: Any) -> None:
    ...

def pygc_set_groups_visibility(_self: Any, groups: Any, expand: Any, new_current: Any) -> Any:
    ...

def replfun(func: Any) -> Any:
    ...

def require(modulename: Any, package: Any = None) -> Any:
    r"""
        Load, or reload a module.
    
        When under heavy development, a user's tool might consist of multiple
        modules. If those are imported using the standard 'import' mechanism,
        there is no guarantee that the Python implementation will re-read
        and re-evaluate the module's Python code. In fact, it usually doesn't.
        What should be done instead is 'reload()'-ing that module.
    
        This is a simple helper function that will do just that: In case the
        module doesn't exist, it 'import's it, and if it does exist,
        'reload()'s it.
    
        The importing module (i.e., the module calling require()) will have
        the loaded module bound to its globals(), under the name 'modulename'.
        (If require() is called from the command line, the importing module
        will be '__main__'.)
    
        For more information, see: <http://www.hexblog.com/?p=749>.
        
    """
    ...

def set_script_timeout(timeout: Any) -> Any:
    r"""Changes the script timeout value. The script wait box dialog will be hidden and shown again when the timeout elapses.
    See also L{disable_script_timeout}.
    
    :param timeout: This value is in seconds.
                    If this value is set to zero then the script will never timeout.
    :returns: Returns the old timeout value
    """
    ...

def struct_unpack(buffer: Any, signed: Any = False, offs: Any = 0) -> Any:
    r"""
        Unpack a buffer given its length and offset using struct.unpack_from().
        This function will know how to unpack the given buffer by using the lookup table '__struct_unpack_table'
        If the buffer is of unknown length then None is returned. Otherwise the unpacked value is returned.
        
    """
    ...

ea_t = int
long_type = int
BADADDR: int  # 18446744073709551615
BADADDR32: int  # 4294967295
BADADDR64: int  # 18446744073709551615
BADSEL: int  # 18446744073709551615
HBF_CALL_WITH_NEW_EXEC: int  # 1
HBF_VOLATILE_METHOD_SET: int  # 2
NW_CLOSEIDB: int  # 2
NW_INITIDA: int  # 4
NW_OPENIDB: int  # 1
NW_REMOVE: int  # 16
NW_TERMIDA: int  # 8
PLUGIN_DBG: int  # 32
PLUGIN_DRAW: int  # 2
PLUGIN_FIX: int  # 128
PLUGIN_HIDE: int  # 16
PLUGIN_KEEP: int  # 2
PLUGIN_MOD: int  # 1
PLUGIN_MULTI: int  # 256
PLUGIN_OK: int  # 1
PLUGIN_PROC: int  # 64
PLUGIN_SEG: int  # 4
PLUGIN_SKIP: int  # 0
PLUGIN_UNL: int  # 8
PY_ICID_BYREF: int  # 1
PY_ICID_INT64: int  # 0
PY_ICID_OPAQUE: int  # 2
SEEK_CUR: int  # 1
SEEK_END: int  # 2
SEEK_SET: int  # 0
SIZE_MAX: int  # 18446744073709551615
ST_OVER_DEBUG_SEG: int  # 1
ST_OVER_LIB_FUNC: int  # 2
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
bisect: module
builtins: module  # <module 'builtins' (built-in)>
ida_idaapi: module
inspect: module
integer_types: tuple  # (<class 'int'>,)
os: module  # <module 'os' from '/usr/lib/python3.10/os.py'>
re: module  # <module 're' from '/usr/lib/python3.10/re.py'>
string_types: tuple  # (<class 'str'>,)
struct: module
sys: module  # <module 'sys' (built-in)>
traceback: module
weakref: module