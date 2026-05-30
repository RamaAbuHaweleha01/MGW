from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""
idautils.py - High level utility functions for IDA

"""

class Strings:
    r"""
        Allows iterating over the string list. The set of strings will not be
        modified, unless asked explicitly at setup()-time. This string list also
        is used by the "String window" so it may be changed when this window is
        updated.
    
        Example:
            s = Strings()
    
            for i in s:
                print("%x: len=%d type=%d -> '%s'" % (i.ea, i.length, i.strtype, str(i)))
    
        
    """
    def StringItem(self, si: Any) -> Any:
        r"""
                Class representing each string item.
                
        """
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
    def __getitem__(self, index: Any) -> Any:
        r"""Returns a string item or None"""
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, default_setup: Any = False) -> Any:
        r"""
                Initializes the Strings enumeration helper class
        
                :param default_setup: Set to True to use default setup (C strings, min len 5, ...)
                
        """
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
    def clear_cache(self) -> Any:
        r"""Clears the string list cache"""
        ...
    def refresh(self) -> Any:
        r"""Refreshes the string list"""
        ...
    def setup(self, strtypes: Any = ..., minlen: Any = 5, only_7bit: Any = True, ignore_instructions: Any = False, display_only_existing_strings: Any = False) -> Any:
        ...

class peutils_t:
    r"""
        PE utility class. Retrieves PE information from the database.
    
        Constants from pe.h
        
    """
    PE_ALT_DBG_FPOS: int  # 18446744073709551615
    PE_ALT_IMAGEBASE: int  # 18446744073709551614
    PE_ALT_NEFLAGS: int  # 18446744073709551612
    PE_ALT_PEHDR_OFF: int  # 18446744073709551613
    PE_ALT_PSXDLL: int  # 18446744073709551610
    PE_ALT_TDS_LOADED: int  # 18446744073709551611
    PE_NODE: str  # $ PE header
    @property
    def header_offset(self) -> Any: ...
    @property
    def imagebase(self) -> Any: ...
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
    def header(self) -> Any:
        ...

def Assemble(ea: Any, line: Any) -> Any:
    r"""
        Assembles one or more lines (does not display an message dialogs)
        If line is a list then this function will attempt to assemble all the lines
        This function will turn on batch mode temporarily so that no messages are displayed on the screen
    
        :param ea:       start address
        :returns: (False, "Error message") or (True, asm_buf) or (True, [asm_buf1, asm_buf2, asm_buf3])
        
    """
    ...

def Chunks(start: Any) -> Any:
    r"""
        Get a list of function chunks
        See also ida_funcs.func_tail_iterator_t
    
        :param start: address of the function
    
        :returns: list of function chunks (tuples of the form (start_ea, end_ea))
                 belonging to the function
        
    """
    ...

def CodeRefsFrom(ea: Any, flow: bool) -> Any:
    r"""
        Get a list of code references from 'ea'
    
        :param ea:   Target address
        :param flow: Follow normal code flow or not
    
        :returns: list of references (may be empty list)
    
        Example::
    
            for ref in CodeRefsFrom(get_screen_ea(), 1):
                print(ref)
        
    """
    ...

def CodeRefsTo(ea: Any, flow: bool) -> Any:
    r"""
        Get a list of code references to 'ea'
    
        :param ea:   Target address
        :param flow: Follow normal code flow or not
    
        :returns: list of references (may be empty list)
    
        Example::
    
            for ref in CodeRefsTo(get_screen_ea(), 1):
                print(ref)
        
    """
    ...

def DataRefsFrom(ea: Any) -> Any:
    r"""
        Get a list of data references from 'ea'
    
        :param ea:   Target address
    
        :returns: list of references (may be empty list)
    
        Example::
    
            for ref in DataRefsFrom(get_screen_ea()):
                print(ref)
        
    """
    ...

def DataRefsTo(ea: Any) -> Any:
    r"""
        Get a list of data references to 'ea'
    
        :param ea:   Target address
    
        :returns: list of references (may be empty list)
    
        Example::
    
            for ref in DataRefsTo(get_screen_ea()):
                print(ref)
        
    """
    ...

def DecodeInstruction(ea: Any) -> Any:
    r"""
        Decodes an instruction and returns an insn_t like class
    
        :param ea: address to decode
        :returns: None or a new insn_t instance
        
    """
    ...

def DecodePrecedingInstruction(ea: Any) -> Any:
    r"""
        Decode preceding instruction in the execution flow.
    
        :param ea: address to decode
        :returns: (None or the decode instruction, farref)
                 farref will contain 'true' if followed an xref, false otherwise
        
    """
    ...

def DecodePreviousInstruction(ea: Any) -> Any:
    r"""
        Decodes the previous instruction and returns an insn_t like class
    
        :param ea: address to decode
        :returns: None or a new insn_t instance
        
    """
    ...

def Entries() -> Any:
    r"""
        Returns a list of entry points (exports)
    
        :returns: List of tuples (index, ordinal, ea, name)
        
    """
    ...

def FuncItems(start: Any) -> Any:
    r"""
        Get a list of function items (instruction or data items inside function boundaries)
        See also ida_funcs.func_item_iterator_t
    
        :param start: address of the function
    
        :returns: ea of each item in the function
        
    """
    ...

def Functions(start: Any = None, end: Any = None) -> Any:
    r"""
        Get a list of functions
    
        :param start: start address (default: inf.min_ea)
        :param end:   end address (default: inf.max_ea)
    
        :returns: list of function entrypoints between start and end
    
        NOTE: The last function that starts before 'end' is included even
        if it extends beyond 'end'. Any function that has its chunks scattered
        in multiple segments will be reported multiple times, once in each segment
        as they are listed.
        
    """
    ...

def GetDataList(ea: Any, count: Any, itemsize: Any = 1) -> Any:
    r"""
        Get data list - INTERNAL USE ONLY
        
    """
    ...

def GetIdbDir() -> Any:
    r"""
        Get IDB directory
    
        This function returns directory path of the current IDB database
        
    """
    ...

def GetInputFileMD5() -> bytes:
    r"""Get input file md5.
    
    """
    ...

def GetInstructionList() -> Any:
    r"""Returns the instruction list of the current processor module"""
    ...

def GetRegisterList() -> Any:
    r"""Returns the register list"""
    ...

def Heads(start: Any = None, end: Any = None) -> Any:
    r"""
        Get a list of heads (instructions or data items)
    
        :param start: start address (default: inf.min_ea)
        :param end:   end address (default: inf.max_ea)
    
        :returns: list of heads between start and end
        
    """
    ...

def MapDataList(ea: Any, length: Any, func: Any, wordsize: Any = 1) -> Any:
    r"""
        Map through a list of data words in the database
    
        :param ea:       start address
        :param length:   number of words to map
        :param func:     mapping function
        :param wordsize: size of words to map [default: 1 byte]
    
        :returns: None
        
    """
    ...

def Modules() -> Any:
    r"""
        Returns a list of module objects with name,size,base and the rebase_to attributes
        
    """
    ...

def Names() -> Any:
    r"""
        Returns a list of names
    
        :returns: List of tuples (ea, name)
        
    """
    ...

def ProcessUiActions(actions: Any, flags: Any = 0) -> Any:
    r"""
        :param actions: A string containing a list of actions separated by semicolon, a list or a tuple
        :param flags: flags to be passed to process_ui_action()
        :returns: Boolean. Returns False if the action list was empty or execute_ui_requests() failed.
        
    """
    ...

def PutDataList(ea: Any, datalist: Any, itemsize: Any = 1) -> Any:
    r"""
        Put data list - INTERNAL USE ONLY
        
    """
    ...

def Segments() -> Any:
    r"""
        Get list of segments (sections) in the binary image
    
        :returns: List of segment start addresses.
        
    """
    ...

def StructMembers(sid: Any) -> Any:
    r"""
        Get a list of structure members information (or stack vars if given a frame).
    
        :param sid: ID of the structure.
    
        :returns: List of tuples (offset_in_bytes, name, size_in_bytes)
    
        NOTE: If 'sid' does not refer to a valid structure, an exception will be raised.
        NOTE: This will not return 'holes' in structures/stack frames; it only returns defined structure members.
        
    """
    ...

def Structs() -> Any:
    r"""
        Get a list of structures
    
        :returns: List of tuples (ordinal, sid, name)
        
    """
    ...

def Threads() -> Any:
    r"""Returns all thread IDs for the current debugee"""
    ...

def XrefTypeName(typecode: Any) -> Any:
    r"""
        Convert cross-reference type codes to readable names
    
        :param typecode: cross-reference type code
        
    """
    ...

def XrefsFrom(ea: Any, flags: Any = 0) -> Any:
    r"""
        Return all references from address 'ea'
    
        :param ea: Reference address
        :param flags: one of ida_xref.XREF_ALL (default), ida_xref.XREF_FAR, ida_xref.XREF_DATA
    
        Example::
               for xref in XrefsFrom(here(), 0):
                   print(xref.type, XrefTypeName(xref.type),                          'from', hex(xref.frm), 'to', hex(xref.to))
        
    """
    ...

def XrefsTo(ea: Any, flags: Any = 0) -> Any:
    r"""
        Return all references to address 'ea'
    
        :param ea: Reference address
        :param flags: one of ida_xref.XREF_ALL (default), ida_xref.XREF_FAR, ida_xref.XREF_DATA
    
        Example::
               for xref in XrefsTo(here(), 0):
                   print(xref.type, XrefTypeName(xref.type),                          'from', hex(xref.frm), 'to', hex(xref.to))
        
    """
    ...

cpu: _cpu  # <idautils._cpu object at 0x717728809de0>
ida_bytes: module
ida_dbg: module
ida_entry: module
ida_funcs: module
ida_ida: _module_wrapper_t
ida_idaapi: module
ida_idd: module
ida_idp: module
ida_kernwin: module
ida_loader: module
ida_nalt: module
ida_name: module
ida_netnode: module
ida_segment: module
ida_strlist: module
ida_typeinf: module
ida_ua: module
ida_xref: module
idc: _module_wrapper_t
os: module  # <module 'os' from '/usr/lib/python3.10/os.py'>
procregs: _procregs  # <idautils._procregs object at 0x717728809e10>
sys: module  # <module 'sys' (built-in)>
types: module