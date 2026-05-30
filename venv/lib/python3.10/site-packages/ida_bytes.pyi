from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Contains functions that deal with individual byte characteristics.

Each byte of the disassembled program is represented by a 32-bit value. We will call this value 'flags'. The structure of the flags is here.
You are not allowed to inspect individual bits of flags and modify them directly. Use special functions to inspect and/or modify flags.
Flags are kept in a virtual array file (*.id1). Addresses (ea) are all 32-bit (or 64-bit) quantities. 
    
"""

class compiled_binpat_t:
    @property
    def bytes(self) -> Any: ...
    @property
    def encidx(self) -> Any: ...
    @property
    def mask(self) -> Any: ...
    @property
    def strlits(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: compiled_binpat_t) -> bool:
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
    def __ne__(self, r: compiled_binpat_t) -> bool:
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
    def all_bytes_defined(self) -> bool:
        ...
    def qclear(self) -> None:
        ...

class compiled_binpat_vec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: compiled_binpat_vec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> compiled_binpat_t:
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
    def __ne__(self, r: compiled_binpat_vec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: compiled_binpat_t) -> None:
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
    def add_unique(self, x: compiled_binpat_t) -> bool:
        ...
    def append(self, x: compiled_binpat_t) -> None:
        ...
    def at(self, _idx: size_t) -> compiled_binpat_t:
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
    def extend(self, x: compiled_binpat_vec_t) -> None:
        ...
    def extract(self) -> compiled_binpat_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: compiled_binpat_t) -> bool:
        ...
    def inject(self, s: compiled_binpat_t, len: size_t) -> None:
        ...
    def insert(self, it: compiled_binpat_t, x: compiled_binpat_t) -> qvector:
        ...
    def parse(self, ea: ida_idaapi.ea_t, text: str, radix: int = -1, strlits_encoding: int = -1) -> compiled_binpat_vec_t:
        r"""Convert user-specified binary string to internal representation.
        
        The 'in' parameter contains space-separated tokens:
        
            *numbers (numeric base is determined by 'radix')
                - if value of number fits a byte, it is considered as a byte
                - if value of number fits a word, it is considered as 2 bytes
                - if value of number fits a dword,it is considered as 4 bytes
            * "..." string constants
            * 'x'  single-character constants
            * ?    variable bytes
        
        Note that string constants are surrounded with double quotes.
        
        Here are a few examples (assuming base 16):
        
            * CD 21          - bytes 0xCD, 0x21
            * 21CD           - bytes 0xCD, 0x21 (little endian ) or 0x21, 0xCD (big-endian)
            * "Hello", 0     - the null terminated string "Hello"
            * L"Hello"       - 'H', 0, 'e', 0, 'l', 0, 'l', 0, 'o', 0
            * B8 ? ? ? ? 90  - byte 0xB8, 4 bytes with any value, byte 0x90
        
        This method will throw an exception if the pattern could not be parsed
        
        :param ea: linear address to convert for (the conversion depends on the
                   address, because the number of bits in a byte depend on the
                   segment type)
        :param text: input text string
        :param radix: numeric base of numbers (8,10,16). If `-1` (the default), then the default radix will be used (see get_default_radix)
        :param strlits_encoding: the target encoding into which the string
                             literals present in 'in', should be encoded.
                             Can be any from [1, get_encoding_qty()), or
                             the special values PBSENC_*
        :returns: a set of patterns
        """
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> compiled_binpat_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: compiled_binpat_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class data_format_t:
    r"""Information about a data format"""
    @property
    def hotkey(self) -> Any: ...
    @property
    def id(self) -> Any: ...
    @property
    def menu_name(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def props(self) -> Any: ...
    @property
    def text_width(self) -> Any: ...
    @property
    def value_size(self) -> Any: ...
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
    def __gt__(self, value: Any) -> Any:
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
    def __le__(self, value: Any) -> Any:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> Any:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __real__init__(self, _self: Any, name: str, value_size: asize_t = 0, menu_name: str = None, props: int = 0, hotkey: str = None, text_width: int = 0) -> Any:
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
    def is_present_in_menus(self) -> bool:
        r"""Should this format be shown in UI menus 
                
        :returns: success
        """
        ...

class data_type_t:
    r"""Information about a data type"""
    @property
    def asm_keyword(self) -> Any: ...
    @property
    def hotkey(self) -> Any: ...
    @property
    def id(self) -> Any: ...
    @property
    def menu_name(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def props(self) -> Any: ...
    @property
    def value_size(self) -> Any: ...
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
    def __gt__(self, value: Any) -> Any:
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
    def __le__(self, value: Any) -> Any:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> Any:
        r"""Return self<value."""
        ...
    def __ne__(self, value: Any) -> bool:
        r"""Return self!=value."""
        ...
    def __new__(self, args: Any, kwargs: Any) -> Any:
        r"""Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __real__init__(self, _self: Any, name: str, value_size: asize_t = 0, menu_name: str = None, hotkey: str = None, asm_keyword: str = None, props: int = 0) -> Any:
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
    def is_present_in_menus(self) -> bool:
        r"""Should this type be shown in UI menus 
                
        :returns: success
        """
        ...

class hidden_range_t:
    @property
    def color(self) -> Any: ...
    @property
    def description(self) -> Any: ...
    @property
    def end_ea(self) -> Any: ...
    @property
    def footer(self) -> Any: ...
    @property
    def header(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def visible(self) -> Any: ...
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

class octet_generator_t:
    @property
    def avail_bits(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def high_byte_first(self) -> Any: ...
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
    def __init__(self, _ea: ida_idaapi.ea_t) -> Any:
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
    def invert_byte_order(self) -> None:
        ...

def add_byte(ea: ida_idaapi.ea_t, value: int) -> None:
    r"""Add a value to one byte of the program. This function works for wide byte processors too. 
            
    :param ea: linear address
    :param value: byte value
    """
    ...

def add_dword(ea: ida_idaapi.ea_t, value: uint64) -> None:
    r"""Add a value to one dword of the program. This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :param ea: linear address
    :param value: byte value
    """
    ...

def add_hidden_range(args: Any) -> bool:
    r"""Mark a range of addresses as hidden. The range will be created in the invisible state with the default color 
            
    :param ea1: linear address of start of the address range
    :param ea2: linear address of end of the address range
    :param description: range parameters
    :param header: range parameters
    :param footer: range parameters
    :param color: the range color
    :returns: success
    """
    ...

def add_mapping(_from: ida_idaapi.ea_t, to: ida_idaapi.ea_t, size: asize_t) -> bool:
    r"""IDA supports memory mapping. References to the addresses from the mapped range use data and meta-data from the mapping range. 
            
    :param to: start of the mapping range (existent address)
    :param size: size of the range
    :returns: success
    """
    ...

def add_qword(ea: ida_idaapi.ea_t, value: uint64) -> None:
    r"""Add a value to one qword of the program. This function does not work for wide byte processors. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :param ea: linear address
    :param value: byte value
    """
    ...

def add_word(ea: ida_idaapi.ea_t, value: uint64) -> None:
    r"""Add a value to one word of the program. This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :param ea: linear address
    :param value: byte value
    """
    ...

def align_flag() -> flags64_t:
    r"""Get a flags64_t representing an alignment directive.
    
    """
    ...

def append_cmt(ea: ida_idaapi.ea_t, str: str, rptble: bool) -> bool:
    r"""Append to an indented comment. Creates a new comment if none exists. Appends a newline character and the specified string otherwise. 
            
    :param ea: linear address
    :param str: comment string to append
    :param rptble: append to repeatable comment?
    :returns: success
    """
    ...

def attach_custom_data_format(dtid: int, dfid: int) -> bool:
    r"""Attach the data format to the data type. 
            
    :param dtid: data type id that can use the data format. 0 means all standard data types. Such data formats can be applied to any data item or instruction operands. For instruction operands, the data_format_t::value_size check is not performed by the kernel.
    :param dfid: data format id
    :returns: true: ok
    :returns: false: no such `dtid`, or no such `dfid', or the data format has already been attached to the data type
    """
    ...

def bin_flag() -> flags64_t:
    r"""Get number flag of the base, regardless of current processor - better to use num_flag()
    
    """
    ...

def bin_search(args: Any) -> Any:
    r"""Search for a set of bytes in the program
    
    This function has the following signatures:
    
        1. bin_search(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, data: compiled_binpat_vec_t, flags: int) -> Tuple[ida_idaapi.ea_t, int]
        2. bin_search(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, image: bytes, mask: bytes, len: int, flags: int) -> ida_idaapi.ea_t
    
    The return value type will differ depending on the form:
    
        1. a tuple `(matched-address, index-in-compiled_binpat_vec_t)` (1st form)
        2. the address of a match, or ida_idaapi.BADADDR if not found (2nd form)
    
    This is a low-level function; more user-friendly alternatives
    are available. Please see 'find_bytes' and 'find_string'.
    
    :param start_ea: linear address, start of range to search
    :param end_ea: linear address, end of range to search (exclusive)
    :param data: (1st form) the prepared data to search for (see parse_binpat_str())
    :param bytes: (2nd form) a set of bytes to match
    :param mask: (2nd form) a mask to apply to the set of bytes
    :param flags: combination of BIN_SEARCH_* flags
    :returns: either a tuple holding both the address of the match and the index of the compiled pattern that matched, or the address of a match (ida_idaapi.BADADDR if not found)
    """
    ...

def byte_flag() -> flags64_t:
    r"""Get a flags64_t representing a byte.
    
    """
    ...

def bytesize(ea: ida_idaapi.ea_t) -> int:
    r"""Get number of bytes required to store a byte at the given address.
    
    """
    ...

def calc_def_align(ea: ida_idaapi.ea_t, mina: int, maxa: int) -> int:
    r"""Calculate the default alignment exponent. 
            
    :param ea: linear address
    :param mina: minimal possible alignment exponent.
    :param maxa: minimal possible alignment exponent.
    """
    ...

def calc_dflags(f: flags64_t, force: bool) -> flags64_t:
    ...

def calc_max_align(endea: ida_idaapi.ea_t) -> int:
    r"""Calculate the maximal possible alignment exponent. 
            
    :param endea: end address of the alignment item.
    :returns: a value in the 0..32 range
    """
    ...

def calc_max_item_end(ea: ida_idaapi.ea_t, how: int = 15) -> ida_idaapi.ea_t:
    r"""Calculate maximal reasonable end address of a new item. This function will limit the item with the current segment bounds. 
            
    :param ea: linear address
    :param how: when to stop the search. A combination of Item end search flags
    :returns: end of new item. If it is not possible to create an item, it will return 'ea'. If operation was cancelled by user, it will return 'ea'
    """
    ...

def calc_min_align(length: asize_t) -> int:
    r"""Calculate the minimal possible alignment exponent. 
            
    :param length: size of the item in bytes.
    :returns: a value in the 1..32 range
    """
    ...

def can_define_item(ea: ida_idaapi.ea_t, length: asize_t, flags: flags64_t) -> bool:
    r"""Can define item (instruction/data) of the specified 'length', starting at 'ea'? 
    * a new item would cross segment boundaries
    * a new item would overlap with existing items (except items specified by 'flags') 
    
    
            
    :param ea: start of the range for the new item
    :param length: length of the new item in bytes
    :param flags: if not 0, then the kernel will ignore the data types specified by the flags and destroy them. For example: 
                     1000 dw 5
                     1002 db 5 ; undef
                     1003 db 5 ; undef
                     1004 dw 5
                     1006 dd 5
                      can_define_item(1000, 6, 0) - false because of dw at 1004 
     can_define_item(1000, 6, word_flag()) - true, word at 1004 is destroyed
    :returns: 1-yes, 0-no
    """
    ...

def change_storage_type(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, stt: storage_type_t) -> error_t:
    r"""Change flag storage type for address range. 
            
    :param start_ea: should be lower than end_ea.
    :param end_ea: does not belong to the range.
    :param stt: storage_type_t
    :returns: error code
    """
    ...

def char_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def chunk_size(ea: ida_idaapi.ea_t) -> int:
    r"""Get size of the contiguous address block containing 'ea'. 
            
    :returns: 0 if 'ea' doesn't belong to the program.
    """
    ...

def chunk_start(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get start of the contiguous address block containing 'ea'. 
            
    :returns: BADADDR if 'ea' doesn't belong to the program.
    """
    ...

def clr_lzero(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Clear toggle lzero bit. This function reset the display of leading zeroes for the specified operand to the default. If the default is not to display leading zeroes, leading zeroes will not be displayed, as vice versa. 
            
    :param ea: the item (insn/data) address
    :param n: the operand number (0-first operand, 1-other operands)
    :returns: success
    """
    ...

def clr_op_type(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Remove operand representation information. (set operand representation to be 'undefined') 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: success
    """
    ...

def code_flag() -> flags64_t:
    r"""FF_CODE
    
    """
    ...

def combine_flags(F: flags64_t) -> flags64_t:
    ...

def create_16bit_data(ea: ida_idaapi.ea_t, length: asize_t) -> bool:
    r"""Convert to 16-bit quantity (take the byte size into account)
    
    """
    ...

def create_32bit_data(ea: ida_idaapi.ea_t, length: asize_t) -> bool:
    r"""Convert to 32-bit quantity (take the byte size into account)
    
    """
    ...

def create_align(ea: ida_idaapi.ea_t, length: asize_t, alignment: int) -> bool:
    r"""Create an alignment item. 
            
    :param ea: linear address
    :param length: size of the item in bytes. 0 means to infer from ALIGNMENT
    :param alignment: alignment exponent. Example: 3 means align to 8 bytes. 0 means to infer from LENGTH It is forbidden to specify both LENGTH and ALIGNMENT as 0.
    :returns: success
    """
    ...

def create_byte(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to byte.
    
    """
    ...

def create_custdata(ea: ida_idaapi.ea_t, length: asize_t, dtid: int, fid: int, force: bool = False) -> bool:
    r"""Convert to custom data type.
    
    """
    ...

def create_data(ea: ida_idaapi.ea_t, dataflag: flags64_t, size: asize_t, tid: tid_t) -> bool:
    r"""Convert to data (byte, word, dword, etc). This function may be used to create arrays. 
            
    :param ea: linear address
    :param dataflag: type of data. Value of function byte_flag(), word_flag(), etc.
    :param size: size of array in bytes. should be divisible by the size of one item of the specified type. for variable sized items it can be specified as 0, and the kernel will try to calculate the size.
    :param tid: type id. If the specified type is a structure, then tid is structure id. Otherwise should be BADNODE.
    :returns: success
    """
    ...

def create_double(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to double.
    
    """
    ...

def create_dword(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to dword.
    
    """
    ...

def create_float(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to float.
    
    """
    ...

def create_oword(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to octaword/xmm word.
    
    """
    ...

def create_packed_real(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to packed decimal real.
    
    """
    ...

def create_qword(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to quadword.
    
    """
    ...

def create_strlit(start: ida_idaapi.ea_t, len: size_t, strtype: int) -> bool:
    r"""Convert to string literal and give a meaningful name. 'start' may be higher than 'end', the kernel will swap them in this case 
            
    :param start: starting address
    :param len: length of the string in bytes. if 0, then get_max_strlit_length() will be used to determine the length
    :param strtype: string type. one of String type codes
    :returns: success
    """
    ...

def create_struct(ea: ida_idaapi.ea_t, length: asize_t, tid: tid_t, force: bool = False) -> bool:
    r"""Convert to struct.
    
    """
    ...

def create_tbyte(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to tbyte.
    
    """
    ...

def create_word(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to word.
    
    """
    ...

def create_yword(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to ymm word.
    
    """
    ...

def create_zword(ea: ida_idaapi.ea_t, length: asize_t, force: bool = False) -> bool:
    r"""Convert to zmm word.
    
    """
    ...

def cust_flag() -> flags64_t:
    r"""Get a flags64_t representing custom type data.
    
    """
    ...

def custfmt_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def dec_flag() -> flags64_t:
    r"""Get number flag of the base, regardless of current processor - better to use num_flag()
    
    """
    ...

def del_hidden_range(ea: ida_idaapi.ea_t) -> bool:
    r"""Delete hidden range. 
            
    :param ea: any address in the hidden range
    :returns: success
    """
    ...

def del_items(ea: ida_idaapi.ea_t, flags: int = 0, nbytes: asize_t = 1, may_destroy: may_destroy_cb_t = None) -> bool:
    r"""Convert item (instruction/data) to unexplored bytes. The whole item (including the head and tail bytes) will be destroyed. It is allowed to pass any address in the item to this function 
            
    :param ea: any address within the first item to delete
    :param flags: combination of Unexplored byte conversion flags
    :param nbytes: number of bytes in the range to be undefined
    :param may_destroy: optional routine invoked before deleting a head item. If callback returns false then item is not to be deleted and operation fails
    :returns: true on sucessful operation, otherwise false
    """
    ...

def del_mapping(ea: ida_idaapi.ea_t) -> None:
    r"""Delete memory mapping range. 
            
    :param ea: any address in the mapped range
    """
    ...

def del_value(ea: ida_idaapi.ea_t) -> None:
    r"""Delete byte value from flags. The corresponding byte becomes uninitialized. 
            
    """
    ...

def detach_custom_data_format(dtid: int, dfid: int) -> bool:
    r"""Detach the data format from the data type. Unregistering a custom data type detaches all attached data formats, no need to detach them explicitly. You still need unregister them. Unregistering a custom data format detaches it from all attached data types. 
            
    :param dtid: data type id to detach data format from
    :param dfid: data format id to detach
    :returns: true: ok
    :returns: false: no such `dtid`, or no such `dfid', or the data format was not attached to the data type
    """
    ...

def disable_flags(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t) -> error_t:
    r"""Deallocate flags for address range. Exit with an error message if not enough disk space (this may occur too). 
            
    :param start_ea: should be lower than end_ea.
    :param end_ea: does not belong to the range.
    :returns: 0 if ok, otherwise return error code
    """
    ...

def double_flag() -> flags64_t:
    r"""Get a flags64_t representing a double.
    
    """
    ...

def dword_flag() -> flags64_t:
    r"""Get a flags64_t representing a double word.
    
    """
    ...

def enable_flags(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, stt: storage_type_t) -> error_t:
    r"""Allocate flags for address range. This function does not change the storage type of existing ranges. Exit with an error message if not enough disk space. 
            
    :param start_ea: should be lower than end_ea.
    :param end_ea: does not belong to the range.
    :param stt: storage_type_t
    :returns: 0 if ok, otherwise an error code
    """
    ...

def enum_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def equal_bytes(ea: ida_idaapi.ea_t, image: uchar, mask: uchar, len: size_t, bin_search_flags: int) -> bool:
    r"""Compare 'len' bytes of the program starting from 'ea' with 'image'. 
            
    :param ea: linear address
    :param image: bytes to compare with
    :param mask: array of mask bytes, it's length is 'len'. if the flag BIN_SEARCH_BITMASK is passsed, 'bitwise AND' is used to compare. if not; 1 means to perform the comparison of the corresponding byte. 0 means not to perform. if mask == nullptr, then all bytes of 'image' will be compared. if mask == SKIP_FF_MASK then 0xFF bytes will be skipped
    :param len: length of block to compare in bytes.
    :param bin_search_flags: combination of Search flags
    :returns: 1: equal
    :returns: 0: not equal
    """
    ...

def f_has_cmt(f: flags64_t, arg2: void) -> bool:
    ...

def f_has_dummy_name(f: flags64_t, arg2: void) -> bool:
    r"""Does the current byte have dummy (auto-generated, with special prefix) name?
    
    """
    ...

def f_has_extra_cmts(f: flags64_t, arg2: void) -> bool:
    ...

def f_has_name(f: flags64_t, arg2: void) -> bool:
    r"""Does the current byte have non-trivial (non-dummy) name?
    
    """
    ...

def f_has_user_name(F: flags64_t, arg2: void) -> bool:
    r"""Does the current byte have user-specified name?
    
    """
    ...

def f_has_xref(f: flags64_t, arg2: void) -> bool:
    r"""Does the current byte have cross-references to it?
    
    """
    ...

def f_is_align(F: flags64_t, arg2: void) -> bool:
    r"""See is_align()
    
    """
    ...

def f_is_byte(F: flags64_t, arg2: void) -> bool:
    r"""See is_byte()
    
    """
    ...

def f_is_code(F: flags64_t, arg2: void) -> bool:
    r"""Does flag denote start of an instruction?
    
    """
    ...

def f_is_custom(F: flags64_t, arg2: void) -> bool:
    r"""See is_custom()
    
    """
    ...

def f_is_data(F: flags64_t, arg2: void) -> bool:
    r"""Does flag denote start of data?
    
    """
    ...

def f_is_double(F: flags64_t, arg2: void) -> bool:
    r"""See is_double()
    
    """
    ...

def f_is_dword(F: flags64_t, arg2: void) -> bool:
    r"""See is_dword()
    
    """
    ...

def f_is_float(F: flags64_t, arg2: void) -> bool:
    r"""See is_float()
    
    """
    ...

def f_is_head(F: flags64_t, arg2: void) -> bool:
    r"""Does flag denote start of instruction OR data?
    
    """
    ...

def f_is_not_tail(F: flags64_t, arg2: void) -> bool:
    r"""Does flag denote tail byte?
    
    """
    ...

def f_is_oword(F: flags64_t, arg2: void) -> bool:
    r"""See is_oword()
    
    """
    ...

def f_is_pack_real(F: flags64_t, arg2: void) -> bool:
    r"""See is_pack_real()
    
    """
    ...

def f_is_qword(F: flags64_t, arg2: void) -> bool:
    r"""See is_qword()
    
    """
    ...

def f_is_strlit(F: flags64_t, arg2: void) -> bool:
    r"""See is_strlit()
    
    """
    ...

def f_is_struct(F: flags64_t, arg2: void) -> bool:
    r"""See is_struct()
    
    """
    ...

def f_is_tail(F: flags64_t, arg2: void) -> bool:
    r"""Does flag denote tail byte?
    
    """
    ...

def f_is_tbyte(F: flags64_t, arg2: void) -> bool:
    r"""See is_tbyte()
    
    """
    ...

def f_is_word(F: flags64_t, arg2: void) -> bool:
    r"""See is_word()
    
    """
    ...

def f_is_yword(F: flags64_t, arg2: void) -> bool:
    r"""See is_yword()
    
    """
    ...

def find_byte(sEA: ida_idaapi.ea_t, size: asize_t, value: uchar, bin_search_flags: int) -> ida_idaapi.ea_t:
    r"""Find forward a byte with the specified value (only 8-bit value from the database). example: ea=4 size=3 will inspect addresses 4, 5, and 6 
            
    :param sEA: linear address
    :param size: number of bytes to inspect
    :param value: value to find
    :param bin_search_flags: combination of Search flags
    :returns: address of byte or BADADDR
    """
    ...

def find_byter(sEA: ida_idaapi.ea_t, size: asize_t, value: uchar, bin_search_flags: int) -> ida_idaapi.ea_t:
    r"""Find reverse a byte with the specified value (only 8-bit value from the database). example: ea=4 size=3 will inspect addresses 6, 5, and 4 
            
    :param sEA: the lower address of the search range
    :param size: number of bytes to inspect
    :param value: value to find
    :param bin_search_flags: combination of Search flags
    :returns: address of byte or BADADDR
    """
    ...

def find_bytes(bs: Any, range_start: int, range_size: typing.Optional[int] = None, range_end: typing.Optional[int] = 18446744073709551615, mask: Any = None, flags: typing.Optional[int] = 8, radix: typing.Optional[int] = 16, strlit_encoding: Any = 0) -> int:
    ...

def find_custom_data_format(name: str) -> int:
    r"""Get id of a custom data format. 
            
    :param name: name of the custom data format
    :returns: id or -1
    """
    ...

def find_custom_data_type(name: str) -> int:
    r"""Get id of a custom data type. 
            
    :param name: name of the custom data type
    :returns: id or -1
    """
    ...

def find_free_chunk(start: ida_idaapi.ea_t, size: asize_t, alignment: asize_t) -> ida_idaapi.ea_t:
    r"""Search for a hole in the addressing space of the program. 
            
    :param start: Address to start searching from
    :param size: Size of the desired empty range
    :param alignment: Alignment bitmask, must be a pow2-1. (for example, 0xF would align the returned range to 16 bytes).
    :returns: Start of the found empty range or BADADDR
    """
    ...

def find_string(_str: str, range_start: int, range_end: typing.Optional[int] = 18446744073709551615, range_size: typing.Optional[int] = None, strlit_encoding: Any = 0, flags: typing.Optional[int] = 8) -> int:
    ...

def float_flag() -> flags64_t:
    r"""Get a flags64_t representing a float.
    
    """
    ...

def flt_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def get_16bit(ea: ida_idaapi.ea_t) -> int:
    r"""Get 16bits of the program at 'ea'. 
            
    :returns: 1 byte (getFullByte()) if the current processor has 16-bit byte, otherwise return get_word()
    """
    ...

def get_32bit(ea: ida_idaapi.ea_t) -> int:
    r"""Get not more than 32bits of the program at 'ea'. 
            
    :returns: 32 bit value, depending on processor_t::nbits:
    * if ( nbits <= 8 ) return get_dword(ea);
    * if ( nbits <= 16) return get_wide_word(ea);
    * return get_wide_byte(ea);
    """
    ...

def get_64bit(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get not more than 64bits of the program at 'ea'. 
            
    :returns: 64 bit value, depending on processor_t::nbits:
    * if ( nbits <= 8 ) return get_qword(ea);
    * if ( nbits <= 16) return get_wide_dword(ea);
    * return get_wide_byte(ea);
    """
    ...

def get_byte(ea: ida_idaapi.ea_t) -> uchar:
    r"""Get one byte (8-bit) of the program at 'ea'. This function works only for 8bit byte processors. 
            
    """
    ...

def get_bytes(ea: ida_idaapi.ea_t, size: int, gmb_flags: int = 1) -> Any:
    r"""Get the specified number of bytes of the program.
    
    :param ea: program address
    :param size: number of bytes to return
    :param gmb_flags: OR'ed combination of GMB_* values (defaults to GMB_READALL)
    :returns: the bytes (as bytes object), or None in case of failure
    """
    ...

def get_bytes_and_mask(ea: ida_idaapi.ea_t, size: int, gmb_flags: int = 1) -> Any:
    r"""Get the specified number of bytes of the program, and a bitmask
    specifying what bytes are defined and what bytes are not.
    
    :param ea: program address
    :param size: number of bytes to return
    :param gmb_flags: OR'ed combination of GMB_* values (defaults to GMB_READALL)
    :returns: a tuple (bytes, mask), or None in case of failure.
             Both 'bytes' and 'mask' are 'str' instances.
    """
    ...

def get_cmt(ea: ida_idaapi.ea_t, rptble: bool) -> str:
    r"""Get an indented comment. 
            
    :param ea: linear address. may point to tail byte, the function will find start of the item
    :param rptble: get repeatable comment?
    :returns: size of comment or -1
    """
    ...

def get_custom_data_format(dfid: int) -> data_format_t:
    r"""Get definition of a registered custom data format. 
            
    :param dfid: data format id
    :returns: data format definition or nullptr
    """
    ...

def get_custom_data_formats(out: intvec_t, dtid: int) -> int:
    r"""Get list of attached custom data formats for the specified data type. 
            
    :param out: buffer for the output. may be nullptr
    :param dtid: data type id
    :returns: number of returned custom data formats. if error, returns -1
    """
    ...

def get_custom_data_type(dtid: int) -> data_type_t:
    r"""Get definition of a registered custom data type. 
            
    :param dtid: data type id
    :returns: data type definition or nullptr
    """
    ...

def get_custom_data_types(args: Any) -> int:
    r"""Get list of registered custom data type ids. 
            
    :param out: buffer for the output. may be nullptr
    :param min_size: minimum value size
    :param max_size: maximum value size
    :returns: number of custom data types with the specified size limits
    """
    ...

def get_data_elsize(ea: ida_idaapi.ea_t, F: flags64_t, ti: opinfo_t = None) -> int:
    r"""Get size of data type specified in flags 'F'. 
            
    :param ea: linear address of the item
    :param F: flags
    :param ti: additional information about the data type. For example, if the current item is a structure instance, then ti->tid is structure id. Otherwise is ignored (may be nullptr). If specified as nullptr, will be automatically retrieved from the database
    :returns: * byte : 1
    * word : 2
    * etc...
    """
    ...

def get_data_value(v: uval_t, ea: ida_idaapi.ea_t, size: asize_t) -> bool:
    r"""Get the value at of the item at 'ea'. This function works with entities up to sizeof(ea_t) (bytes, word, etc) 
            
    :param v: pointer to the result. may be nullptr
    :param ea: linear address
    :param size: size of data to read. If 0, then the item type at 'ea' will be used
    :returns: success
    """
    ...

def get_db_byte(ea: ida_idaapi.ea_t) -> uchar:
    r"""Get one byte (8-bit) of the program at 'ea' from the database. Works even if the debugger is active. See also get_dbg_byte() to read the process memory directly. This function works only for 8bit byte processors. 
            
    """
    ...

def get_default_radix() -> int:
    r"""Get default base of number for the current processor. 
            
    :returns: 2, 8, 10, 16
    """
    ...

def get_dword(ea: ida_idaapi.ea_t) -> int:
    r"""Get one dword (32-bit) of the program at 'ea'. This function takes into account order of bytes specified in idainfo::is_be() This function works only for 8bit byte processors. 
            
    """
    ...

def get_enum_id(ea: ida_idaapi.ea_t, n: int) -> uchar:
    r"""Get enum id of 'enum' operand. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL one of the operands
    :returns: id of enum or BADNODE
    """
    ...

def get_first_hidden_range() -> hidden_range_t:
    r"""Get pointer to the first hidden range. 
            
    :returns: ptr to hidden range or nullptr
    """
    ...

def get_flags(ea: ida_idaapi.ea_t) -> flags64_t:
    r"""Get flags value for address 'ea'. The byte value is not included in the flags. This function should be used if the operand types of any operand beyond the first two operands is required. This function is more expensive to use than get_flags32() 
            
    :returns: 0 if address is not present in the program
    """
    ...

def get_flags32(ea: ida_idaapi.ea_t) -> flags64_t:
    r"""Get only 32 low bits of flags. This function returns the most commonly used bits of the flags. However, it does not return the operand info for the operands beyond the first two operands (0,1). If you need to deal with the operands (2..n), then use get_flags(). It is customary to assign the return value to the variable named "F32", to distinguish is from 64-bit flags. 
            
    :returns: 0 if address is not present in the program
    """
    ...

def get_flags_by_size(size: size_t) -> flags64_t:
    r"""Get flags from size (in bytes). Supported sizes: 1, 2, 4, 8, 16, 32. For other sizes returns 0 
            
    """
    ...

def get_flags_ex(ea: ida_idaapi.ea_t, how: int) -> flags64_t:
    r"""Get flags for the specified address, extended form.
    
    """
    ...

def get_forced_operand(ea: ida_idaapi.ea_t, n: int) -> str:
    r"""Get forced operand. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number
    :returns: size of forced operand or -1
    """
    ...

def get_full_data_elsize(ea: ida_idaapi.ea_t, F: flags64_t, ti: opinfo_t = None) -> int:
    r"""Get full size of data type specified in flags 'F'. takes into account processors with wide bytes e.g. returns 2 for a byte element with 16-bit bytes 
            
    """
    ...

def get_full_flags(ea: ida_idaapi.ea_t) -> flags64_t:
    r"""Get full flags value for address 'ea'. This function returns the byte value in the flags as well. See FF_IVL and MS_VAL. This function is more expensive to use than get_flags() 
            
    :returns: 0 if address is not present in the program
    """
    ...

def get_hidden_range(ea: ida_idaapi.ea_t) -> hidden_range_t:
    r"""Get pointer to hidden range structure, in: linear address. 
            
    :param ea: any address in the hidden range
    """
    ...

def get_hidden_range_num(ea: ida_idaapi.ea_t) -> int:
    r"""Get number of a hidden range. 
            
    :param ea: any address in the hidden range
    :returns: number of hidden range (0..get_hidden_range_qty()-1)
    """
    ...

def get_hidden_range_qty() -> int:
    r"""Get number of hidden ranges.
    
    """
    ...

def get_item_end(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the end address of the item at 'ea'. The returned address doesn't belong to the current item. Unexplored bytes are counted as 1 byte entities. 
            
    """
    ...

def get_item_flag(_from: ida_idaapi.ea_t, n: int, ea: ida_idaapi.ea_t, appzero: bool) -> flags64_t:
    r"""Get flag of the item at 'ea' even if it is a tail byte of some array or structure. This function is used to get flags of structure members or array elements. 
            
    :param n: operand number which refers to 'ea' or OPND_ALL for one of the operands
    :param ea: the referenced address
    :param appzero: append a struct field name if the field offset is zero? meaningful only if the name refers to a structure.
    :returns: flags or 0 (if failed)
    """
    ...

def get_item_head(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the start address of the item at 'ea'. If there is no current item, then 'ea' will be returned (see definition at the end of bytes.hpp source) 
            
    """
    ...

def get_item_refinfo(ri: refinfo_t, ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Get refinfo of the item at 'ea'. This function works for a regular offset operand as well as for a tail byte of a structure variable (in this case refinfo to corresponding structure member will be returned) 
            
    :param ri: refinfo holder
    :param ea: the item address
    :param n: operand number which refers to 'ea' or OPND_ALL for one of the operands
    :returns: success
    """
    ...

def get_item_size(ea: ida_idaapi.ea_t) -> int:
    r"""Get size of item (instruction/data) in bytes. Unexplored bytes have length of 1 byte. This function returns 0 only for BADADDR. 
            
    """
    ...

def get_last_hidden_range() -> hidden_range_t:
    r"""Get pointer to the last hidden range. 
            
    :returns: ptr to hidden range or nullptr
    """
    ...

def get_manual_insn(ea: ida_idaapi.ea_t) -> str:
    r"""Retrieve the user-specified string for the manual instruction. 
            
    :param ea: linear address of the instruction or data item
    :returns: size of manual instruction or -1
    """
    ...

def get_mapping(n: size_t) -> Any:
    r"""Get memory mapping range by its number. 
            
    :param n: number of mapping range (0..get_mappings_qty()-1)
    :returns: false if the specified range doesn't exist, otherwise returns `from`, `to`, `size`
    """
    ...

def get_mappings_qty() -> int:
    r"""Get number of mappings.
    
    """
    ...

def get_max_strlit_length(ea: ida_idaapi.ea_t, strtype: int, options: int = 0) -> int:
    r"""Determine maximum length of string literal.
    If the string literal has a length prefix (e.g., STRTYPE_LEN2 has a two-byte length prefix), the length of that prefix (i.e., 2) will be part of the returned value.
    
    :param ea: starting address
    :param strtype: string type. one of String type codes
    :param options: combination of string literal length options
    :returns: length of the string in octets (octet==8bit)
    """
    ...

def get_next_hidden_range(ea: ida_idaapi.ea_t) -> hidden_range_t:
    r"""Get pointer to next hidden range. 
            
    :param ea: any address in the program
    :returns: ptr to hidden range or nullptr if next hidden range doesn't exist
    """
    ...

def get_octet(ogen: octet_generator_t) -> uchar:
    ...

def get_operand_flag(typebits: uint8, n: int) -> flags64_t:
    r"""Place operand `n`'s type flag in the right nibble of a 64-bit flags set.
    
    :param typebits: the type bits (one of `FF_N_`)
    :param n: the operand number
    :returns: the shift to the nibble
    """
    ...

def get_operand_type_shift(n: int) -> int:
    r"""Get the shift in `flags64_t` for the nibble representing operand `n`'s type
    Note: n must be < UA_MAXOP, and is not checked
    
    :param n: the operand number
    :returns: the shift to the nibble
    """
    ...

def get_opinfo(buf: opinfo_t, ea: ida_idaapi.ea_t, n: int, flags: flags64_t) -> opinfo_t:
    r"""Get additional information about an operand representation. 
            
    :param buf: buffer to receive the result. may not be nullptr
    :param ea: linear address of item
    :param n: number of operand, 0 or 1
    :param flags: flags of the item
    :returns: nullptr if no additional representation information
    """
    ...

def get_optype_flags0(F: flags64_t) -> flags64_t:
    r"""Get flags for first operand.
    
    """
    ...

def get_optype_flags1(F: flags64_t) -> flags64_t:
    r"""Get flags for second operand.
    
    """
    ...

def get_original_byte(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get original byte value (that was before patching). This function works for wide byte processors too. 
            
    """
    ...

def get_original_dword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get original dword (that was before patching) This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_original_qword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get original qword value (that was before patching) This function DOESN'T work for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_original_word(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get original word value (that was before patching). This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_possible_item_varsize(ea: ida_idaapi.ea_t, tif: tinfo_t) -> int:
    r"""Return the possible size of the item at EA of type TIF if TIF is the variable structure. 
            
    :param ea: the linear address of the item
    :param tif: the item type
    :returns: the possible size
    :returns: asize_t(-1): TIF is not a variable structure
    """
    ...

def get_predef_insn_cmt(ins: insn_t) -> str:
    r"""Get predefined comment. 
            
    :param ins: current instruction information
    :returns: size of comment or -1
    """
    ...

def get_prev_hidden_range(ea: ida_idaapi.ea_t) -> hidden_range_t:
    r"""Get pointer to previous hidden range. 
            
    :param ea: any address in the program
    :returns: ptr to hidden range or nullptr if previous hidden range doesn't exist
    """
    ...

def get_qword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one qword (64-bit) of the program at 'ea'. This function takes into account order of bytes specified in idainfo::is_be() This function works only for 8bit byte processors. 
            
    """
    ...

def get_radix(F: flags64_t, n: int) -> int:
    r"""Get radix of the operand, in: flags. If the operand is not a number, returns get_default_radix() 
            
    :param F: flags
    :param n: number of operand (0, 1, -1)
    :returns: 2, 8, 10, 16
    """
    ...

def get_strlit_contents(ea: ida_idaapi.ea_t, len: int, type: int, flags: int = 0) -> Any:
    r"""Get contents of string literal, as UTF-8-encoded codepoints.
    It works even if the string has not been created in the database yet.
    
    Note that the returned value will be of type 'bytes'; if
    you want auto-conversion to unicode strings (that is: real Python
    strings), you should probably be using the idautils.Strings class.
    
    :param ea: linear address of the string
    :param len: length of the string in bytes (including terminating 0)
    :param type: type of the string. Represents both the character encoding,
                 <u>and</u> the 'type' of string at the given location.
    :param flags: combination of STRCONV_..., to perform output conversion.
    :returns: a bytes-filled str object.
    """
    ...

def get_stroff_path(args: Any) -> Any:
    r"""Get the structure offset path for operand `n`, at the
    specified address.
    
    This function has the following signatures:
    
        1. get_stroff_path(ea: ida_idaapi.ea_t, n : int) -> Tuple[List[int], int]
        2. get_stroff_path(path: tid_array, delta: sval_pointer, ea: ida_idaapi.ea_t, n : int) (backward-compatibility only)
    
    :param ea: address where the operand holds a path to a structure offset (1st form)
    :param n: operand number (1st form)
    :returns: a tuple holding a (list_of_tid_t's, delta_within_the_last_type), or (None, None)
    """
    ...

def get_wide_byte(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one wide byte of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. 
            
    """
    ...

def get_wide_dword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get two wide words (4 'bytes') of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_wide_word(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one wide word (2 'byte') of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_word(ea: ida_idaapi.ea_t) -> ushort:
    r"""Get one word (16-bit) of the program at 'ea'. This function takes into account order of bytes specified in idainfo::is_be() This function works only for 8bit byte processors. 
            
    """
    ...

def get_zero_ranges(zranges: rangeset_t, range: range_t) -> bool:
    r"""Return set of ranges with zero initialized bytes. The returned set includes only big zero initialized ranges (at least >1KB). Some zero initialized byte ranges may be not included. Only zero bytes that use the sparse storage method (STT_MM) are reported. 
            
    :param zranges: pointer to the return value. cannot be nullptr
    :param range: the range of addresses to verify. can be nullptr - means all ranges
    :returns: true if the result is a non-empty set
    """
    ...

def getn_hidden_range(n: int) -> hidden_range_t:
    r"""Get pointer to hidden range structure, in: number of hidden range. 
            
    :param n: number of hidden range, is in range 0..get_hidden_range_qty()-1
    """
    ...

def has_any_name(F: flags64_t) -> bool:
    r"""Does the current byte have any name?
    
    """
    ...

def has_auto_name(F: flags64_t) -> bool:
    r"""Does the current byte have auto-generated (no special prefix) name?
    
    """
    ...

def has_cmt(F: flags64_t) -> bool:
    r"""Does the current byte have an indented comment?
    
    """
    ...

def has_dummy_name(F: flags64_t) -> bool:
    r"""Does the current byte have dummy (auto-generated, with special prefix) name?
    
    """
    ...

def has_extra_cmts(F: flags64_t) -> bool:
    r"""Does the current byte have additional anterior or posterior lines?
    
    """
    ...

def has_immd(F: flags64_t) -> bool:
    r"""Has immediate value?
    
    """
    ...

def has_name(F: flags64_t) -> bool:
    r"""Does the current byte have non-trivial (non-dummy) name?
    
    """
    ...

def has_user_name(F: flags64_t) -> bool:
    r"""Does the current byte have user-specified name?
    
    """
    ...

def has_value(F: flags64_t) -> bool:
    r"""Do flags contain byte value?
    
    """
    ...

def has_xref(F: flags64_t) -> bool:
    r"""Does the current byte have cross-references to it?
    
    """
    ...

def hex_flag() -> flags64_t:
    r"""Get number flag of the base, regardless of current processor - better to use num_flag()
    
    """
    ...

def is_align(F: flags64_t) -> bool:
    r"""FF_ALIGN
    
    """
    ...

def is_attached_custom_data_format(dtid: int, dfid: int) -> bool:
    r"""Is the custom data format attached to the custom data type? 
            
    :param dtid: data type id
    :param dfid: data format id
    :returns: true or false
    """
    ...

def is_bnot(ea: ida_idaapi.ea_t, F: flags64_t, n: int) -> bool:
    r"""Should we negate the operand?. asm_t::a_bnot should be defined in the idp module in order to work with this function 
            
    """
    ...

def is_byte(F: flags64_t) -> bool:
    r"""FF_BYTE
    
    """
    ...

def is_char(F: flags64_t, n: int) -> bool:
    r"""is character constant?
    
    """
    ...

def is_char0(F: flags64_t) -> bool:
    r"""Is the first operand character constant? (example: push 'a')
    
    """
    ...

def is_char1(F: flags64_t) -> bool:
    r"""Is the second operand character constant? (example: mov al, 'a')
    
    """
    ...

def is_code(F: flags64_t) -> bool:
    r"""Does flag denote start of an instruction?
    
    """
    ...

def is_custfmt(F: flags64_t, n: int) -> bool:
    r"""is custom data format?
    
    """
    ...

def is_custfmt0(F: flags64_t) -> bool:
    r"""Does the first operand use a custom data representation?
    
    """
    ...

def is_custfmt1(F: flags64_t) -> bool:
    r"""Does the second operand use a custom data representation?
    
    """
    ...

def is_custom(F: flags64_t) -> bool:
    r"""FF_CUSTOM
    
    """
    ...

def is_data(F: flags64_t) -> bool:
    r"""Does flag denote start of data?
    
    """
    ...

def is_defarg(F: flags64_t, n: int) -> bool:
    r"""is defined?
    
    """
    ...

def is_defarg0(F: flags64_t) -> bool:
    r"""Is the first operand defined? Initially operand has no defined representation.
    
    """
    ...

def is_defarg1(F: flags64_t) -> bool:
    r"""Is the second operand defined? Initially operand has no defined representation.
    
    """
    ...

def is_double(F: flags64_t) -> bool:
    r"""FF_DOUBLE
    
    """
    ...

def is_dword(F: flags64_t) -> bool:
    r"""FF_DWORD
    
    """
    ...

def is_enum(F: flags64_t, n: int) -> bool:
    r"""is enum?
    
    """
    ...

def is_enum0(F: flags64_t) -> bool:
    r"""Is the first operand a symbolic constant (enum member)?
    
    """
    ...

def is_enum1(F: flags64_t) -> bool:
    r"""Is the second operand a symbolic constant (enum member)?
    
    """
    ...

def is_flag_for_operand(F: flags64_t, typebits: uint8, n: int) -> bool:
    r"""Check that the 64-bit flags set has the expected type for operand `n`.
    
    :param F: the flags
    :param typebits: the type bits (one of `FF_N_`)
    :param n: the operand number
    :returns: success
    """
    ...

def is_float(F: flags64_t) -> bool:
    r"""FF_FLOAT
    
    """
    ...

def is_float0(F: flags64_t) -> bool:
    r"""Is the first operand a floating point number?
    
    """
    ...

def is_float1(F: flags64_t) -> bool:
    r"""Is the second operand a floating point number?
    
    """
    ...

def is_flow(F: flags64_t) -> bool:
    r"""Does the previous instruction exist and pass execution flow to the current byte?
    
    """
    ...

def is_fltnum(F: flags64_t, n: int) -> bool:
    r"""is floating point number?
    
    """
    ...

def is_forced_operand(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Is operand manually defined?. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number
    """
    ...

def is_func(F: flags64_t) -> bool:
    r"""Is function start?
    
    """
    ...

def is_head(F: flags64_t) -> bool:
    r"""Does flag denote start of instruction OR data?
    
    """
    ...

def is_invsign(ea: ida_idaapi.ea_t, F: flags64_t, n: int) -> bool:
    r"""Should sign of n-th operand inverted during output?. allowed values of n: 0-first operand, 1-other operands 
            
    """
    ...

def is_loaded(ea: ida_idaapi.ea_t) -> bool:
    r"""Does the specified address have a byte value (is initialized?)
    
    """
    ...

def is_lzero(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Display leading zeroes? Display leading zeroes in operands. The global switch for the leading zeroes is in idainfo::s_genflags Note: the leading zeroes doesn't work if for the target assembler octal numbers start with 0. 
            
    :param ea: the item (insn/data) address
    :param n: the operand number (0-first operand, 1-other operands)
    :returns: success
    """
    ...

def is_manual(F: flags64_t, n: int) -> bool:
    r"""is forced operand? (use is_forced_operand())
    
    """
    ...

def is_manual_insn(ea: ida_idaapi.ea_t) -> bool:
    r"""Is the instruction overridden? 
            
    :param ea: linear address of the instruction or data item
    """
    ...

def is_mapped(ea: ida_idaapi.ea_t) -> bool:
    r"""Is the specified address 'ea' present in the program?
    
    """
    ...

def is_not_tail(F: flags64_t) -> bool:
    r"""Does flag denote tail byte?
    
    """
    ...

def is_numop(F: flags64_t, n: int) -> bool:
    r"""is number (bin, oct, dec, hex)?
    
    """
    ...

def is_numop0(F: flags64_t) -> bool:
    r"""Is the first operand a number (i.e. binary, octal, decimal or hex?)
    
    """
    ...

def is_numop1(F: flags64_t) -> bool:
    r"""Is the second operand a number (i.e. binary, octal, decimal or hex?)
    
    """
    ...

def is_off(F: flags64_t, n: int) -> bool:
    r"""is offset?
    
    """
    ...

def is_off0(F: flags64_t) -> bool:
    r"""Is the first operand offset? (example: push offset xxx)
    
    """
    ...

def is_off1(F: flags64_t) -> bool:
    r"""Is the second operand offset? (example: mov ax, offset xxx)
    
    """
    ...

def is_oword(F: flags64_t) -> bool:
    r"""FF_OWORD
    
    """
    ...

def is_pack_real(F: flags64_t) -> bool:
    r"""FF_PACKREAL
    
    """
    ...

def is_qword(F: flags64_t) -> bool:
    r"""FF_QWORD
    
    """
    ...

def is_same_data_type(F1: flags64_t, F2: flags64_t) -> bool:
    r"""Do the given flags specify the same data type?
    
    """
    ...

def is_seg(F: flags64_t, n: int) -> bool:
    r"""is segment?
    
    """
    ...

def is_seg0(F: flags64_t) -> bool:
    r"""Is the first operand segment selector? (example: push seg seg001)
    
    """
    ...

def is_seg1(F: flags64_t) -> bool:
    r"""Is the second operand segment selector? (example: mov dx, seg dseg)
    
    """
    ...

def is_stkvar(F: flags64_t, n: int) -> bool:
    r"""is stack variable?
    
    """
    ...

def is_stkvar0(F: flags64_t) -> bool:
    r"""Is the first operand a stack variable?
    
    """
    ...

def is_stkvar1(F: flags64_t) -> bool:
    r"""Is the second operand a stack variable?
    
    """
    ...

def is_strlit(F: flags64_t) -> bool:
    r"""FF_STRLIT
    
    """
    ...

def is_stroff(F: flags64_t, n: int) -> bool:
    r"""is struct offset?
    
    """
    ...

def is_stroff0(F: flags64_t) -> bool:
    r"""Is the first operand an offset within a struct?
    
    """
    ...

def is_stroff1(F: flags64_t) -> bool:
    r"""Is the second operand an offset within a struct?
    
    """
    ...

def is_struct(F: flags64_t) -> bool:
    r"""FF_STRUCT
    
    """
    ...

def is_suspop(ea: ida_idaapi.ea_t, F: flags64_t, n: int) -> bool:
    r"""is suspicious operand?
    
    """
    ...

def is_tail(F: flags64_t) -> bool:
    r"""Does flag denote tail byte?
    
    """
    ...

def is_tbyte(F: flags64_t) -> bool:
    r"""FF_TBYTE
    
    """
    ...

def is_unknown(F: flags64_t) -> bool:
    r"""Does flag denote unexplored byte?
    
    """
    ...

def is_varsize_item(ea: ida_idaapi.ea_t, F: flags64_t, ti: opinfo_t = None, itemsize: asize_t = None) -> int:
    r"""Is the item at 'ea' variable size?. 
            
    :param ea: linear address of the item
    :param F: flags
    :param ti: additional information about the data type. For example, if the current item is a structure instance, then ti->tid is structure id. Otherwise is ignored (may be nullptr). If specified as nullptr, will be automatically retrieved from the database
    :param itemsize: if not nullptr and the item is varsize, itemsize will contain the calculated item size (for struct types, the minimal size is returned)
    :returns: 1: varsize item
    :returns: 0: fixed item
    :returns: -1: error (bad data definition)
    """
    ...

def is_word(F: flags64_t) -> bool:
    r"""FF_WORD
    
    """
    ...

def is_yword(F: flags64_t) -> bool:
    r"""FF_YWORD
    
    """
    ...

def is_zword(F: flags64_t) -> bool:
    r"""FF_ZWORD
    
    """
    ...

def leading_zero_important(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Check if leading zeroes are important.
    
    """
    ...

def nbits(ea: ida_idaapi.ea_t) -> int:
    r"""Get number of bits in a byte at the given address. 
            
    :returns: processor_t::dnbits() if the address doesn't belong to a segment, otherwise the result depends on the segment type
    """
    ...

def next_addr(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next address in the program (i.e. next address which has flags). 
            
    :returns: BADADDR if no such address exist.
    """
    ...

def next_chunk(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the first address of next contiguous chunk in the program. 
            
    :returns: BADADDR if next chunk doesn't exist.
    """
    ...

def next_head(ea: ida_idaapi.ea_t, maxea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get start of next defined item. 
            
    :param ea: begin search at this address
    :param maxea: not included in the search range
    :returns: BADADDR if none exists.
    """
    ...

def next_inited(ea: ida_idaapi.ea_t, maxea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Find the next initialized address.
    
    """
    ...

def next_not_tail(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of next non-tail byte. 
            
    :returns: BADADDR if none exists.
    """
    ...

def next_that(ea: ida_idaapi.ea_t, maxea: ida_idaapi.ea_t, testf: testf_t) -> ida_idaapi.ea_t:
    r"""Find next address with a flag satisfying the function 'testf'. 
            
    :param ea: start searching at this address + 1
    :param maxea: not included in the search range.
    :param testf: test function to find next address
    :returns: the found address or BADADDR.
    """
    ...

def next_unknown(ea: ida_idaapi.ea_t, maxea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Similar to next_that(), but will find the next address that is unexplored.
    
    """
    ...

def next_visea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next visible address. 
            
    :returns: BADADDR if none exists.
    """
    ...

def num_flag() -> flags64_t:
    r"""Get number of default base (bin, oct, dec, hex) 
            
    """
    ...

def oct_flag() -> flags64_t:
    r"""Get number flag of the base, regardless of current processor - better to use num_flag()
    
    """
    ...

def off_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def op_adds_xrefs(F: flags64_t, n: int) -> bool:
    r"""Should processor module create xrefs from the operand?. Currently 'offset', 'structure offset', 'stack' and 'enum' operands create xrefs 
            
    """
    ...

def op_based_stroff(insn: insn_t, n: int, opval: adiff_t, base: ida_idaapi.ea_t) -> bool:
    r"""Set operand representation to be 'struct offset' if the operand likely points to a structure member. For example, let's there is a structure at 1000 1000 stru_1000 Elf32_Sym <...> the operand #8 will be represented as '#Elf32_Sym.st_size' after the call of 'op_based_stroff(..., 8, 0x1000)' By the way, after the call of 'op_plain_offset(..., 0x1000)' it will be represented as '#(stru_1000.st_size - 0x1000)' 
            
    :param insn: the instruction
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :param opval: operand value (usually op_t::value or op_t::addr)
    :param base: base reference
    :returns: success
    """
    ...

def op_bin(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to bin_flag()
    
    """
    ...

def op_chr(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to char_flag()
    
    """
    ...

def op_custfmt(ea: ida_idaapi.ea_t, n: int, fid: int) -> bool:
    r"""Set custom data format for operand (fid-custom data format id)
    
    """
    ...

def op_dec(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to dec_flag()
    
    """
    ...

def op_enum(ea: ida_idaapi.ea_t, n: int, id: tid_t, serial: uchar = 0) -> bool:
    r"""Set operand representation to be enum type If applied to unexplored bytes, converts them to 16/32bit word data 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :param id: id of enum
    :param serial: the serial number of the constant in the enumeration, usually 0. the serial numbers are used if the enumeration contains several constants with the same value
    :returns: success
    """
    ...

def op_flt(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to flt_flag()
    
    """
    ...

def op_hex(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to hex_flag()
    
    """
    ...

def op_num(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to num_flag()
    
    """
    ...

def op_oct(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to oct_flag()
    
    """
    ...

def op_seg(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Set operand representation to be 'segment'. If applied to unexplored bytes, converts them to 16/32bit word data 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: success
    """
    ...

def op_stkvar(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Set operand representation to be 'stack variable'. Should be applied to an instruction within a function. Should be applied after creating a stack var using insn_t::create_stkvar(). 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: success
    """
    ...

def op_stroff(args: Any) -> bool:
    r"""Set operand representation to be 'struct offset'.
    
    This function has the following signatures:
    
        1. op_stroff(ins: ida_ua.insn_t, n: int, path: List[int], delta: int)
        2. op_stroff(ins: ida_ua.insn_t, n: int, path: ida_pro.tid_array, path_len: int, delta: int) (backward-compatibility only)
    
    Here is an example using this function:
    
        ins = ida_ua.insn_t()
        if ida_ua.decode_insn(ins, some_address):
            operand = 0
            path = [ida_typeinf.get_named_type_tid("my_stucture_t")] # a one-element path
            ida_bytes.op_stroff(ins, operand, path, 0)
    """
    ...

def oword_flag() -> flags64_t:
    r"""Get a flags64_t representing a octaword.
    
    """
    ...

def packreal_flag() -> flags64_t:
    r"""Get a flags64_t representing a packed decimal real.
    
    """
    ...

def parse_binpat_str(out: compiled_binpat_vec_t, ea: ida_idaapi.ea_t, _in: str, radix: int, strlits_encoding: int = 0) -> bool:
    r"""Deprecated.
    
    Please use compiled_binpat_vec_t.from_pattern() instead.
    """
    ...

def patch_byte(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a byte of the program. The original value of the byte is saved and can be obtained by get_original_byte(). This function works for wide byte processors too. 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_bytes(ea: ida_idaapi.ea_t, buf: void) -> None:
    r"""Patch the specified number of bytes of the program. Original values of bytes are saved and are available with get_original...() functions. See also put_bytes(). 
            
    :param ea: linear address
    :param buf: buffer with new values of bytes
    """
    ...

def patch_dword(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a dword of the program. The original value of the dword is saved and can be obtained by get_original_dword(). This function DOESN'T work for wide byte processors. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_qword(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a qword of the program. The original value of the qword is saved and can be obtained by get_original_qword(). This function DOESN'T work for wide byte processors. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_word(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a word of the program. The original value of the word is saved and can be obtained by get_original_word(). This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def prev_addr(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get previous address in the program. 
            
    :returns: BADADDR if no such address exist.
    """
    ...

def prev_chunk(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the last address of previous contiguous chunk in the program. 
            
    :returns: BADADDR if previous chunk doesn't exist.
    """
    ...

def prev_head(ea: ida_idaapi.ea_t, minea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get start of previous defined item. 
            
    :param ea: begin search at this address
    :param minea: included in the search range
    :returns: BADADDR if none exists.
    """
    ...

def prev_inited(ea: ida_idaapi.ea_t, minea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Find the previous initialized address.
    
    """
    ...

def prev_not_tail(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of previous non-tail byte. 
            
    :returns: BADADDR if none exists.
    """
    ...

def prev_that(ea: ida_idaapi.ea_t, minea: ida_idaapi.ea_t, testf: testf_t) -> ida_idaapi.ea_t:
    r"""Find previous address with a flag satisfying the function 'testf'. 
            
    :param ea: start searching from this address - 1.
    :param minea: included in the search range.
    :param testf: test function to find previous address
    :returns: the found address or BADADDR.
    """
    ...

def prev_unknown(ea: ida_idaapi.ea_t, minea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Similar to prev_that(), but will find the previous address that is unexplored.
    
    """
    ...

def prev_visea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get previous visible address. 
            
    :returns: BADADDR if none exists.
    """
    ...

def print_strlit_type(strtype: int, flags: int = 0) -> Any:
    r"""Get string type information: the string type name (possibly decorated with hotkey markers), and the tooltip.
    
    :param strtype: the string type
    :param flags: or'ed PSTF_* constants
    :returns: length of generated text
    """
    ...

def put_byte(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Set value of one byte of the program. This function modifies the database. If the debugger is active then the debugged process memory is patched too. 
            
    :param ea: linear address
    :param x: byte value
    :returns: true if the database has been modified
    """
    ...

def put_bytes(ea: ida_idaapi.ea_t, buf: void) -> None:
    r"""Modify the specified number of bytes of the program. This function does not save the original values of bytes. See also patch_bytes(). 
            
    :param ea: linear address
    :param buf: buffer with new values of bytes
    """
    ...

def put_dword(ea: ida_idaapi.ea_t, x: uint64) -> None:
    r"""Set value of one dword of the program. This function takes into account order of bytes specified in idainfo::is_be() This function works for wide byte processors too. 
            
    :param ea: linear address
    :param x: dword value
    """
    ...

def put_qword(ea: ida_idaapi.ea_t, x: uint64) -> None:
    r"""Set value of one qword (8 bytes) of the program. This function takes into account order of bytes specified in idainfo::is_be() This function DOESN'T works for wide byte processors. 
            
    :param ea: linear address
    :param x: qword value
    """
    ...

def put_word(ea: ida_idaapi.ea_t, x: uint64) -> None:
    r"""Set value of one word of the program. This function takes into account order of bytes specified in idainfo::is_be() This function works for wide byte processors too. 
            
    """
    ...

def qword_flag() -> flags64_t:
    r"""Get a flags64_t representing a quad word.
    
    """
    ...

def register_custom_data_format(df: Any) -> Any:
    r"""Registers a custom data format with a given data type.
    
    :param df: an instance of data_format_t
    :returns: < 0 if failed to register
    :returns: > 0 data format id
    """
    ...

def register_custom_data_type(dt: Any) -> Any:
    r"""Registers a custom data type.
    
    :param dt: an instance of the data_type_t class
    :returns: < 0 if failed to register
    :returns: > 0 data type id
    """
    ...

def register_data_types_and_formats(formats: Any) -> Any:
    r"""
        Registers multiple data types and formats at once.
        To register one type/format at a time use register_custom_data_type/register_custom_data_format
    
        It employs a special table of types and formats described below:
    
        The 'formats' is a list of tuples. If a tuple has one element then it is the format to be registered with dtid=0
        If the tuple has more than one element, then tuple[0] is the data type and tuple[1:] are the data formats. For example:
        many_formats = [
          (pascal_data_type(), pascal_data_format()),
          (simplevm_data_type(), simplevm_data_format()),
          (makedword_data_format(),),
          (simplevm_data_format(),)
        ]
        The first two tuples describe data types and their associated formats.
        The last two tuples describe two data formats to be used with built-in data types.
        The data format may be attached to several data types. The id of the
        data format is stored in the first data_format_t object. For example:
        assert many_formats[1][1] != -1
        assert many_formats[2][0] != -1
        assert many_formats[3][0] == -1
        
    """
    ...

def revert_byte(ea: ida_idaapi.ea_t) -> bool:
    r"""Revert patched byte 
            
    :returns: true: byte was patched before and reverted now
    """
    ...

def seg_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def set_cmt(ea: ida_idaapi.ea_t, comm: str, rptble: bool) -> bool:
    r"""Set an indented comment. 
            
    :param ea: linear address
    :param comm: comment string
    * nullptr: do nothing (return 0)
    * "" : delete comment
    :param rptble: is repeatable?
    :returns: success
    """
    ...

def set_forced_operand(ea: ida_idaapi.ea_t, n: int, op: str) -> bool:
    r"""Set forced operand. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number
    :param op: text of operand
    * nullptr: do nothing (return 0)
    * "" : delete forced operand
    :returns: success
    """
    ...

def set_immd(ea: ida_idaapi.ea_t) -> bool:
    r"""Set 'has immediate operand' flag. Returns true if the FF_IMMD bit was not set and now is set 
            
    """
    ...

def set_lzero(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Set toggle lzero bit. This function changes the display of leading zeroes for the specified operand. If the default is not to display leading zeroes, this function will display them and vice versa. 
            
    :param ea: the item (insn/data) address
    :param n: the operand number (0-first operand, 1-other operands)
    :returns: success
    """
    ...

def set_manual_insn(ea: ida_idaapi.ea_t, manual_insn: str) -> None:
    r"""Set manual instruction string. 
            
    :param ea: linear address of the instruction or data item
    :param manual_insn: "" - delete manual string. nullptr - do nothing
    """
    ...

def set_op_type(ea: ida_idaapi.ea_t, type: flags64_t, n: int) -> bool:
    r"""(internal function) change representation of operand(s). 
            
    :param ea: linear address
    :param type: new flag value (should be obtained from char_flag(), num_flag() and similar functions)
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: 1: ok
    :returns: 0: failed (applied to a tail byte)
    """
    ...

def set_opinfo(ea: ida_idaapi.ea_t, n: int, flag: flags64_t, ti: opinfo_t, suppress_events: bool = False) -> bool:
    r"""Set additional information about an operand representation. This function is a low level one. Only the kernel should use it. 
            
    :param ea: linear address of the item
    :param n: number of operand, 0 or 1 (see the note below)
    :param flag: flags of the item
    :param ti: additional representation information
    :param suppress_events: do not generate changing_op_type and op_type_changed events
    :returns: success
    """
    ...

def stkvar_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def strlit_flag() -> flags64_t:
    r"""Get a flags64_t representing a string literal.
    
    """
    ...

def stroff_flag() -> flags64_t:
    r"""see FF_opbits
    
    """
    ...

def stru_flag() -> flags64_t:
    r"""Get a flags64_t representing a struct.
    
    """
    ...

def tbyte_flag() -> flags64_t:
    r"""Get a flags64_t representing a tbyte.
    
    """
    ...

def toggle_bnot(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Toggle binary negation of operand. also see is_bnot()
    
    """
    ...

def toggle_lzero(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Toggle lzero bit. 
            
    :param ea: the item (insn/data) address
    :param n: the operand number (0-first operand, 1-other operands)
    :returns: success
    """
    ...

def toggle_sign(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Toggle sign of n-th operand. allowed values of n: 0-first operand, 1-other operands 
            
    """
    ...

def unregister_custom_data_format(dfid: Any) -> Any:
    r"""Unregisters a custom data format
    
    :param dfid: data format id
    :returns: Boolean
    """
    ...

def unregister_custom_data_type(dtid: Any) -> Any:
    r"""Unregisters a custom data type.
    
    :param dtid: the data type id
    :returns: Boolean
    """
    ...

def unregister_data_types_and_formats(formats: Any) -> Any:
    r"""
        As opposed to register_data_types_and_formats(), this function
        unregisters multiple data types and formats at once.
        
    """
    ...

def update_hidden_range(ha: hidden_range_t) -> bool:
    r"""Update hidden range information in the database. You cannot use this function to change the range boundaries 
            
    :param ha: range to update
    :returns: success
    """
    ...

def use_mapping(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Translate address according to current mappings. 
            
    :param ea: address to translate
    :returns: translated address
    """
    ...

def visit_patched_bytes(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, callable: Any) -> Any:
    r"""Enumerates patched bytes in the given range and invokes a callable
    
    :param ea1: start address
    :param ea2: end address
    :param callable: a Python callable with the following prototype:
                     callable(ea, fpos, org_val, patch_val).
                     If the callable returns non-zero then that value will be
                     returned to the caller and the enumeration will be
                     interrupted.
    :returns: Zero if the enumeration was successful or the return
             value of the callback if enumeration was interrupted.
    """
    ...

def word_flag() -> flags64_t:
    r"""Get a flags64_t representing a word.
    
    """
    ...

def yword_flag() -> flags64_t:
    r"""Get a flags64_t representing a ymm word.
    
    """
    ...

def zword_flag() -> flags64_t:
    r"""Get a flags64_t representing a zmm word.
    
    """
    ...

ALOPT_APPEND: int  # 32
ALOPT_IGNCLT: int  # 4
ALOPT_IGNHEADS: int  # 1
ALOPT_IGNPRINT: int  # 2
ALOPT_MAX4K: int  # 8
ALOPT_ONLYTERM: int  # 16
BIN_SEARCH_BACKWARD: int  # 16
BIN_SEARCH_BITMASK: int  # 32
BIN_SEARCH_CASE: int  # 1
BIN_SEARCH_FORWARD: int  # 0
BIN_SEARCH_INITED: int  # 4
BIN_SEARCH_NOBREAK: int  # 2
BIN_SEARCH_NOCASE: int  # 0
BIN_SEARCH_NOSHOW: int  # 8
DELIT_DELNAMES: int  # 2
DELIT_EXPAND: int  # 1
DELIT_KEEPFUNC: int  # 32
DELIT_NOCMT: int  # 16
DELIT_NOTRUNC: int  # 4
DELIT_NOUNAME: int  # 8
DELIT_SIMPLE: int  # 0
DTP_NODUP: int  # 1
DT_TYPE: int  # -268435456
FF_0CHAR: int  # 3145728
FF_0CUST: int  # 13631488
FF_0ENUM: int  # 8388608
FF_0FLT: int  # 12582912
FF_0FOP: int  # 9437184
FF_0NUMB: int  # 6291456
FF_0NUMD: int  # 2097152
FF_0NUMH: int  # 1048576
FF_0NUMO: int  # 7340032
FF_0OFF: int  # 5242880
FF_0SEG: int  # 4194304
FF_0STK: int  # 11534336
FF_0STRO: int  # 10485760
FF_0VOID: int  # 0
FF_1CHAR: int  # 50331648
FF_1CUST: int  # 218103808
FF_1ENUM: int  # 134217728
FF_1FLT: int  # 201326592
FF_1FOP: int  # 150994944
FF_1NUMB: int  # 100663296
FF_1NUMD: int  # 33554432
FF_1NUMH: int  # 16777216
FF_1NUMO: int  # 117440512
FF_1OFF: int  # 83886080
FF_1SEG: int  # 67108864
FF_1STK: int  # 184549376
FF_1STRO: int  # 167772160
FF_1VOID: int  # 0
FF_ALIGN: int  # -1342177280
FF_ANYNAME: int  # 49152
FF_BNOT: int  # 262144
FF_BYTE: int  # 0
FF_CODE: int  # 1536
FF_COMM: int  # 2048
FF_CUSTOM: int  # -805306368
FF_DATA: int  # 1024
FF_DOUBLE: int  # -1879048192
FF_DWORD: int  # 536870912
FF_FLOAT: int  # -2147483648
FF_FLOW: int  # 65536
FF_FUNC: int  # 268435456
FF_IMMD: int  # 1073741824
FF_IVL: int  # 256
FF_JUMP: int  # -2147483648
FF_LABL: int  # 32768
FF_LINE: int  # 8192
FF_NAME: int  # 16384
FF_N_CHAR: int  # 3
FF_N_CUST: int  # 13
FF_N_ENUM: int  # 8
FF_N_FLT: int  # 12
FF_N_FOP: int  # 9
FF_N_NUMB: int  # 6
FF_N_NUMD: int  # 2
FF_N_NUMH: int  # 1
FF_N_NUMO: int  # 7
FF_N_OFF: int  # 5
FF_N_SEG: int  # 4
FF_N_STK: int  # 11
FF_N_STRO: int  # 10
FF_N_VOID: int  # 0
FF_OWORD: int  # 1879048192
FF_PACKREAL: int  # -1610612736
FF_QWORD: int  # 805306368
FF_REF: int  # 4096
FF_SIGN: int  # 131072
FF_STRLIT: int  # 1342177280
FF_STRUCT: int  # 1610612736
FF_TAIL: int  # 512
FF_TBYTE: int  # 1073741824
FF_UNK: int  # 0
FF_UNUSED: int  # 524288
FF_WORD: int  # 268435456
FF_YWORD: int  # -536870912
FF_ZWORD: int  # -268435456
GFE_32BIT: int  # 4
GFE_IDB_VALUE: int  # 2
GFE_VALUE: int  # 1
GMB_READALL: int  # 1
GMB_WAITBOX: int  # 2
ITEM_END_CANCEL: int  # 16
ITEM_END_FIXUP: int  # 1
ITEM_END_INITED: int  # 2
ITEM_END_NAME: int  # 4
ITEM_END_XREF: int  # 8
MS_0TYPE: int  # 15728640
MS_1TYPE: int  # 251658240
MS_CLS: int  # 1536
MS_CODE: int  # -268435456
MS_COMM: int  # 1046528
MS_N_TYPE: int  # 15
MS_VAL: int  # 255
OPND_ALL: int  # 15
OPND_MASK: int  # 15
OPND_OUTER: int  # 128
PBSENC_ALL: int  # -1
PBSENC_DEF1BPU: int  # 0
PSTF_ATTRIB: int  # 16
PSTF_ENC: int  # 8
PSTF_HOTKEY: int  # 4
PSTF_ONLY_ENC: int  # 11
PSTF_TBRIEF: int  # 1
PSTF_TINLIN: int  # 2
PSTF_TMASK: int  # 3
PSTF_TNORM: int  # 0
STRCONV_ESCAPE: int  # 1
STRCONV_INCLLEN: int  # 4
STRCONV_REPLCHAR: int  # 2
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
ida_nalt: module
ida_range: module
typing: module
weakref: module