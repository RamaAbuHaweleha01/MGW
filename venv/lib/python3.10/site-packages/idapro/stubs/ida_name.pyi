from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with names.

A non-tail address of the program may have a name. Tail addresses (i.e. the addresses in the middle of an instruction or data item) cannot have names. 
    
"""

class NearestName:
    r"""
        Utility class to help find the nearest name in a given ea/name dictionary
        
    """
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
    def __getitem__(self, index: Any) -> ea_name_t:
        r"""Returns the tupple (ea, name, index)"""
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, ea_names: Any) -> Any:
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
    def find(self, ea: Any) -> Any:
        r"""
                Returns a tupple (ea, name, pos) that is the nearest to the passed ea
                If no name is matched then None is returned
                
        """
        ...
    def update(self, ea_names: Any) -> Any:
        r"""Updates the ea/names map"""
        ...

class ea_name_t:
    @property
    def ea(self) -> Any: ...
    @property
    def name(self) -> Any: ...
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

class ea_name_vec_t:
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
    def __getitem__(self, i: size_t) -> ea_name_t:
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
    def __setitem__(self, i: size_t, v: ea_name_t) -> None:
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
    def append(self, x: ea_name_t) -> None:
        ...
    def at(self, _idx: size_t) -> ea_name_t:
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
    def extend(self, x: ea_name_vec_t) -> None:
        ...
    def extract(self) -> ea_name_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: ea_name_t, len: size_t) -> None:
        ...
    def insert(self, it: ea_name_t, x: ea_name_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> ea_name_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: ea_name_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

def append_struct_fields(disp: adiff_t, n: int, path: tid_t, flags: flags64_t, delta: adiff_t, appzero: bool) -> str:
    r"""Append names of struct fields to a name if the name is a struct name. 
            
    :param disp: displacement from the name
    :param n: operand number in which the name appears
    :param path: path in the struct. path is an array of id's. maximal length of array is MAXSTRUCPATH. the first element of the array is the structure id. consecutive elements are id's of used union members (if any).
    :param flags: the input flags. they will be returned if the struct cannot be found.
    :param delta: delta to add to displacement
    :param appzero: should append a struct field name if the displacement is zero?
    :returns: flags of the innermost struct member or the input flags
    """
    ...

def calc_gtn_flags(fromaddr: Any, ea: Any) -> Any:
    r"""
        Calculate flags for get_ea_name() function
    
        :param fromaddr: the referring address. May be BADADDR.
        :param ea: linear address
    
        :returns: flags
        
    """
    ...

def cleanup_name(ea: ida_idaapi.ea_t, name: str, flags: int = 0) -> str:
    ...

def del_debug_names(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> None:
    ...

def del_global_name(ea: ida_idaapi.ea_t) -> bool:
    ...

def del_local_name(ea: ida_idaapi.ea_t) -> bool:
    ...

def demangle_name(name: str, disable_mask: int, demreq: demreq_type_t = 2) -> str:
    r"""Demangle a name. 
            
    :param name: name to demangle
    :param disable_mask: bits to inhibit parts of demangled name (see MNG_). by the M_COMPILER bits a specific compiler can be selected (see MT_).
    :param demreq: the request type demreq_type_t
    :returns: ME_... or MT__ bitmasks from demangle.hpp
    """
    ...

def extract_name(line: str, x: int) -> str:
    r"""Extract a name or address from the specified string. 
            
    :param line: input string
    :param x: x coordinate of cursor
    :returns: -1 if cannot extract. otherwise length of the name
    """
    ...

def force_name(ea: ida_idaapi.ea_t, name: str, flags: int = 0) -> bool:
    ...

def get_colored_demangled_name(ea: ida_idaapi.ea_t, inhibitor: int, demform: int, gtn_flags: int = 0) -> str:
    ...

def get_colored_long_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    ...

def get_colored_name(ea: ida_idaapi.ea_t) -> str:
    ...

def get_colored_short_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    ...

def get_cp_validity(args: Any) -> bool:
    r"""Is the given codepoint (or range) acceptable in the given context? If 'endcp' is not BADCP, it is considered to be the end of the range: [cp, endcp), and is not included in the range 
            
    """
    ...

def get_debug_name(ea_ptr: ea_t, how: debug_name_how_t) -> str:
    ...

def get_debug_name_ea(name: str) -> ida_idaapi.ea_t:
    ...

def get_debug_names(args: Any) -> Any:
    ...

def get_demangled_name(ea: ida_idaapi.ea_t, inhibitor: int, demform: int, gtn_flags: int = 0) -> str:
    ...

def get_ea_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    r"""Get name at the specified address. 
            
    :param ea: linear address
    :param gtn_flags: how exactly the name should be retrieved. combination of bits for get_ea_name() function. There is a convenience bits
    :returns: success
    """
    ...

def get_long_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    ...

def get_mangled_name_type(name: str) -> mangled_name_type_t:
    ...

def get_name(ea: ida_idaapi.ea_t) -> str:
    ...

def get_name_base_ea(_from: ida_idaapi.ea_t, to: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of the name used in the expression for the address 
            
    :param to: the referenced address
    :returns: address of the name used to represent the operand
    """
    ...

def get_name_color(_from: ida_idaapi.ea_t, ea: ida_idaapi.ea_t) -> color_t:
    r"""Calculate flags for get_ea_name() function.
    
    Get name color. 
            
    :param ea: linear address
    """
    ...

def get_name_ea(_from: ida_idaapi.ea_t, name: str) -> ida_idaapi.ea_t:
    r"""Get the address of a name. This function resolves a name into an address. It can handle regular global and local names, as well as debugger names. 
            
    :param name: any name in the program or nullptr
    :returns: address of the name or BADADDR
    """
    ...

def get_name_expr(_from: ida_idaapi.ea_t, n: int, ea: ida_idaapi.ea_t, off: int, flags: int = 1) -> str:
    r"""Convert address to name expression (name with a displacement). This function takes into account fixup information and returns a colored name expression (in the form <name> +/- <offset>). It also knows about structure members and arrays. If the specified address doesn't have a name, a dummy name is generated. 
            
    :param n: number of referencing operand. for data items specify 0
    :param ea: address to convert to name expression
    :param off: the value of name expression. this parameter is used only to check that the name expression will have the wanted value. 'off' may be equal to BADADDR but this is discouraged because it prohibits checks.
    :param flags: Name expression flags
    :returns: < 0 if address is not valid, no segment or other failure. otherwise the length of the name expression in characters.
    """
    ...

def get_name_value(_from: ida_idaapi.ea_t, name: str) -> uval_t:
    r"""Get value of the name. This function knows about: regular names, enums, special segments, etc. 
            
    :param name: any name in the program or nullptr
    :returns: Name value result codes
    """
    ...

def get_nice_colored_name(ea: ida_idaapi.ea_t, flags: int = 0) -> str:
    r"""Get a nice colored name at the specified address. Ex:
    * segment:sub+offset
    * segment:sub:local_label
    * segment:label
    * segment:address
    * segment:address+offset
    
    
    
    :param ea: linear address
    :param flags: Nice colored name flags
    :returns: the length of the generated name in bytes.
    """
    ...

def get_nlist_ea(idx: size_t) -> ida_idaapi.ea_t:
    ...

def get_nlist_idx(ea: ida_idaapi.ea_t) -> int:
    ...

def get_nlist_name(idx: size_t) -> str:
    ...

def get_nlist_size() -> int:
    ...

def get_short_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    ...

def get_visible_name(ea: ida_idaapi.ea_t, gtn_flags: int = 0) -> str:
    ...

def hide_name(ea: ida_idaapi.ea_t) -> None:
    r"""Remove name from the list of names 
            
    :param ea: address of the name
    """
    ...

def is_ident(name: str) -> bool:
    r"""Is a valid name? (including ::MangleChars)
    
    """
    ...

def is_ident_cp(cp: wchar32_t) -> bool:
    r"""Can a character appear in a name? (present in ::NameChars or ::MangleChars)
    
    """
    ...

def is_in_nlist(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_name_defined_locally(args: Any) -> bool:
    r"""Is the name defined locally in the specified function? 
            
    :param pfn: pointer to function
    :param name: name to check
    :param ignore_name_def: which names to ignore when checking
    :param ea1: the starting address of the range inside the function (optional)
    :param ea2: the ending address of the range inside the function (optional)
    :returns: true if the name has been defined
    """
    ...

def is_public_name(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_strlit_cp(cp: wchar32_t, specific_ranges: rangeset_crefvec_t = None) -> bool:
    r"""Can a character appear in a string literal (present in ::StrlitChars) If 'specific_ranges' are specified, those will be used instead of the ones corresponding to the current culture (only if ::StrlitChars is configured to use the current culture) 
            
    """
    ...

def is_uname(name: str) -> bool:
    r"""Is valid user-specified name? (valid name & !dummy prefix). 
            
    :param name: name to test. may be nullptr.
    :returns: 1: yes
    :returns: 0: no
    """
    ...

def is_valid_cp(cp: wchar32_t, kind: nametype_t, data: void = None) -> bool:
    r"""Is the given codepoint acceptable in the given context?
    
    """
    ...

def is_valid_typename(name: str) -> bool:
    r"""Is valid type name? 
            
    :param name: name to test. may be nullptr.
    :returns: 1: yes
    :returns: 0: no
    """
    ...

def is_visible_cp(cp: wchar32_t) -> bool:
    r"""Can a character be displayed in a name? (present in ::NameChars)
    
    """
    ...

def is_weak_name(ea: ida_idaapi.ea_t) -> bool:
    ...

def make_name_auto(ea: ida_idaapi.ea_t) -> bool:
    ...

def make_name_non_public(ea: ida_idaapi.ea_t) -> None:
    ...

def make_name_non_weak(ea: ida_idaapi.ea_t) -> None:
    ...

def make_name_public(ea: ida_idaapi.ea_t) -> None:
    ...

def make_name_user(ea: ida_idaapi.ea_t) -> bool:
    ...

def make_name_weak(ea: ida_idaapi.ea_t) -> None:
    ...

def rebuild_nlist() -> None:
    ...

def reorder_dummy_names() -> None:
    r"""Renumber dummy names.
    
    """
    ...

def set_cp_validity(args: Any) -> None:
    r"""Mark the given codepoint (or range) as acceptable or unacceptable in the given context If 'endcp' is not BADCP, it is considered to be the end of the range: [cp, endcp), and is not included in the range 
            
    """
    ...

def set_debug_name(ea: ida_idaapi.ea_t, name: str) -> bool:
    ...

def set_dummy_name(_from: ida_idaapi.ea_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Give an autogenerated (dummy) name. Autogenerated names have special prefixes (loc_...). 
            
    :param ea: linear address
    :returns: 1: ok, dummy name is generated or the byte already had a name
    :returns: 0: failure, invalid address or tail byte
    """
    ...

def set_name(ea: ida_idaapi.ea_t, name: str, flags: int = 0) -> bool:
    r"""Set or delete name of an item at the specified address. An item can be anything: instruction, function, data byte, word, string, structure, etc... Include name into the list of names. 
            
    :param ea: linear address. do nothing if ea is not valid (return 0). tail bytes can't have names.
    :param name: new name.
    * nullptr: do nothing (return 0).
    * "" : delete name.
    * otherwise this is a new name.
    :param flags: Set name flags. If a bit is not specified, then the corresponding action is not performed and the name will retain the same bits as before calling this function. For new names, default is: non-public, non-weak, non-auto.
    :returns: 1: ok, name is changed
    :returns: 0: failure, a warning is displayed
    """
    ...

def show_name(ea: ida_idaapi.ea_t) -> None:
    r"""Insert name to the list of names.
    
    """
    ...

def validate_name(name: str, type: nametype_t, flags: int = 1) -> Any:
    r"""Validate a name. If SN_NOCHECK is specified, this function replaces all invalid characters in the name with SUBSTCHAR. However, it will return false if name is valid but not allowed to be an identifier (is a register name).
    
    :param name: ptr to name. the name will be modified
    :param type: the type of name we want to validate
    :param flags: see SN_*
    :returns: success
    """
    ...

CN_KEEP_TRAILING_DIGITS: int  # 1
CN_KEEP_UNDERSCORES: int  # 2
DEBNAME_EXACT: int  # 0
DEBNAME_LOWER: int  # 1
DEBNAME_NICE: int  # 3
DEBNAME_UPPER: int  # 2
DQT_COMPILER: int  # 0
DQT_FULL: int  # 2
DQT_NAME_TYPE: int  # 1
DQT_NPURGED_2: int  # -2
DQT_NPURGED_4: int  # -4
DQT_NPURGED_8: int  # -8
FUNC_IMPORT_PREFIX: str  # __imp_
GETN_APPZERO: int  # 1
GETN_NODUMMY: int  # 4
GETN_NOFIXUP: int  # 2
GNCN_NOCOLOR: int  # 2
GNCN_NODBGNM: int  # 256
GNCN_NOFUNC: int  # 8
GNCN_NOLABEL: int  # 4
GNCN_NOSEG: int  # 1
GNCN_PREFDBG: int  # 512
GNCN_REQFUNC: int  # 64
GNCN_REQNAME: int  # 128
GNCN_SEGNUM: int  # 32
GNCN_SEG_FUNC: int  # 16
GN_COLORED: int  # 2
GN_DEMANGLED: int  # 4
GN_ISRET: int  # 128
GN_LOCAL: int  # 64
GN_LONG: int  # 32
GN_NOT_DUMMY: int  # 512
GN_NOT_ISRET: int  # 256
GN_SHORT: int  # 16
GN_STRICT: int  # 8
GN_VISIBLE: int  # 1
MANGLED_CODE: int  # 0
MANGLED_DATA: int  # 1
MANGLED_UNKNOWN: int  # 2
MAXNAMELEN: int  # 512
ME_ERRAUTO: int  # -7
ME_FRAME: int  # -5
ME_ILLSTR: int  # -3
ME_INTERR: int  # -1
ME_NOCOMP: int  # -6
ME_NOERROR_LIMIT: int  # -10
ME_NOHASHMEM: int  # -8
ME_NOSTRMEM: int  # -9
ME_PARAMERR: int  # -2
ME_SMALLANS: int  # -4
MNG_CALC_VALID: int  # 1979711488
MNG_COMPILER_MSK: int  # 1879048192
MNG_DEFFAR: int  # 2
MNG_DEFHUGE: int  # 4
MNG_DEFNEAR: int  # 0
MNG_DEFNEARANY: int  # 1
MNG_DEFNONE: int  # 6
MNG_DEFPTR64: int  # 5
MNG_DROP_IMP: int  # 8388608
MNG_IGN_ANYWAY: int  # 33554432
MNG_IGN_JMP: int  # 67108864
MNG_LONG_FORM: int  # 104857607
MNG_MOVE_JMP: int  # 134217728
MNG_NOBASEDT: int  # 128
MNG_NOCALLC: int  # 256
MNG_NOCLOSUR: int  # 32768
MNG_NOCSVOL: int  # 16384
MNG_NODEFINIT: int  # 8
MNG_NOECSU: int  # 8192
MNG_NOMANAGE: int  # 131072
MNG_NOMODULE: int  # 262144
MNG_NOPOSTFC: int  # 512
MNG_NOPTRTYP: int  # 7
MNG_NOPTRTYP16: int  # 3
MNG_NORETTYPE: int  # 64
MNG_NOSCTYP: int  # 1024
MNG_NOSTVIR: int  # 4096
MNG_NOTHROW: int  # 2048
MNG_NOTYPE: int  # 32
MNG_NOUNALG: int  # 65536
MNG_NOUNDERSCORE: int  # 16
MNG_PTRMSK: int  # 7
MNG_SHORT_FORM: int  # 245612135
MNG_SHORT_S: int  # 1048576
MNG_SHORT_U: int  # 2097152
MNG_ZPT_SPACE: int  # 4194304
MT_BORLAN: int  # 536870912
MT_CASTING: int  # 4194304
MT_CDECL: int  # 2
MT_CLRCALL: int  # 11
MT_CLRCDTOR: int  # 5242880
MT_CONSTR: int  # 2097152
MT_DEFAULT: int  # 1
MT_DESTR: int  # 3145728
MT_DMDCALL: int  # 12
MT_FASTCALL: int  # 5
MT_FORTRAN: int  # 7
MT_GCC3: int  # 1610612736
MT_GNU: int  # 1342177280
MT_INTERRUPT: int  # 9
MT_LOCALNAME: int  # 15
MT_MEMBER: int  # 128
MT_MSCOMP: int  # 268435456
MT_MSFASTCALL: int  # 10
MT_OPERAT: int  # 1048576
MT_OTHER: int  # 1073741824
MT_PARMAX: int  # 255
MT_PARSHF: int  # 8
MT_PASCAL: int  # 3
MT_PRIVATE: int  # 64
MT_PROTECT: int  # 96
MT_PUBLIC: int  # 32
MT_REGCALL: int  # 14
MT_RTTI: int  # 192
MT_STDCALL: int  # 4
MT_SYSCALL: int  # 8
MT_THISCALL: int  # 6
MT_VECTORCALL: int  # 13
MT_VISAGE: int  # 1879048192
MT_VOIDARG: int  # 130816
MT_VTABLE: int  # 160
MT_WATCOM: int  # 805306368
M_ANONNSP: int  # 33554432
M_AUTOCRT: int  # 524288
M_CLASS: int  # 224
M_COMPILER: int  # 1879048192
M_DBGNAME: int  # 134217728
M_ELLIPSIS: int  # 65536
M_PARMSK: int  # 65280
M_PRCMSK: int  # 15
M_SAVEREGS: int  # 16
M_STATIC: int  # 131072
M_THUNK: int  # 16777216
M_TMPLNAM: int  # 67108864
M_TRUNCATE: int  # 8388608
M_TYPMASK: int  # 7340032
M_VIRTUAL: int  # 262144
NT_ABS: int  # 5
NT_BMASK: int  # 8
NT_BYTE: int  # 1
NT_ENUM: int  # 4
NT_LOCAL: int  # 2
NT_NONE: int  # 0
NT_REGVAR: int  # 9
NT_SEG: int  # 6
NT_STKVAR: int  # 3
NT_STROFF: int  # 7
SN_AUTO: int  # 32
SN_CHECK: int  # 0
SN_DELTAIL: int  # 8192
SN_FORCE: int  # 2048
SN_IDBENC: int  # 1024
SN_LOCAL: int  # 512
SN_MULTI: int  # 16384
SN_MULTI_FORCE: int  # 32768
SN_NOCHECK: int  # 1
SN_NODUMMY: int  # 4096
SN_NOLIST: int  # 128
SN_NON_AUTO: int  # 64
SN_NON_PUBLIC: int  # 4
SN_NON_WEAK: int  # 16
SN_NOWARN: int  # 256
SN_PUBLIC: int  # 2
SN_WEAK: int  # 8
SWIG_PYTHON_LEGACY_BOOL: int  # 1
UCDR_MANGLED: int  # 4
UCDR_NAME: int  # 2
UCDR_STRLIT: int  # 1
UCDR_TYPE: int  # 8
VNT_IDENT: int  # 6
VNT_STRLIT: int  # 1
VNT_TYPE: int  # 8
VNT_UDTMEM: int  # 2
VNT_VISIBLE: int  # 2
annotations: _Feature
bisect: module
cvar: swigvarlink
ida_idaapi: module
ignore_glabel: int  # 4
ignore_llabel: int  # 2
ignore_none: int  # 0
ignore_regvar: int  # 1
ignore_stkvar: int  # 3
weakref: module