from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""File I/O functions for IDA.

You should not use standard C file I/O functions in modules. Use functions from this header, pro.h and fpro.h instead.
This file also declares a call_system() function. 
    
"""

class choose_ioport_parser_t:
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
    def parse(self, param: str, line: str) -> bool:
        r""":returns: true: and fill PARAM with a displayed string
        :returns: false: and empty PARAM to skip the current device
        :returns: false: and fill PARAM with an error message
        """
        ...

class file_enumerator_t:
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
    def visit_file(self, file: str) -> int:
        ...

class generic_linput_t:
    @property
    def blocksize(self) -> Any: ...
    @property
    def filesize(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def read(self, off: qoff64_t, buffer: void, nbytes: size_t) -> ssize_t:
        ...

class ioports_fallback_t:
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
    def handle(self, ports: ioports_t, line: str) -> bool:
        r""":param ports: i/o port definitions
        :param line: input line to parse
        :returns: success or fills ERRBUF with an error message
        """
        ...

def choose_ioport_device2(_device: str, file: str, parse_params: choose_ioport_parser_t) -> bool:
    ...

def close_linput(li: linput_t) -> None:
    ...

def create_bytearray_linput(s: str) -> linput_t:
    ...

def create_generic_linput(gl: generic_linput_t) -> linput_t:
    ...

def create_memory_linput(start: ida_idaapi.ea_t, size: asize_t) -> linput_t:
    ...

def enumerate_files(path: Any, fname: Any, callback: Any) -> Any:
    r"""Enumerate files in the specified directory while the callback returns 0.
    
    :param path: directory to enumerate files in
    :param fname: mask of file names to enumerate
    :param callback: a callable object that takes the filename as
                     its first argument and it returns 0 to continue
                     enumeration or non-zero to stop enumeration.
    :returns: tuple(code, fname) : If the callback returns non-zero, or None in case of script errors
    """
    ...

def fopenA(file: str) -> FILE:
    ...

def fopenM(file: str) -> FILE:
    ...

def fopenRB(file: str) -> FILE:
    ...

def fopenRT(file: str) -> FILE:
    ...

def fopenWB(file: str) -> FILE:
    ...

def fopenWT(file: str) -> FILE:
    ...

def get_ida_subdirs(subdir: str, flags: int = 0) -> qstrvec_t:
    r"""Get list of directories in which to find a specific IDA resource (see IDA subdirectories). The order of the resulting list is as follows: 
         [$IDAUSR/subdir (0..N entries)]
         $IDADIR/subdir
    
    
            
    :param subdir: name of the resource to list (can be nullptr)
    :param flags: Subdirectory modification flags bits
    :returns: number of directories appended to 'dirs'
    """
    ...

def get_linput_type(li: linput_t) -> linput_type_t:
    ...

def get_special_folder(csidl: int) -> str:
    r"""Get a folder location by CSIDL (see Common CSIDLs). Path should be of at least MAX_PATH size 
            
    """
    ...

def get_user_idadir() -> str:
    r"""Get user ida related directory. 
         if $IDAUSR is defined:
            - the first element in $IDAUSR
         else
            - default user directory ($HOME/.idapro or %APPDATA%Hex-Rays/IDA Pro)
    
    
            
    """
    ...

def getsysfile(filename: str, subdir: str) -> str:
    r"""Search for IDA system file. This function searches for a file in:
    0. each directory specified by IDAUSR%
    1. ida directory [+ subdir]
    
    
    and returns the first match. 
            
    :param filename: name of file to search
    :param subdir: if specified, the file is looked for in the specified subdirectory of the ida directory first (see IDA subdirectories)
    :returns: nullptr if not found, otherwise a pointer to full file name.
    """
    ...

def idadir(subdir: str) -> str:
    r"""Get IDA directory (if subdir==nullptr) or the specified subdirectory (see IDA subdirectories) 
            
    """
    ...

def open_linput(file: str, remote: bool) -> linput_t:
    ...

def qlgetz(li: linput_t, fpos: int64) -> str:
    ...

def read_ioports(ports: ioports_t, device: str, file: str, callback: ioports_fallback_t = None) -> ssize_t:
    ...

CFG_SUBDIR: str  # cfg
CSIDL_APPDATA: int  # 26
CSIDL_LOCAL_APPDATA: int  # 28
CSIDL_PROGRAM_FILES: int  # 38
CSIDL_PROGRAM_FILESX86: int  # 42
CSIDL_PROGRAM_FILES_COMMON: int  # 43
IDA_SUBDIR_IDADIR_FIRST: int  # 2
IDA_SUBDIR_IDP: int  # 1
IDA_SUBDIR_ONLY_EXISTING: int  # 4
IDC_SUBDIR: str  # idc
IDP_SUBDIR: str  # procs
IDS_SUBDIR: str  # ids
LDR_SUBDIR: str  # loaders
LINPUT_GENERIC: int  # 4
LINPUT_LOCAL: int  # 1
LINPUT_NONE: int  # 0
LINPUT_PROCMEM: int  # 3
LINPUT_RFILE: int  # 2
LOC_CLOSE: int  # 0
LOC_KEEP: int  # 2
LOC_UNMAKE: int  # 1
PLG_SUBDIR: str  # plugins
SIG_SUBDIR: str  # sig
SWIG_PYTHON_LEGACY_BOOL: int  # 1
THM_SUBDIR: str  # themes
TIL_SUBDIR: str  # til
VAULT_CACHE_FNAME: str  # .vault_cache
VAULT_CACHE_SUBDIR: str  # .vault
annotations: _Feature
ida_idaapi: module
weakref: module