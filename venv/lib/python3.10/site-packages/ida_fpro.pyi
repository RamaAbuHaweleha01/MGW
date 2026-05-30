from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""System independent counterparts of FILE* related functions from Clib.

You should not use C standard I/O functions in your modules. The reason: Each module compiled with Borland (and statically linked to Borland's library) will host a copy of the FILE * information.
So, if you open a file in the plugin and pass the handle to the kernel, the kernel will not be able to use it.
If you really need to use the standard functions, define USE_STANDARD_FILE_FUNCTIONS. In this case do not mix them with q... functions. 
    
"""

class qfile_t:
    r"""A helper class to work with FILE related functions."""
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
    def close(self) -> Any:
        r"""Closes the file"""
        ...
    def filename(self) -> Any:
        ...
    def flush(self) -> Any:
        ...
    def from_capsule(self, pycapsule: Any) -> qfile_t:
        ...
    def from_fp(self, fp: FILE) -> qfile_t:
        ...
    def get_byte(self) -> Any:
        r"""Reads a single byte from the file. Returns None if EOF or the read byte"""
        ...
    def get_fp(self) -> FILE:
        ...
    def gets(self, len: Any) -> Any:
        r"""Reads a line from the input file. Returns the read line or None
        
        :param len: the maximum line length
        """
        ...
    def open(self, filename: Any, mode: Any) -> Any:
        r"""Opens a file
        
        :param filename: the file name
        :param mode: The mode string, ala fopen() style
        :returns: Boolean
        """
        ...
    def opened(self) -> Any:
        r"""Checks if the file is opened or not"""
        ...
    def put_byte(self) -> Any:
        r"""Writes a single byte to the file
        
        :param chr: the byte value
        """
        ...
    def puts(self, str: str) -> int:
        ...
    def read(self, size: Any) -> Any:
        r"""Reads from the file. Returns the buffer or None
        
        :param size: the maximum number of bytes to read
        :returns: a str, or None
        """
        ...
    def readbytes(self, size: Any, big_endian: Any) -> Any:
        r"""Similar to read() but it respect the endianness
        
        :param size: the maximum number of bytes to read
        :param big_endian: endianness
        :returns: a str, or None
        """
        ...
    def seek(self, offset: Any, whence: Any = 0) -> Any:
        r"""Set input source position
        
        :param offset: the seek offset
        :param whence: the position to seek from
        :returns: the new position (not 0 as fseek!)
        """
        ...
    def size(self) -> int64:
        ...
    def tell(self) -> Any:
        r"""Returns the current position"""
        ...
    def tmpfile(self) -> Any:
        r"""A static method to construct an instance using a temporary file"""
        ...
    def write(self, buf: Any) -> Any:
        r"""Writes to the file. Returns 0 or the number of bytes written
        
        :param buf: the str to write
        :returns: result code
        """
        ...
    def writebytes(self, size: Any, big_endian: Any) -> Any:
        r"""Similar to write() but it respect the endianness
        
        :param buf: the str to write
        :param big_endian: endianness
        :returns: result code
        """
        ...

def qfclose(fp: FILE) -> int:
    ...

def qfile_t_from_capsule(pycapsule: Any) -> qfile_t:
    ...

def qfile_t_from_fp(fp: FILE) -> qfile_t:
    ...

def qfile_t_tmpfile() -> Any:
    r"""A static method to construct an instance using a temporary file"""
    ...

QMOVE_CROSS_FS: int  # 1
QMOVE_OVERWRITE: int  # 2
QMOVE_OVR_RO: int  # 4
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module