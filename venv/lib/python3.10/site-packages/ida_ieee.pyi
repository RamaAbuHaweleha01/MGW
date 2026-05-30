from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""IEEE floating point functions.

"""

class fpvalue_shorts_array_t:
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
    def __getitem__(self, i: size_t) -> int:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
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
    def __setitem__(self, i: size_t, v: int) -> None:
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> str:
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

class fpvalue_t:
    @property
    def bytes(self) -> Any: ...
    @property
    def float(self) -> Any: ...
    @property
    def int64(self) -> Any: ...
    @property
    def shorts(self) -> Any: ...
    @property
    def sval(self) -> Any: ...
    @property
    def uint64(self) -> Any: ...
    @property
    def w(self) -> Any: ...
    def __add__(self, o: fpvalue_t) -> fpvalue_t:
        ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: fpvalue_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: fpvalue_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: Any) -> int:
        ...
    def __gt__(self, r: fpvalue_t) -> bool:
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
        ...
    def __le__(self, r: fpvalue_t) -> bool:
        ...
    def __lt__(self, r: fpvalue_t) -> bool:
        ...
    def __mul__(self, o: fpvalue_t) -> fpvalue_t:
        ...
    def __ne__(self, r: fpvalue_t) -> bool:
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
    def __setitem__(self, i: Any, v: Any) -> None:
        ...
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> str:
        ...
    def __sub__(self, o: fpvalue_t) -> fpvalue_t:
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
    def __truediv__(self, o: fpvalue_t) -> fpvalue_t:
        ...
    def assign(self, r: fpvalue_t) -> None:
        ...
    def clear(self) -> None:
        ...
    def compare(self, r: fpvalue_t) -> int:
        ...
    def copy(self) -> fpvalue_t:
        ...
    def eabs(self) -> None:
        r"""Calculate absolute value.
        
        """
        ...
    def fadd(self, y: fpvalue_t) -> fpvalue_error_t:
        r"""Arithmetic operations.
        
        """
        ...
    def fdiv(self, y: fpvalue_t) -> fpvalue_error_t:
        ...
    def fmul(self, y: fpvalue_t) -> fpvalue_error_t:
        ...
    def from_10bytes(self, fpval: void) -> fpvalue_error_t:
        r"""Conversions for 10-byte floating point values.
        
        """
        ...
    def from_12bytes(self, fpval: void) -> fpvalue_error_t:
        r"""Conversions for 12-byte floating point values.
        
        """
        ...
    def from_int64(self, x: int64) -> None:
        ...
    def from_str(self, p: str) -> fpvalue_error_t:
        r"""Convert string to IEEE. 
                
        """
        ...
    def from_sval(self, x: int) -> None:
        r"""Convert integer to IEEE.
        
        """
        ...
    def from_uint64(self, x: uint64) -> None:
        ...
    def fsub(self, y: fpvalue_t) -> fpvalue_error_t:
        ...
    def get_kind(self) -> fpvalue_kind_t:
        r"""Get value kind.
        
        """
        ...
    def is_negative(self) -> bool:
        r"""Is negative value?
        
        """
        ...
    def mul_pow2(self, power_of_2: int) -> fpvalue_error_t:
        r"""Multiply by a power of 2.
        
        """
        ...
    def negate(self) -> None:
        r"""Negate.
        
        """
        ...
    def new_from_str(self, p: str) -> fpvalue_t:
        ...
    def to_10bytes(self, fpval: void) -> fpvalue_error_t:
        ...
    def to_12bytes(self, fpval: void) -> fpvalue_error_t:
        ...
    def to_int64(self, round: bool = False) -> fpvalue_error_t:
        ...
    def to_str(self, args: Any) -> None:
        r"""Convert IEEE to string. 
                
        :param buf: the output buffer
        :param bufsize: the size of the output buffer
        :param mode: broken down into:
        * low byte: number of digits after '.'
        * second byte: FPNUM_LENGTH
        * third byte: FPNUM_DIGITS
        """
        ...
    def to_sval(self, round: bool = False) -> fpvalue_error_t:
        r"""Convert IEEE to integer (+-0.5 if round)
        
        """
        ...
    def to_uint64(self, round: bool = False) -> fpvalue_error_t:
        ...

def ecleaz(x: eNI) -> None:
    ...

EONE: bytes
ETWO: bytes
EZERO: bytes
E_SPECIAL_EXP: int  # 32767
FPVAL_NWORDS: int  # 8
FPV_BADARG: int  # 0
FPV_NAN: int  # 2
FPV_NINF: int  # 4
FPV_NORM: int  # 1
FPV_PINF: int  # 3
IEEE_E: int  # 1
IEEE_EXONE: int  # 16383
IEEE_M: int  # 2
IEEE_NI: int  # 11
MAXEXP_DOUBLE: int  # 1024
MAXEXP_FLOAT: int  # 128
MAXEXP_LNGDBL: int  # 16384
REAL_ERROR_BADDATA: int  # -3
REAL_ERROR_BADSTR: int  # 3
REAL_ERROR_FORMAT: int  # -1
REAL_ERROR_FPOVER: int  # 2
REAL_ERROR_INTOVER: int  # 5
REAL_ERROR_OK: int  # 1
REAL_ERROR_RANGE: int  # -2
REAL_ERROR_ZERODIV: int  # 4
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
cvar: swigvarlink  # (MAXEXP_LNGDBL, MAXEXP_DOUBLE, MAXEXP_FLOAT)
ida_idaapi: module
weakref: module