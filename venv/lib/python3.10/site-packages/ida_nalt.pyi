from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Definitions of various information kept in netnodes.

Each address in the program has a corresponding netnode: netnode(ea).
If we have no information about an address, the corresponding netnode is not created. Otherwise we will create a netnode and save information in it. All variable length information (names, comments, offset information, etc) is stored in the netnode.
Don't forget that some information is already stored in the flags (bytes.hpp)
netnode. 
    
"""

class array_parameters_t:
    @property
    def alignment(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def lineitems(self) -> Any: ...
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
    def __init__(self, _f: int = 1, _l: int = 0, _a: int = -1) -> Any:
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
    def is_default(self) -> bool:
        ...

class custom_data_type_ids_fids_array:
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
    def __getitem__(self, i: size_t) -> int:
        ...
    def __gt__(self, value: Any) -> Any:
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
    def __setitem__(self, i: size_t, v: short) -> None:
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

class custom_data_type_ids_t:
    @property
    def dtid(self) -> Any: ...
    @property
    def fids(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def get_dtid(self) -> tid_t:
        ...
    def set(self, tid: tid_t) -> None:
        ...

class enum_const_t:
    @property
    def serial(self) -> Any: ...
    @property
    def tid(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

class opinfo_t:
    @property
    def cd(self) -> Any: ...
    @property
    def ec(self) -> Any: ...
    @property
    def path(self) -> Any: ...
    @property
    def ri(self) -> Any: ...
    @property
    def strtype(self) -> Any: ...
    @property
    def tid(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

class printop_t:
    @property
    def aflags(self) -> Any: ...
    @property
    def features(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def is_ti_valid(self) -> Any: ...
    @property
    def suspop(self) -> Any: ...
    @property
    def ti(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def get_ti(self) -> opinfo_t:
        ...
    def is_aflags_initialized(self) -> bool:
        ...
    def is_f64(self) -> bool:
        ...
    def is_ti_initialized(self) -> bool:
        ...
    def set_aflags_initialized(self, v: bool = True) -> None:
        ...
    def set_ti_initialized(self, v: bool = True) -> None:
        ...

class refinfo_t:
    @property
    def base(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def target(self) -> Any: ...
    @property
    def tdelta(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def init(self, args: Any) -> None:
        ...
    def is_custom(self) -> bool:
        ...
    def is_no_ones(self) -> bool:
        ...
    def is_no_zeros(self) -> bool:
        ...
    def is_pastend(self) -> bool:
        ...
    def is_rvaoff(self) -> bool:
        ...
    def is_selfref(self) -> bool:
        ...
    def is_signed(self) -> bool:
        ...
    def is_subtract(self) -> bool:
        ...
    def is_target_optional(self) -> bool:
        r"""< is_reftype_target_optional()
        
        """
        ...
    def no_base_xref(self) -> bool:
        ...
    def set_type(self, rt: reftype_t) -> None:
        ...
    def type(self) -> reftype_t:
        ...

class strpath_ids_array:
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
    def __getitem__(self, i: size_t) -> int:
        ...
    def __gt__(self, value: Any) -> Any:
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
    def __setitem__(self, i: size_t, v: int) -> None:
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

class strpath_t:
    @property
    def delta(self) -> Any: ...
    @property
    def ids(self) -> Any: ...
    @property
    def len(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

class switch_info_t:
    SWITCH_INFO_VERSION: int  # 2
    @property
    def custom(self) -> Any: ...
    @property
    def defjump(self) -> Any: ...
    @property
    def elbase(self) -> Any: ...
    @property
    def expr_ea(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def ind_lowcase(self) -> Any: ...
    @property
    def jcases(self) -> Any: ...
    @property
    def jumps(self) -> Any: ...
    @property
    def lowcase(self) -> Any: ...
    @property
    def marks(self) -> Any: ...
    @property
    def ncases(self) -> Any: ...
    @property
    def regdtype(self) -> Any: ...
    @property
    def regnum(self) -> Any: ...
    @property
    def startea(self) -> Any: ...
    @property
    def values(self) -> Any: ...
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
    def __swig_destroy__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def assign(self, other: switch_info_t) -> None:
        ...
    def clear(self) -> None:
        ...
    def get_jrange_vrange(self, jrange: range_t = None, vrange: range_t = None) -> bool:
        r"""get separate parts of the switch
        
        """
        ...
    def get_jtable_element_size(self) -> int:
        ...
    def get_jtable_size(self) -> int:
        ...
    def get_lowcase(self) -> int:
        ...
    def get_shift(self) -> int:
        r"""See SWI_SHIFT_MASK. possible answers: 0..3. 
                
        """
        ...
    def get_version(self) -> int:
        ...
    def get_vtable_element_size(self) -> int:
        ...
    def has_default(self) -> bool:
        ...
    def has_elbase(self) -> bool:
        ...
    def is_custom(self) -> bool:
        ...
    def is_indirect(self) -> bool:
        ...
    def is_nolowcase(self) -> bool:
        ...
    def is_sparse(self) -> bool:
        ...
    def is_subtract(self) -> bool:
        ...
    def is_user_defined(self) -> bool:
        ...
    def set_elbase(self, base: ida_idaapi.ea_t) -> None:
        ...
    def set_expr(self, r: int, dt: op_dtype_t) -> None:
        ...
    def set_jtable_element_size(self, size: int) -> None:
        ...
    def set_jtable_size(self, size: int) -> None:
        ...
    def set_shift(self, shift: int) -> None:
        r"""See SWI_SHIFT_MASK.
        
        """
        ...
    def set_vtable_element_size(self, size: int) -> None:
        ...
    def use_std_table(self) -> bool:
        ...

def add_encoding(encname: str) -> int:
    ...

def clr__bnot0(ea: ida_idaapi.ea_t) -> None:
    ...

def clr__bnot1(ea: ida_idaapi.ea_t) -> None:
    ...

def clr__invsign0(ea: ida_idaapi.ea_t) -> None:
    ...

def clr__invsign1(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_abits(ea: ida_idaapi.ea_t, bits: aflags_t) -> None:
    ...

def clr_align_flow(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_colored_item(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_fixed_spd(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_has_lname(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_has_ti(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_has_ti0(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_has_ti1(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_libitem(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_lzero0(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_lzero1(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_noret(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_notcode(ea: ida_idaapi.ea_t) -> None:
    r"""Clear not-code mark.
    
    """
    ...

def clr_notproc(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_retfp(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_terse_struc(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_tilcmt(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_usemodsp(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_usersp(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_userti(ea: ida_idaapi.ea_t) -> None:
    ...

def clr_zstroff(ea: ida_idaapi.ea_t) -> None:
    ...

def dbg_get_input_path() -> str:
    r"""Get debugger input file name/path (see LFLG_DBG_NOPATH)
    
    """
    ...

def del_absbase(ea: ida_idaapi.ea_t) -> None:
    ...

def del_aflags(ea: ida_idaapi.ea_t) -> None:
    ...

def del_alignment(ea: ida_idaapi.ea_t) -> None:
    ...

def del_array_parameters(ea: ida_idaapi.ea_t) -> None:
    ...

def del_custom_data_type_ids(ea: ida_idaapi.ea_t) -> None:
    ...

def del_encoding(idx: int) -> bool:
    ...

def del_ind_purged(ea: ida_idaapi.ea_t) -> None:
    ...

def del_item_color(ea: ida_idaapi.ea_t) -> bool:
    ...

def del_op_tinfo(ea: ida_idaapi.ea_t, n: int) -> None:
    ...

def del_refinfo(ea: ida_idaapi.ea_t, n: int) -> bool:
    ...

def del_source_linnum(ea: ida_idaapi.ea_t) -> None:
    ...

def del_str_type(ea: ida_idaapi.ea_t) -> None:
    ...

def del_switch_info(ea: ida_idaapi.ea_t) -> None:
    ...

def del_switch_parent(ea: ida_idaapi.ea_t) -> None:
    ...

def del_tinfo(ea: ida_idaapi.ea_t) -> None:
    ...

def delete_imports() -> None:
    ...

def ea2node(ea: ida_idaapi.ea_t) -> nodeidx_t:
    r"""Get netnode for the specified address.
    
    """
    ...

def encoding_from_strtype(strtype: int) -> str:
    ...

def end_ea2node(ea: ida_idaapi.ea_t) -> nodeidx_t:
    ...

def enum_import_names(mod_index: Any, callback: Any) -> Any:
    r"""Enumerate imports from a specific module.
    Please refer to list_imports.py example.
    
    :param mod_index: The module index
    :param callback: A callable object that will be invoked with an ea, name (could be None) and ordinal.
    :returns: 1-finished ok, -1 on error, otherwise callback return value (<=0)
    """
    ...

def find_custom_refinfo(name: str) -> int:
    r"""Get id of a custom refinfo type.
    
    """
    ...

def get_abi_name() -> Any:
    ...

def get_absbase(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_aflags(ea: ida_idaapi.ea_t) -> aflags_t:
    ...

def get_alignment(ea: ida_idaapi.ea_t) -> int:
    ...

def get_archive_path() -> str:
    r"""Get archive file path from which input file was extracted.
    
    """
    ...

def get_array_parameters(out: array_parameters_t, ea: ida_idaapi.ea_t) -> ssize_t:
    ...

def get_asm_inc_file() -> str:
    r"""Get name of the include file.
    
    """
    ...

def get_custom_data_type_ids(cdis: custom_data_type_ids_t, ea: ida_idaapi.ea_t) -> int:
    ...

def get_custom_refinfo(crid: int) -> custom_refinfo_handler_t:
    r"""Get definition of a registered custom refinfo type.
    
    """
    ...

def get_default_encoding_idx(bpu: int) -> int:
    ...

def get_elapsed_secs() -> int:
    r"""Get seconds database stayed open.
    
    """
    ...

def get_encoding_bpu(idx: int) -> int:
    ...

def get_encoding_bpu_by_name(encname: str) -> int:
    ...

def get_encoding_name(idx: int) -> str:
    ...

def get_encoding_qty() -> int:
    ...

def get_gotea() -> ida_idaapi.ea_t:
    ...

def get_ida_notepad_text() -> str:
    r"""Get notepad text.
    
    """
    ...

def get_idb_ctime() -> time_t:
    r"""Get database creation timestamp.
    
    """
    ...

def get_idb_nopens() -> int:
    r"""Get number of times the database is opened.
    
    """
    ...

def get_ids_modnode() -> netnode:
    r"""Get ids modnode.
    
    """
    ...

def get_imagebase() -> ida_idaapi.ea_t:
    r"""Get image base address.
    
    """
    ...

def get_import_module_name(mod_index: Any) -> Any:
    r"""Returns the name of an imported module given its index
    
    :param mod_index: the module index
    :returns: None or the module name
    """
    ...

def get_import_module_qty() -> uint:
    ...

def get_ind_purged(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_initial_ida_version() -> str:
    r"""Get version of ida which created the database (string format like "7.5")
    
    """
    ...

def get_initial_idb_version() -> ushort:
    r"""Get initial version of the database (numeric format like 700)
    
    """
    ...

def get_initial_version() -> ushort:
    r"""Get initial version of the database (numeric format like 700)
    
    """
    ...

def get_input_file_path() -> str:
    r"""Get full path of the input file.
    
    """
    ...

def get_item_color(ea: ida_idaapi.ea_t) -> bgcolor_t:
    ...

def get_loader_format_name() -> str:
    r"""Get file format name for loader modules.
    
    """
    ...

def get_op_tinfo(tif: tinfo_t, ea: ida_idaapi.ea_t, n: int) -> bool:
    ...

def get_outfile_encoding_idx() -> int:
    ...

def get_refinfo(ri: refinfo_t, ea: ida_idaapi.ea_t, n: int) -> bool:
    ...

def get_reftype_by_size(size: size_t) -> reftype_t:
    r"""Get REF_... constant from size Supported sizes: 1,2,4,8,16 For other sizes returns reftype_t(-1) 
            
    """
    ...

def get_root_filename() -> str:
    r"""Get file name only of the input file.
    
    """
    ...

def get_source_linnum(ea: ida_idaapi.ea_t) -> int:
    ...

def get_srcdbg_paths() -> str:
    r"""Get source debug paths.
    
    """
    ...

def get_srcdbg_undesired_paths() -> str:
    r"""Get user-closed source files.
    
    """
    ...

def get_str_encoding_idx(strtype: int) -> uchar:
    ...

def get_str_term1(strtype: int) -> char:
    ...

def get_str_term2(strtype: int) -> char:
    ...

def get_str_type(ea: ida_idaapi.ea_t) -> int:
    ...

def get_str_type_code(strtype: int) -> uchar:
    ...

def get_str_type_prefix_length(strtype: int) -> int:
    ...

def get_strid(ea: ida_idaapi.ea_t) -> tid_t:
    ...

def get_strtype_bpu(strtype: int) -> int:
    ...

def get_switch_info(args: Any) -> ssize_t:
    ...

def get_switch_parent(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_tinfo(tif: tinfo_t, ea: ida_idaapi.ea_t) -> bool:
    ...

def getnode(ea: ida_idaapi.ea_t) -> netnode:
    ...

def has_aflag_linnum(flags: aflags_t) -> bool:
    ...

def has_aflag_lname(flags: aflags_t) -> bool:
    ...

def has_aflag_ti(flags: aflags_t) -> bool:
    ...

def has_aflag_ti0(flags: aflags_t) -> bool:
    ...

def has_aflag_ti1(flags: aflags_t) -> bool:
    ...

def has_lname(ea: ida_idaapi.ea_t) -> bool:
    ...

def has_ti(ea: ida_idaapi.ea_t) -> bool:
    ...

def has_ti0(ea: ida_idaapi.ea_t) -> bool:
    ...

def has_ti1(ea: ida_idaapi.ea_t) -> bool:
    ...

def hide_border(ea: ida_idaapi.ea_t) -> None:
    ...

def hide_item(ea: ida_idaapi.ea_t) -> None:
    ...

def is__bnot0(ea: ida_idaapi.ea_t) -> bool:
    ...

def is__bnot1(ea: ida_idaapi.ea_t) -> bool:
    ...

def is__invsign0(ea: ida_idaapi.ea_t) -> bool:
    ...

def is__invsign1(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_aflag__bnot0(flags: aflags_t) -> bool:
    ...

def is_aflag__bnot1(flags: aflags_t) -> bool:
    ...

def is_aflag__invsign0(flags: aflags_t) -> bool:
    ...

def is_aflag__invsign1(flags: aflags_t) -> bool:
    ...

def is_aflag_align_flow(flags: aflags_t) -> bool:
    ...

def is_aflag_colored_item(flags: aflags_t) -> bool:
    ...

def is_aflag_data_guessed_by_hexrays(flags: aflags_t) -> bool:
    ...

def is_aflag_fixed_spd(flags: aflags_t) -> bool:
    ...

def is_aflag_func_guessed_by_hexrays(flags: aflags_t) -> bool:
    ...

def is_aflag_hidden_border(flags: aflags_t) -> bool:
    ...

def is_aflag_hidden_item(flags: aflags_t) -> bool:
    ...

def is_aflag_libitem(flags: aflags_t) -> bool:
    ...

def is_aflag_lzero0(flags: aflags_t) -> bool:
    ...

def is_aflag_lzero1(flags: aflags_t) -> bool:
    ...

def is_aflag_manual_insn(flags: aflags_t) -> bool:
    ...

def is_aflag_noret(flags: aflags_t) -> bool:
    ...

def is_aflag_notcode(flags: aflags_t) -> bool:
    ...

def is_aflag_notproc(flags: aflags_t) -> bool:
    ...

def is_aflag_public_name(flags: aflags_t) -> bool:
    ...

def is_aflag_retfp(flags: aflags_t) -> bool:
    ...

def is_aflag_terse_struc(flags: aflags_t) -> bool:
    ...

def is_aflag_tilcmt(flags: aflags_t) -> bool:
    ...

def is_aflag_type_determined_by_hexrays(flags: aflags_t) -> bool:
    ...

def is_aflag_type_guessed_by_hexrays(flags: aflags_t) -> bool:
    ...

def is_aflag_type_guessed_by_ida(flags: aflags_t) -> bool:
    ...

def is_aflag_usersp(flags: aflags_t) -> bool:
    ...

def is_aflag_userti(flags: aflags_t) -> bool:
    ...

def is_aflag_weak_name(flags: aflags_t) -> bool:
    ...

def is_aflag_zstroff(flags: aflags_t) -> bool:
    ...

def is_align_flow(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_colored_item(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_data_guessed_by_hexrays(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_finally_visible_item(ea: ida_idaapi.ea_t) -> bool:
    r"""Is instruction visible?
    
    """
    ...

def is_fixed_spd(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_func_guessed_by_hexrays(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_hidden_border(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_hidden_item(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_libitem(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_lzero0(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_lzero1(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_noret(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_notcode(ea: ida_idaapi.ea_t) -> bool:
    r"""Is the address marked as not-code?
    
    """
    ...

def is_notproc(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_pascal(strtype: int) -> bool:
    ...

def is_reftype_target_optional(type: reftype_t) -> bool:
    r"""Can the target be calculated using operand value?
    
    """
    ...

def is_retfp(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_terse_struc(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_tilcmt(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_type_determined_by_hexrays(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_type_guessed_by_hexrays(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_type_guessed_by_ida(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_usersp(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_userti(ea: ida_idaapi.ea_t) -> bool:
    ...

def is_visible_item(ea: ida_idaapi.ea_t) -> bool:
    r"""Test visibility of item at given ea.
    
    """
    ...

def is_zstroff(ea: ida_idaapi.ea_t) -> bool:
    ...

def make_str_type(type_code: uchar, encoding_idx: int, term1: uchar = 0, term2: uchar = 0) -> int:
    ...

def node2ea(ndx: nodeidx_t) -> ida_idaapi.ea_t:
    ...

def rename_encoding(idx: int, encname: str) -> bool:
    ...

def retrieve_input_file_crc32() -> int:
    r"""Get input file crc32 stored in the database. it can be used to check that the input file has not been changed. 
            
    """
    ...

def retrieve_input_file_md5() -> bytes:
    r"""Get input file md5.
    
    """
    ...

def retrieve_input_file_sha256() -> bytes:
    r"""Get input file sha256.
    
    """
    ...

def retrieve_input_file_size() -> int:
    r"""Get size of input file in bytes.
    
    """
    ...

def set__bnot0(ea: ida_idaapi.ea_t) -> None:
    ...

def set__bnot1(ea: ida_idaapi.ea_t) -> None:
    ...

def set__invsign0(ea: ida_idaapi.ea_t) -> None:
    ...

def set__invsign1(ea: ida_idaapi.ea_t) -> None:
    ...

def set_abits(ea: ida_idaapi.ea_t, bits: aflags_t) -> None:
    ...

def set_absbase(ea: ida_idaapi.ea_t, x: ida_idaapi.ea_t) -> None:
    ...

def set_aflags(ea: ida_idaapi.ea_t, flags: aflags_t) -> None:
    ...

def set_align_flow(ea: ida_idaapi.ea_t) -> None:
    ...

def set_alignment(ea: ida_idaapi.ea_t, x: int) -> None:
    ...

def set_archive_path(file: str) -> bool:
    r"""Set archive file path from which input file was extracted.
    
    """
    ...

def set_array_parameters(ea: ida_idaapi.ea_t, _in: array_parameters_t) -> None:
    ...

def set_asm_inc_file(file: str) -> bool:
    r"""Set name of the include file.
    
    """
    ...

def set_colored_item(ea: ida_idaapi.ea_t) -> None:
    ...

def set_custom_data_type_ids(ea: ida_idaapi.ea_t, cdis: custom_data_type_ids_t) -> None:
    ...

def set_data_guessed_by_hexrays(ea: ida_idaapi.ea_t) -> None:
    ...

def set_default_encoding_idx(bpu: int, idx: int) -> bool:
    ...

def set_fixed_spd(ea: ida_idaapi.ea_t) -> None:
    ...

def set_func_guessed_by_hexrays(ea: ida_idaapi.ea_t) -> None:
    ...

def set_gotea(gotea: ida_idaapi.ea_t) -> None:
    ...

def set_has_lname(ea: ida_idaapi.ea_t) -> None:
    ...

def set_has_ti(ea: ida_idaapi.ea_t) -> None:
    ...

def set_has_ti0(ea: ida_idaapi.ea_t) -> None:
    ...

def set_has_ti1(ea: ida_idaapi.ea_t) -> None:
    ...

def set_ida_notepad_text(text: str, size: size_t = 0) -> None:
    r"""Set notepad text.
    
    """
    ...

def set_ids_modnode(id: netnode) -> None:
    r"""Set ids modnode.
    
    """
    ...

def set_imagebase(base: ida_idaapi.ea_t) -> None:
    r"""Set image base address.
    
    """
    ...

def set_item_color(ea: ida_idaapi.ea_t, color: bgcolor_t) -> None:
    ...

def set_libitem(ea: ida_idaapi.ea_t) -> None:
    ...

def set_loader_format_name(name: str) -> None:
    r"""Set file format name for loader modules.
    
    """
    ...

def set_lzero0(ea: ida_idaapi.ea_t) -> None:
    ...

def set_lzero1(ea: ida_idaapi.ea_t) -> None:
    ...

def set_noret(ea: ida_idaapi.ea_t) -> None:
    ...

def set_notcode(ea: ida_idaapi.ea_t) -> None:
    r"""Mark address so that it cannot be converted to instruction.
    
    """
    ...

def set_notproc(ea: ida_idaapi.ea_t) -> None:
    ...

def set_op_tinfo(ea: ida_idaapi.ea_t, n: int, tif: tinfo_t) -> bool:
    ...

def set_outfile_encoding_idx(idx: int) -> bool:
    ...

def set_refinfo(args: Any) -> bool:
    ...

def set_refinfo_ex(ea: ida_idaapi.ea_t, n: int, ri: refinfo_t) -> bool:
    ...

def set_retfp(ea: ida_idaapi.ea_t) -> None:
    ...

def set_root_filename(file: str) -> None:
    r"""Set full path of the input file.
    
    """
    ...

def set_source_linnum(ea: ida_idaapi.ea_t, lnnum: int) -> None:
    ...

def set_srcdbg_paths(paths: str) -> None:
    r"""Set source debug paths.
    
    """
    ...

def set_srcdbg_undesired_paths(paths: str) -> None:
    r"""Set user-closed source files.
    
    """
    ...

def set_str_encoding_idx(strtype: int, encoding_idx: int) -> int:
    ...

def set_str_type(ea: ida_idaapi.ea_t, x: int) -> None:
    ...

def set_switch_info(ea: ida_idaapi.ea_t, _in: switch_info_t) -> None:
    ...

def set_switch_parent(ea: ida_idaapi.ea_t, x: ida_idaapi.ea_t) -> None:
    ...

def set_terse_struc(ea: ida_idaapi.ea_t) -> None:
    ...

def set_tilcmt(ea: ida_idaapi.ea_t) -> None:
    ...

def set_tinfo(ea: ida_idaapi.ea_t, tif: tinfo_t) -> bool:
    ...

def set_type_determined_by_hexrays(ea: ida_idaapi.ea_t) -> None:
    ...

def set_type_guessed_by_ida(ea: ida_idaapi.ea_t) -> None:
    ...

def set_usemodsp(ea: ida_idaapi.ea_t) -> None:
    ...

def set_usersp(ea: ida_idaapi.ea_t) -> None:
    ...

def set_userti(ea: ida_idaapi.ea_t) -> None:
    ...

def set_visible_item(ea: ida_idaapi.ea_t, visible: bool) -> None:
    r"""Change visibility of item at given ea.
    
    """
    ...

def set_zstroff(ea: ida_idaapi.ea_t) -> None:
    ...

def switch_info_t__from_ptrval__(ptrval: size_t) -> switch_info_t:
    ...

def unhide_border(ea: ida_idaapi.ea_t) -> None:
    ...

def unhide_item(ea: ida_idaapi.ea_t) -> None:
    ...

def upd_abits(ea: ida_idaapi.ea_t, clr_bits: aflags_t, set_bits: aflags_t) -> None:
    ...

def uses_aflag_modsp(flags: aflags_t) -> bool:
    ...

def uses_modsp(ea: ida_idaapi.ea_t) -> bool:
    ...

AFL_ALIGNFLOW: int  # 16777216
AFL_BNOT0: int  # 256
AFL_BNOT1: int  # 512
AFL_COLORED: int  # 262144
AFL_FIXEDSPD: int  # 8388608
AFL_HIDDEN: int  # 16
AFL_HR_DETERMINED: int  # -1073741824
AFL_HR_GUESSED_DATA: int  # -2147483648
AFL_HR_GUESSED_FUNC: int  # 1073741824
AFL_IDA_GUESSED: int  # 0
AFL_LIB: int  # 1024
AFL_LINNUM: int  # 1
AFL_LNAME: int  # 16384
AFL_LZERO0: int  # 65536
AFL_LZERO1: int  # 131072
AFL_MANUAL: int  # 32
AFL_NOBRD: int  # 64
AFL_NORET: int  # 4194304
AFL_NOTCODE: int  # 268435456
AFL_NOTPROC: int  # 536870912
AFL_PUBNAM: int  # 4
AFL_RETFP: int  # 67108864
AFL_SIGN0: int  # 1048576
AFL_SIGN1: int  # 2097152
AFL_TERSESTR: int  # 524288
AFL_TI: int  # 2048
AFL_TI0: int  # 4096
AFL_TI1: int  # 8192
AFL_TILCMT: int  # 32768
AFL_TYPE_GUESSED: int  # -1040187392
AFL_USEMODSP: int  # 134217728
AFL_USERSP: int  # 2
AFL_USERTI: int  # 33554432
AFL_WEAKNAM: int  # 8
AFL_ZSTROFF: int  # 128
AP_ALLOWDUPS: int  # 1
AP_ARRAY: int  # 8
AP_IDXBASEMASK: int  # 240
AP_IDXBIN: int  # 48
AP_IDXDEC: int  # 0
AP_IDXHEX: int  # 16
AP_IDXOCT: int  # 32
AP_INDEX: int  # 4
AP_SIGNED: int  # 2
BPU_1B: int  # 1
BPU_2B: int  # 2
BPU_4B: int  # 4
GOTEA_NODE_IDX: int  # 0
GOTEA_NODE_NAME: str  # $ got
IDB_DESKTOPS_NODE_NAME: str  # $ desktops
IDB_DESKTOPS_TAG: str  # S
MAXSTRUCPATH: int  # 32
NALT_ABSBASE: int  # 10
NALT_AFLAGS: int  # 8
NALT_ALIGN: int  # 17
NALT_COLOR: int  # 20
NALT_CREF_FROM: str  # x
NALT_CREF_TO: str  # X
NALT_DREF_FROM: str  # d
NALT_DREF_TO: str  # D
NALT_ENUM0: int  # 11
NALT_ENUM1: int  # 12
NALT_GR_LAYX: str  # p
NALT_LINNUM: int  # 9
NALT_PURGE: int  # 15
NALT_STRTYPE: int  # 16
NALT_STRUCT: int  # 3
NALT_SWITCH: int  # 1
NSUP_ARGEAS: int  # 30
NSUP_ARRAY: int  # 5
NSUP_CMT: int  # 0
NSUP_CUSTDT: int  # 28
NSUP_EX_FLAGS: int  # 37
NSUP_FOP1: int  # 2
NSUP_FOP2: int  # 3
NSUP_FOP3: int  # 7
NSUP_FOP4: int  # 18
NSUP_FOP5: int  # 19
NSUP_FOP6: int  # 20
NSUP_FOP7: int  # 31
NSUP_FOP8: int  # 32
NSUP_FRAME: int  # 1089536
NSUP_FTAILS: int  # 28672
NSUP_GROUP: int  # 32768
NSUP_GROUPS: int  # 29
NSUP_GR_INFO: str  # g
NSUP_GR_LAYT: str  # l
NSUP_JINFO: int  # 4
NSUP_LLABEL: int  # 20480
NSUP_MANUAL: int  # 8192
NSUP_OMFGRP: int  # 6
NSUP_OPTYPES: int  # 36864
NSUP_OREF0: int  # 12
NSUP_OREF1: int  # 13
NSUP_OREF2: int  # 14
NSUP_OREF3: int  # 24
NSUP_OREF4: int  # 25
NSUP_OREF5: int  # 26
NSUP_OREF6: int  # 35
NSUP_OREF7: int  # 36
NSUP_ORIGFMD: int  # 1085440
NSUP_POINTS: int  # 4096
NSUP_REF0: int  # 9
NSUP_REF1: int  # 10
NSUP_REF2: int  # 11
NSUP_REF3: int  # 21
NSUP_REF4: int  # 22
NSUP_REF5: int  # 23
NSUP_REF6: int  # 33
NSUP_REF7: int  # 34
NSUP_REGARG: int  # 24576
NSUP_REGVAR: int  # 16384
NSUP_REPCMT: int  # 1
NSUP_SEGTRANS: int  # 17
NSUP_STROFF0: int  # 15
NSUP_STROFF1: int  # 16
NSUP_SWITCH: int  # 8
NSUP_TYPEINFO: int  # 12288
NSUP_XREFPOS: int  # 27
PATCH_TAG: str  # P
POF_IS_F64: int  # 4
POF_VALID_AFLAGS: int  # 2
POF_VALID_TI: int  # 1
REFINFO_CUSTOM: int  # 64
REFINFO_NOBASE: int  # 128
REFINFO_NO_ONES: int  # 2048
REFINFO_NO_ZEROS: int  # 1024
REFINFO_PASTEND: int  # 32
REFINFO_RVAOFF: int  # 16
REFINFO_SELFREF: int  # 4096
REFINFO_SIGNEDOP: int  # 512
REFINFO_SUBTRACT: int  # 256
REFINFO_TYPE: int  # 15
REF_HIGH16: int  # 6
REF_HIGH8: int  # 5
REF_LAST: int  # 10
REF_LOW16: int  # 4
REF_LOW8: int  # 3
REF_OFF16: int  # 1
REF_OFF32: int  # 2
REF_OFF64: int  # 9
REF_OFF8: int  # 10
RIDX_ABINAME: int  # 1350
RIDX_ARCHIVE_PATH: int  # 1351
RIDX_C_MACROS: int  # 66
RIDX_DBG_BINPATHS: int  # 1328
RIDX_DUALOP_GRAPH: int  # 1300
RIDX_DUALOP_TEXT: int  # 1301
RIDX_FILE_FORMAT_NAME: int  # 1
RIDX_GROUPS: int  # 64
RIDX_H_PATH: int  # 65
RIDX_IDA_VERSION: int  # 1303
RIDX_INCLUDE: int  # 1100
RIDX_MD5: int  # 1302
RIDX_NOTEPAD: int  # 68
RIDX_PROBLEMS: int  # 1352
RIDX_SELECTORS: int  # 2
RIDX_SHA256: int  # 1349
RIDX_SMALL_IDC: int  # 1200
RIDX_SMALL_IDC_OLD: int  # 67
RIDX_SRCDBG_PATHS: int  # 1306
RIDX_SRCDBG_UNDESIRED: int  # 1353
RIDX_STR_ENCODINGS: int  # 1305
STRENC_DEFAULT: int  # 0
STRENC_NONE: int  # 255
STRLYT_MASK: int  # 252
STRLYT_PASCAL1: int  # 1
STRLYT_PASCAL2: int  # 2
STRLYT_PASCAL4: int  # 3
STRLYT_SHIFT: int  # 2
STRLYT_TERMCHR: int  # 0
STRTYPE_C: int  # 0
STRTYPE_C_16: int  # 1
STRTYPE_C_32: int  # 2
STRTYPE_LEN2: int  # 8
STRTYPE_LEN2_16: int  # 9
STRTYPE_LEN2_32: int  # 10
STRTYPE_LEN4: int  # 12
STRTYPE_LEN4_16: int  # 13
STRTYPE_LEN4_32: int  # 14
STRTYPE_PASCAL: int  # 4
STRTYPE_PASCAL_16: int  # 5
STRTYPE_PASCAL_32: int  # 6
STRTYPE_TERMCHR: int  # 0
STRWIDTH_1B: int  # 0
STRWIDTH_2B: int  # 1
STRWIDTH_4B: int  # 2
STRWIDTH_MASK: int  # 3
SWIG_PYTHON_LEGACY_BOOL: int  # 1
SWI_CUSTOM: int  # 16384
SWI_DEFRET: int  # 1048576
SWI_DEF_IN_TBL: int  # 32
SWI_ELBASE: int  # 512
SWI_HXNOLOWCASE: int  # 262144
SWI_INDIRECT: int  # 65536
SWI_J32: int  # 4
SWI_JMPINSN: int  # 4194304
SWI_JMP_INV: int  # 64
SWI_JSIZE: int  # 1024
SWI_SELFREL: int  # 2097152
SWI_SEPARATE: int  # 4096
SWI_SHIFT_MASK: int  # 384
SWI_SIGNED: int  # 8192
SWI_SPARSE: int  # 1
SWI_STDTBL: int  # 524288
SWI_SUBTRACT: int  # 131072
SWI_USER: int  # 16
SWI_V32: int  # 2
SWI_VERSION: int  # 8388608
SWI_VSIZE: int  # 2048
SWI_VSPLIT: int  # 8
V695_REF_OFF8: int  # 0
V695_REF_VHIGH: int  # 7
V695_REF_VLOW: int  # 8
annotations: _Feature
cvar: swigvarlink
ida_idaapi: module
weakref: module