from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Contains definition of the interface to IDD modules.

The interface consists of structures describing the target debugged processor and a debugging API. 
    
"""

class Appcall__:
    APPCALL_DEBEV: int  # 2
    APPCALL_MANUAL: int  # 1
    APPCALL_TIMEOUT: int  # 4
    @property
    def Consts(self) -> Any: ...
    def UTF16(self, s: Any) -> Any:
        ...
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
    def __getattr__(self, name_or_ea: Any) -> Any:
        r"""Allows you to call functions as if they were member functions (by returning a callable object)"""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, idx: Any) -> register_info_t:
        r"""
                Use self[func_name] syntax if the function name contains invalid characters for an attribute name
                See __getattr___
                
        """
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
    def array(self, type_name: Any) -> Any:
        r"""Defines an array type. Later you need to pack() / unpack()"""
        ...
    def buffer(self, str: Any = None, size: Any = 0, fill: Any = '\x00') -> Any:
        r"""
                Creates a string buffer. The returned value (r) will be a byref object.
                Use r.value to get the contents and r.size to get the buffer's size
                
        """
        ...
    def byref(self, val: Any) -> Any:
        r"""
                Method to create references to immutable objects
                Currently we support references to int/strings
                Objects need not be passed by reference (this will be done automatically)
                
        """
        ...
    def cleanup_appcall(self, tid: Any = 0) -> drc_t:
        r"""Cleanup after manual appcall. 
                
        :param tid: thread to use. NO_THREAD means to use the current thread The application state is restored as it was before calling the last appcall(). Nested appcalls are supported.
        :returns: eOk if successful, otherwise an error code
        """
        ...
    def cstr(self, val: Any) -> Any:
        ...
    def get_appcall_options(self) -> Any:
        r"""Return the global Appcall options"""
        ...
    def int64(self, v: Any) -> Any:
        r"""Whenever a 64bit number is needed use this method to construct an object"""
        ...
    def obj(self, kwds: Any) -> Any:
        r"""Returns an empty object or objects with attributes as passed via its keywords arguments"""
        ...
    def proto(self, name_or_ea: Any, proto_or_tinfo: Any, flags: Any = None) -> Any:
        r"""
                Allows you to instantiate an appcall (callable object) with the desired prototype
                :param name_or_ea: The name of the function (will be resolved with LocByName())
                :param proto_or_tinfo: function prototype as a string or type of the function as tinfo_t object
                :returns: a callbable Appcall instance with the given prototypes and flags, or
                          an exception if the prototype could not be parsed or the address is not resolvable.
                
        """
        ...
    def set_appcall_options(self, opt: Any) -> Any:
        r"""Method to change the Appcall options globally (not per Appcall)"""
        ...
    def typedobj(self, typedecl_or_tinfo: Any, ea: Any = None) -> Any:
        r"""
                Returns an appcall object for a type (can be given as tinfo_t object or
                as a string declaration)
                One can then use retrieve() member method
                :param ea: Optional parameter that later can be used to retrieve the type
                :returns: Appcall object or raises ValueError exception
                
        """
        ...
    def unicode(self, s: Any) -> Any:
        ...
    def valueof(self, name: Any, default: Any = 0) -> Any:
        r"""
                If the name could not be resolved then the default value will be returned
        
                :returns: the numeric value of a given name string.
                
        """
        ...

class Appcall_array__:
    r"""This class is used with Appcall.array() method"""
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
    def __init__(self, tp: Any) -> Any:
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
    def pack(self, L: Any) -> Any:
        r"""Packs a list or tuple into a byref buffer"""
        ...
    def try_to_convert_to_list(self, obj: Any) -> Any:
        r"""Is this object a list? We check for the existance of attribute zero and attribute self.size-1"""
        ...
    def unpack(self, buf: Any, as_list: Any = True) -> Any:
        r"""Unpacks an array back into a list or an object"""
        ...

class Appcall_callable__:
    r"""
        Helper class to issue appcalls using a natural syntax:
          appcall.FunctionNameInTheDatabase(arguments, ....)
        or
          appcall["Function@8"](arguments, ...)
        or
          f8 = appcall["Function@8"]
          f8(arg1, arg2, ...)
        or
          o = appcall.obj()
          i = byref(5)
          appcall.funcname(arg1, i, "hello", o)
        
    """
    @property
    def ea(self) -> Any: ...
    @property
    def fields(self) -> Any: ...
    @property
    def options(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def tif(self) -> Any: ...
    @property
    def timeout(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __call__(self, args: Any) -> Any:
        r"""Make object callable. We redirect execution to idaapi.appcall()"""
        ...
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
    def __init__(self, ea: Any, tinfo_or_typestr: Any = None, fields: Any = None) -> Any:
        r"""Initializes an appcall with a given function ea"""
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
    def retrieve(self, src: Any = None, flags: Any = 0) -> Any:
        r"""
                Unpacks a typed object from the database if an ea is given or from a string if a string was passed
                :param src: the address of the object or a string
                :returns: Returns a tuple of boolean and object or error number (Bool, Error | Object).
                
        """
        ...
    def store(self, obj: Any, dest_ea: Any = None, base_ea: Any = 0, flags: Any = 0) -> Any:
        r"""
                Packs an object into a given ea if provided or into a string if no address was passed.
                :param obj: The object to pack
                :param dest_ea: If packing to idb this will be the store location
                :param base_ea: If packing to a buffer, this will be the base that will be used to relocate the pointers
        
                :returns: Tuple(Boolean, packed_string or error code) if packing to a string
                :returns: a return code is returned (0 indicating success) if packing to the database
                
        """
        ...

class Appcall_consts__:
    r"""
        Helper class used by Appcall.Consts attribute
        It is used to retrieve constants via attribute access
        
    """
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
    def __getattr__(self, attr: Any) -> Any:
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
    def __init__(self, default: Any = None) -> Any:
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

class bptaddr_t:
    @property
    def hea(self) -> Any: ...
    @property
    def kea(self) -> Any: ...
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

class call_stack_info_t:
    @property
    def callea(self) -> Any: ...
    @property
    def fp(self) -> Any: ...
    @property
    def funcea(self) -> Any: ...
    @property
    def funcok(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: call_stack_info_t) -> bool:
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
    def __ne__(self, r: call_stack_info_t) -> bool:
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

class call_stack_info_vec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: call_stack_info_vec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __ne__(self, r: call_stack_info_vec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: call_stack_info_t) -> None:
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
    def add_unique(self, x: call_stack_info_t) -> bool:
        ...
    def append(self, x: call_stack_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: call_stack_info_vec_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: call_stack_info_t) -> bool:
        ...
    def inject(self, s: call_stack_info_t, len: size_t) -> None:
        ...
    def insert(self, it: call_stack_info_t, x: call_stack_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: call_stack_info_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class call_stack_t(call_stack_info_vec_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: call_stack_info_vec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __ne__(self, r: call_stack_info_vec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: call_stack_info_t) -> None:
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
    def add_unique(self, x: call_stack_info_t) -> bool:
        ...
    def append(self, x: call_stack_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: call_stack_info_vec_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: call_stack_info_t) -> bool:
        ...
    def inject(self, s: call_stack_info_t, len: size_t) -> None:
        ...
    def insert(self, it: call_stack_info_t, x: call_stack_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: call_stack_info_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class debapp_attrs_t:
    @property
    def addrsize(self) -> Any: ...
    @property
    def cbsize(self) -> Any: ...
    @property
    def is_be(self) -> Any: ...
    @property
    def platform(self) -> Any: ...
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

class debug_event_t:
    @property
    def ea(self) -> Any: ...
    @property
    def handled(self) -> Any: ...
    @property
    def pid(self) -> Any: ...
    @property
    def tid(self) -> Any: ...
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
    def bpt(self) -> bptaddr_t:
        r"""EXCEPTION
        
        """
        ...
    def bpt_ea(self) -> ida_idaapi.ea_t:
        r"""On some systems with special memory mappings the triggered ea might be different from the actual ea. Calculate the address to use. 
                
        """
        ...
    def clear(self) -> None:
        r"""clear the dependent information (see below), set event code to NO_EVENT
        
        """
        ...
    def clear_all(self) -> None:
        ...
    def copy(self, r: debug_event_t) -> debug_event_t:
        ...
    def eid(self) -> event_id_t:
        r"""Event code.
        
        """
        ...
    def exc(self) -> excinfo_t:
        ...
    def exit_code(self) -> int:
        r"""THREAD_STARTED (thread name) LIB_UNLOADED (unloaded library name) INFORMATION (will be displayed in the output window if not empty) 
                
        """
        ...
    def info(self) -> str:
        r"""BREAKPOINT
        
        """
        ...
    def is_bitness_changed(self) -> bool:
        r"""process bitness
        
        """
        ...
    def modinfo(self) -> modinfo_t:
        r"""Information that depends on the event code:
        
        < PROCESS_STARTED, PROCESS_ATTACHED, LIB_LOADED PROCESS_EXITED, THREAD_EXITED 
                
        """
        ...
    def set_bitness_changed(self, on: bool = True) -> None:
        ...
    def set_bpt(self) -> bptaddr_t:
        ...
    def set_eid(self, id: event_id_t) -> None:
        r"""Set event code. If the new event code is compatible with the old one then the dependent information (see below) will be preserved. Otherwise the event will be cleared and the new event code will be set. 
                
        """
        ...
    def set_exception(self) -> excinfo_t:
        ...
    def set_exit_code(self, id: event_id_t, code: int) -> None:
        ...
    def set_info(self, id: event_id_t) -> str:
        ...
    def set_modinfo(self, id: event_id_t) -> modinfo_t:
        ...

class debugger_t:
    ev_appcall: int  # 34
    ev_attach_process: int  # 4
    ev_bin_search: int  # 42
    ev_check_bpt: int  # 24
    ev_cleanup_appcall: int  # 35
    ev_close_file: int  # 28
    ev_dbg_enable_trace: int  # 38
    ev_detach_process: int  # 5
    ev_eval_lowcnd: int  # 36
    ev_exit_process: int  # 9
    ev_get_debapp_attrs: int  # 6
    ev_get_debmod_extensions: int  # 32
    ev_get_debug_event: int  # 10
    ev_get_dynamic_register_set: int  # 43
    ev_get_memory_info: int  # 21
    ev_get_processes: int  # 2
    ev_get_srcinfo_path: int  # 41
    ev_init_debugger: int  # 0
    ev_is_tracing_enabled: int  # 39
    ev_map_address: int  # 31
    ev_open_file: int  # 27
    ev_read_file: int  # 29
    ev_read_memory: int  # 22
    ev_read_registers: int  # 18
    ev_rebase_if_required_to: int  # 7
    ev_request_pause: int  # 8
    ev_resume: int  # 11
    ev_rexec: int  # 40
    ev_send_ioctl: int  # 37
    ev_set_backwards: int  # 12
    ev_set_dbg_options: int  # 44
    ev_set_exception_info: int  # 13
    ev_set_resume_mode: int  # 17
    ev_start_process: int  # 3
    ev_suspended: int  # 14
    ev_term_debugger: int  # 1
    ev_thread_continue: int  # 16
    ev_thread_get_sreg_base: int  # 20
    ev_thread_suspend: int  # 15
    ev_update_bpts: int  # 25
    ev_update_call_stack: int  # 33
    ev_update_lowcnds: int  # 26
    ev_write_file: int  # 30
    ev_write_memory: int  # 23
    ev_write_register: int  # 19
    @property
    def bpt_bytes(self) -> Any: ...
    @property
    def bpt_size(self) -> Any: ...
    @property
    def default_regclasses(self) -> Any: ...
    @property
    def filetype(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def id(self) -> Any: ...
    @property
    def memory_page_size(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def nregisters(self) -> Any: ...
    @property
    def processor(self) -> Any: ...
    @property
    def regclasses(self) -> Any: ...
    @property
    def registers(self) -> Any: ...
    @property
    def resume_modes(self) -> Any: ...
    @property
    def version(self) -> Any: ...
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
    def attach_process(self, pid: pid_t, event_id: int, dbg_proc_flags: int) -> drc_t:
        ...
    def bin_search(self, start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, data: compiled_binpat_vec_t, srch_flags: int) -> drc_t:
        ...
    def cache_block_size(self) -> int:
        ...
    def can_continue_from_bpt(self) -> bool:
        ...
    def can_debug_standalone_dlls(self) -> bool:
        ...
    def check_bpt(self, bptvc: int, type: bpttype_t, ea: ida_idaapi.ea_t, len: int) -> drc_t:
        ...
    def cleanup_appcall(self, tid: thid_t) -> drc_t:
        ...
    def close_file(self, fn: int) -> None:
        ...
    def dbg_enable_trace(self, tid: thid_t, enable: bool, trace_flags: int) -> bool:
        ...
    def detach_process(self) -> drc_t:
        ...
    def eval_lowcnd(self, tid: thid_t, ea: ida_idaapi.ea_t) -> drc_t:
        ...
    def exit_process(self) -> drc_t:
        ...
    def fake_memory(self) -> bool:
        ...
    def get_debapp_attrs(self, out_pattrs: debapp_attrs_t) -> bool:
        ...
    def get_debmod_extensions(self) -> None:
        ...
    def get_debug_event(self, event: debug_event_t, timeout_ms: int) -> gdecode_t:
        ...
    def get_dynamic_register_set(self, regset: dynamic_register_set_t) -> bool:
        ...
    def get_memory_info(self, ranges: meminfo_vec_t) -> drc_t:
        ...
    def get_processes(self, procs: procinfo_vec_t) -> drc_t:
        ...
    def get_srcinfo_path(self, path: str, base: ida_idaapi.ea_t) -> bool:
        ...
    def has_appcall(self) -> bool:
        ...
    def has_attach_process(self) -> bool:
        ...
    def has_check_bpt(self) -> bool:
        ...
    def has_detach_process(self) -> bool:
        ...
    def has_get_processes(self) -> bool:
        ...
    def has_map_address(self) -> bool:
        ...
    def has_open_file(self) -> bool:
        ...
    def has_request_pause(self) -> bool:
        ...
    def has_rexec(self) -> bool:
        ...
    def has_set_exception_info(self) -> bool:
        ...
    def has_set_resume_mode(self) -> bool:
        ...
    def has_soft_bpt(self) -> bool:
        ...
    def has_thread_continue(self) -> bool:
        ...
    def has_thread_get_sreg_base(self) -> bool:
        ...
    def has_thread_suspend(self) -> bool:
        ...
    def has_update_call_stack(self) -> bool:
        ...
    def have_set_options(self) -> bool:
        ...
    def init_debugger(self, hostname: str, portnum: int, password: str) -> bool:
        ...
    def is_remote(self) -> bool:
        ...
    def is_resmod_avail(self, resmod: int) -> bool:
        ...
    def is_safe(self) -> bool:
        ...
    def is_tracing_enabled(self, tid: thid_t, tracebit: int) -> bool:
        ...
    def is_ttd(self) -> bool:
        ...
    def map_address(self, off: ida_idaapi.ea_t, regs: regval_t, regnum: int) -> ida_idaapi.ea_t:
        ...
    def may_disturb(self) -> bool:
        ...
    def may_take_exit_snapshot(self) -> bool:
        ...
    def must_have_hostname(self) -> bool:
        ...
    def open_file(self, file: str, fsize: uint64, readonly: bool) -> int:
        ...
    def read_file(self, fn: int, off: qoff64_t, buf: void, size: size_t) -> ssize_t:
        ...
    def read_memory(self, nbytes: size_t, ea: ida_idaapi.ea_t, buffer: void, size: size_t) -> drc_t:
        ...
    def read_registers(self, tid: thid_t, clsmask: int, values: regval_t) -> drc_t:
        ...
    def rebase_if_required_to(self, new_base: ida_idaapi.ea_t) -> None:
        ...
    def regs(self, idx: int) -> register_info_t:
        ...
    def request_pause(self) -> drc_t:
        ...
    def resume(self, event: debug_event_t) -> drc_t:
        ...
    def rexec(self, cmdline: str) -> int:
        ...
    def send_ioctl(self, fn: int, buf: void, poutbuf: void, poutsize: ssize_t) -> drc_t:
        ...
    def set_backwards(self, backwards: bool) -> drc_t:
        ...
    def set_exception_info(self, info: exception_info_t, qty: int) -> None:
        ...
    def set_resume_mode(self, tid: thid_t, resmod: resume_mode_t) -> drc_t:
        ...
    def start_process(self, path: str, args: str, envs: launch_env_t, startdir: str, dbg_proc_flags: int, input_path: str, input_file_crc32: int) -> drc_t:
        ...
    def supports_debthread(self) -> bool:
        ...
    def supports_lowcnds(self) -> bool:
        ...
    def suspended(self, dlls_added: bool, thr_names: thread_name_vec_t = None) -> None:
        ...
    def term_debugger(self) -> bool:
        ...
    def thread_continue(self, tid: thid_t) -> drc_t:
        ...
    def thread_get_sreg_base(self, answer: ea_t, tid: thid_t, sreg_value: int) -> drc_t:
        ...
    def thread_suspend(self, tid: thid_t) -> drc_t:
        ...
    def update_bpts(self, nbpts: int, bpts: update_bpt_info_t, nadd: int, ndel: int) -> drc_t:
        ...
    def update_call_stack(self, tid: thid_t, trace: call_stack_t) -> drc_t:
        ...
    def update_lowcnds(self, nupdated: int, lowcnds: lowcnd_t, nlowcnds: int) -> drc_t:
        ...
    def use_memregs(self) -> bool:
        ...
    def use_sregs(self) -> bool:
        ...
    def virtual_threads(self) -> bool:
        ...
    def write_file(self, fn: int, off: qoff64_t, buf: void) -> ssize_t:
        ...
    def write_memory(self, nbytes: size_t, ea: ida_idaapi.ea_t, buffer: void, size: size_t) -> drc_t:
        ...
    def write_register(self, tid: thid_t, regidx: int, value: regval_t) -> drc_t:
        ...

class dyn_register_info_array:
    @property
    def count(self) -> Any: ...
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
    def __ge__(self, value: Any) -> Any:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __getitem__(self, i: size_t) -> register_info_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: register_info_t, _count: size_t) -> Any:
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
    def __setitem__(self, i: size_t, v: register_info_t) -> None:
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

class exception_info_t:
    @property
    def code(self) -> Any: ...
    @property
    def desc(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
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
    def break_on(self) -> bool:
        r"""Should we break on the exception?
        
        """
        ...
    def handle(self) -> bool:
        r"""Should we handle the exception?
        
        """
        ...

class excinfo_t:
    @property
    def can_cont(self) -> Any: ...
    @property
    def code(self) -> Any: ...
    @property
    def ea(self) -> Any: ...
    @property
    def info(self) -> Any: ...
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

class excvec_t:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __setitem__(self, i: size_t, v: exception_info_t) -> None:
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
    def append(self, x: exception_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: excvec_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: exception_info_t, len: size_t) -> None:
        ...
    def insert(self, it: exception_info_t, x: exception_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: excvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class launch_env_t:
    @property
    def merge(self) -> Any: ...
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
    def envs(self) -> Any:
        ...
    def set(self, envvar: str, value: str) -> None:
        ...

class meminfo_vec_t(meminfo_vec_template_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: meminfo_vec_template_t) -> bool:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __ne__(self, r: meminfo_vec_template_t) -> bool:
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
    def __setitem__(self, i: size_t, v: memory_info_t) -> None:
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
    def add_unique(self, x: memory_info_t) -> bool:
        ...
    def append(self, x: memory_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: meminfo_vec_template_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: memory_info_t) -> bool:
        ...
    def inject(self, s: memory_info_t, len: size_t) -> None:
        ...
    def insert(self, it: memory_info_t, x: memory_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: meminfo_vec_template_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class meminfo_vec_template_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: meminfo_vec_template_t) -> bool:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __ne__(self, r: meminfo_vec_template_t) -> bool:
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
    def __setitem__(self, i: size_t, v: memory_info_t) -> None:
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
    def add_unique(self, x: memory_info_t) -> bool:
        ...
    def append(self, x: memory_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: meminfo_vec_template_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: memory_info_t) -> bool:
        ...
    def inject(self, s: memory_info_t, len: size_t) -> None:
        ...
    def insert(self, it: memory_info_t, x: memory_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: meminfo_vec_template_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class memory_info_t:
    @property
    def bitness(self) -> Any: ...
    @property
    def end_ea(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def perm(self) -> Any: ...
    @property
    def sbase(self) -> Any: ...
    @property
    def sclass(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: memory_info_t) -> bool:
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
    def __ne__(self, r: memory_info_t) -> bool:
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

class modinfo_t:
    @property
    def base(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def rebase_to(self) -> Any: ...
    @property
    def size(self) -> Any: ...
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

class process_info_t:
    @property
    def name(self) -> Any: ...
    @property
    def pid(self) -> Any: ...
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

class procinfo_vec_t:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __setitem__(self, i: size_t, v: process_info_t) -> None:
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
    def append(self, x: process_info_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: procinfo_vec_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: process_info_t, len: size_t) -> None:
        ...
    def insert(self, it: process_info_t, x: process_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: procinfo_vec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class register_info_t:
    @property
    def bit_strings(self) -> Any: ...
    @property
    def default_bit_strings_mask(self) -> Any: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def register_class(self) -> Any: ...
    @property
    def register_class_mask(self) -> Any: ...
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

class regval_t:
    @property
    def ival(self) -> Any: ...
    @property
    def rvtype(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: regval_t) -> bool:
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
    def __ne__(self, r: regval_t) -> bool:
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
    def bytes(self, args: Any) -> bytevec_t:
        ...
    def clear(self) -> None:
        r"""Clear register value.
        
        """
        ...
    def get_data(self, args: Any) -> None:
        ...
    def get_data_size(self) -> int:
        ...
    def pyval(self, dtype: op_dtype_t) -> Any:
        ...
    def set_bytes(self, args: Any) -> bytevec_t:
        ...
    def set_float(self, v: bytevec_t) -> None:
        ...
    def set_int(self, x: uint64) -> None:
        ...
    def set_pyval(self, o: Any, dtype: op_dtype_t) -> bool:
        ...
    def set_unavailable(self) -> None:
        ...
    def swap(self, r: regval_t) -> None:
        r"""Set this = r and r = this.
        
        """
        ...
    def use_bytevec(self) -> bool:
        ...

class regvals_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: regvals_t) -> bool:
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
    def __getitem__(self, i: size_t) -> register_info_t:
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
    def __ne__(self, r: regvals_t) -> bool:
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
    def __setitem__(self, i: size_t, v: regval_t) -> None:
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
    def add_unique(self, x: regval_t) -> bool:
        ...
    def append(self, x: regval_t) -> None:
        ...
    def at(self, _idx: size_t) -> regval_t:
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
    def extend(self, x: regvals_t) -> None:
        ...
    def extract(self) -> regval_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: regval_t) -> bool:
        ...
    def inject(self, s: regval_t, len: size_t) -> None:
        ...
    def insert(self, it: regval_t, x: regval_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> regval_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: regvals_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class scattered_segm_t:
    @property
    def end_ea(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
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

class thread_name_t:
    @property
    def name(self) -> Any: ...
    @property
    def tid(self) -> Any: ...
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

def appcall(func_ea: ida_idaapi.ea_t, tid: thid_t, _type_or_none: bytevec_t, _fields: bytevec_t, arg_list: Any) -> Any:
    ...

def can_exc_continue(ev: debug_event_t) -> bool:
    ...

def cleanup_appcall(tid: thid_t) -> drc_t:
    r"""Cleanup after manual appcall. 
            
    :param tid: thread to use. NO_THREAD means to use the current thread The application state is restored as it was before calling the last appcall(). Nested appcalls are supported.
    :returns: eOk if successful, otherwise an error code
    """
    ...

def cpu2ieee(ieee_out: fpvalue_t, cpu_fpval: void, size: int) -> int:
    r"""Convert a floating point number in CPU native format to IDA's internal format. 
            
    :param ieee_out: output buffer
    :param cpu_fpval: floating point number in CPU native format
    :param size: size of cpu_fpval in bytes (size of the input buffer)
    :returns: Floating point/IEEE Conversion codes
    """
    ...

def dbg_appcall(retval: idc_value_t, func_ea: ida_idaapi.ea_t, tid: thid_t, ptif: tinfo_t, argv: idc_value_t, argnum: size_t) -> error_t:
    r"""Call a function from the debugged application. 
            
    :param retval: function return value
    * for APPCALL_MANUAL, r will hold the new stack point value
    * for APPCALL_DEBEV, r will hold the exception information upon failure and the return code will be eExecThrow
    :param func_ea: address to call
    :param tid: thread to use. NO_THREAD means to use the current thread
    :param ptif: pointer to type of the function to call
    :param argv: array of arguments
    :param argnum: number of actual arguments
    :returns: eOk if successful, otherwise an error code
    """
    ...

def dbg_can_query(*args: Any, **kwargs: Any) -> Any:
    ...

def dbg_get_memory_info() -> Any:
    r"""This function returns the memory configuration of a debugged process.
    
    :returns: tuple(start_ea, end_ea, name, sclass, sbase, bitness, perm), or None if no debugger is active
    """
    ...

def dbg_get_name() -> Any:
    r"""This function returns the current debugger's name.
    
    :returns: Debugger name or None if no debugger is active
    """
    ...

def dbg_get_registers() -> Any:
    r"""This function returns the register definition from the currently loaded debugger.
    Basically, it returns an array of structure similar to to idd.hpp / register_info_t
    
    :returns: None if no debugger is loaded
    :returns: tuple(name, flags, class, dtype, bit_strings, default_bit_strings_mask)
              The bit_strings can be a tuple of strings or None (if the register does not have bit_strings)
    """
    ...

def dbg_get_thread_sreg_base(tid: Any, sreg_value: Any) -> Any:
    r"""Returns the segment register base value
    
    :param tid: thread id
    :param sreg_value: segment register (selector) value
    :returns: The base as an 'ea', or None on failure
    """
    ...

def dbg_read_memory(ea: Any, sz: Any) -> Any:
    r"""Reads from the debugee's memory at the specified ea
    
    :param ea: the debuggee's memory address
    :param sz: the amount of data to read
    :returns: The read buffer (as bytes), or None on failure
    """
    ...

def dbg_write_memory(ea: Any, buffer: Any) -> Any:
    r"""Writes a buffer to the debugee's memory
    
    :param ea: the debuggee's memory address
    :param buf: a bytes object to write
    :returns: Boolean
    """
    ...

def get_dbg() -> debugger_t:
    ...

def get_debug_event_name(dev: debug_event_t) -> str:
    r"""get debug event name
    
    """
    ...

def get_event_bpt_hea(ev: debug_event_t) -> ida_idaapi.ea_t:
    ...

def get_event_exc_code(ev: debug_event_t) -> uint:
    ...

def get_event_exc_ea(ev: debug_event_t) -> ida_idaapi.ea_t:
    ...

def get_event_exc_info(ev: debug_event_t) -> str:
    ...

def get_event_info(ev: debug_event_t) -> str:
    ...

def get_event_module_base(ev: debug_event_t) -> ida_idaapi.ea_t:
    ...

def get_event_module_name(ev: debug_event_t) -> str:
    ...

def get_event_module_size(ev: debug_event_t) -> int:
    ...

def ieee2cpu(cpu_fpval_out: void, ieee: fpvalue_t, size: int) -> int:
    r"""Convert a floating point number in IDA's internal format to CPU native format. 
            
    :param cpu_fpval_out: output buffer
    :param ieee: floating point number of IDA's internal format
    :param size: size of cpu_fpval in bytes (size of the output buffer)
    :returns: Floating point/IEEE Conversion codes
    """
    ...

def set_debug_event_code(ev: debug_event_t, id: event_id_t) -> None:
    ...

APPCALL_DEBEV: int  # 2
APPCALL_MANUAL: int  # 1
APPCALL_TIMEOUT: int  # 4
Appcall: Appcall__  # <ida_idd.Appcall__ object at 0x717729761cf0>
BBLK_TRACE: int  # 8
BITNESS_CHANGED: int  # -2147483648
BPT_BAD_ADDR: int  # 4
BPT_BAD_ALIGN: int  # 3
BPT_BAD_LEN: int  # 5
BPT_BAD_TYPE: int  # 2
BPT_DEFAULT: int  # 12
BPT_EXEC: int  # 8
BPT_INTERNAL_ERR: int  # 1
BPT_OK: int  # 0
BPT_PAGE_OK: int  # 10
BPT_RDWR: int  # 3
BPT_READ: int  # 2
BPT_READ_ERROR: int  # 7
BPT_SKIP: int  # 9
BPT_SOFT: int  # 4
BPT_TOO_MANY: int  # 6
BPT_WRITE: int  # 1
BPT_WRITE_ERROR: int  # 8
BREAKPOINT: int  # 5
DBG_FLAG_ADD_ENVS: int  # 134217728
DBG_FLAG_ANYSIZE_HWBPT: int  # 4194304
DBG_FLAG_CAN_CONT_BPT: int  # 16
DBG_FLAG_CLEAN_EXIT: int  # 256
DBG_FLAG_CONNSTRING: int  # 8192
DBG_FLAG_DEBTHREAD: int  # 524288
DBG_FLAG_DEBUG_DLL: int  # 1048576
DBG_FLAG_DISABLE_ASLR: int  # 536870912
DBG_FLAG_DONT_DISTURB: int  # 64
DBG_FLAG_EXITSHOTOK: int  # 65536
DBG_FLAG_FAKE_ATTACH: int  # 4
DBG_FLAG_FAKE_MEMORY: int  # 2097152
DBG_FLAG_FAST_STEP: int  # 67108864
DBG_FLAG_FULL_INSTR_BPT: int  # 2147483648
DBG_FLAG_HWDATBPT_ONE: int  # 8
DBG_FLAG_LAZY_WATCHPTS: int  # 33554432
DBG_FLAG_LOWCNDS: int  # 262144
DBG_FLAG_MANMEMINFO: int  # 32768
DBG_FLAG_MERGE_ENVS: int  # 268435456
DBG_FLAG_NEEDPORT: int  # 32
DBG_FLAG_NOHOST: int  # 2
DBG_FLAG_NOPARAMETERS: int  # 2048
DBG_FLAG_NOPASSWORD: int  # 4096
DBG_FLAG_NOSTARTDIR: int  # 1024
DBG_FLAG_PREFER_SWBPTS: int  # 16777216
DBG_FLAG_REMOTE: int  # 1
DBG_FLAG_SAFE: int  # 128
DBG_FLAG_SMALLBLKS: int  # 16384
DBG_FLAG_TRACER_MODULE: int  # 8388608
DBG_FLAG_TTD: int  # 1073741824
DBG_FLAG_USE_SREGS: int  # 512
DBG_FLAG_VIRTHREADS: int  # 131072
DBG_HAS_APPCALL: int  # 17592186044416
DBG_HAS_ATTACH_PROCESS: int  # 8589934592
DBG_HAS_CHECK_BPT: int  # 2199023255552
DBG_HAS_DETACH_PROCESS: int  # 17179869184
DBG_HAS_GET_PROCESSES: int  # 4294967296
DBG_HAS_MAP_ADDRESS: int  # 70368744177664
DBG_HAS_OPEN_FILE: int  # 4398046511104
DBG_HAS_REQUEST_PAUSE: int  # 34359738368
DBG_HAS_REXEC: int  # 35184372088832
DBG_HAS_SET_EXCEPTION_INFO: int  # 68719476736
DBG_HAS_SET_RESUME_MODE: int  # 549755813888
DBG_HAS_THREAD_CONTINUE: int  # 274877906944
DBG_HAS_THREAD_GET_SREG_BASE: int  # 1099511627776
DBG_HAS_THREAD_SUSPEND: int  # 137438953472
DBG_HAS_UPDATE_CALL_STACK: int  # 8796093022208
DBG_HIDE_WINDOW: int  # 32
DBG_NO_ASLR: int  # 128
DBG_NO_TRACE: int  # 16
DBG_PROC_32BIT: int  # 4
DBG_PROC_64BIT: int  # 8
DBG_PROC_IS_DLL: int  # 1
DBG_PROC_IS_GUI: int  # 2
DBG_RESMOD_STEP_BACKINTO: int  # 256
DBG_RESMOD_STEP_HANDLE: int  # 128
DBG_RESMOD_STEP_INTO: int  # 1
DBG_RESMOD_STEP_OUT: int  # 4
DBG_RESMOD_STEP_OVER: int  # 2
DBG_RESMOD_STEP_SRCINTO: int  # 8
DBG_RESMOD_STEP_SRCOUT: int  # 32
DBG_RESMOD_STEP_SRCOVER: int  # 16
DBG_RESMOD_STEP_USER: int  # 64
DBG_SUSPENDED: int  # 64
DEBUGGER_ID_6811_EMULATOR: int  # 7
DEBUGGER_ID_ARM_IPHONE_USER: int  # 5
DEBUGGER_ID_ARM_LINUX_USER: int  # 11
DEBUGGER_ID_ARM_MACOS_USER: int  # 17
DEBUGGER_ID_DALVIK_USER: int  # 15
DEBUGGER_ID_GDB_USER: int  # 8
DEBUGGER_ID_TRACE_REPLAYER: int  # 12
DEBUGGER_ID_WINDBG: int  # 9
DEBUGGER_ID_X86_DOSBOX_EMULATOR: int  # 10
DEBUGGER_ID_X86_IA32_BOCHS: int  # 6
DEBUGGER_ID_X86_IA32_LINUX_USER: int  # 1
DEBUGGER_ID_X86_IA32_MACOSX_USER: int  # 3
DEBUGGER_ID_X86_IA32_WIN32_USER: int  # 0
DEBUGGER_ID_X86_PIN_TRACER: int  # 14
DEBUGGER_ID_XNU_USER: int  # 16
DEF_ADDRSIZE: int  # 8
DRC_CRC: int  # 2
DRC_ERROR: int  # -7
DRC_EVENTS: int  # 3
DRC_FAILED: int  # -1
DRC_IDBSEG: int  # -4
DRC_NETERR: int  # -2
DRC_NOCHG: int  # -6
DRC_NOFILE: int  # -3
DRC_NONE: int  # 0
DRC_NOPROC: int  # -5
DRC_OK: int  # 1
EXCEPTION: int  # 7
EXC_BREAK: int  # 1
EXC_HANDLE: int  # 2
EXC_MSG: int  # 4
EXC_SILENT: int  # 8
FUNC_TRACE: int  # 4
IDD_INTERFACE_VERSION: int  # 31
INFORMATION: int  # 10
INSN_TRACE: int  # 2
LIB_LOADED: int  # 8
LIB_UNLOADED: int  # 9
NO_EVENT: int  # 0
NO_PROCESS: int  # 4294967295
NO_THREAD: int  # 0
PROCESS_ATTACHED: int  # 11
PROCESS_DETACHED: int  # 12
PROCESS_EXITED: int  # 2
PROCESS_STARTED: int  # 1
PROCESS_SUSPENDED: int  # 13
REGISTER_ADDRESS: int  # 16
REGISTER_CS: int  # 32
REGISTER_CUSTFMT: int  # 256
REGISTER_FP: int  # 8
REGISTER_IP: int  # 2
REGISTER_NOLF: int  # 128
REGISTER_READONLY: int  # 1
REGISTER_SP: int  # 4
REGISTER_SS: int  # 64
RESMOD_BACKINTO: int  # 9
RESMOD_HANDLE: int  # 8
RESMOD_INTO: int  # 1
RESMOD_MAX: int  # 10
RESMOD_NONE: int  # 0
RESMOD_OUT: int  # 3
RESMOD_OVER: int  # 2
RESMOD_SRCINTO: int  # 4
RESMOD_SRCOUT: int  # 6
RESMOD_SRCOVER: int  # 5
RESMOD_USER: int  # 7
RQ_IDAIDLE: int  # 128
RQ_IGNWERR: int  # 4
RQ_MASKING: int  # 1
RQ_NOSUSP: int  # 0
RQ_PROCEXIT: int  # 64
RQ_RESMOD: int  # 61440
RQ_RESMOD_SHIFT: int  # 12
RQ_RESUME: int  # 512
RQ_SILENT: int  # 8
RQ_SUSPEND: int  # 2
RQ_SUSPRUN: int  # 256
RQ_SWSCREEN: int  # 16
RQ_VERBOSE: int  # 0
RQ__NOTHRRF: int  # 32
RVT_FLOAT: int  # -1
RVT_INT: int  # -2
RVT_UNAVAILABLE: int  # -3
STATUS_MASK: int  # -268435456
STEP: int  # 6
STEP_TRACE: int  # 1
SWIG_PYTHON_LEGACY_BOOL: int  # 1
THREAD_EXITED: int  # 4
THREAD_STARTED: int  # 3
TRACE_FULL: int  # 14
annotations: _Feature
cvar: swigvarlink
ida_idaapi: module
ida_range: module
ida_typeinf: module
types: module
weakref: module