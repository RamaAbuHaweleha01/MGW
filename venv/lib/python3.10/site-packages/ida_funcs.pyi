from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Routines for working with functions within the disassembled program.

This file also contains routines for working with library signatures (e.g. FLIRT).
Each function consists of function chunks. At least one function chunk must be present in the function definition - the function entry chunk. Other chunks are called function tails. There may be several of them for a function.
A function tail is a continuous range of addresses. It can be used in the definition of one or more functions. One function using the tail is singled out and called the tail owner. This function is considered as 'possessing' the tail. get_func() on a tail address will return the function possessing the tail. You can enumerate the functions using the tail by using func_parent_iterator_t.
Each function chunk in the disassembly is represented as an "range" (a range of addresses, see range.hpp for details) with characteristics.
A function entry must start with an instruction (code) byte. 
    
"""

class dyn_ea_array:
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
    def __getitem__(self, i: size_t) -> regarg_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: int, _count: size_t) -> Any:
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

class dyn_range_array:
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
    def __getitem__(self, i: size_t) -> regarg_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: range_t, _count: size_t) -> Any:
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
    def __setitem__(self, i: size_t, v: range_t) -> None:
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

class dyn_regarg_array:
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
    def __getitem__(self, i: size_t) -> regarg_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: regarg_t, _count: size_t) -> Any:
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
    def __setitem__(self, i: size_t, v: regarg_t) -> None:
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

class dyn_regvar_array:
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
    def __getitem__(self, i: size_t) -> regarg_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: regvar_t, _count: size_t) -> Any:
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
    def __setitem__(self, i: size_t, v: regvar_t) -> None:
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

class dyn_stkpnt_array:
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
    def __getitem__(self, i: size_t) -> regarg_t:
        ...
    def __gt__(self, value: Any) -> Any:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _data: stkpnt_t, _count: size_t) -> Any:
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
    def __setitem__(self, i: size_t, v: stkpnt_t) -> None:
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

class func_item_iterator_t:
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
    def __iter__(self) -> Any:
        r"""
                Provide an iterator on code items
                
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
    def __next__(self, func: testf_t) -> bool:
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
    def addresses(self) -> Any:
        r"""
                Provide an iterator on addresses contained within the function
                
        """
        ...
    def chunk(self) -> range_t:
        ...
    def code_items(self) -> Any:
        r"""
                Provide an iterator on code items contained within the function
                
        """
        ...
    def current(self) -> ida_idaapi.ea_t:
        ...
    def data_items(self) -> Any:
        r"""
                Provide an iterator on data items contained within the function
                
        """
        ...
    def decode_preceding_insn(self, visited: eavec_t, p_farref: bool, out: insn_t) -> bool:
        ...
    def decode_prev_insn(self, out: insn_t) -> bool:
        ...
    def first(self) -> bool:
        ...
    def head_items(self) -> Any:
        r"""
                Provide an iterator on item heads contained within the function
                
        """
        ...
    def last(self) -> bool:
        ...
    def next(self, func: testf_t) -> bool:
        ...
    def next_addr(self) -> bool:
        ...
    def next_code(self) -> bool:
        ...
    def next_data(self) -> bool:
        ...
    def next_head(self) -> bool:
        ...
    def next_not_tail(self) -> bool:
        ...
    def not_tails(self) -> Any:
        r"""
                Provide an iterator on non-tail addresses contained within the function
                
        """
        ...
    def prev(self, func: testf_t) -> bool:
        ...
    def prev_addr(self) -> bool:
        ...
    def prev_code(self) -> bool:
        ...
    def prev_data(self) -> bool:
        ...
    def prev_head(self) -> bool:
        ...
    def prev_not_tail(self) -> bool:
        ...
    def set(self, args: Any) -> bool:
        r"""Set a function range. if pfn == nullptr then a segment range will be set.
        
        """
        ...
    def set_ea(self, _ea: ida_idaapi.ea_t) -> bool:
        ...
    def set_range(self, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> bool:
        r"""Set an arbitrary range.
        
        """
        ...
    def succ(self, func: testf_t) -> bool:
        r"""Similar to next(), but succ() iterates the chunks from low to high addresses, while next() iterates through chunks starting at the function entry chunk 
                
        """
        ...
    def succ_code(self) -> bool:
        ...

class func_parent_iterator_t:
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
    def __iter__(self) -> Any:
        r"""
                Provide an iterator on function parents
                
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
    def __next__(self) -> bool:
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
    def first(self) -> bool:
        ...
    def last(self) -> bool:
        ...
    def next(self) -> bool:
        ...
    def parent(self) -> ida_idaapi.ea_t:
        ...
    def prev(self) -> bool:
        ...
    def reset_fnt(self, _fnt: func_t) -> None:
        ...
    def set(self, _fnt: func_t) -> bool:
        ...

class func_t:
    @property
    def argsize(self) -> Any: ...
    @property
    def color(self) -> Any: ...
    @property
    def end_ea(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def fpd(self) -> Any: ...
    @property
    def frame(self) -> Any: ...
    @property
    def frame_object(self) -> Any: ...
    @property
    def frregs(self) -> Any: ...
    @property
    def frsize(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def owner(self) -> Any: ...
    @property
    def pntqty(self) -> Any: ...
    @property
    def points(self) -> Any: ...
    @property
    def prototype(self) -> Any: ...
    @property
    def referers(self) -> Any: ...
    @property
    def refqty(self) -> Any: ...
    @property
    def regargqty(self) -> Any: ...
    @property
    def regargs(self) -> Any: ...
    @property
    def regvarqty(self) -> Any: ...
    @property
    def regvars(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def tailqty(self) -> Any: ...
    @property
    def tails(self) -> Any: ...
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
    def __get_points__(self) -> dynamic_wrapped_array_t:
        ...
    def __get_referers__(self) -> dynamic_wrapped_array_t:
        ...
    def __get_regargs__(self) -> dynamic_wrapped_array_t:
        ...
    def __get_regvars__(self) -> dynamic_wrapped_array_t:
        ...
    def __get_tails__(self) -> dynamic_wrapped_array_t:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: range_t) -> bool:
        ...
    def __init__(self, start: ida_idaapi.ea_t = 0, end: ida_idaapi.ea_t = 0, f: flags64_t = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __iter__(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).__iter__()
                
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
    def addresses(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).addresses()
                
        """
        ...
    def analyzed_sp(self) -> bool:
        r"""Has SP-analysis been performed?
        
        """
        ...
    def clear(self) -> None:
        r"""Set start_ea, end_ea to 0.
        
        """
        ...
    def code_items(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).code_items()
                
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
    def data_items(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).data_items()
                
        """
        ...
    def does_return(self) -> bool:
        r"""Does function return?
        
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
    def get_frame_object(self) -> Any:
        r"""Retrieve the function frame, in the form of a structure
        where frame offsets that are accessed by the program, as well
        as areas for "saved registers" and "return address", are
        represented by structure members.
        
        If the function has no associated frame, return None
        
        :returns: a ida_typeinf.tinfo_t object representing the frame, or None
        """
        ...
    def get_name(self) -> Any:
        r"""Get the function name
        
        :returns: the function name
        """
        ...
    def get_prototype(self) -> Any:
        r"""Retrieve the function prototype.
        
        Once you have obtained the prototype, you can:
        
        * retrieve the return type through ida_typeinf.tinfo_t.get_rettype()
        * iterate on the arguments using ida_typeinf.tinfo_t.iter_func()
        
        If the function has no associated prototype, return None
        
        :returns: a ida_typeinf.tinfo_t object representing the prototype, or None
        """
        ...
    def head_items(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).head_items()
                
        """
        ...
    def intersect(self, r: range_t) -> None:
        r"""Assign the range_t to the intersection between the range_t and 'r'.
        
        """
        ...
    def is_far(self) -> bool:
        r"""Is a far function?
        
        """
        ...
    def need_prolog_analysis(self) -> bool:
        r"""Needs prolog analysis?
        
        """
        ...
    def not_tails(self) -> Any:
        r"""
                Alias for func_item_iterator_t(self).not_tails()
                
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

class func_tail_iterator_t:
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
    def __iter__(self) -> Any:
        r"""
                Provide an iterator on function tails
                
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
    def __next__(self) -> bool:
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
    def chunk(self) -> range_t:
        ...
    def first(self) -> bool:
        ...
    def last(self) -> bool:
        ...
    def main(self) -> bool:
        ...
    def next(self) -> bool:
        ...
    def prev(self) -> bool:
        ...
    def set(self, args: Any) -> bool:
        ...
    def set_ea(self, ea: ida_idaapi.ea_t) -> bool:
        ...
    def set_range(self, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> bool:
        ...

class lock_func:
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
    def __init__(self, _pfn: func_t) -> Any:
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

class lock_func_with_tails_t:
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
    def __init__(self, pfn: func_t) -> Any:
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

class regarg_t:
    @property
    def name(self) -> Any: ...
    @property
    def reg(self) -> Any: ...
    @property
    def type(self) -> Any: ...
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
    def swap(self, r: regarg_t) -> None:
        ...

def add_func(args: Any) -> bool:
    r"""Add a new function. If the function end address is BADADDR, then IDA will try to determine the function bounds by calling find_func_bounds(..., FIND_FUNC_DEFINE). 
            
    :param ea1: start address
    :param ea2: end address
    :returns: success
    """
    ...

def add_func_ex(pfn: func_t) -> bool:
    r"""Add a new function. If the fn->end_ea is BADADDR, then IDA will try to determine the function bounds by calling find_func_bounds(..., FIND_FUNC_DEFINE). 
            
    :param pfn: ptr to filled function structure
    :returns: success
    """
    ...

def add_regarg(pfn: func_t, reg: int, tif: tinfo_t, name: str) -> None:
    ...

def append_func_tail(pfn: func_t, ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> bool:
    r"""Append a new tail chunk to the function definition. If the tail already exists, then it will simply be added to the function tail list Otherwise a new tail will be created and its owner will be set to be our function If a new tail cannot be created, then this function will fail. 
            
    :param pfn: pointer to the function
    :param ea1: start of the tail. If a tail already exists at the specified address it must start at 'ea1'
    :param ea2: end of the tail. If a tail already exists at the specified address it must end at 'ea2'. If specified as BADADDR, IDA will determine the end address itself.
    """
    ...

def apply_idasgn_to(signame: str, ea: ida_idaapi.ea_t, is_startup: bool) -> int:
    r"""Apply a signature file to the specified address. 
            
    :param signame: short name of signature file (the file name without path)
    :param ea: address to apply the signature
    :param is_startup: if set, then the signature is treated as a startup one for startup signature ida doesn't rename the first function of the applied module.
    :returns: Library function codes
    """
    ...

def apply_startup_sig(ea: ida_idaapi.ea_t, startup: str) -> bool:
    r"""Apply a startup signature file to the specified address. 
            
    :param ea: address to apply the signature to; usually idainfo::start_ea
    :param startup: the name of the signature file without path and extension
    :returns: true if successfully applied the signature
    """
    ...

def calc_func_size(pfn: func_t) -> int:
    r"""Calculate function size. This function takes into account all fragments of the function. 
            
    :param pfn: ptr to function structure
    """
    ...

def calc_idasgn_state(n: int) -> int:
    r"""Get state of a signature in the list of planned signatures 
            
    :param n: number of signature in the list (0..get_idasgn_qty()-1)
    :returns: state of signature or IDASGN_BADARG
    """
    ...

def calc_thunk_func_target(args: Any) -> int:
    r"""Calculate target of a thunk function. 
            
    :param pfn: pointer to function (may not be nullptr)
    :returns: the target function or BADADDR
    """
    ...

def del_func(ea: ida_idaapi.ea_t) -> bool:
    r"""Delete a function. 
            
    :param ea: any address in the function entry chunk
    :returns: success
    """
    ...

def del_idasgn(n: int) -> int:
    r"""Remove signature from the list of planned signatures. 
            
    :param n: number of signature in the list (0..get_idasgn_qty()-1)
    :returns: IDASGN_OK, IDASGN_BADARG, IDASGN_APPLIED
    """
    ...

def f_any(arg1: flags64_t, arg2: void) -> bool:
    r"""Helper function to accept any address.
    
    """
    ...

def find_func_bounds(nfn: func_t, flags: int) -> int:
    r"""Determine the boundaries of a new function. This function tries to find the start and end addresses of a new function. It calls the module with processor_t::func_bounds in order to fine tune the function boundaries. 
            
    :param nfn: structure to fill with information \ nfn->start_ea points to the start address of the new function.
    :param flags: Find function bounds flags
    :returns: Find function bounds result codes
    """
    ...

def free_regarg(v: regarg_t) -> None:
    ...

def func_contains(pfn: func_t, ea: ida_idaapi.ea_t) -> bool:
    r"""Does the given function contain the given address?
    
    """
    ...

def func_does_return(callee: ida_idaapi.ea_t) -> bool:
    r"""Does the function return?. To calculate the answer, FUNC_NORET flag and is_noret() are consulted The latter is required for imported functions in the .idata section. Since in .idata we have only function pointers but not functions, we have to introduce a special flag for them. 
            
    """
    ...

def func_parent_iterator_set(fpi: func_parent_iterator_t, pfn: func_t) -> bool:
    ...

def func_t__from_ptrval__(ptrval: size_t) -> Optional[func_t]:
    ...

def func_tail_iterator_set(fti: func_tail_iterator_t, pfn: func_t, ea: ida_idaapi.ea_t) -> bool:
    ...

def func_tail_iterator_set_ea(fti: func_tail_iterator_t, ea: ida_idaapi.ea_t) -> bool:
    ...

def get_current_idasgn() -> int:
    r"""Get number of the the current signature. 
            
    :returns: 0..n-1
    """
    ...

def get_fchunk(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to function chunk structure by address. 
            
    :param ea: any address in a function chunk
    :returns: ptr to a function chunk or nullptr. This function may return a function entry as well as a function tail.
    """
    ...

def get_fchunk_num(ea: ida_idaapi.ea_t) -> int:
    r"""Get ordinal number of a function chunk in the global list of function chunks. 
            
    :param ea: any address in the function chunk
    :returns: number of function chunk (0..get_fchunk_qty()-1). -1 means 'no function chunk at the specified address'.
    """
    ...

def get_fchunk_qty() -> int:
    r"""Get total number of function chunks in the program.
    
    """
    ...

def get_fchunk_referer(ea: int, idx: Any) -> Any:
    ...

def get_func(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to function structure by address. 
            
    :param ea: any address in a function
    :returns: ptr to a function or nullptr. This function returns a function entry chunk.
    """
    ...

def get_func_bitness(pfn: func_t) -> int:
    r"""Get function bitness (which is equal to the function segment bitness). pfn==nullptr => returns 0 
            
    :returns: 0: 16
    :returns: 1: 32
    :returns: 2: 64
    """
    ...

def get_func_bits(pfn: func_t) -> int:
    r"""Get number of bits in the function addressing.
    
    """
    ...

def get_func_bytes(pfn: func_t) -> int:
    r"""Get number of bytes in the function addressing.
    
    """
    ...

def get_func_chunknum(pfn: func_t, ea: ida_idaapi.ea_t) -> int:
    r"""Get the containing tail chunk of 'ea'. 
            
    :returns: -1: means 'does not contain ea'
    :returns: 0: means the 'pfn' itself contains ea
    :returns: >0: the number of the containing function tail chunk
    """
    ...

def get_func_cmt(pfn: func_t, repeatable: bool) -> str:
    r"""Get function comment. 
            
    :param pfn: ptr to function structure
    :param repeatable: get repeatable comment?
    :returns: size of comment or -1 In fact this function works with function chunks too.
    """
    ...

def get_func_name(ea: ida_idaapi.ea_t) -> str:
    r"""Get function name. 
            
    :param ea: any address in the function
    :returns: length of the function name
    """
    ...

def get_func_num(ea: ida_idaapi.ea_t) -> int:
    r"""Get ordinal number of a function. 
            
    :param ea: any address in the function
    :returns: number of function (0..get_func_qty()-1). -1 means 'no function at the specified address'.
    """
    ...

def get_func_qty() -> int:
    r"""Get total number of functions in the program.
    
    """
    ...

def get_func_ranges(ranges: rangeset_t, pfn: func_t) -> ida_idaapi.ea_t:
    r"""Get function ranges. 
            
    :param ranges: buffer to receive the range info
    :param pfn: ptr to function structure
    :returns: end address of the last function range (BADADDR-error)
    """
    ...

def get_idasgn_desc(n: Any) -> Any:
    r"""Get information about a signature in the list.
    It returns: (name of signature, names of optional libraries)
    
    See also: get_idasgn_desc_with_matches
    
    :param n: number of signature in the list (0..get_idasgn_qty()-1)
    :returns: None on failure or tuple(signame, optlibs)
    """
    ...

def get_idasgn_desc_with_matches(n: Any) -> Any:
    r"""Get information about a signature in the list.
    It returns: (name of signature, names of optional libraries, number of matches)
    
    :param n: number of signature in the list (0..get_idasgn_qty()-1)
    :returns: None on failure or tuple(signame, optlibs, nmatches)
    """
    ...

def get_idasgn_qty() -> int:
    r"""Get number of signatures in the list of planned and applied signatures. 
            
    :returns: 0..n
    """
    ...

def get_idasgn_title(name: str) -> str:
    r"""Get full description of the signature by its short name. 
            
    :param name: short name of a signature
    :returns: size of signature description or -1
    """
    ...

def get_next_fchunk(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to the next function chunk in the global list. 
            
    :param ea: any address in the program
    :returns: ptr to function chunk or nullptr if next function chunk doesn't exist
    """
    ...

def get_next_func(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to the next function. 
            
    :param ea: any address in the program
    :returns: ptr to function or nullptr if next function doesn't exist
    """
    ...

def get_next_func_addr(pfn: func_t, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_prev_fchunk(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to the previous function chunk in the global list. 
            
    :param ea: any address in the program
    :returns: ptr to function chunk or nullptr if previous function chunk doesn't exist
    """
    ...

def get_prev_func(ea: ida_idaapi.ea_t) -> Optional[func_t]:
    r"""Get pointer to the previous function. 
            
    :param ea: any address in the program
    :returns: ptr to function or nullptr if previous function doesn't exist
    """
    ...

def get_prev_func_addr(pfn: func_t, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def getn_fchunk(n: int) -> Optional[func_t]:
    r"""Get pointer to function chunk structure by number. 
            
    :param n: number of function chunk, is in range 0..get_fchunk_qty()-1
    :returns: ptr to a function chunk or nullptr. This function may return a function entry as well as a function tail.
    """
    ...

def getn_func(n: size_t) -> Optional[func_t]:
    r"""Get pointer to function structure by number. 
            
    :param n: number of function, is in range 0..get_func_qty()-1
    :returns: ptr to a function or nullptr. This function returns a function entry chunk.
    """
    ...

def is_finally_visible_func(pfn: func_t) -> bool:
    r"""Is the function visible (event after considering SCF_SHHID_FUNC)?
    
    """
    ...

def is_func_entry(pfn: func_t) -> bool:
    r"""Does function describe a function entry chunk?
    
    """
    ...

def is_func_locked(pfn: func_t) -> bool:
    r"""Is the function pointer locked?
    
    """
    ...

def is_func_tail(pfn: func_t) -> bool:
    r"""Does function describe a function tail chunk?
    
    """
    ...

def is_same_func(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t) -> bool:
    r"""Do two addresses belong to the same function?
    
    """
    ...

def is_visible_func(pfn: func_t) -> bool:
    r"""Is the function visible (not hidden)?
    
    """
    ...

def lock_func_range(pfn: func_t, lock: bool) -> None:
    r"""Lock function pointer Locked pointers are guaranteed to remain valid until they are unlocked. Ranges with locked pointers cannot be deleted or moved. 
            
    """
    ...

def plan_to_apply_idasgn(fname: str) -> int:
    r"""Add a signature file to the list of planned signature files. 
            
    :param fname: file name. should not contain directory part.
    :returns: 0 if failed, otherwise number of planned (and applied) signatures
    """
    ...

def read_regargs(pfn: func_t) -> None:
    ...

def reanalyze_function(args: Any) -> None:
    r"""Reanalyze a function. This function plans to analyzes all chunks of the given function. Optional parameters (ea1, ea2) may be used to narrow the analyzed range. 
            
    :param pfn: pointer to a function
    :param ea1: start of the range to analyze
    :param ea2: end of range to analyze
    :param analyze_parents: meaningful only if pfn points to a function tail. if true, all tail parents will be reanalyzed. if false, only the given tail will be reanalyzed.
    """
    ...

def reanalyze_noret_flag(ea: ida_idaapi.ea_t) -> bool:
    r"""Plan to reanalyze noret flag. This function does not remove FUNC_NORET if it is already present. It just plans to reanalysis. 
            
    """
    ...

def remove_func_tail(pfn: func_t, tail_ea: ida_idaapi.ea_t) -> bool:
    r"""Remove a function tail. If the tail belongs only to one function, it will be completely removed. Otherwise if the function was the tail owner, the first function using this tail becomes the owner of the tail. 
            
    :param pfn: pointer to the function
    :param tail_ea: any address inside the tail to remove
    """
    ...

def set_func_cmt(pfn: func_t, cmt: str, repeatable: bool) -> bool:
    r"""Set function comment. This function works with function chunks too. 
            
    :param pfn: ptr to function structure
    :param cmt: comment string, may be multiline (with '
    '). Use empty str ("") to delete comment
    :param repeatable: set repeatable comment?
    """
    ...

def set_func_end(ea: ida_idaapi.ea_t, newend: ida_idaapi.ea_t) -> bool:
    r"""Move function chunk end address. 
            
    :param ea: any address in the function
    :param newend: new end address of the function
    :returns: success
    """
    ...

def set_func_name_if_jumpfunc(pfn: func_t, oldname: str) -> int:
    r"""Give a meaningful name to function if it consists of only 'jump' instruction. 
            
    :param pfn: pointer to function (may be nullptr)
    :param oldname: old name of function. if old name was in "j_..." form, then we may discard it and set a new name. if oldname is not known, you may pass nullptr.
    :returns: success
    """
    ...

def set_func_start(ea: ida_idaapi.ea_t, newstart: ida_idaapi.ea_t) -> int:
    r"""Move function chunk start address. 
            
    :param ea: any address in the function
    :param newstart: new end address of the function
    :returns: Function move result codes
    """
    ...

def set_noret_insn(insn_ea: ida_idaapi.ea_t, noret: bool) -> bool:
    r"""Signal a non-returning instruction. This function can be used by the processor module to tell the kernel about non-returning instructions (like call exit). The kernel will perform the global function analysis and find out if the function returns at all. This analysis will be done at the first call to func_does_return() 
            
    :returns: true if the instruction 'noret' flag has been changed
    """
    ...

def set_tail_owner(fnt: func_t, new_owner: ida_idaapi.ea_t) -> bool:
    r"""Set a new owner of a function tail. The new owner function must be already referring to the tail (after append_func_tail). 
            
    :param fnt: pointer to the function tail
    :param new_owner: the entry point of the new owner function
    """
    ...

def set_visible_func(pfn: func_t, visible: bool) -> None:
    r"""Set visibility of function.
    
    """
    ...

def try_to_add_libfunc(ea: ida_idaapi.ea_t) -> int:
    r"""Apply the currently loaded signature file to the specified address. If a library function is found, then create a function and name it accordingly. 
            
    :param ea: any address in the program
    :returns: Library function codes
    """
    ...

def update_func(pfn: func_t) -> bool:
    r"""Update information about a function in the database (func_t). You must not change the function start and end addresses using this function. Use set_func_start() and set_func_end() for it. 
            
    :param pfn: ptr to function structure
    :returns: success
    """
    ...

FIND_FUNC_DEFINE: int  # 1
FIND_FUNC_EXIST: int  # 2
FIND_FUNC_IGNOREFN: int  # 2
FIND_FUNC_KEEPBD: int  # 4
FIND_FUNC_NORMAL: int  # 0
FIND_FUNC_OK: int  # 1
FIND_FUNC_UNDEF: int  # 0
FUNC_BOTTOMBP: int  # 256
FUNC_CATCH: int  # 1048576
FUNC_FAR: int  # 2
FUNC_FRAME: int  # 16
FUNC_FUZZY_SP: int  # 2048
FUNC_HIDDEN: int  # 64
FUNC_LIB: int  # 4
FUNC_LUMINA: int  # 65536
FUNC_NORET: int  # 1
FUNC_NORET_PENDING: int  # 512
FUNC_OUTLINE: int  # 131072
FUNC_PROLOG_OK: int  # 4096
FUNC_PURGED_OK: int  # 16384
FUNC_REANALYZE: int  # 262144
FUNC_SP_READY: int  # 1024
FUNC_STATICDEF: int  # 8
FUNC_TAIL: int  # 32768
FUNC_THUNK: int  # 128
FUNC_UNWIND: int  # 524288
FUNC_USERFAR: int  # 32
IDASGN_APPLIED: int  # 2
IDASGN_BADARG: int  # 1
IDASGN_CURRENT: int  # 3
IDASGN_OK: int  # 0
IDASGN_PLANNED: int  # 4
LIBFUNC_DELAY: int  # 2
LIBFUNC_FOUND: int  # 0
LIBFUNC_NONE: int  # 1
MOVE_FUNC_BADSTART: int  # 2
MOVE_FUNC_NOCODE: int  # 1
MOVE_FUNC_NOFUNC: int  # 3
MOVE_FUNC_OK: int  # 0
MOVE_FUNC_REFUSED: int  # 4
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
ida_range: module
weakref: module