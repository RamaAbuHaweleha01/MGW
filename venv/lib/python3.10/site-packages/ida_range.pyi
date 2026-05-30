from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Contains the definition of range_t.

A range is a non-empty continuous range of addresses (specified by its start and end addresses, the end address is excluded from the range).
Ranges are stored in the Btree part of the IDA database. To learn more about Btrees (Balanced Trees): [http://www.bluerwhite.org/btree/](http://www.bluerwhite.org/btree/) 
    
"""

class array_of_rangesets:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: array_of_rangesets) -> bool:
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
    def __getitem__(self, i: size_t) -> rangeset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: array_of_rangesets) -> bool:
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
    def __setitem__(self, i: size_t, v: rangeset_t) -> None:
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
    def add_unique(self, x: rangeset_t) -> bool:
        ...
    def append(self, x: rangeset_t) -> None:
        ...
    def at(self, _idx: size_t) -> rangeset_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> rangeset_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> rangeset_t:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: array_of_rangesets) -> None:
        ...
    def extract(self) -> rangeset_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: rangeset_t) -> bool:
        ...
    def inject(self, s: rangeset_t, len: size_t) -> None:
        ...
    def insert(self, it: rangeset_t, x: rangeset_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> rangeset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: array_of_rangesets) -> None:
        ...
    def truncate(self) -> None:
        ...

class range_t:
    @property
    def end_ea(self) -> Any: ...
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
    def __init__(self, ea1: ida_idaapi.ea_t = 0, ea2: ida_idaapi.ea_t = 0) -> Any:
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
    def intersect(self, r: range_t) -> bool:
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

class rangeset_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, aset: rangeset_t) -> bool:
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
    def __getitem__(self, idx: Any) -> rangeset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        r"""Get the number of range_t elements in the set.
        
        """
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, aset: rangeset_t) -> bool:
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
    def add(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. add(range: const range_t &) -> bool
            1. add(start: ida_idaapi.ea_t, _end: ida_idaapi.ea_t) -> bool
            2. add(aset: const rangeset_t &) -> bool
        
        # 0: add(range: const range_t &) -> bool
        
        Add an address range to the set. If 'range' intersects an existing element e, then e is extended to include 'range', and any superfluous elements (subsets of e) are removed. 
                
        :returns: false if 'range' was not added (the set was unchanged)
        
        # 1: add(start: ida_idaapi.ea_t, _end: ida_idaapi.ea_t) -> bool
        
        Create a new range_t from 'start' and 'end' and add it to the set.
        
        
        # 2: add(aset: const rangeset_t &) -> bool
        
        Add each element of 'aset' to the set. 
                
        :returns: false if no elements were added (the set was unchanged)
        
        """
        ...
    def as_rangevec(self) -> rangevec_t:
        r"""Return underlying rangevec_t object.
        
        """
        ...
    def begin(self) -> rangeset_t:
        r"""Get an iterator that points to the first element in the set.
        
        """
        ...
    def cached_range(self) -> Optional[range_t]:
        r"""When searching the rangeset, we keep a cached element to help speed up searches. 
                
        :returns: a pointer to the cached element
        """
        ...
    def clear(self) -> None:
        r"""Delete all elements from the set. See qvector::clear()
        
        """
        ...
    def contains(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. contains(ea: ida_idaapi.ea_t) -> bool
            1. contains(aset: const rangeset_t &) -> bool
        
        # 0: contains(ea: ida_idaapi.ea_t) -> bool
        
        Does an element of the rangeset contain 'ea'? See range_t::contains(ea_t)
        
        
        # 1: contains(aset: const rangeset_t &) -> bool
        
        Is every element in 'aset' contained in an element of this rangeset?. See range_t::contains(range_t)
        
        
        """
        ...
    def empty(self) -> bool:
        r"""Does the set have zero elements.
        
        """
        ...
    def end(self) -> rangeset_t:
        r"""Get an iterator that points to the end of the set. (This is NOT the last element)
        
        """
        ...
    def find_range(self, ea: ida_idaapi.ea_t) -> Optional[range_t]:
        r"""Get the element from the set that contains 'ea'. 
                
        :returns: nullptr if there is no such element
        """
        ...
    def getrange(self, idx: int) -> range_t:
        r"""Get the range_t at index 'idx'.
        
        """
        ...
    def has_common(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. has_common(range: const range_t &) -> bool
            1. has_common(aset: const rangeset_t &) -> bool
        
        # 0: has_common(range: const range_t &) -> bool
        
        Is there an ea in 'range' that is also in the rangeset?
        
        
        # 1: has_common(aset: const rangeset_t &) -> bool
        
        Does any element of 'aset' overlap with an element in this rangeset?. See range_t::overlaps()
        
        
        """
        ...
    def includes(self, range: range_t) -> bool:
        r"""Is every ea in 'range' contained in the rangeset?
        
        """
        ...
    def intersect(self, aset: rangeset_t) -> bool:
        r"""Set the rangeset to its intersection with 'aset'. 
                
        :returns: false if the set was unchanged
        """
        ...
    def is_equal(self, aset: rangeset_t) -> bool:
        r"""Do this rangeset and 'aset' have identical elements?
        
        """
        ...
    def is_subset_of(self, aset: rangeset_t) -> bool:
        r"""Is every element in the rangeset contained in an element of 'aset'?
        
        """
        ...
    def lastrange(self) -> range_t:
        r"""Get the last range_t in the set.
        
        """
        ...
    def next_addr(self, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Get the smallest ea_t value greater than 'ea' contained in the rangeset.
        
        """
        ...
    def next_range(self, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Get the smallest ea_t value greater than 'ea' that is not in the same range as 'ea'.
        
        """
        ...
    def nranges(self) -> int:
        r"""Get the number of range_t elements in the set.
        
        """
        ...
    def prev_addr(self, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Get the largest ea_t value less than 'ea' contained in the rangeset.
        
        """
        ...
    def prev_range(self, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
        r"""Get the largest ea_t value less than 'ea' that is not in the same range as 'ea'.
        
        """
        ...
    def sub(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. sub(range: const range_t &) -> bool
            1. sub(ea: ida_idaapi.ea_t) -> bool
            2. sub(aset: const rangeset_t &) -> bool
        
        # 0: sub(range: const range_t &) -> bool
        
        Subtract an address range from the set. All subsets of 'range' will be removed, and all elements that intersect 'range' will be truncated/split so they do not include 'range'. 
                
        :returns: false if 'range' was not subtracted (the set was unchanged)
        
        # 1: sub(ea: ida_idaapi.ea_t) -> bool
        
        Subtract an ea (an range of size 1) from the set. See sub(const range_t &)
        
        
        # 2: sub(aset: const rangeset_t &) -> bool
        
        Subtract each range in 'aset' from the set 
                
        :returns: false if nothing was subtracted (the set was unchanged)
        
        """
        ...
    def swap(self, r: rangeset_t) -> None:
        r"""Set this = 'r' and 'r' = this. See qvector::swap()
        
        """
        ...

class rangevec_base_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: rangevec_base_t) -> bool:
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
    def __getitem__(self, i: size_t) -> rangeset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: rangevec_base_t) -> bool:
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
    def add_unique(self, x: range_t) -> bool:
        ...
    def append(self, x: range_t) -> None:
        ...
    def at(self, _idx: size_t) -> rangeset_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> rangeset_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> rangeset_t:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: rangevec_base_t) -> None:
        ...
    def extract(self) -> rangeset_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: range_t) -> bool:
        ...
    def inject(self, s: range_t, len: size_t) -> None:
        ...
    def insert(self, it: range_t, x: range_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> rangeset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: rangevec_base_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class rangevec_t(rangevec_base_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: rangevec_base_t) -> bool:
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
    def __getitem__(self, i: size_t) -> rangeset_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __len__(self) -> int:
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: rangevec_base_t) -> bool:
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
    def add_unique(self, x: range_t) -> bool:
        ...
    def append(self, x: range_t) -> None:
        ...
    def at(self, _idx: size_t) -> rangeset_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> rangeset_t:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> rangeset_t:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: rangevec_base_t) -> None:
        ...
    def extract(self) -> rangeset_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: range_t) -> bool:
        ...
    def inject(self, s: range_t, len: size_t) -> None:
        ...
    def insert(self, it: range_t, x: range_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> rangeset_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: rangevec_base_t) -> None:
        ...
    def truncate(self) -> None:
        ...

def range_t_print(cb: range_t) -> str:
    r"""Helper function. Should not be called directly!
    
    """
    ...

RANGE_KIND_FUNC: int  # 1
RANGE_KIND_HIDDEN_RANGE: int  # 3
RANGE_KIND_SEGMENT: int  # 2
RANGE_KIND_UNKNOWN: int  # 0
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module