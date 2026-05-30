from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Type information in IDA.

In IDA, types are represented by and manipulated through tinfo_t objects.
A tinfo_t can represent a simple type (e.g., `int`, `float`), a complex type (a structure, enum, union, typedef), or even an array, or a function prototype.
The key types in this file are:

* til_t - a type info library. Holds type information in serialized form.
* tinfo_t - information about a type (simple, complex, ...)


# Glossary
All throughout this file, there are certain terms that will keep appearing:

* udt: "user-defined type": a structure or union - but not enums. See udt_type_data_t
* udm: "udt member": i.e., a structure or union member. See udm_t
* edm: "enum member": i.e., an enumeration member - i.e., an enumerator. See edm_t


# Under the hood
The tinfo_t type provides a lot of useful methods already, but it's possible to achieve even more by retrieving its contents into the container classes:

* udt_type_data_t - for structures & unions. See tinfo_t::get_udt_details . Essentially, a vector of udm_t
* enum_type_data_t - for enumerations. See tinfo_t::get_enum_details . Essentially, a vector of edm_t
* ptr_type_data_t - for pointers. See tinfo_t::get_ptr_details
* array_type_data_t - for arrays. See tinfo_t::get_array_details
* func_type_data_t - for function prototypes. See tinfo_t::get_func_details
* bitfield_type_data_t - for bitfields. See tinfo_t::get_bitfield_details


# Attached & detached tinfo_t objects
tinfo_t objects can be attached to a til_t library, or can be created without using any til_t.
Here is an example, assigning a function prototype:
func_type_data_t func_info;
funcarg_t argc; argc.name = "argc"; argc.type = tinfo_t(BT_INT); func_info.push_back(argc);
funcarg_t argv; argc.name = "argv"; argc.type = tinfo_t("const char **"); func_info.push_back(argv)
tinfo_t tif; if ( tif.create_func(func_info) ) { ea_t ea = // get address of "main" apply_tinfo(ea, tif, TINFO_DEFINITE); }
This code manipulates a "detached" tinfo_t object, which does not depend on any til_t file. However, any complex type will require a til_t file. In IDA, there is always a default til_t file for each idb file. This til_t file can be specified by nullptr.
On the other hand, the following code manipulates an "attached" tinfo_t object, and any operation that modifies it, will also modify it in the hosting til_t:
tinfo_t tif; Load type from the "Local Types" til_t. Note: we could have used `get_idati()` instead of nullptr if ( tif.get_named_type(nullptr, "my_struct_t") ) tif.add_udm("extra_field", "unsigned long long");
You can check if a tinfo_t instance is attached to a type in a til_t file by calling tinfo_t::is_typeref 
    
"""

class aloc_visitor_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def visit_location(self, v: argloc_t, off: int, size: int) -> int:
        ...

class argloc_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: argloc_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: argloc_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: argloc_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: argloc_t) -> bool:
        ...
    def __lt__(self, r: argloc_t) -> bool:
        ...
    def __ne__(self, r: argloc_t) -> bool:
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
    def advance(self, delta: int) -> bool:
        r"""Move the location to point 'delta' bytes further.
        
        """
        ...
    def align_reg_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set register offset to align it to the upper part of _SLOTSIZE.
        
        """
        ...
    def align_stkoff_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set stack offset to align to the upper part of _SLOTSIZE.
        
        """
        ...
    def atype(self) -> argloc_type_t:
        r"""Get type (Argument location types)
        
        """
        ...
    def calc_offset(self) -> int:
        r"""Calculate offset that can be used to compare 2 similar arglocs.
        
        """
        ...
    def compare(self, r: argloc_t) -> int:
        ...
    def consume_rrel(self, p: rrel_t) -> None:
        r"""Set register-relative location - can't be nullptr.
        
        """
        ...
    def consume_scattered(self, p: scattered_aloc_t) -> None:
        r"""Set distributed argument location.
        
        """
        ...
    def get_biggest(self) -> argloc_t:
        r"""Get largest element in internal union.
        
        """
        ...
    def get_custom(self) -> None:
        r"""Get custom argloc info. Use if atype() == ALOC_CUSTOM 
                
        """
        ...
    def get_ea(self) -> ida_idaapi.ea_t:
        r"""Get the global address. Use when atype() == ALOC_STATIC 
                
        """
        ...
    def get_reginfo(self) -> int:
        r"""Get all register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def get_rrel(self) -> rrel_t:
        r"""Get register-relative info. Use when atype() == ALOC_RREL 
                
        """
        ...
    def has_reg(self) -> bool:
        r"""TRUE if argloc has a register part.
        
        """
        ...
    def has_stkoff(self) -> bool:
        r"""TRUE if argloc has a stack part.
        
        """
        ...
    def in_stack(self) -> bool:
        r"""TRUE if argloc is in stack entirely.
        
        """
        ...
    def is_badloc(self) -> bool:
        r"""See ALOC_NONE.
        
        """
        ...
    def is_custom(self) -> bool:
        r"""See ALOC_CUSTOM.
        
        """
        ...
    def is_ea(self) -> bool:
        r"""See ALOC_STATIC.
        
        """
        ...
    def is_fragmented(self) -> bool:
        r"""is_scattered() || is_reg2()
        
        """
        ...
    def is_mixed_scattered(self) -> bool:
        r"""mixed scattered: consists of register and stack parts
        
        """
        ...
    def is_reg(self) -> bool:
        r"""is_reg1() || is_reg2()
        
        """
        ...
    def is_reg1(self) -> bool:
        r"""See ALOC_REG1.
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""See ALOC_REG2.
        
        """
        ...
    def is_rrel(self) -> bool:
        r"""See ALOC_RREL.
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""See ALOC_DIST.
        
        """
        ...
    def is_stkoff(self) -> bool:
        r"""See ALOC_STACK.
        
        """
        ...
    def reg1(self) -> int:
        r"""Get the register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def reg2(self) -> int:
        r"""Get info for the second register. Use when atype() == ALOC_REG2 
                
        """
        ...
    def regoff(self) -> int:
        r"""Get offset from the beginning of the register in bytes. Use when atype() == ALOC_REG1 
                
        """
        ...
    def scattered(self) -> scattered_aloc_t:
        r"""Get scattered argument info. Use when atype() == ALOC_DIST 
                
        """
        ...
    def set_badloc(self) -> None:
        r"""Set to invalid location.
        
        """
        ...
    def set_ea(self, _ea: ida_idaapi.ea_t) -> None:
        r"""Set static ea location.
        
        """
        ...
    def set_reg1(self, reg: int, off: int = 0) -> None:
        r"""Set register location.
        
        """
        ...
    def set_reg2(self, _reg1: int, _reg2: int) -> None:
        r"""Set secondary register location.
        
        """
        ...
    def set_stkoff(self, off: int) -> None:
        r"""Set stack offset location.
        
        """
        ...
    def stkoff(self) -> int:
        r"""Get the stack offset. Use if atype() == ALOC_STACK 
                
        """
        ...
    def swap(self, r: argloc_t) -> None:
        r"""Assign this == r and r == this.
        
        """
        ...

class argpart_t(argloc_t):
    @property
    def off(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: argloc_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: argloc_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: argloc_t) -> bool:
        ...
    def __init__(self, args: Any) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: argloc_t) -> bool:
        ...
    def __lt__(self, r: argpart_t) -> bool:
        ...
    def __ne__(self, r: argloc_t) -> bool:
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
    def advance(self, delta: int) -> bool:
        r"""Move the location to point 'delta' bytes further.
        
        """
        ...
    def align_reg_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set register offset to align it to the upper part of _SLOTSIZE.
        
        """
        ...
    def align_stkoff_high(self, size: size_t, _slotsize: size_t) -> None:
        r"""Set stack offset to align to the upper part of _SLOTSIZE.
        
        """
        ...
    def atype(self) -> argloc_type_t:
        r"""Get type (Argument location types)
        
        """
        ...
    def bad_offset(self) -> bool:
        r"""Does this argpart have a valid offset?
        
        """
        ...
    def bad_size(self) -> bool:
        r"""Does this argpart have a valid size?
        
        """
        ...
    def calc_offset(self) -> int:
        r"""Calculate offset that can be used to compare 2 similar arglocs.
        
        """
        ...
    def compare(self, r: argloc_t) -> int:
        ...
    def consume_rrel(self, p: rrel_t) -> None:
        r"""Set register-relative location - can't be nullptr.
        
        """
        ...
    def consume_scattered(self, p: scattered_aloc_t) -> None:
        r"""Set distributed argument location.
        
        """
        ...
    def get_biggest(self) -> argloc_t:
        r"""Get largest element in internal union.
        
        """
        ...
    def get_custom(self) -> None:
        r"""Get custom argloc info. Use if atype() == ALOC_CUSTOM 
                
        """
        ...
    def get_ea(self) -> ida_idaapi.ea_t:
        r"""Get the global address. Use when atype() == ALOC_STATIC 
                
        """
        ...
    def get_reginfo(self) -> int:
        r"""Get all register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def get_rrel(self) -> rrel_t:
        r"""Get register-relative info. Use when atype() == ALOC_RREL 
                
        """
        ...
    def has_reg(self) -> bool:
        r"""TRUE if argloc has a register part.
        
        """
        ...
    def has_stkoff(self) -> bool:
        r"""TRUE if argloc has a stack part.
        
        """
        ...
    def in_stack(self) -> bool:
        r"""TRUE if argloc is in stack entirely.
        
        """
        ...
    def is_badloc(self) -> bool:
        r"""See ALOC_NONE.
        
        """
        ...
    def is_custom(self) -> bool:
        r"""See ALOC_CUSTOM.
        
        """
        ...
    def is_ea(self) -> bool:
        r"""See ALOC_STATIC.
        
        """
        ...
    def is_fragmented(self) -> bool:
        r"""is_scattered() || is_reg2()
        
        """
        ...
    def is_mixed_scattered(self) -> bool:
        r"""mixed scattered: consists of register and stack parts
        
        """
        ...
    def is_reg(self) -> bool:
        r"""is_reg1() || is_reg2()
        
        """
        ...
    def is_reg1(self) -> bool:
        r"""See ALOC_REG1.
        
        """
        ...
    def is_reg2(self) -> bool:
        r"""See ALOC_REG2.
        
        """
        ...
    def is_rrel(self) -> bool:
        r"""See ALOC_RREL.
        
        """
        ...
    def is_scattered(self) -> bool:
        r"""See ALOC_DIST.
        
        """
        ...
    def is_stkoff(self) -> bool:
        r"""See ALOC_STACK.
        
        """
        ...
    def reg1(self) -> int:
        r"""Get the register info. Use when atype() == ALOC_REG1 or ALOC_REG2 
                
        """
        ...
    def reg2(self) -> int:
        r"""Get info for the second register. Use when atype() == ALOC_REG2 
                
        """
        ...
    def regoff(self) -> int:
        r"""Get offset from the beginning of the register in bytes. Use when atype() == ALOC_REG1 
                
        """
        ...
    def scattered(self) -> scattered_aloc_t:
        r"""Get scattered argument info. Use when atype() == ALOC_DIST 
                
        """
        ...
    def set_badloc(self) -> None:
        r"""Set to invalid location.
        
        """
        ...
    def set_ea(self, _ea: ida_idaapi.ea_t) -> None:
        r"""Set static ea location.
        
        """
        ...
    def set_reg1(self, reg: int, off: int = 0) -> None:
        r"""Set register location.
        
        """
        ...
    def set_reg2(self, _reg1: int, _reg2: int) -> None:
        r"""Set secondary register location.
        
        """
        ...
    def set_stkoff(self, off: int) -> None:
        r"""Set stack offset location.
        
        """
        ...
    def stkoff(self) -> int:
        r"""Get the stack offset. Use if atype() == ALOC_STACK 
                
        """
        ...
    def swap(self, r: argpart_t) -> None:
        r"""Assign this = r and r = this.
        
        """
        ...

class argpartvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: argpartvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: argpartvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: argpart_t) -> None:
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
    def add_unique(self, x: argpart_t) -> bool:
        ...
    def append(self, x: argpart_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: argpartvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: argpart_t) -> bool:
        ...
    def inject(self, s: argpart_t, len: size_t) -> None:
        ...
    def insert(self, it: argpart_t, x: argpart_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: argpartvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class argtinfo_helper_t:
    @property
    def reserved(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def has_delay_slot(self, arg0: ida_idaapi.ea_t) -> bool:
        r"""The call instruction with a delay slot?.
        
        """
        ...
    def is_stkarg_load(self, insn: insn_t, src: int, dst: int) -> bool:
        r"""Is the current insn a stkarg load?. if yes:
        * src: index of the source operand in insn_t::ops
        * dst: index of the destination operand in insn_t::ops insn_t::ops[dst].addr is expected to have the stack offset 
        
        
                
        """
        ...
    def set_op_tinfo(self, insn: insn_t, x: op_t, tif: tinfo_t, name: str) -> bool:
        r"""Set the operand type as specified.
        
        """
        ...
    def use_arg_tinfos(self, caller: ida_idaapi.ea_t, fti: func_type_data_t, rargs: funcargvec_t) -> None:
        r"""This function is to be called by the processor module in response to ev_use_arg_types. 
                
        """
        ...

class array_type_data_t:
    @property
    def base(self) -> Any: ...
    @property
    def elem_type(self) -> Any: ...
    @property
    def nelems(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, b: size_t = 0, n: size_t = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def swap(self, r: array_type_data_t) -> None:
        r"""set this = r and r = this
        
        """
        ...

class bitfield_type_data_t:
    @property
    def is_unsigned(self) -> Any: ...
    @property
    def nbytes(self) -> Any: ...
    @property
    def width(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: bitfield_type_data_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: bitfield_type_data_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: bitfield_type_data_t) -> bool:
        ...
    def __init__(self, _nbytes: uchar = 0, _width: uchar = 0, _is_unsigned: bool = False) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: bitfield_type_data_t) -> bool:
        ...
    def __lt__(self, r: bitfield_type_data_t) -> bool:
        ...
    def __ne__(self, r: bitfield_type_data_t) -> bool:
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
    def compare(self, r: bitfield_type_data_t) -> int:
        ...
    def is_valid_bitfield(self) -> bool:
        ...
    def swap(self, r: bitfield_type_data_t) -> None:
        ...

class callregs_t:
    FPREGS: int  # 1
    GPREGS: int  # 0
    @property
    def fpregs(self) -> Any: ...
    @property
    def gpregs(self) -> Any: ...
    @property
    def nregs(self) -> Any: ...
    @property
    def policy(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def append_registers(self, kind: reg_kind_t, first_reg: int, last_reg: int) -> None:
        ...
    def by_slots(self) -> bool:
        ...
    def init_regs(self, cc: callcnv_t) -> bool:
        r"""Init policy & registers for given CC.
        
        """
        ...
    def regcount(self, cc: callcnv_t) -> int:
        r"""Get max number of registers may be used in a function call.
        
        """
        ...
    def reginds(self, gp_ind: int, fp_ind: int, r: int) -> bool:
        r"""Get register indexes within GP/FP arrays. (-1 -> is not present in the corresponding array) 
                
        """
        ...
    def reset(self) -> None:
        r"""Set policy and registers to invalid values.
        
        """
        ...
    def set(self, _policy: argreg_policy_t, gprs: int, fprs: int) -> None:
        r"""Init policy & registers (arrays are -1-terminated)
        
        """
        ...
    def set_registers(self, kind: reg_kind_t, first_reg: int, last_reg: int) -> None:
        ...
    def swap(self, r: callregs_t) -> None:
        r"""swap two instances
        
        """
        ...

class const_aloc_visitor_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def visit_location(self, v: argloc_t, off: int, size: int) -> int:
        ...

class custom_callcnv_t:
    @property
    def abibits(self) -> Any: ...
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
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def calc_arglocs(self, fti: func_type_data_t) -> bool:
        r"""Calculate the argument locations. This function must fill all fti->at(i).argloc instances. It may be called for variadic functions too, in calc_varglocs fails. 
                
        :param fti: function prototype
        :returns: success
        """
        ...
    def calc_purged_bytes(self, args: Any) -> int:
        r"""Calculate the number of purged bytes 
                
        :param fti: function prototype
        :param call_ea: address of the call instruction (not used yet)
        """
        ...
    def calc_retloc(self, fti: func_type_data_t) -> bool:
        r"""Calculate the location of the return value. This function must fill fti->retloc. 
                
        :param fti: function prototype
        :returns: success
        """
        ...
    def calc_varglocs(self, fti: func_type_data_t, regs: regobjs_t, stkargs: relobj_t, nfixed: int) -> bool:
        r"""Calculate the argument locations for a variadic function. This function must fill all fti->at(i).argloc instances and provide more detailed info about registers and stkargs. 
                
        :param fti: function prototype
        :param regs: buffer for hidden register arguments, may be nullptr
        :param stkargs: buffer for hidden stack arguments, may be nullptr
        :param nfixed: number of fixed arguments
        :returns: success
        """
        ...
    def decorate_name(self, name: str, should_decorate: bool, cc: callcnv_t, type: tinfo_t) -> str:
        r"""Function to be overloaded for custom calling conventions.
        
        Decorate a function name. Some compilers decorate names depending on the calling convention. This function provides the means to handle it for custom callcnvs. Please note that this is about name decoration (C), not name mangling (C++). 
                
        """
        ...
    def find_varargs(self, fti: func_type_data_t, call_ea: ida_idaapi.ea_t, blk: mblock_t) -> ssize_t:
        r"""Discover variadic arguments. This function is called only for variadic functions. It is currently used by the decompiler. 
                
        :param fti: function prototype. find_varargs() should append the discovered variadic arguments to it.
        :param call_ea: address of the call instruction
        :param blk: microcode block with the call instruction
        :returns: >0 - total number of arguments after the call <0 - failure ==0 - means to use the standard algorithm to discover variadic args
        """
        ...
    def get_cc_regs(self, out: callregs_t) -> bool:
        r"""Retrieve generic information about call registers.
        
        """
        ...
    def get_stkarg_area_info(self, out: stkarg_area_info_t) -> bool:
        r"""Retrieve generic information about stack arguments.
        
        """
        ...
    def is_purging(self) -> bool:
        ...
    def is_usercall(self) -> bool:
        ...
    def is_vararg(self) -> bool:
        ...
    def lower_func_type(self, fti: func_type_data_t) -> int:
        r"""Lower a function type. See lower_type() for more explanations. 
                
        :param fti: function prototype
        :returns: <0-failure, >=0-ok, 2-made substantial changes
        """
        ...
    def validate_func(self, fti: func_type_data_t, reterr: str) -> bool:
        r"""Validate a function prototype. This function is used during parsing or deserializing a function prototype to verify semantic limitations of the prototype (for example, returning arrays is forbidden in C) 
                
        :param fti: function prototype
        :param reterr: buffer for error message
        """
        ...

class custom_data_type_info_t:
    @property
    def dtid(self) -> Any: ...
    @property
    def fid(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class edm_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: edm_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        r"""Create an enumerator, with the specified name and value
        
        This constructor has the following signatures:
        
            1. edm_t(edm: edm_t)
            2. edm_t(name: str, value: int, cmt: str=None)
        
        :param name:  Enumerator name. Must not be empty (1st form)
        :param value: Enumerator value (1st form)
        :param cmt:   Enumerator repeatable comment. May be empty (1st form)
        :param edm:   An enum member to copy
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: edm_t) -> bool:
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
    def empty(self) -> bool:
        ...
    def get_tid(self) -> tid_t:
        ...
    def swap(self, r: edm_t) -> None:
        ...

class edmvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: edmvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: edmvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: edm_t) -> None:
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
    def add_unique(self, x: edm_t) -> bool:
        ...
    def append(self, x: edm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: edmvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: edm_t) -> bool:
        ...
    def inject(self, s: edm_t, len: size_t) -> None:
        ...
    def insert(self, it: edm_t, x: edm_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: edmvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class enum_member_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: edm_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        r"""Create an enumerator, with the specified name and value
        
        This constructor has the following signatures:
        
            1. edm_t(edm: edm_t)
            2. edm_t(name: str, value: int, cmt: str=None)
        
        :param name:  Enumerator name. Must not be empty (1st form)
        :param value: Enumerator value (1st form)
        :param cmt:   Enumerator repeatable comment. May be empty (1st form)
        :param edm:   An enum member to copy
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: edm_t) -> bool:
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
    def empty(self) -> bool:
        ...
    def get_tid(self) -> tid_t:
        ...
    def swap(self, r: edm_t) -> None:
        ...

class enum_member_vec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: edmvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: edmvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: edm_t) -> None:
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
    def add_unique(self, x: edm_t) -> bool:
        ...
    def append(self, x: edm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: edmvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: edm_t) -> bool:
        ...
    def inject(self, s: edm_t, len: size_t) -> None:
        ...
    def insert(self, it: edm_t, x: edm_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: edmvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class enum_type_data_t(edmvec_t):
    @property
    def bte(self) -> Any: ...
    @property
    def group_sizes(self) -> Any: ...
    @property
    def taenum_bits(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: edmvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: edmvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: edm_t) -> None:
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
    def add_constant(self, name: str, value: uint64, cmt: str = None) -> None:
        r"""add constant for regular enum
        
        """
        ...
    def add_unique(self, x: edm_t) -> bool:
        ...
    def all_constants(self) -> Any:
        r"""
                Generate tupples of all constants except of bitmasks.
                Each tupple is:
                [0] constant index
                [1] enum member index of group start
                [2] group size
                In case of regular enum the second element of tupple is 0 and the third element of tupple is the number of enum members.
                
        """
        ...
    def all_groups(self, skip_trivial: Any = False) -> Any:
        r"""
                Generate tuples for bitmask enum groups.
                Each tupple is:
                [0] enum member index of group start
                [1] group size
                Tupples may include or not the group with 1 element.
                
        """
        ...
    def append(self, x: edm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def calc_mask(self) -> uint64:
        ...
    def calc_nbytes(self) -> int:
        r"""get the width of enum in bytes
        
        """
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: edmvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def find_member(self, args: Any) -> ssize_t:
        r"""This function has the following signatures:
        
            0. find_member(name: str, from: size_t=0, to: size_t=size_t(-1)) -> ssize_t
            1. find_member(value: uint64, serial: uchar, from: size_t=0, to: size_t=size_t(-1), vmask: uint64=uint64(-1)) -> ssize_t
        
        # 0: find_member(name: str, from: size_t=0, to: size_t=size_t(-1)) -> ssize_t
        
        find member (constant or bmask) by name
        
        
        # 1: find_member(value: uint64, serial: uchar, from: size_t=0, to: size_t=size_t(-1), vmask: uint64=uint64(-1)) -> ssize_t
        
        find member (constant or bmask) by value
        
        
        """
        ...
    def front(self) -> Any:
        ...
    def get_constant_group(self, args: Any) -> Any:
        r"""get group parameters for the constant, valid for bitmask enum 
                
        :param group_start_index: index of the group mask
        :param group_size: group size (>=1)
        :param idx: constant index
        :returns: success
        """
        ...
    def get_enum_radix(self) -> int:
        r"""Get enum constant radix 
                
        :returns: radix or 1 for BTE_CHAR
        """
        ...
    def get_max_serial(self, value: uint64) -> uchar:
        r"""return the maximum serial for the value
        
        """
        ...
    def get_serial(self, index: size_t) -> uchar:
        r"""returns serial for the constant
        
        """
        ...
    def get_value_repr(self, repr: value_repr_t) -> tinfo_code_t:
        r"""get enum radix and other representation info 
                
        :param repr: value display info
        """
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: edm_t) -> bool:
        ...
    def has_lzero(self) -> bool:
        ...
    def inject(self, s: edm_t, len: size_t) -> None:
        ...
    def insert(self, it: edm_t, x: edm_t) -> qvector:
        ...
    def is_bf(self) -> bool:
        r"""is bitmask or ordinary enum?
        
        """
        ...
    def is_bin(self) -> bool:
        ...
    def is_char(self) -> bool:
        ...
    def is_dec(self) -> bool:
        ...
    def is_group_mask_at(self, idx: size_t) -> bool:
        r"""is the enum member at IDX a non-trivial group mask? a trivial group consist of one bit and has just one member, which can be considered as a mask or a bitfield constant 
                
        :param idx: index
        :returns: success
        """
        ...
    def is_hex(self) -> bool:
        ...
    def is_number_signed(self) -> bool:
        ...
    def is_oct(self) -> bool:
        ...
    def is_sbin(self) -> bool:
        ...
    def is_shex(self) -> bool:
        ...
    def is_soct(self) -> bool:
        ...
    def is_udec(self) -> bool:
        ...
    def is_valid_group_sizes(self) -> bool:
        r"""is valid group sizes
        
        """
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def set_enum_radix(self, radix: int, sign: bool) -> None:
        r"""Set radix to display constants 
                
        :param radix: radix with the special case 1 to display as character
        """
        ...
    def set_lzero(self, on: bool) -> None:
        ...
    def set_nbytes(self, nbytes: int) -> bool:
        r"""set enum width (nbytes)
        
        """
        ...
    def set_value_repr(self, repr: value_repr_t) -> None:
        r"""set enum radix and other representation info 
                
        :param repr: value display info
        """
        ...
    def size(self) -> int:
        ...
    def store_64bit_values(self) -> bool:
        ...
    def swap(self, r: enum_type_data_t) -> None:
        r"""swap two instances
        
        """
        ...
    def truncate(self) -> None:
        ...

class func_type_data_t(funcargvec_t):
    @property
    def flags(self) -> Any: ...
    @property
    def retloc(self) -> Any: ...
    @property
    def rettype(self) -> Any: ...
    @property
    def spoiled(self) -> Any: ...
    @property
    def stkargs(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: funcargvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: funcargvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: funcarg_t) -> None:
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
    def add_unique(self, x: funcarg_t) -> bool:
        ...
    def append(self, x: funcarg_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def dump(self, praloc_bits: int = 2) -> bool:
        r"""Dump information that is not always visible in the function prototype. (argument locations, return location, total stkarg size) 
                
        """
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: funcargvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def find_argument(self, args: Any) -> ssize_t:
        r"""find argument by name
        
        """
        ...
    def front(self) -> Any:
        ...
    def get_call_method(self) -> int:
        ...
    def get_explicit_cc(self) -> callcnv_t:
        ...
    def grow(self, args: Any) -> None:
        ...
    def guess_cc(self, purged: int, cc_flags: int) -> callcnv_t:
        r"""Guess function calling convention use the following info: argument locations and 'stkargs' 
                
        """
        ...
    def has(self, x: funcarg_t) -> bool:
        ...
    def inject(self, s: funcarg_t, len: size_t) -> None:
        ...
    def insert(self, it: funcarg_t, x: funcarg_t) -> qvector:
        ...
    def is_const(self) -> bool:
        ...
    def is_ctor(self) -> bool:
        ...
    def is_dtor(self) -> bool:
        ...
    def is_golang_cc(self) -> bool:
        ...
    def is_high(self) -> bool:
        ...
    def is_noret(self) -> bool:
        ...
    def is_pure(self) -> bool:
        ...
    def is_static(self) -> bool:
        ...
    def is_swift_cc(self) -> bool:
        ...
    def is_user_cc(self) -> bool:
        ...
    def is_vararg_cc(self) -> bool:
        ...
    def is_virtual(self) -> bool:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def set_cc(self, cc: callcnv_t) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: func_type_data_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class funcarg_t:
    @property
    def argloc(self) -> Any: ...
    @property
    def cmt(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: funcarg_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        r"""Create a function argument, with the specified name and type.
        
        This constructor has the following signatures:
        
            1. funcarg_t(name: str, type, argloc: argloc_t)
            2. funcarg_t(funcarg: funcarg_t)
        
        In the 1st form, the 'type' descriptor, can be one of:
        
            * type_t: if the type is simple (integral/floating/bool). E.g., `BTF_INT`
            * tinfo_t: can handle more complex types (structures, pointers, arrays, ...)
            * str: a C type declaration
        
        If an input argument is incorrect, the constructor may raise an exception
        
        :param name: a valid argument name. May not be empty (1st form).
        :param type: the member type (1st form).
        :param argloc: the argument location. Can be empty (1st form).
        :param funcarg: a funcarg_t to copy
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: funcarg_t) -> bool:
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

class funcargvec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: funcargvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: funcargvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: funcarg_t) -> None:
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
    def add_unique(self, x: funcarg_t) -> bool:
        ...
    def append(self, x: funcarg_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: funcargvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: funcarg_t) -> bool:
        ...
    def inject(self, s: funcarg_t, len: size_t) -> None:
        ...
    def insert(self, it: funcarg_t, x: funcarg_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: funcargvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class ida_lowertype_helper_t(lowertype_helper_t):
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, _tif: tinfo_t, _ea: ida_idaapi.ea_t, _pb: int) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def func_has_stkframe_hole(self, candidate: tinfo_t, candidate_data: func_type_data_t) -> bool:
        ...
    def get_func_purged_bytes(self, candidate: tinfo_t, arg3: func_type_data_t) -> int:
        ...

class lowertype_helper_t:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def func_has_stkframe_hole(self, candidate: tinfo_t, candidate_data: func_type_data_t) -> bool:
        ...
    def get_func_purged_bytes(self, candidate: tinfo_t, candidate_data: func_type_data_t) -> int:
        ...

class predicate_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def should_display(self, til: til_t, name: str, type: type_t, fields: p_list) -> bool:
        ...

class ptr_type_data_t:
    @property
    def based_ptr_size(self) -> Any: ...
    @property
    def closure(self) -> Any: ...
    @property
    def delta(self) -> Any: ...
    @property
    def obj_type(self) -> Any: ...
    @property
    def parent(self) -> Any: ...
    @property
    def taptr_bits(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: ptr_type_data_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: ptr_type_data_t) -> bool:
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
    def is_code_ptr(self) -> bool:
        r"""Are we pointing to code?
        
        """
        ...
    def is_shifted(self) -> bool:
        ...
    def swap(self, r: ptr_type_data_t) -> None:
        r"""Set this = r and r = this.
        
        """
        ...

class reginfovec_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: reginfovec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: reginfovec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: reg_info_t) -> None:
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
    def add_unique(self, x: reg_info_t) -> bool:
        ...
    def append(self, args: Any) -> None:
        ...
    def at(self, i: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: reginfovec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: reg_info_t) -> bool:
        ...
    def inject(self, s: reg_info_t, len: size_t) -> None:
        ...
    def insert(self, it: reg_info_t, x: reg_info_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: reginfovec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class regobj_t:
    @property
    def regidx(self) -> Any: ...
    @property
    def relocate(self) -> Any: ...
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
    def __ge__(self, value: Any) -> bool:
        r"""Return self>=value."""
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def size(self) -> int:
        ...

class regobjs_t(regobjvec_t):
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
    def __getitem__(self, i: size_t) -> udm_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: regobj_t) -> None:
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
    def append(self, x: regobj_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: regobjvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: regobj_t, len: size_t) -> None:
        ...
    def insert(self, it: regobj_t, x: regobj_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: regobjvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class regobjvec_t:
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
    def __getitem__(self, i: size_t) -> udm_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: regobj_t) -> None:
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
    def append(self, x: regobj_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: regobjvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: regobj_t, len: size_t) -> None:
        ...
    def insert(self, it: regobj_t, x: regobj_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: regobjvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class rrel_t:
    @property
    def off(self) -> Any: ...
    @property
    def reg(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class scattered_aloc_t(argpartvec_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: argpartvec_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: argpartvec_t) -> bool:
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
    def __setitem__(self, i: size_t, v: argpart_t) -> None:
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
    def add_unique(self, x: argpart_t) -> bool:
        ...
    def append(self, x: argpart_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: argpartvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: argpart_t) -> bool:
        ...
    def inject(self, s: argpart_t, len: size_t) -> None:
        ...
    def insert(self, it: argpart_t, x: argpart_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: argpartvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class simd_info_t:
    @property
    def memtype(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def tif(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def match_pattern(self, pattern: simd_info_t) -> bool:
        ...

class stkarg_area_info_t:
    @property
    def cb(self) -> Any: ...
    @property
    def linkage_area(self) -> Any: ...
    @property
    def shadow_size(self) -> Any: ...
    @property
    def stkarg_offset(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class text_sink_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class til_symbol_t:
    @property
    def name(self) -> Any: ...
    @property
    def til(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, n: str = None, t: til_t = None) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class til_t:
    @property
    def cc(self) -> Any: ...
    @property
    def desc(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def nbases(self) -> Any: ...
    @property
    def nrefs(self) -> Any: ...
    @property
    def nstreams(self) -> Any: ...
    @property
    def streams(self) -> Any: ...
    @property
    def type_names(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: til_t) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, value: Any) -> bool:
        r"""Return self<value."""
        ...
    def __ne__(self, r: til_t) -> bool:
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
    def base(self, n: int) -> til_t:
        ...
    def find_base(self, n: str) -> til_t:
        r"""Find the base til with the provided name 
                
        :param n: the base til name
        :returns: the found til_t, or nullptr
        """
        ...
    def get_named_type(self, name: Any) -> bool:
        r"""Retrieves a tinfo_t representing the named type in this type library.
        
        :param name: a type name
        :returns: a new tinfo_t object, or None if not found
        """
        ...
    def get_numbered_type(self, ordinal: Any) -> Any:
        r"""Retrieves a tinfo_t representing the numbered type in this type library.
        
        :param ordinal: a type ordinal
        :returns: a new tinfo_t object, or None if not found
        """
        ...
    def get_type_names(self) -> Any:
        ...
    def import_type(self, src: Any) -> Any:
        r"""Import a type (and all its dependencies) into this type info library.
        
        :param src: The type to import
        :returns: the imported copy, or None
        """
        ...
    def is_dirty(self) -> bool:
        r"""Has the til been modified? (TIL_MOD)
        
        """
        ...
    def named_types(self) -> Any:
        r"""Returns a generator over the named types contained in this
        type library.
        
        Every iteration returns a fresh new tinfo_t object
        
        :returns: a tinfo_t-producing generator
        """
        ...
    def numbered_types(self) -> Any:
        r"""Returns a generator over the numbered types contained in this
        type library.
        
        Every iteration returns a fresh new tinfo_t object
        
        :returns: a tinfo_t-producing generator
        """
        ...
    def set_dirty(self) -> None:
        r"""Mark the til as modified (TIL_MOD)
        
        """
        ...

class til_type_ref_t:
    @property
    def bf_bitoff(self) -> Any: ...
    @property
    def bucket_start(self) -> Any: ...
    @property
    def cb(self) -> Any: ...
    @property
    def cursor(self) -> Any: ...
    @property
    def edm(self) -> Any: ...
    @property
    def fa(self) -> Any: ...
    @property
    def is_detached(self) -> Any: ...
    @property
    def is_forward(self) -> Any: ...
    @property
    def is_writable(self) -> Any: ...
    @property
    def kind(self) -> Any: ...
    @property
    def last_udm_offset(self) -> Any: ...
    @property
    def memidx(self) -> Any: ...
    @property
    def nmembers(self) -> Any: ...
    @property
    def offset(self) -> Any: ...
    @property
    def ordinal(self) -> Any: ...
    @property
    def tif(self) -> Any: ...
    @property
    def total_size(self) -> Any: ...
    @property
    def udm(self) -> Any: ...
    @property
    def unpadded_size(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def clear(self) -> None:
        ...
    def is_enum(self) -> bool:
        ...
    def is_func(self) -> bool:
        ...
    def is_struct(self) -> bool:
        ...
    def is_typedef(self) -> bool:
        ...
    def is_udt(self) -> bool:
        ...
    def is_union(self) -> bool:
        ...
    def on_member(self) -> bool:
        ...

class tinfo_t:
    ENUMBM_AUTO: int  # 2
    ENUMBM_OFF: int  # 0
    ENUMBM_ON: int  # 1
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: tinfo_t) -> bool:
        ...
    def __format__(self, format_spec: Any) -> Any:
        r"""Default object formatter."""
        ...
    def __ge__(self, r: tinfo_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, r: tinfo_t) -> bool:
        ...
    def __init__(self, args: Any, ordinal: Any = None, name: Any = None, tid: Any = None, til: Any = None) -> Any:
        r"""Create a type object with the provided argumens.
        
        This constructor has the following signatures:
        
            1. tinfo_t(decl_type: type_t)
            2. tinfo_t(decl: str, til: til_t = None, pt_flags: int = 0)
        
        The latter form will create the type object by parsing the type declaration
        
        Alternatively, you can use a form accepting the following keyword arguments:
        
        * ordinal: int
        * name: str
        * tid: int
        * til: til_t=None # `None` means `get_idati()`
        
        E.g.,
        
        * tinfo_t(ordinal=3)
        * tinfo_t(ordinal=10, til=get_idati())
        * tinfo_t(name="mytype_t")
        * tinfo_t(name="thattype_t", til=my_other_til)
        * tinfo_t(tid=ida_nalt.get_strid(some_address))
        
        The constructor may raise an exception if data was invalid, or if parsing failed.
        
        :param decl_type: A simple type
        :param decl: A valid C declaration
        :param til: A type library, or `None` to use the (`get_idati()`) default
        :param ordinal: An ordinal in the type library
        :param name: A valid type name
        :param pt_flags: Parsing flags
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, r: tinfo_t) -> bool:
        ...
    def __lt__(self, r: tinfo_t) -> bool:
        ...
    def __ne__(self, r: tinfo_t) -> bool:
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
    def __str__(self) -> str:
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
    def add_edm(self, args: Any) -> Any:
        r"""Add an enumerator to the current enumeration.
        
        When creating a new enumeration from scratch, you might
        want to first call `create_enum()`
        
        This method has the following signatures:
        
            1. add_edm(edm: edm_t, bmask: int = -1, etf_flags: int = 0, idx: int = -1)
            2. add_edm(name: str, value: int, bmask: int = -1, etf_flags: int = 0, idx: int = -1)
        
        If an input argument is incorrect, the constructor may raise an exception
        
        :param edm:       The member, fully initialized (1st form)
        :param name:      Enumerator name - must not be empty
        :param value:     Enumerator value
        :param bmask:     A bitmask to which the enumerator belongs
        :param etf_flags: an OR'ed combination of ETF_ flags
        :param idx:       the index in the edm array where the new udm should be placed.
                          If the specified index cannot be honored because it would spoil
                          the edm sorting order, it is silently ignored.
        """
        ...
    def add_funcarg(self, farg: funcarg_t, etf_flags: uint = 0, idx: ssize_t = -1) -> tinfo_code_t:
        r"""Add a function argument. 
                
        :param farg: argument to add
        :param etf_flags: type changing flags flags
        :param idx: the index in the funcarg array where the new funcarg should be placed. if the specified index cannot be honored because it would spoil the funcarg sorting order, it is silently ignored.
        """
        ...
    def add_udm(self, args: Any) -> Any:
        r"""Add a member to the current structure/union.
        
        When creating a new structure/union from scratch, you might
        want to first call `create_udt()`
        
        This method has the following signatures:
        
            1. add_udm(udm: udm_t, etf_flags: int = 0, times: int = 1, idx: int = -1)
            2. add_udm(name: str, type: type_t | tinfo_t | str, offset: int = 0, etf_flags: int = 0, times: int = 1, idx: int = -1)
        
        In the 2nd form, the 'type' descriptor, can be one of:
        
        * type_t: if the type is simple (integral/floating/bool). E.g., `BTF_INT`
        * tinfo_t: can handle more complex types (structures, pointers, arrays, ...)
        * str: a C type declaration
        
        If an input argument is incorrect, the constructor may raise an exception
        
        :param udm:       The member, fully initialized (1st form)
        :param name:      Member name - must not be empty
        :param type:      Member type
        :param offset:    the member offset in bits. It is the caller's responsibility
                          to specify correct offsets.
        :param etf_flags: an OR'ed combination of ETF_ flags
        :param times:     how many times to add the new member
        :param idx:       the index in the udm array where the new udm should be placed.
                          If the specified index cannot be honored because it would spoil
                          the udm sorting order, it is silently ignored.
        """
        ...
    def append_covered(self, out: rangeset_t, offset: uint64 = 0) -> bool:
        r"""Calculate set of covered bytes for the type 
                
        :param out: pointer to the output buffer. covered bytes will be appended to it.
        :param offset: delta in bytes to add to all calculations. used internally during recurion.
        """
        ...
    def calc_enum_mask(self) -> uint64:
        ...
    def calc_gaps(self, out: rangeset_t) -> bool:
        r"""Calculate set of padding bytes for the type 
                
        :param out: pointer to the output buffer; old buffer contents will be lost.
        """
        ...
    def calc_purged_bytes(self) -> int:
        r"""BT_FUNC: Calculate number of purged bytes
        
        """
        ...
    def calc_score(self) -> int:
        r"""Calculate the type score (the higher - the nicer is the type)
        
        """
        ...
    def calc_udt_aligns(self, sudt_flags: int = 4) -> bool:
        r"""Calculate the udt alignments using the field offsets/sizes and the total udt size This function does not work on typerefs 
                
        """
        ...
    def change_sign(self, sign: type_sign_t) -> bool:
        r"""Change the type sign. Works only for the types that may have sign.
        
        """
        ...
    def clear(self) -> None:
        r"""Clear contents of this tinfo, and remove from the type system.
        
        """
        ...
    def clr_const(self) -> bool:
        ...
    def clr_const_volatile(self) -> bool:
        ...
    def clr_decl_const_volatile(self) -> None:
        ...
    def clr_volatile(self) -> bool:
        ...
    def compare(self, r: tinfo_t) -> int:
        ...
    def compare_with(self, r: tinfo_t, tcflags: int = 0) -> bool:
        r"""Compare two types, based on given flags (see tinfo_t comparison flags)
        
        """
        ...
    def convert_array_to_ptr(self) -> bool:
        r"""Convert an array into a pointer. type[] => type * 
                
        """
        ...
    def copy(self) -> tinfo_t:
        ...
    def copy_type(self, args: Any) -> tinfo_code_t:
        ...
    def create_array(self, args: Any) -> bool:
        ...
    def create_bitfield(self, args: Any) -> bool:
        ...
    def create_enum(self, args: Any) -> bool:
        r"""Create an empty enum.
        
        """
        ...
    def create_forward_decl(self, til: til_t, decl_type: type_t, name: str, ntf_flags: int = 0) -> tinfo_code_t:
        r"""Create a forward declaration. decl_type: BTF_STRUCT, BTF_UNION, or BTF_ENUM 
                
        """
        ...
    def create_func(self, args: Any) -> bool:
        ...
    def create_ptr(self, args: Any) -> bool:
        ...
    def create_simple_type(self, decl_type: type_t) -> bool:
        ...
    def create_typedef(self, args: Any) -> None:
        ...
    def create_udt(self, args: Any) -> bool:
        r"""Create an empty structure/union.
        
        """
        ...
    def del_attr(self, key: str, make_copy: bool = True) -> bool:
        r"""Del a type attribute. typerefs cannot be modified by this function.
        
        """
        ...
    def del_attrs(self) -> None:
        r"""Del all type attributes. typerefs cannot be modified by this function.
        
        """
        ...
    def del_edm(self, args: Any) -> Any:
        r"""Delete an enumerator with the specified name
        or the specified index, in the specified tinfo_t object.
        
        This method has the following signatures:
        
            1. del_edm(name: str) -> int
            2. del_edm(index: int) -> int
        
        :param name: an enumerator name (1st form)
        :param index: an enumerator index (2nd form)
        :returns: TERR_OK in case of success, or another TERR_* value in case of error
        """
        ...
    def del_edm_by_value(self, value: int, etf_flags: int = 0, bmask: int = 18446744073709551615, serial: int = 0) -> Any:
        r"""Delete an enumerator with the specified value,
        in the specified tinfo_t object.
        
        :param value: the enumerator value
        :returns: TERR_OK in case of success, or another TERR_* value in case of error
        """
        ...
    def del_edms(self, idx1: size_t, idx2: size_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Delete enum members 
                
        :param idx1: index in edmvec_t
        :param idx2: index in edmvec_t or size_t(-1)
        :param etf_flags: etf_flag_t Delete enum members in [idx1, idx2)
        """
        ...
    def del_funcarg(self, idx: size_t, etf_flags: uint = 0) -> tinfo_code_t:
        ...
    def del_funcargs(self, idx1: size_t, idx2: size_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Delete function arguments 
                
        :param idx1: index in funcargvec_t
        :param idx2: index in funcargvec_t or size_t(-1)
        :param etf_flags: etf_flag_t Delete function arguments in [idx1, idx2)
        """
        ...
    def del_udm(self, index: size_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Delete a structure/union member.
        
        """
        ...
    def del_udms(self, idx1: size_t, idx2: size_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Delete structure/union members in the range [idx1, idx2)
        
        """
        ...
    def deserialize(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. deserialize(til: const til_t *, ptype: const type_t **, pfields: const p_list **=nullptr, pfldcmts: const p_list **=nullptr, cmt: str=nullptr) -> bool
            1. deserialize(til: const til_t *, ptype: const qtype *, pfields: const qtype *=nullptr, pfldcmts: const qtype *=nullptr, cmt: str=nullptr) -> bool
        
        # 0: deserialize(til: const til_t *, ptype: const type_t **, pfields: const p_list **=nullptr, pfldcmts: const p_list **=nullptr, cmt: str=nullptr) -> bool
        
        Deserialize a type string into a tinfo_t object.
        
        
        # 1: deserialize(til: const til_t *, ptype: const qtype *, pfields: const qtype *=nullptr, pfldcmts: const qtype *=nullptr, cmt: str=nullptr) -> bool
        
        Deserialize a type string into a tinfo_t object.
        
        
        """
        ...
    def detach(self) -> bool:
        r"""Detach tinfo_t from the underlying type. After calling this finction, tinfo_t will lose its link to the underlying named or numbered type (if any) and will become a reference to a unique type. After that, any modifications to tinfo_t will affect only its type. 
                
        """
        ...
    def dstr(self) -> str:
        r"""Function to facilitate debugging.
        
        """
        ...
    def edit_edm(self, args: Any) -> tinfo_code_t:
        r"""Change constant value and/or bitmask 
                
        :param idx: index in edmvec_t
        :param value: old or new value
        :param bmask: old or new bitmask
        :param etf_flags: etf_flag_t
        """
        ...
    def empty(self) -> bool:
        r"""Was tinfo_t initialized with some type info or not?
        
        """
        ...
    def equals_to(self, r: tinfo_t) -> bool:
        ...
    def expand_udt(self, idx: size_t, delta: adiff_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Expand/shrink a structure by adding/removing a gap before the specified member.
        For regular structures, either the gap can be accommodated by aligning the next member with an alignment directive, or an explicit "gap" member will be inserted. Also note that it is impossible to add a gap at the end of a regular structure.
        When it comes to fixed-layout structures, there is no need to either add new "gap" members or align existing members, since all members have a fixed offset. It is possible to add a gap at the end of a fixed-layout structure, by passing `-1` as index.
        
        :param idx: index of the member
        :param delta: number of bytes to add or remove
        :param etf_flags: etf_flag_t
        """
        ...
    def find_edm(self, args: Any) -> ssize_t:
        ...
    def find_udm(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. find_udm(udm: udm_t *, strmem_flags: int) -> int
            1. find_udm(offset: uint64, strmem_flags: int=0) -> int
            2. find_udm(name: str, strmem_flags: int=0) -> int
        
        # 0: find_udm(udm: udm_t *, strmem_flags: int) -> int
        
        BTF_STRUCT,BTF_UNION: Find a udt member.
        * at the specified offset (STRMEM_OFFSET)
        * with the specified index (STRMEM_INDEX)
        * with the specified type (STRMEM_TYPE)
        * with the specified name (STRMEM_NAME)
        
        
        
        :returns: the index of the found member or -1
        
        # 1: find_udm(offset: uint64, strmem_flags: int=0) -> int
        
        BTF_STRUCT,BTF_UNION: Find an udt member at the specified offset 
                
        :returns: the index of the found member or -1
        
        # 2: find_udm(name: str, strmem_flags: int=0) -> int
        
        BTF_STRUCT,BTF_UNION: Find an udt member by name 
                
        :returns: the index of the found member or -1
        
        """
        ...
    def find_udt_member(self, args: Any) -> int:
        r"""This function has the following signatures:
        
            0. find_udm(udm: udm_t *, strmem_flags: int) -> int
            1. find_udm(offset: uint64, strmem_flags: int=0) -> int
            2. find_udm(name: str, strmem_flags: int=0) -> int
        
        # 0: find_udm(udm: udm_t *, strmem_flags: int) -> int
        
        BTF_STRUCT,BTF_UNION: Find a udt member.
        * at the specified offset (STRMEM_OFFSET)
        * with the specified index (STRMEM_INDEX)
        * with the specified type (STRMEM_TYPE)
        * with the specified name (STRMEM_NAME)
        
        
        
        :returns: the index of the found member or -1
        
        # 1: find_udm(offset: uint64, strmem_flags: int=0) -> int
        
        BTF_STRUCT,BTF_UNION: Find an udt member at the specified offset 
                
        :returns: the index of the found member or -1
        
        # 2: find_udm(name: str, strmem_flags: int=0) -> int
        
        BTF_STRUCT,BTF_UNION: Find an udt member by name 
                
        :returns: the index of the found member or -1
        
        """
        ...
    def force_tid(self) -> tid_t:
        r"""Get the type tid. Create if it does not exist yet. If the type comes from a base til, the type will be copied to the local til and a new tid will be created for it. (if the type comes from a base til, it does not have a tid yet). If the type comes from the local til, this function is equivalent to get_tid() 
                
        :returns: tid or BADADDR
        """
        ...
    def get_alias_target(self) -> int:
        r"""Get type alias If the type has no alias, return 0. 
                
        """
        ...
    def get_alignment(self) -> int:
        r"""Get type alignment This function returns the effective type alignment. Zero means error. 
                
        """
        ...
    def get_array_details(self, ai: array_type_data_t) -> bool:
        r"""Get the array specific info.
        
        """
        ...
    def get_array_element(self) -> tinfo_t:
        r"""BT_ARRAY: get type of array element. See also get_ptrarr_object()
        
        """
        ...
    def get_array_nelems(self) -> int:
        r"""BT_ARRAY: get number of elements (-1 means error)
        
        """
        ...
    def get_attr(self, key: str, all_attrs: bool = True) -> Any:
        r"""Get a type attribute.
        
        """
        ...
    def get_attrs(self, tav: type_attrs_t, all_attrs: bool = False) -> bool:
        r"""Get type attributes (all_attrs: include attributes of referenced types, if any)
        
        """
        ...
    def get_bit_buckets(self, buckets: range64vec_t) -> bool:
        r"""::BT_STRUCT: get bit buckets Bit buckets are used to layout bitfields 
                
        :returns: false if wrong type was passed
        """
        ...
    def get_bitfield_details(self, bi: bitfield_type_data_t) -> bool:
        r"""Get the bitfield specific info.
        
        """
        ...
    def get_by_edm_name(self, mname: str, til: til_t = None) -> ssize_t:
        r"""Retrieve enum tinfo using enum member name 
                
        :param mname: enum type member name
        :param til: type library
        :returns: member index, otherwise returns -1. If the function fails, THIS object becomes empty.
        """
        ...
    def get_declalign(self) -> uchar:
        r"""Get declared alignment of the type.
        
        """
        ...
    def get_decltype(self) -> type_t:
        r"""Get declared type (without resolving type references; they are returned as is). Obviously this is a very fast function and should be used instead of get_realtype() if possible. Please note that for typerefs this function will return BTF_TYPEDEF. To determine if a typeref is a typedef, use is_typedef() 
                
        """
        ...
    def get_edm(self, args: Any) -> Any:
        r"""Retrieve an enumerator with either the specified name
        or the specified index, in the specified tinfo_t object.
        
        This function has the following signatures:
        
            1. get_edm(index: int)
            2. get_edm(name: str)
        
        :param index: an enumerator index (1st form).
        :param name: an enumerator name (2nd form).
        :returns: a tuple (int, edm_t), or (-1, None) if member not found
        """
        ...
    def get_edm_by_name(self, mname: str, til: til_t = None) -> ssize_t:
        r"""Retrieve enum tinfo using enum member name 
                
        :param mname: enum type member name
        :param til: type library
        :returns: member index, otherwise returns -1. If the function fails, THIS object becomes empty.
        """
        ...
    def get_edm_by_tid(self, edm: edm_t, tid: tid_t) -> ssize_t:
        ...
    def get_edm_by_value(self, value: int, bmask: int = 18446744073709551615, serial: int = 0) -> Any:
        r"""Retrieve an enumerator with the specified value,
        in the specified tinfo_t object.
        
        :param value: the enumerator value
        :returns: a tuple (int, edm_t), or (-1, None) if member not found
        """
        ...
    def get_edm_tid(self, idx: size_t) -> tid_t:
        r"""Get enum member TID 
                
        :param idx: enum member index
        :returns: tid or BADADDR The tid is used to collect xrefs to the member, it can be passed to xref-related functions instead of the address.
        """
        ...
    def get_enum_base_type(self) -> type_t:
        r"""Get enum base type (convert enum to integer type) Returns BT_UNK if failed to convert 
                
        """
        ...
    def get_enum_details(self, ei: enum_type_data_t) -> bool:
        r"""Get the enum specific info.
        
        """
        ...
    def get_enum_nmembers(self) -> int:
        r"""Get number of enum members. 
                
        :returns: BADSIZE if error
        """
        ...
    def get_enum_radix(self) -> int:
        r"""Get enum constant radix 
                
        :returns: radix or 1 for BTE_CHAR enum_type_data_t::get_enum_radix()
        """
        ...
    def get_enum_repr(self, repr: value_repr_t) -> tinfo_code_t:
        r"""Set the representation of enum members. 
                
        :param repr: value_repr_t
        """
        ...
    def get_enum_width(self) -> int:
        r"""Get enum width 
                
        :returns: width of enum base type in bytes, 0 - unspecified, or -1 enum_type_data_t::calc_nbytes()
        """
        ...
    def get_final_element(self) -> tinfo_t:
        r"""repeat recursively: if an array, return the type of its element; else return the type itself.
        
        """
        ...
    def get_final_ordinal(self) -> int:
        r"""Get final type ordinal (0 if none)
        
        """
        ...
    def get_final_type_name(self) -> bool:
        r"""Use in the case of typedef chain (TYPE1 -> TYPE2 -> TYPE3...TYPEn). 
                
        :returns: the name of the last type in the chain (TYPEn). if there is no chain, returns TYPE1
        """
        ...
    def get_forward_type(self) -> type_t:
        r"""Get type of a forward declaration. For a forward declaration this function returns its base type. In other cases it returns BT_UNK 
                
        """
        ...
    def get_frame_func(self) -> ida_idaapi.ea_t:
        r"""Get function address for the frame.
        
        """
        ...
    def get_func_details(self, fi: func_type_data_t, gtd: gtd_func_t = 0) -> bool:
        r"""Get only the function specific info for this tinfo_t.
        
        """
        ...
    def get_func_frame(self, pfn: func_t) -> bool:
        r"""Create a tinfo_t object for the function frame 
                
        :param pfn: function
        """
        ...
    def get_innermost_member_type(self, bitoffset: uint64) -> tinfo_t:
        r"""Get the innermost member type at the given offset 
                
        :param bitoffset: bit offset into the structure
        :returns: the: innermost member type
        """
        ...
    def get_innermost_udm(self, bitoffset: uint64) -> tinfo_t:
        r"""Get the innermost member at the given offset 
                
        :param bitoffset: bit offset into the structure
        :returns: udt: with the innermost member
        :returns: empty: type if it is not a struct type or OFFSET could not be found
        """
        ...
    def get_methods(self, methods: udtmembervec_t) -> bool:
        r"""BT_COMPLEX: get a list of member functions declared in this udt. 
                
        :returns: false if no member functions exist
        """
        ...
    def get_modifiers(self) -> type_t:
        ...
    def get_named_type(self, args: Any) -> bool:
        r"""This function has the following signatures:
        
            0. get_named_type(til: const til_t *, name: str, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true, try_ordinal: bool=true) -> bool
            1. get_named_type(name: str, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true, try_ordinal: bool=true) -> bool
        
        # 0: get_named_type(til: const til_t *, name: str, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true, try_ordinal: bool=true) -> bool
        
        Create a tinfo_t object for an existing named type. 
                
        
        # 1: get_named_type(name: str, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true, try_ordinal: bool=true) -> bool
        
        
        """
        ...
    def get_nargs(self) -> int:
        r"""BT_FUNC or BT_PTR BT_FUNC: Calculate number of arguments (-1 - error)
        
        """
        ...
    def get_next_type_name(self) -> bool:
        r"""Use In the case of typedef chain (TYPE1 -> TYPE2 -> TYPE3...TYPEn). 
                
        :returns: the name of the next type in the chain (TYPE2). if there is no chain, returns failure
        """
        ...
    def get_nice_type_name(self) -> bool:
        r"""Get the beautified type name. Get the referenced name and apply regular expressions from goodname.cfg to beautify the name 
                
        """
        ...
    def get_nth_arg(self, n: int) -> tinfo_t:
        r"""BT_FUNC or BT_PTR BT_FUNC: Get type of n-th arg (-1 means return type, see get_rettype())
        
        """
        ...
    def get_numbered_type(self, args: Any) -> Any:
        r"""This function has the following signatures:
        
            0. get_numbered_type(til: const til_t *, ordinal: int, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true) -> bool
            1. get_numbered_type(ordinal: int, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true) -> bool
        
        # 0: get_numbered_type(til: const til_t *, ordinal: int, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true) -> bool
        
        Create a tinfo_t object for an existing ordinal type. 
                
        
        # 1: get_numbered_type(ordinal: int, decl_type: type_t=BTF_TYPEDEF, resolve: bool=true) -> bool
        
        
        """
        ...
    def get_onemember_type(self) -> tinfo_t:
        r"""For objects consisting of one member entirely: return type of the member.
        
        """
        ...
    def get_ordinal(self) -> int:
        r"""Get type ordinal (only if the type was created as a numbered type, 0 if none)
        
        """
        ...
    def get_pointed_object(self) -> tinfo_t:
        r"""BT_PTR: get type of pointed object. If the current type is not a pointer, return empty type info. See also get_ptrarr_object() and remove_pointer() 
                
        """
        ...
    def get_ptr_details(self, pi: ptr_type_data_t) -> bool:
        r"""Get the pointer info.
        
        """
        ...
    def get_ptrarr_object(self) -> tinfo_t:
        r"""BT_PTR & BT_ARRAY: get the pointed object or array element. If the current type is not a pointer or array, return empty type info. 
                
        """
        ...
    def get_ptrarr_objsize(self) -> int:
        r"""BT_PTR & BT_ARRAY: get size of pointed object or array element. On error returns -1
        
        """
        ...
    def get_realtype(self, full: bool = False) -> type_t:
        r"""Get the resolved base type. Deserialization options:
        * if full=true, the referenced type will be deserialized fully, this may not always be desirable (slows down things)
        * if full=false, we just return the base type, the referenced type will be resolved again later if necessary (this may lead to multiple resolvings of the same type) imho full=false is a better approach because it does not perform unnecessary actions just in case. however, in some cases the caller knows that it is very likely that full type info will be required. in those cases full=true makes sense 
        
        
                
        """
        ...
    def get_rettype(self) -> tinfo_t:
        r"""BT_FUNC or BT_PTR BT_FUNC: Get the function's return type
        
        """
        ...
    def get_sign(self) -> type_sign_t:
        r"""Get type sign.
        
        """
        ...
    def get_size(self, p_effalign: uint32 = None, gts_code: int = 0) -> int:
        r"""Get the type size in bytes. 
                
        :param p_effalign: buffer for the alignment value
        :param gts_code: combination of GTS_... constants
        :returns: BADSIZE in case of problems
        """
        ...
    def get_stkvar(self, insn: insn_t, x: op_t, v: int) -> ssize_t:
        r"""Retrieve frame tinfo for a stack variable 
                
        :param insn: the instruction
        :param x: reference to instruction operand, may be nullptr
        :param v: immediate value in the operand (usually x.addr)
        :returns: returns the member index, otherwise returns -1. if the function fails, THIS object becomes empty.
        """
        ...
    def get_stock(self, id: stock_type_id_t) -> tinfo_t:
        r"""Get stock type information. This function can be used to get tinfo_t for some common types. The same tinfo_t will be returned for the same id, thus saving memory and increasing the speed Please note that retrieving the STI_SIZE_T or STI_SSIZE_T stock type, will also have the side-effect of adding that type to the 'idati' TIL, under the well-known name 'size_t' or 'ssize_t' (respectively). The same is valid for STI_COMPLEX64 and STI_COMPLEX64 stock types with names 'complex64_t' and 'complex128_t' (respectively). 
                
        """
        ...
    def get_tid(self) -> tid_t:
        r"""Get the type tid Each type in the local type library has a so-called `tid` associated with it. The tid is used to collect xrefs to the type. The tid is created when the type is created in the local type library and does not change afterwards. It can be passed to xref-related functions instead of the address. 
                
        :returns: tid or BADADDR
        """
        ...
    def get_til(self) -> til_t:
        r"""Get the type library for tinfo_t.
        
        """
        ...
    def get_type_by_tid(self, tid: tid_t) -> bool:
        ...
    def get_type_cmt(self) -> int:
        r"""Get type comment 
                
        :returns: 0-failed, 1-returned regular comment, 2-returned repeatable comment
        """
        ...
    def get_type_name(self) -> bool:
        r"""Does a type refer to a name?. If yes, fill the provided buffer with the type name and return true. Names are returned for numbered types too: either a user-defined nice name or, if a user-provided name does not exist, an ordinal name (like #xx, see create_numbered_type_name()). 
                
        """
        ...
    def get_type_rptcmt(self) -> bool:
        r"""Get type comment only if it is repeatable.
        
        """
        ...
    def get_udm(self, args: Any) -> Any:
        r"""Retrieve a structure/union member with either the specified name
        or the specified index, in the specified tinfo_t object.
        
        This function has the following signatures:
        
            1. get_udm(index: int)
            2. get_udm(name: str)
        
        :param index: a member index (1st form)
        :param name: a member name (2nd form)
        :returns: a tuple (int, udm_t), or (-1, None) if member not found
        """
        ...
    def get_udm_by_offset(self, offset: int) -> Any:
        r"""Retrieve a structure/union member with the specified offset,
        in the specified tinfo_t object.
        
        :param offset: the member offset
        :returns: a tuple (int, udm_t), or (-1, None) if member not found
        """
        ...
    def get_udm_by_tid(self, udm: udm_t, tid: tid_t) -> ssize_t:
        ...
    def get_udm_tid(self, idx: size_t) -> tid_t:
        r"""Get udt member TID 
                
        :param idx: the index of udt the member
        :returns: tid or BADADDR The tid is used to collect xrefs to the member, it can be passed to xref-related functions instead of the address.
        """
        ...
    def get_udt_details(self, udt: udt_type_data_t, gtd: gtd_udt_t = 0) -> bool:
        r"""Get the udt specific info.
        
        """
        ...
    def get_udt_nmembers(self) -> int:
        r"""Get number of udt members. -1-error.
        
        """
        ...
    def get_udt_taudt_bits(self) -> int:
        r"""Get udt_type_data_t::taudt_bits.
        
        """
        ...
    def get_unpadded_size(self) -> int:
        r"""Get the type size in bytes without the final padding, in bytes. For some UDTs get_unpadded_size() != get_size() 
                
        """
        ...
    def has_details(self) -> bool:
        r"""Does this type refer to a nontrivial type?
        
        """
        ...
    def has_union(self) -> bool:
        r"""Has a member of type "union"?
        
        """
        ...
    def has_vftable(self) -> bool:
        r"""Has a vftable?
        
        """
        ...
    def is_aliased(self) -> bool:
        ...
    def is_anonymous_udt(self) -> bool:
        r"""Is an anonymous struct/union? We assume that types with names are anonymous if the name starts with $ 
                
        """
        ...
    def is_arithmetic(self) -> bool:
        r"""is_type_arithmetic(get_realtype())
        
        """
        ...
    def is_array(self) -> bool:
        r"""is_type_array(get_realtype())
        
        """
        ...
    def is_bitfield(self) -> bool:
        r"""is_type_bitfld(get_realtype())
        
        """
        ...
    def is_bitmask_enum(self) -> bool:
        r"""Is bitmask enum? 
                
        :returns: true for bitmask enum and false in other cases enum_type_data_t::is_bf()
        """
        ...
    def is_bool(self) -> bool:
        r"""is_type_bool(get_realtype())
        
        """
        ...
    def is_castable_to(self, target: tinfo_t) -> bool:
        ...
    def is_char(self) -> bool:
        r"""is_type_char(get_realtype())
        
        """
        ...
    def is_complex(self) -> bool:
        r"""is_type_complex(get_realtype())
        
        """
        ...
    def is_const(self) -> bool:
        r"""is_type_const(get_realtype())
        
        """
        ...
    def is_correct(self) -> bool:
        r"""Is the type object correct?. It is possible to create incorrect types. For example, we can define a function that returns an enum and then delete the enum type. If this function returns false, the type should not be used in disassembly. Please note that this function does not verify all involved types: for example, pointers to undefined types are permitted. 
                
        """
        ...
    def is_cpp_struct(self) -> bool:
        r"""Is a c++ object, not simple pod type.
        
        """
        ...
    def is_decl_array(self) -> bool:
        r"""is_type_array(get_decltype())
        
        """
        ...
    def is_decl_bitfield(self) -> bool:
        r"""is_type_bitfld(get_decltype())
        
        """
        ...
    def is_decl_bool(self) -> bool:
        r"""is_type_bool(get_decltype())
        
        """
        ...
    def is_decl_char(self) -> bool:
        r"""is_type_char(get_decltype())
        
        """
        ...
    def is_decl_complex(self) -> bool:
        r"""is_type_complex(get_decltype())
        
        """
        ...
    def is_decl_const(self) -> bool:
        r"""is_type_const(get_decltype())
        
        """
        ...
    def is_decl_double(self) -> bool:
        r"""is_type_double(get_decltype())
        
        """
        ...
    def is_decl_enum(self) -> bool:
        r"""is_type_enum(get_decltype())
        
        """
        ...
    def is_decl_float(self) -> bool:
        r"""is_type_float(get_decltype())
        
        """
        ...
    def is_decl_floating(self) -> bool:
        r"""is_type_floating(get_decltype())
        
        """
        ...
    def is_decl_func(self) -> bool:
        r"""is_type_func(get_decltype())
        
        """
        ...
    def is_decl_int(self) -> bool:
        r"""is_type_int(get_decltype())
        
        """
        ...
    def is_decl_int128(self) -> bool:
        r"""is_type_int128(get_decltype())
        
        """
        ...
    def is_decl_int16(self) -> bool:
        r"""is_type_int16(get_decltype())
        
        """
        ...
    def is_decl_int32(self) -> bool:
        r"""is_type_int32(get_decltype())
        
        """
        ...
    def is_decl_int64(self) -> bool:
        r"""is_type_int64(get_decltype())
        
        """
        ...
    def is_decl_last(self) -> bool:
        r"""is_typeid_last(get_decltype())
        
        """
        ...
    def is_decl_ldouble(self) -> bool:
        r"""is_type_ldouble(get_decltype())
        
        """
        ...
    def is_decl_paf(self) -> bool:
        r"""is_type_paf(get_decltype())
        
        """
        ...
    def is_decl_partial(self) -> bool:
        r"""is_type_partial(get_decltype())
        
        """
        ...
    def is_decl_ptr(self) -> bool:
        r"""is_type_ptr(get_decltype())
        
        """
        ...
    def is_decl_struct(self) -> bool:
        r"""is_type_struct(get_decltype())
        
        """
        ...
    def is_decl_sue(self) -> bool:
        r"""is_type_sue(get_decltype())
        
        """
        ...
    def is_decl_tbyte(self) -> bool:
        r"""is_type_tbyte(get_decltype())
        
        """
        ...
    def is_decl_typedef(self) -> bool:
        r"""is_type_typedef(get_decltype())
        
        """
        ...
    def is_decl_uchar(self) -> bool:
        r"""is_type_uchar(get_decltype())
        
        """
        ...
    def is_decl_udt(self) -> bool:
        r"""is_type_struni(get_decltype())
        
        """
        ...
    def is_decl_uint(self) -> bool:
        r"""is_type_uint(get_decltype())
        
        """
        ...
    def is_decl_uint128(self) -> bool:
        r"""is_type_uint128(get_decltype())
        
        """
        ...
    def is_decl_uint16(self) -> bool:
        r"""is_type_uint16(get_decltype())
        
        """
        ...
    def is_decl_uint32(self) -> bool:
        r"""is_type_uint32(get_decltype())
        
        """
        ...
    def is_decl_uint64(self) -> bool:
        r"""is_type_uint64(get_decltype())
        
        """
        ...
    def is_decl_union(self) -> bool:
        r"""is_type_union(get_decltype())
        
        """
        ...
    def is_decl_unknown(self) -> bool:
        r"""is_type_unknown(get_decltype())
        
        """
        ...
    def is_decl_void(self) -> bool:
        r"""is_type_void(get_decltype())
        
        """
        ...
    def is_decl_volatile(self) -> bool:
        r"""is_type_volatile(get_decltype())
        
        """
        ...
    def is_double(self) -> bool:
        r"""is_type_double(get_realtype())
        
        """
        ...
    def is_empty_enum(self) -> bool:
        r"""Is an empty enum? (has no constants)
        
        """
        ...
    def is_empty_udt(self) -> bool:
        r"""Is an empty struct/union? (has no fields)
        
        """
        ...
    def is_enum(self) -> bool:
        r"""is_type_enum(get_realtype())
        
        """
        ...
    def is_ext_arithmetic(self) -> bool:
        r"""is_type_ext_arithmetic(get_realtype()) 
                
        """
        ...
    def is_ext_integral(self) -> bool:
        r"""is_type_ext_integral(get_realtype())
        
        """
        ...
    def is_fixed_struct(self) -> bool:
        r"""Is a structure with fixed offsets?
        
        """
        ...
    def is_float(self) -> bool:
        r"""is_type_float(get_realtype())
        
        """
        ...
    def is_floating(self) -> bool:
        r"""is_type_floating(get_realtype())
        
        """
        ...
    def is_forward_decl(self) -> bool:
        r"""Is this a forward declaration?. Forward declarations are placeholders: the type definition does not exist 
                
        """
        ...
    def is_forward_enum(self) -> bool:
        ...
    def is_forward_struct(self) -> bool:
        ...
    def is_forward_union(self) -> bool:
        ...
    def is_frame(self) -> bool:
        r"""Is a function frame?
        
        """
        ...
    def is_from_subtil(self) -> bool:
        r"""Was the named type found in some base type library (not the top level type library)?. If yes, it usually means that the type comes from some loaded type library, not the local type library for the database 
                
        """
        ...
    def is_func(self) -> bool:
        r"""is_type_func(get_realtype())
        
        """
        ...
    def is_funcptr(self) -> bool:
        r"""Is this pointer to a function?
        
        """
        ...
    def is_high_func(self) -> bool:
        r"""BT_FUNC: Is high level type?
        
        """
        ...
    def is_int(self) -> bool:
        r"""is_type_int(get_realtype())
        
        """
        ...
    def is_int128(self) -> bool:
        r"""is_type_int128(get_realtype())
        
        """
        ...
    def is_int16(self) -> bool:
        r"""is_type_int16(get_realtype())
        
        """
        ...
    def is_int32(self) -> bool:
        r"""is_type_int32(get_realtype())
        
        """
        ...
    def is_int64(self) -> bool:
        r"""is_type_int64(get_realtype())
        
        """
        ...
    def is_integral(self) -> bool:
        r"""is_type_integral(get_realtype())
        
        """
        ...
    def is_ldouble(self) -> bool:
        r"""is_type_ldouble(get_realtype())
        
        """
        ...
    def is_manually_castable_to(self, target: tinfo_t) -> bool:
        ...
    def is_msstruct(self) -> bool:
        r"""Is gcc msstruct attribute applied.
        
        """
        ...
    def is_one_fpval(self) -> bool:
        r"""Floating value or an object consisting of one floating member entirely.
        
        """
        ...
    def is_paf(self) -> bool:
        r"""is_type_paf(get_realtype())
        
        """
        ...
    def is_partial(self) -> bool:
        r"""is_type_partial(get_realtype())
        
        """
        ...
    def is_ptr(self) -> bool:
        r"""is_type_ptr(get_realtype())
        
        """
        ...
    def is_ptr_or_array(self) -> bool:
        r"""is_type_ptr_or_array(get_realtype())
        
        """
        ...
    def is_punknown(self) -> bool:
        r"""Is "_UNKNOWN *"?. This function does not check the pointer attributes and type modifiers.
        
        """
        ...
    def is_purging_cc(self) -> bool:
        r"""is_purging_cc(get_cc())
        
        """
        ...
    def is_pvoid(self) -> bool:
        r"""Is "void *"?. This function does not check the pointer attributes and type modifiers.
        
        """
        ...
    def is_scalar(self) -> bool:
        r"""Does the type represent a single number?
        
        """
        ...
    def is_shifted_ptr(self) -> bool:
        r"""Is a shifted pointer?
        
        """
        ...
    def is_signed(self) -> bool:
        r"""Is this a signed type?
        
        """
        ...
    def is_small_udt(self) -> bool:
        r"""Is a small udt? (can fit a register or a pair of registers)
        
        """
        ...
    def is_sse_type(self) -> bool:
        r"""Is a SSE vector type?
        
        """
        ...
    def is_struct(self) -> bool:
        r"""is_type_struct(get_realtype())
        
        """
        ...
    def is_sue(self) -> bool:
        r"""is_type_sue(get_realtype())
        
        """
        ...
    def is_tbyte(self) -> bool:
        r"""is_type_tbyte(get_realtype())
        
        """
        ...
    def is_tuple(self) -> bool:
        r"""Is a tuple?
        
        """
        ...
    def is_typedef(self) -> bool:
        r"""Is this a typedef?. This function will return true for a reference to a local type that is declared as a typedef. 
                
        """
        ...
    def is_typeref(self) -> bool:
        r"""Is this type a type reference?.
        
        """
        ...
    def is_uchar(self) -> bool:
        r"""is_type_uchar(get_realtype())
        
        """
        ...
    def is_udm_by_til(self, idx: size_t) -> bool:
        r"""Was the member created due to the type system 
                
        :param idx: index of the member
        """
        ...
    def is_udt(self) -> bool:
        r"""is_type_struni(get_realtype())
        
        """
        ...
    def is_uint(self) -> bool:
        r"""is_type_uint(get_realtype())
        
        """
        ...
    def is_uint128(self) -> bool:
        r"""is_type_uint128(get_realtype())
        
        """
        ...
    def is_uint16(self) -> bool:
        r"""is_type_uint16(get_realtype())
        
        """
        ...
    def is_uint32(self) -> bool:
        r"""is_type_uint32(get_realtype())
        
        """
        ...
    def is_uint64(self) -> bool:
        r"""is_type_uint64(get_realtype())
        
        """
        ...
    def is_unaligned_struct(self) -> bool:
        r"""Is an unaligned struct.
        
        """
        ...
    def is_union(self) -> bool:
        r"""is_type_union(get_realtype())
        
        """
        ...
    def is_unknown(self) -> bool:
        r"""is_type_unknown(get_realtype())
        
        """
        ...
    def is_unsigned(self) -> bool:
        r"""Is this an unsigned type?
        
        """
        ...
    def is_user_cc(self) -> bool:
        r"""is_user_cc(get_cc())
        
        """
        ...
    def is_vararg_cc(self) -> bool:
        r"""is_vararg_cc(get_cc())
        
        """
        ...
    def is_varmember(self) -> bool:
        r"""Can the type be of a variable struct member? This function checks for: is_array() && array.nelems==0 Such a member can be only the very last member of a structure 
                
        """
        ...
    def is_varstruct(self) -> bool:
        r"""Is a variable-size structure?
        
        """
        ...
    def is_vftable(self) -> bool:
        r"""Is a vftable type?
        
        """
        ...
    def is_void(self) -> bool:
        r"""is_type_void(get_realtype())
        
        """
        ...
    def is_volatile(self) -> bool:
        r"""is_type_volatile(get_realtype())
        
        """
        ...
    def is_well_defined(self) -> bool:
        r"""!(empty()) && !(is_decl_partial()) && !(is_punknown())
        
        """
        ...
    def iter_enum(self) -> Any:
        r"""Iterate on the members composing this enumeration.
        
        Example:
        
            til = ida_typeinf.get_idati()
            tif = til.get_named_type("my_enum")
            for edm in tif.iter_enum():
                print(f"{edm.name} = {edm.value}")
        
        Will raise an exception if this type is not an enumeration
        
        :returns: a edm_t-producing generator
        """
        ...
    def iter_func(self) -> Any:
        r"""Iterate on the arguments contained in this function prototype
        
        Example:
        
            address = ...
            func = ida_funcs.get_func(address)
            func_type = func.prototype
            for arg in func_type.iter_func():
                print(f"{arg.name}, of type {arg.type}")
        
        Will raise an exception if this type is not a function
        
        :returns: a funcarg_t-producing generator
        """
        ...
    def iter_struct(self) -> Any:
        r"""Iterate on the members composing this structure.
        
        Example:
        
            til = ida_typeinf.get_idati()
            tif = til.get_named_type("my_struc")
            for udm in tif.iter_struct():
                print(f"{udm.name} at bit offset {udm.offset}")
        
        Will raise an exception if this type is not a structure.
        
        :returns: a udm_t-producing generator
        """
        ...
    def iter_udt(self) -> Any:
        r"""Iterate on the members composing this structure, or union.
        
        Example:
        
            til = ida_typeinf.get_idati()
            tif = til.get_named_type("my_type")
            for udm in tif.iter_udt():
                print(f"{udm.name} at bit offset {udm.offset} with type {udm.type}")
        
        Will raise an exception if this type is not a structure, or union
        
        :returns: a udm_t-producing generator
        """
        ...
    def iter_union(self) -> Any:
        r"""Iterate on the members composing this union.
        
        Example:
        
            til = ida_typeinf.get_idati()
            tif = til.get_named_type("my_union")
            for udm in tif.iter_union():
                print(f"{udm.name}, with type {udm.type}")
        
        Will raise an exception if this type is not a union.
        
        :returns: a udm_t-producing generator
        """
        ...
    def parse(self, decl: str, til: til_t = None, pt_flags: int = 0) -> bool:
        r"""Convenience function to parse a string with a type declaration 
                
        :param decl: a type declaration
        :param til: type library to use
        :param pt_flags: combination of Type parsing flags bits
        """
        ...
    def present(self) -> bool:
        r"""Is the type really present? (not a reference to a missing type, for example)
        
        """
        ...
    def read_bitfield_value(self, v: uint64, bitoff: int) -> uint64:
        ...
    def remove_ptr_or_array(self) -> bool:
        r"""Replace the current type with the ptr obj or array element. This function performs one of the following conversions:
        * type[] => type
        * type* => type If the conversion is performed successfully, return true 
        
        
                
        """
        ...
    def rename_edm(self, idx: size_t, name: str, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Rename a enum member 
                
        :param idx: index in edmvec_t
        :param name: new name
        :param etf_flags: etf_flag_t ETF_FORCENAME may be used in case of TERR_ALIEN_NAME
        """
        ...
    def rename_funcarg(self, index: size_t, name: str, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Rename a function argument. The new name must be unique. 
                
        :param index: argument index in the function array
        :param name: new name
        :param etf_flags: etf_flag_t
        """
        ...
    def rename_type(self, name: str, ntf_flags: int = 0) -> tinfo_code_t:
        r"""Rename a type 
                
        :param name: new type name
        :param ntf_flags: Flags for named types
        """
        ...
    def rename_udm(self, index: size_t, name: str, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Rename a structure/union member. The new name must be unique. 
                
        """
        ...
    def requires_qualifier(self, name: str, offset: uint64) -> bool:
        r"""Requires full qualifier? (name is not unique) 
                
        :param name: field name
        :param offset: field offset in bits
        :returns: if the name is not unique, returns true
        """
        ...
    def save_type(self, args: Any) -> tinfo_code_t:
        ...
    def serialize(self, args: Any) -> Any:
        r"""Serialize tinfo_t object into a type string.
        
        """
        ...
    def set_attr(self, ta: type_attr_t, may_overwrite: bool = True) -> bool:
        r"""Set a type attribute. If necessary, a new typid will be created.
        
        """
        ...
    def set_attrs(self, tav: type_attrs_t) -> bool:
        r"""Set type attributes. If necessary, a new typid will be created. this function modifies tav! (returns old attributes, if any) 
                
        :returns: false: bad attributes
        """
        ...
    def set_const(self) -> None:
        ...
    def set_declalign(self, declalign: uchar) -> bool:
        ...
    def set_edm_cmt(self, idx: size_t, cmt: str, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set a comment for an enum member. Such comments are always considered as repeatable. 
                
        :param idx: index in edmvec_t
        :param cmt: comment
        :param etf_flags: etf_flag_t
        """
        ...
    def set_enum_is_bitmask(self, args: Any) -> tinfo_code_t:
        ...
    def set_enum_radix(self, radix: int, sign: bool, etf_flags: uint = 0) -> None:
        r"""Set enum radix to display constants 
                
        :param radix: radix 2, 4, 8, 16, with the special case 1 to display as character
        :param sign: display as signed or unsigned
        :param etf_flags: etf_flag_t
        """
        ...
    def set_enum_repr(self, repr: value_repr_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set the representation of enum members. 
                
        :param repr: value_repr_t
        :param etf_flags: etf_flag_t
        """
        ...
    def set_enum_sign(self, sign: type_sign_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set enum sign 
                
        :param sign: type_sign_t
        :param etf_flags: etf_flag_t
        """
        ...
    def set_enum_width(self, nbytes: int, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set the width of enum base type 
                
        :param nbytes: width of enum base type, allowed values: 0 (unspecified),1,2,4,8,16,32,64
        :param etf_flags: etf_flag_t
        """
        ...
    def set_fixed_struct(self, on: bool = True) -> tinfo_code_t:
        r"""Declare struct member offsets as fixed. For such structures, IDA will not recalculate the member offsets. If a member does not fit into its place anymore, it will be deleted. This function works only with structures (not unions). 
                
        """
        ...
    def set_func_cc(self, cc: callcnv_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set function calling convention.
        
        """
        ...
    def set_func_retloc(self, argloc: argloc_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set location of function return value. 
                
        :param argloc: new location for the return value
        :param etf_flags: etf_flag_t
        :returns: tinfo_code_t
        """
        ...
    def set_func_rettype(self, tif: tinfo_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set function return type . 
                
        :param tif: new type for the return type
        :param etf_flags: etf_flag_t
        :returns: tinfo_code_t
        """
        ...
    def set_funcarg_loc(self, index: size_t, argloc: argloc_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set location of a function argument. 
                
        :param index: argument index in the function array
        :param argloc: new location for the argument
        :param etf_flags: etf_flag_t
        :returns: tinfo_code_t
        """
        ...
    def set_funcarg_type(self, index: size_t, tif: tinfo_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set type of a function argument. 
                
        :param index: argument index in the function array
        :param tif: new type for the argument
        :param etf_flags: etf_flag_t
        :returns: tinfo_code_t
        """
        ...
    def set_methods(self, methods: udtmembervec_t) -> bool:
        r"""BT_COMPLEX: set the list of member functions. This function consumes 'methods' (makes it empty). 
                
        :returns: false if this type is not a udt, or if the given list is empty
        """
        ...
    def set_modifiers(self, mod: type_t) -> None:
        ...
    def set_named_type(self, til: til_t, name: str, ntf_flags: int = 0) -> tinfo_code_t:
        ...
    def set_numbered_type(self, til: til_t, ord: int, ntf_flags: int = 0, name: str = None) -> tinfo_code_t:
        ...
    def set_struct_size(self, new_size: size_t) -> tinfo_code_t:
        r"""Explicitly specify the struct size. This function works only with fixed structures. The new struct size can be equal or higher the unpadded struct size (IOW, all existing members should fit into the specified size). 
                
        :param new_size: new structure size in bytes
        """
        ...
    def set_symbol_type(self, til: til_t, name: str, ntf_flags: int = 0) -> tinfo_code_t:
        ...
    def set_tuple(self, on: bool = True) -> None:
        r"""Declare struct as a tuple. Currently, tuples in IDA behave the same way as structures but they are returned in a different manner from functions. Also, 2 different tuples having the same members are considered to be equal. This function works only with structures (not unions). 
                
        """
        ...
    def set_type_alias(self, dest_ord: int) -> bool:
        r"""Set type alias Redirects all references to source type to the destination type. This is equivalent to instantaneous replacement all references to srctype by dsttype. 
                
        """
        ...
    def set_type_alignment(self, declalign: uchar, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set type alignment.
        
        """
        ...
    def set_type_cmt(self, cmt: str, is_regcmt: bool = False, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set type comment This function works only for non-trivial types 
                
        """
        ...
    def set_udm_by_til(self, idx: size_t, on: bool = True, etf_flags: uint = 0) -> tinfo_code_t:
        r"""The member is created due to the type system 
                
        :param idx: index of the member
        :param etf_flags: etf_flag_t
        """
        ...
    def set_udm_cmt(self, index: size_t, cmt: str, is_regcmt: bool = False, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set a comment for a structure/union member. A member may have just one comment, and it is either repeatable or regular. 
                
        """
        ...
    def set_udm_repr(self, index: size_t, repr: value_repr_t, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set the representation of a structure/union member.
        
        """
        ...
    def set_udm_type(self, index: size_t, tif: tinfo_t, etf_flags: uint = 0, repr: value_repr_t = None) -> tinfo_code_t:
        r"""Set type of a structure/union member. 
                
        :param index: member index in the udm array
        :param tif: new type for the member
        :param etf_flags: etf_flag_t
        :param repr: new representation for the member (optional)
        :returns: tinfo_code_t
        """
        ...
    def set_udt_alignment(self, sda: int, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set declared structure alignment (sda) This alignment supersedes the alignment returned by get_declalign() and is really used when calculating the struct layout. However, the effective structure alignment may differ from `sda` because of packing. The type editing functions (they accept etf_flags) may overwrite this attribute. 
                
        """
        ...
    def set_udt_pack(self, pack: int, etf_flags: uint = 0) -> tinfo_code_t:
        r"""Set structure packing. The value controls how little a structure member alignment can be. Example: if pack=1, then it is possible to align a double to a byte. __attribute__((aligned(1))) double x; However, if pack=3, a double will be aligned to 8 (2**3) even if requested to be aligned to a byte. pack==0 will have the same effect. The type editing functions (they accept etf_flags) may overwrite this attribute. 
                
        """
        ...
    def set_volatile(self) -> None:
        ...
    def swap(self, r: tinfo_t) -> None:
        r"""Assign this = r and r = this.
        
        """
        ...
    def write_bitfield_value(self, dst: uint64, v: uint64, bitoff: int) -> uint64:
        ...

class tinfo_visitor_t:
    @property
    def state(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __hash__(self) -> Any:
        r"""Return hash(self)."""
        ...
    def __init__(self, s: int = 0) -> Any:
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def apply_to(self, tif: tinfo_t, out: type_mods_t = None, name: str = None, cmt: str = None) -> int:
        r"""Call this function to initiate the traversal.
        
        """
        ...
    def prune_now(self) -> None:
        r"""To refuse to visit children of the current type, use this:
        
        """
        ...
    def visit_type(self, out: type_mods_t, tif: tinfo_t, name: str, cmt: str) -> int:
        r"""Visit a subtype. this function must be implemented in the derived class. it may optionally fill out with the new type info. this can be used to modify types (in this case the 'out' argument of apply_to() may not be nullptr) return 0 to continue the traversal. return !=0 to stop the traversal. 
                
        """
        ...

class type_attr_t:
    @property
    def key(self) -> Any: ...
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
    def __ge__(self, r: type_attr_t) -> bool:
        ...
    def __getattribute__(self, name: Any) -> Any:
        r"""Return getattr(self, name)."""
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, r: type_attr_t) -> bool:
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

class type_attrs_t:
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
    def __getitem__(self, i: size_t) -> udm_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: type_attr_t) -> None:
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
    def append(self, x: type_attr_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: type_attrs_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: type_attr_t, len: size_t) -> None:
        ...
    def insert(self, it: type_attr_t, x: type_attr_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: type_attrs_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class type_mods_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def type(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def clear(self) -> None:
        ...
    def has_cmt(self) -> bool:
        ...
    def has_info(self) -> bool:
        ...
    def has_name(self) -> bool:
        ...
    def has_type(self) -> bool:
        ...
    def is_rptcmt(self) -> bool:
        ...
    def set_new_cmt(self, c: str, rptcmt: bool) -> None:
        ...
    def set_new_name(self, n: str) -> None:
        ...
    def set_new_type(self, t: tinfo_t) -> None:
        r"""The visit_type() function may optionally save the modified type info. Use the following functions for that. The new name and comment will be applied only if the current tinfo element has storage for them. 
                
        """
        ...

class typedef_type_data_t:
    @property
    def is_ordref(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def ordinal(self) -> Any: ...
    @property
    def resolve(self) -> Any: ...
    @property
    def til(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def swap(self, r: typedef_type_data_t) -> None:
        ...

class udm_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def effalign(self) -> Any: ...
    @property
    def fda(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def offset(self) -> Any: ...
    @property
    def repr(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def tafld_bits(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udm_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        r"""Create a structure/union member, with the specified name and type.
        
        This constructor has the following signatures:
        
            1. udm_t(udm: udm_t)
            2. udm_t(name: str, type, offset: int)
        
        The 'type' descriptor, can be one of:
        
        * type_t: if the type is simple (integral/floating/bool). E.g., `BTF_INT`
        * tinfo_t: can handle more complex types (structures, pointers, arrays, ...)
        * str: a C type declaration
        
        If an input argument is incorrect, the constructor may raise an exception
        The size will be computed automatically.
        
        :param udm: a source udm_t
        :param name: a valid member name. Must not be empty.
        :param type: the member type
        :param offset: the member offset in bits. It is the caller's responsibility
               to specify correct offsets.
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, r: udm_t) -> bool:
        ...
    def __ne__(self, r: udm_t) -> bool:
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
    def begin(self) -> uint64:
        ...
    def can_be_dtor(self) -> bool:
        ...
    def can_rename(self) -> bool:
        ...
    def clr_baseclass(self) -> None:
        ...
    def clr_method(self) -> None:
        ...
    def clr_unaligned(self) -> None:
        ...
    def clr_vftable(self) -> None:
        ...
    def clr_virtbase(self) -> None:
        ...
    def compare_with(self, r: udm_t, tcflags: int) -> bool:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> uint64:
        ...
    def is_anonymous_udm(self) -> bool:
        ...
    def is_baseclass(self) -> bool:
        ...
    def is_bitfield(self) -> bool:
        ...
    def is_by_til(self) -> bool:
        ...
    def is_gap(self) -> bool:
        ...
    def is_method(self) -> bool:
        ...
    def is_regcmt(self) -> bool:
        ...
    def is_retaddr(self) -> bool:
        ...
    def is_savregs(self) -> bool:
        ...
    def is_special_member(self) -> bool:
        ...
    def is_unaligned(self) -> bool:
        ...
    def is_vftable(self) -> bool:
        ...
    def is_virtbase(self) -> bool:
        ...
    def is_zero_bitfield(self) -> bool:
        ...
    def set_baseclass(self, on: bool = True) -> None:
        ...
    def set_by_til(self, on: bool = True) -> None:
        ...
    def set_method(self, on: bool = True) -> None:
        ...
    def set_regcmt(self, on: bool = True) -> None:
        ...
    def set_retaddr(self, on: bool = True) -> None:
        ...
    def set_savregs(self, on: bool = True) -> None:
        ...
    def set_unaligned(self, on: bool = True) -> None:
        ...
    def set_value_repr(self, r: value_repr_t) -> None:
        ...
    def set_vftable(self, on: bool = True) -> None:
        ...
    def set_virtbase(self, on: bool = True) -> None:
        ...
    def swap(self, r: udm_t) -> None:
        ...

class udm_visitor_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __disown__(self) -> Any:
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def visit_udm(self, tid: tid_t, tif: tinfo_t, udt: udt_type_data_t, idx: ssize_t) -> int:
        r""":param tid: udt tid
        :param tif: udt type info (may be nullptr for corrupted idbs)
        :param udt: udt type data (may be nullptr for corrupted idbs)
        :param idx: the index of udt the member (may be -1 if udm was not found)
        """
        ...

class udt_member_t:
    @property
    def cmt(self) -> Any: ...
    @property
    def effalign(self) -> Any: ...
    @property
    def fda(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def offset(self) -> Any: ...
    @property
    def repr(self) -> Any: ...
    @property
    def size(self) -> Any: ...
    @property
    def tafld_bits(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udm_t) -> bool:
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
    def __gt__(self, value: Any) -> bool:
        r"""Return self>value."""
        ...
    def __init__(self, args: Any) -> Any:
        r"""Create a structure/union member, with the specified name and type.
        
        This constructor has the following signatures:
        
            1. udm_t(udm: udm_t)
            2. udm_t(name: str, type, offset: int)
        
        The 'type' descriptor, can be one of:
        
        * type_t: if the type is simple (integral/floating/bool). E.g., `BTF_INT`
        * tinfo_t: can handle more complex types (structures, pointers, arrays, ...)
        * str: a C type declaration
        
        If an input argument is incorrect, the constructor may raise an exception
        The size will be computed automatically.
        
        :param udm: a source udm_t
        :param name: a valid member name. Must not be empty.
        :param type: the member type
        :param offset: the member offset in bits. It is the caller's responsibility
               to specify correct offsets.
        """
        ...
    def __init_subclass__(self, *args: Any, **kwargs: Any) -> Any:
        r"""This method is called when a class is subclassed.
        
        The default implementation does nothing. It may be
        overridden to extend subclasses.
        
        """
        ...
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
        ...
    def __lt__(self, r: udm_t) -> bool:
        ...
    def __ne__(self, r: udm_t) -> bool:
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
    def begin(self) -> uint64:
        ...
    def can_be_dtor(self) -> bool:
        ...
    def can_rename(self) -> bool:
        ...
    def clr_baseclass(self) -> None:
        ...
    def clr_method(self) -> None:
        ...
    def clr_unaligned(self) -> None:
        ...
    def clr_vftable(self) -> None:
        ...
    def clr_virtbase(self) -> None:
        ...
    def compare_with(self, r: udm_t, tcflags: int) -> bool:
        ...
    def empty(self) -> bool:
        ...
    def end(self) -> uint64:
        ...
    def is_anonymous_udm(self) -> bool:
        ...
    def is_baseclass(self) -> bool:
        ...
    def is_bitfield(self) -> bool:
        ...
    def is_by_til(self) -> bool:
        ...
    def is_gap(self) -> bool:
        ...
    def is_method(self) -> bool:
        ...
    def is_regcmt(self) -> bool:
        ...
    def is_retaddr(self) -> bool:
        ...
    def is_savregs(self) -> bool:
        ...
    def is_special_member(self) -> bool:
        ...
    def is_unaligned(self) -> bool:
        ...
    def is_vftable(self) -> bool:
        ...
    def is_virtbase(self) -> bool:
        ...
    def is_zero_bitfield(self) -> bool:
        ...
    def set_baseclass(self, on: bool = True) -> None:
        ...
    def set_by_til(self, on: bool = True) -> None:
        ...
    def set_method(self, on: bool = True) -> None:
        ...
    def set_regcmt(self, on: bool = True) -> None:
        ...
    def set_retaddr(self, on: bool = True) -> None:
        ...
    def set_savregs(self, on: bool = True) -> None:
        ...
    def set_unaligned(self, on: bool = True) -> None:
        ...
    def set_value_repr(self, r: value_repr_t) -> None:
        ...
    def set_vftable(self, on: bool = True) -> None:
        ...
    def set_virtbase(self, on: bool = True) -> None:
        ...
    def swap(self, r: udm_t) -> None:
        ...

class udt_type_data_t(udtmembervec_t, udtmembervec_template_t):
    @property
    def effalign(self) -> Any: ...
    @property
    def is_union(self) -> Any: ...
    @property
    def pack(self) -> Any: ...
    @property
    def sda(self) -> Any: ...
    @property
    def taudt_bits(self) -> Any: ...
    @property
    def total_size(self) -> Any: ...
    @property
    def unpadded_size(self) -> Any: ...
    @property
    def version(self) -> Any: ...
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udtmembervec_template_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: udtmembervec_template_t) -> bool:
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
    def __setitem__(self, i: size_t, v: udm_t) -> None:
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
    def add_member(self, _name: str, _type: tinfo_t, _offset: uint64 = 0) -> udm_t:
        r"""Add a new member to a structure or union. This function just pushes a new member to the back of the structure/union member vector.
        
        :param _name: Member name. Must not be nullptr.
        :param _type: Member type. Must not be empty.
        :param _offset: Member offset in bits. It is the caller's responsibility to specify correct offsets.
        :returns: { Reference to the newly added member }
        """
        ...
    def add_unique(self, x: udm_t) -> bool:
        ...
    def append(self, x: udm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: udtmembervec_template_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def find_member(self, args: Any) -> ssize_t:
        r"""This function has the following signatures:
        
            0. find_member(pattern_udm: udm_t *, strmem_flags: int) -> ssize_t
            1. find_member(name: str) -> ssize_t
            2. find_member(bit_offset: uint64) -> ssize_t
        
        # 0: find_member(pattern_udm: udm_t *, strmem_flags: int) -> ssize_t
        
        tinfo_t::find_udm 
                
        :returns: the index of the found member or -1
        
        # 1: find_member(name: str) -> ssize_t
        
        
        # 2: find_member(bit_offset: uint64) -> ssize_t
        
        
        """
        ...
    def front(self) -> Any:
        ...
    def get_best_fit_member(self, disp: Any) -> Any:
        r"""Get the member that is most likely referenced by the specified offset.
        
        :param disp: the byte offset
        :returns: a tuple (int, udm_t), or (-1, None) if member not found
        """
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: udm_t) -> bool:
        ...
    def inject(self, s: udm_t, len: size_t) -> None:
        ...
    def insert(self, it: udm_t, x: udm_t) -> qvector:
        ...
    def is_cppobj(self) -> bool:
        ...
    def is_fixed(self) -> bool:
        ...
    def is_last_baseclass(self, idx: size_t) -> bool:
        ...
    def is_msstruct(self) -> bool:
        ...
    def is_tuple(self) -> bool:
        ...
    def is_unaligned(self) -> bool:
        ...
    def is_vftable(self) -> bool:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def set_fixed(self, on: bool = True) -> None:
        ...
    def set_tuple(self, on: bool = True) -> None:
        ...
    def set_vftable(self, on: bool = True) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: udt_type_data_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class udtmembervec_t(udtmembervec_template_t):
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udtmembervec_template_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: udtmembervec_template_t) -> bool:
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
    def __setitem__(self, i: size_t, v: udm_t) -> None:
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
    def add_unique(self, x: udm_t) -> bool:
        ...
    def append(self, x: udm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: udtmembervec_template_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: udm_t) -> bool:
        ...
    def inject(self, s: udm_t, len: size_t) -> None:
        ...
    def insert(self, it: udm_t, x: udm_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: udtmembervec_template_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class udtmembervec_template_t:
    def __delattr__(self, name: Any) -> Any:
        r"""Implement delattr(self, name)."""
        ...
    def __dir__(self) -> Any:
        r"""Default dir() implementation."""
        ...
    def __eq__(self, r: udtmembervec_template_t) -> bool:
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
    def __getitem__(self, i: size_t) -> udm_t:
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
    def __ne__(self, r: udtmembervec_template_t) -> bool:
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
    def __setitem__(self, i: size_t, v: udm_t) -> None:
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
    def add_unique(self, x: udm_t) -> bool:
        ...
    def append(self, x: udm_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: udtmembervec_template_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def find(self, args: Any) -> qvector:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def has(self, x: udm_t) -> bool:
        ...
    def inject(self, s: udm_t, len: size_t) -> None:
        ...
    def insert(self, it: udm_t, x: udm_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: udtmembervec_template_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class valstr_t:
    @property
    def info(self) -> Any: ...
    @property
    def length(self) -> Any: ...
    @property
    def members(self) -> Any: ...
    @property
    def oneline(self) -> Any: ...
    @property
    def props(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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

class valstrs_t(valstrvec_t):
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
    def __getitem__(self, i: size_t) -> udm_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: valstr_t) -> None:
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
    def append(self, x: valstr_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: valstrvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: valstr_t, len: size_t) -> None:
        ...
    def insert(self, it: valstr_t, x: valstr_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: valstrvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class valstrvec_t:
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
    def __getitem__(self, i: size_t) -> udm_t:
        ...
    def __gt__(self, value: Any) -> bool:
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
    def __setitem__(self, i: size_t, v: valstr_t) -> None:
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
    def append(self, x: valstr_t) -> None:
        ...
    def at(self, _idx: size_t) -> udm_t:
        ...
    def back(self) -> Any:
        ...
    def begin(self, args: Any) -> uint64:
        ...
    def capacity(self) -> int:
        ...
    def clear(self) -> None:
        ...
    def empty(self) -> bool:
        ...
    def end(self, args: Any) -> uint64:
        ...
    def erase(self, args: Any) -> qvector:
        ...
    def extend(self, x: valstrvec_t) -> None:
        ...
    def extract(self) -> udm_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: valstr_t, len: size_t) -> None:
        ...
    def insert(self, it: valstr_t, x: valstr_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> udm_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: valstrvec_t) -> None:
        ...
    def truncate(self) -> None:
        ...

class value_repr_t:
    @property
    def ap(self) -> Any: ...
    @property
    def bits(self) -> Any: ...
    @property
    def cd(self) -> Any: ...
    @property
    def delta(self) -> Any: ...
    @property
    def ri(self) -> Any: ...
    @property
    def strtype(self) -> Any: ...
    @property
    def type_ordinal(self) -> Any: ...
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
    def __gt__(self, value: Any) -> bool:
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
    def __le__(self, value: Any) -> bool:
        r"""Return self<=value."""
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
    def __sizeof__(self) -> Any:
        r"""Size of object in memory, in bytes."""
        ...
    def __str__(self) -> str:
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
        ...
    def empty(self) -> bool:
        ...
    def from_opinfo(self, flags: flags64_t, afl: aflags_t, opinfo: opinfo_t, _ap: array_parameters_t) -> bool:
        ...
    def get_vtype(self) -> uint64:
        ...
    def has_lzeroes(self) -> bool:
        ...
    def has_tabform(self) -> bool:
        ...
    def init_ap(self, _ap: array_parameters_t) -> None:
        ...
    def is_custom(self) -> bool:
        ...
    def is_enum(self) -> bool:
        ...
    def is_offset(self) -> bool:
        ...
    def is_signed(self) -> bool:
        ...
    def is_strlit(self) -> bool:
        ...
    def is_stroff(self) -> bool:
        ...
    def is_typref(self) -> bool:
        ...
    def parse_value_repr(self, args: Any) -> bool:
        ...
    def set_ap(self, _ap: array_parameters_t) -> None:
        ...
    def set_lzeroes(self, on: bool) -> None:
        ...
    def set_signed(self, on: bool) -> None:
        ...
    def set_tabform(self, on: bool) -> None:
        ...
    def set_vtype(self, vt: uint64) -> None:
        ...
    def swap(self, r: value_repr_t) -> None:
        ...

def add_til(name: str, flags: int) -> int:
    r"""Load a til file and add it the database type libraries list. IDA will also apply function prototypes for matching function names. 
            
    :param name: til name
    :param flags: combination of Load TIL flags
    :returns: one of Load TIL result codes
    """
    ...

def alloc_type_ordinal(ti: til_t) -> int:
    r"""alloc_type_ordinals(ti, 1)
    
    """
    ...

def alloc_type_ordinals(ti: til_t, qty: int) -> int:
    r"""Allocate a range of ordinal numbers for new types. 
            
    :param ti: type library
    :param qty: number of ordinals to allocate
    :returns: the first ordinal. 0 means failure.
    """
    ...

def append_abi_opts(abi_opts: str, user_level: bool = False) -> bool:
    r"""Add/remove/check ABI option General form of full abi name: abiname-opt1-opt2-... or -opt1-opt2-... 
            
    :param abi_opts: - ABI options to add/remove in form opt1-opt2-...
    :param user_level: - initiated by user if TRUE (==SETCOMP_BY_USER)
    :returns: success
    """
    ...

def append_argloc(out: qtype, vloc: argloc_t) -> bool:
    r"""Serialize argument location 
            
    """
    ...

def append_tinfo_covered(out: rangeset_t, typid: typid_t, offset: uint64) -> bool:
    ...

def apply_callee_tinfo(caller: ida_idaapi.ea_t, tif: tinfo_t) -> bool:
    r"""Apply the type of the called function to the calling instruction. This function will append parameter comments and rename the local variables of the calling function. It also stores information about the instructions that initialize call arguments in the database. Use get_arg_addrs() to retrieve it if necessary. Alternatively it is possible to hook to processor_t::arg_addrs_ready event. 
            
    :param caller: linear address of the calling instruction. must belong to a function.
    :param tif: type info
    :returns: success
    """
    ...

def apply_cdecl(til: til_t, ea: ida_idaapi.ea_t, decl: str, flags: int = 0) -> bool:
    r"""Apply the specified type to the address. This function parses the declaration and calls apply_tinfo() 
            
    :param til: type library
    :param ea: linear address
    :param decl: type declaration in C form
    :param flags: flags to pass to apply_tinfo (TINFO_DEFINITE is always passed)
    :returns: success
    """
    ...

def apply_named_type(ea: ida_idaapi.ea_t, name: str) -> bool:
    r"""Apply the specified named type to the address. 
            
    :param ea: linear address
    :param name: the type name, e.g. "FILE"
    :returns: success
    """
    ...

def apply_once_tinfo_and_name(dea: ida_idaapi.ea_t, tif: tinfo_t, name: str) -> bool:
    r"""Apply the specified type and name to the address. This function checks if the address already has a type. If the old type 
    does not exist or the new type is 'better' than the old type, then the 
    new type will be applied. A type is considered better if it has more 
    information (e.g. BTMT_STRUCT is better than BT_INT). 
    The same logic is with the name: if the address already have a meaningful 
    name, it will be preserved. Only if the old name does not exist or it 
    is a dummy name like byte_123, it will be replaced by the new name. 
            
    :param dea: linear address
    :param tif: new type
    :param name: new name for the address
    :returns: success
    """
    ...

def apply_tinfo(ea: ida_idaapi.ea_t, tif: tinfo_t, flags: int) -> bool:
    r"""Apply the specified type to the specified address. This function sets the type and tries to convert the item at the specified address to conform the type. 
            
    :param ea: linear address
    :param tif: new type
    :param flags: combination of Apply tinfo flags
    :returns: success
    """
    ...

def apply_tinfo_to_stkarg(insn: insn_t, x: op_t, v: int, tif: tinfo_t, name: str) -> bool:
    r"""Helper function for the processor modules. to be called from processor_t::use_stkarg_type 
            
    """
    ...

def apply_type(til: til_t, type: bytes, fields: bytes, ea: ida_idaapi.ea_t, flags: int) -> bool:
    r"""Apply the specified type to the address
    
    :param til: Type info library. 'None' can be used.
    :param type: type string
    :param fields: fields string (may be empty or None)
    :param ea: the address of the object
    :param flags: combination of TINFO_... constants or 0
    :returns: Boolean
    """
    ...

def begin_type_updating(utp: update_type_t) -> None:
    r"""Mark the beginning of a large update operation on the types. Can be used with add_enum_member(), add_struc_member, etc... Also see end_type_updating() 
            
    """
    ...

def calc_arglocs(fti: func_type_data_t) -> bool:
    ...

def calc_c_cpp_name(name: str, type: tinfo_t, ccn_flags: int) -> str:
    r"""Get C or C++ form of the name. 
            
    :param name: original (mangled or decorated) name
    :param type: name type if known, otherwise nullptr
    :param ccn_flags: one of C/C++ naming flags
    """
    ...

def calc_number_of_children(loc: argloc_t, tif: tinfo_t, dont_deref_ptr: bool = False) -> int:
    r"""Calculate max number of lines of a formatted c data, when expanded (PTV_EXPAND). 
            
    :param loc: location of the data (ALOC_STATIC or ALOC_CUSTOM)
    :param tif: type info
    :param dont_deref_ptr: consider 'ea' as the ptr value
    :returns: 0: data is not expandable
    :returns: -1: error, see qerrno
    :returns: else: the max number of lines
    """
    ...

def calc_retloc(args: Any) -> bool:
    r"""This function has the following signatures:
    
        0. calc_retloc(fti: func_type_data_t *) -> bool
        1. calc_retloc(retloc: argloc_t *, rettype: const tinfo_t &, cc: callcnv_t) -> bool
    
    # 0: calc_retloc(fti: func_type_data_t *) -> bool
    
    
    # 1: calc_retloc(retloc: argloc_t *, rettype: const tinfo_t &, cc: callcnv_t) -> bool
    
    
    """
    ...

def calc_tinfo_gaps(out: rangeset_t, typid: typid_t) -> bool:
    ...

def calc_type_size(til: til_t, type: bytes) -> Any:
    r"""Returns the size of a type
    :param til: Type info library. 'None' can be passed.
    :param type: serialized type byte string
    :returns: The size of the type (None on failure)
    """
    ...

def calc_varglocs(fti: func_type_data_t, regs: regobjs_t, stkargs: relobj_t, nfixed: int) -> bool:
    ...

def choose_local_tinfo(ti: til_t, title: str, func: local_tinfo_predicate_t = None, def_ord: int = 0, ud: void = None) -> int:
    r"""Choose a type from the local type library. 
            
    :param ti: pointer to til
    :param title: title of listbox to display
    :param func: predicate to select types to display (maybe nullptr)
    :param def_ord: ordinal to position cursor before choose
    :param ud: user data
    :returns: == 0 means nothing is chosen, otherwise an ordinal number
    """
    ...

def choose_local_tinfo_and_delta(delta: int32, ti: til_t, title: str, func: local_tinfo_predicate_t = None, def_ord: int = 0, ud: void = None) -> int:
    r"""Choose a type from the local type library and specify the pointer shift value. 
            
    :param delta: pointer shift value
    :param ti: pointer to til
    :param title: title of listbox to display
    :param func: predicate to select types to display (maybe nullptr)
    :param def_ord: ordinal to position cursor before choose
    :param ud: user data
    :returns: == 0 means nothing is chosen, otherwise an ordinal number
    """
    ...

def choose_named_type(out_sym: til_symbol_t, root_til: til_t, title: str, ntf_flags: int, predicate: predicate_t = None) -> bool:
    r"""Choose a type from a type library. 
            
    :param out_sym: pointer to be filled with the chosen type
    :param root_til: pointer to starting til (the function will inspect the base tils if allowed by flags)
    :param title: title of listbox to display
    :param ntf_flags: combination of Flags for named types
    :param predicate: predicate to select types to display (maybe nullptr)
    :returns: false if nothing is chosen, otherwise true
    """
    ...

def clear_tinfo_t(_this: tinfo_t) -> None:
    ...

def compact_til(ti: til_t) -> bool:
    r"""Collect garbage in til. Must be called before storing the til. 
            
    :returns: true if any memory was freed
    """
    ...

def compare_tinfo(t1: typid_t, t2: typid_t, tcflags: int) -> bool:
    ...

def convert_pt_flags_to_hti(pt_flags: int) -> int:
    r"""Convert Type parsing flags to Type formatting flags. Type parsing flags lesser than 0x10 don't have stable meaning and will be ignored (more on these flags can be seen in idc.idc) 
            
    """
    ...

def copy_named_type(dsttil: til_t, srctil: til_t, name: str) -> int:
    r"""Copy a named type from one til to another. This function will copy the specified type and all dependent types from the source type library to the destination library. 
            
    :param dsttil: Destination til. It must have original types enabled
    :param srctil: Source til.
    :param name: name of the type to copy
    :returns: ordinal number of the copied type. 0 means error
    """
    ...

def copy_tinfo_t(_this: tinfo_t, r: tinfo_t) -> None:
    ...

def create_enum_type(enum_name: str, ei: enum_type_data_t, enum_width: int, sign: type_sign_t, convert_to_bitmask: bool, enum_cmt: str = None) -> tid_t:
    r"""Create type enum 
            
    :param enum_name: type name
    :param ei: enum type data
    :param enum_width: the width of an enum element allowed values: 0 (unspecified),1,2,4,8,16,32,64
    :param sign: enum sign
    :param convert_to_bitmask: try convert enum to bitmask enum
    :param enum_cmt: enum type comment
    :returns: enum TID
    """
    ...

def create_numbered_type_name(ord: int) -> str:
    r"""Create anonymous name for numbered type. This name can be used to reference a numbered type by its ordinal Ordinal names have the following format: '#' + set_de(ord) Returns: -1 if error, otherwise the name length 
            
    """
    ...

def create_tinfo(_this: tinfo_t, bt: type_t, bt2: type_t, ptr: void) -> bool:
    ...

def decorate_name(args: Any) -> str:
    r"""Decorate/undecorate a C symbol name. 
            
    :param out: output buffer
    :param name: name of symbol
    :param should_decorate: true-decorate name, false-undecorate
    :param cc: calling convention
    :param type: name type (nullptr-unknown)
    :returns: success
    """
    ...

def default_compiler() -> comp_t:
    r"""Get compiler specified by inf.cc.
    
    """
    ...

def del_named_type(ti: til_t, name: str, ntf_flags: int) -> bool:
    r"""Delete information about a symbol. 
            
    :param ti: type library
    :param name: name of symbol
    :param ntf_flags: combination of Flags for named types
    :returns: success
    """
    ...

def del_numbered_type(ti: til_t, ordinal: int) -> bool:
    r"""Delete a numbered type.
    
    """
    ...

def del_til(name: str) -> bool:
    r"""Unload a til file.
    
    """
    ...

def del_tinfo_attr(tif: tinfo_t, key: str, make_copy: bool) -> bool:
    ...

def del_vftable_ea(ordinal: int) -> bool:
    r"""Delete the address of a vftable instance for a vftable type. 
            
    :param ordinal: ordinal number of a vftable type.
    :returns: success
    """
    ...

def deref_ptr(ptr_ea: ea_t, tif: tinfo_t, closure_obj: ea_t = None) -> bool:
    r"""Dereference a pointer. 
            
    :param ptr_ea: in/out parameter
    * in: address of the pointer
    * out: the pointed address
    :param tif: type of the pointer
    :param closure_obj: closure object (not used yet)
    :returns: success
    """
    ...

def deserialize_tinfo(tif: tinfo_t, til: til_t, ptype: type_t, pfields: p_list, pfldcmts: p_list, cmt: str = None) -> bool:
    ...

def detach_tinfo_t(_this: tinfo_t) -> bool:
    ...

def dstr_tinfo(tif: tinfo_t) -> str:
    ...

def dump_func_type_data(fti: func_type_data_t, praloc_bits: int) -> str:
    r"""Use func_type_data_t::dump()
    
    """
    ...

def enable_numbered_types(ti: til_t, enable: bool) -> bool:
    r"""Enable the use of numbered types in til. Currently it is impossible to disable numbered types once they are enabled 
            
    """
    ...

def end_type_updating(utp: update_type_t) -> None:
    r"""Mark the end of a large update operation on the types (see begin_type_updating())
    
    """
    ...

def extract_argloc(vloc: argloc_t, ptype: type_t, forbid_stkoff: bool) -> bool:
    r"""Deserialize an argument location. Argument FORBID_STKOFF checks location type. It can be used, for example, to check the return location of a function that cannot return a value in the stack 
            
    """
    ...

def find_custom_callcnv(name: str) -> callcnv_t:
    r"""Find a calling convention by its name 
            
    :returns: CM_CC_INVALID is not found
    """
    ...

def find_tinfo_udt_member(udm: udm_t, typid: typid_t, strmem_flags: int) -> int:
    ...

def first_named_type(ti: til_t, ntf_flags: int) -> str:
    r"""Enumerate types. 
            
    :param ti: type library. nullptr means the local type library for the current database.
    :param ntf_flags: combination of Flags for named types
    :returns: Type or symbol names, depending of ntf_flags. Returns mangled names. Never returns anonymous types. To include them, enumerate types by ordinals.
    """
    ...

def for_all_arglocs(vv: aloc_visitor_t, vloc: argloc_t, size: int, off: int = 0) -> int:
    r"""Compress larger argloc types and initiate the aloc visitor.
    
    """
    ...

def for_all_const_arglocs(vv: const_aloc_visitor_t, vloc: argloc_t, size: int, off: int = 0) -> int:
    r"""See for_all_arglocs()
    
    """
    ...

def free_til(ti: til_t) -> None:
    r"""Free memory allocated by til.
    
    """
    ...

def func_has_stkframe_hole(ea: ida_idaapi.ea_t, fti: func_type_data_t) -> bool:
    r"""Looks for a hole at the beginning of the stack arguments. Will make use of the IDB's func_t function at that place (if present) to help determine the presence of such a hole. 
            
    """
    ...

def gcc_layout() -> bool:
    r"""Should use the struct/union layout as done by gcc?
    
    """
    ...

def gen_decorate_name(name: str, should_decorate: bool, cc: callcnv_t, type: tinfo_t) -> str:
    r"""Generic function for decorate_name() (may be used in IDP modules)
    
    """
    ...

def gen_use_arg_tinfos(_this: argtinfo_helper_t, caller: ida_idaapi.ea_t, fti: func_type_data_t, rargs: funcargvec_t) -> None:
    r"""Do not call this function directly, use argtinfo_helper_t.
    
    """
    ...

def get_abi_name() -> str:
    r"""Get ABI name. 
            
    :returns: length of the name (>=0)
    """
    ...

def get_alias_target(ti: til_t, ordinal: int) -> int:
    r"""Find the final alias destination. If the ordinal has not been aliased, return the specified ordinal itself If failed, returns 0. 
            
    """
    ...

def get_arg_addrs(caller: ida_idaapi.ea_t) -> Any:
    r"""Retrieve addresses of argument initialization instructions
    
    :param caller: the address of the call instruction
    :returns: list of instruction addresses
    """
    ...

def get_base_type(t: type_t) -> type_t:
    r"""Get get basic type bits (TYPE_BASE_MASK)
    
    """
    ...

def get_c_header_path() -> str:
    r"""Get the include directory path of the target compiler.
    
    """
    ...

def get_c_macros() -> str:
    r"""Get predefined macros for the target compiler.
    
    """
    ...

def get_comp(comp: comp_t) -> comp_t:
    r"""Get compiler bits.
    
    """
    ...

def get_compiler_abbr(id: comp_t) -> str:
    r"""Get abbreviated compiler name.
    
    """
    ...

def get_compiler_name(id: comp_t) -> str:
    r"""Get full compiler name.
    
    """
    ...

def get_compilers(ids: compvec_t, names: qstrvec_t, abbrs: qstrvec_t) -> None:
    r"""Get names of all built-in compilers.
    
    """
    ...

def get_custom_callcnv(callcnv: callcnv_t) -> custom_callcnv_t:
    r"""Retrieve custom calling convention details.
    
    """
    ...

def get_custom_callcnvs(names: qstrvec_t, codes: callcnvs_t) -> int:
    r"""Get all custom calling conventions 
            
    :param names: output buffer for the convention names
    :param codes: output buffer for the convention codes The two output buffers correspond to each other.
    :returns: number of the calling conventions added to the output buffers
    """
    ...

def get_enum_member_expr(tif: tinfo_t, serial: int, value: uint64) -> str:
    r"""Return a C expression that can be used to represent an enum member. If the value does not correspond to any single enum member, this function tries to find a bitwise combination of enum members that correspond to it. If more than half of value bits do not match any enum members, it fails. 
            
    :param tif: enumeration type
    :param serial: which enumeration member to use (0 means the first with the given value)
    :param value: value to search in the enumeration type
    :returns: success
    """
    ...

def get_full_type(t: type_t) -> type_t:
    r"""Get basic type bits + type flags (TYPE_FULL_MASK)
    
    """
    ...

def get_idainfo_by_type(tif: tinfo_t) -> Any:
    r"""Extract information from a tinfo_t. 
            
    :param tif: the type to inspect
    """
    ...

def get_idainfo_by_udm(args: Any) -> bool:
    r"""Calculate IDA info from udt member 
            
    :param udm: udt member
    :param refinfo_ea: if specified will be used to adjust the refinfo_t data
    """
    ...

def get_idati() -> til_t:
    r"""Pointer to the local type library - this til is private for each IDB file Functions that accept til_t* default to `idati` when is nullptr provided. 
            
    """
    ...

def get_named_type(til: til_t, name: str, ntf_flags: int) -> bool:
    r"""Get a type data by its name.
    
    :param til: Type library
    :param name: the type name
    :param ntf_flags: a combination of NTF_* constants
    :returns: tuple(code, type_str, fields_str, cmt, field_cmts, sclass, value) on success, or None on failure
    """
    ...

def get_named_type64(til: til_t, name: str, ntf_flags: int = 0) -> Any:
    r"""Get a named type from a type library.
    
    Please use til_t.get_named_type instead.
    """
    ...

def get_named_type_tid(name: str) -> tid_t:
    r"""Get named local type TID 
            
    :param name: type name
    :returns: TID or BADADDR
    """
    ...

def get_numbered_type(til: til_t, ordinal: int) -> Any:
    r"""Get a type from a type library, by its ordinal
    
    Please use til_t.get_numbered_type instead.
    """
    ...

def get_numbered_type_name(ti: til_t, ordinal: int) -> str:
    r"""Get type name (if exists) by its ordinal. If the type is anonymous, returns "". If failed, returns nullptr 
            
    """
    ...

def get_ordinal_count(ti: til_t = None) -> int:
    r"""Get number of allocated ordinals. 
            
    :param ti: type library; nullptr means the local types for the current database.
    :returns: 0 if ordinals have not been enabled for the til.
    """
    ...

def get_ordinal_limit(ti: til_t = None) -> int:
    r"""Get number of allocated ordinals + 1. If there are no allocated ordinals, return 0. To enumerate all ordinals, use: for ( uint32 i = 1; i < limit; ++i ) 
            
    :param ti: type library; nullptr means the local types for the current database.
    :returns: uint32(-1) if ordinals have not been enabled for the til. For local types (idati), ordinals are always enabled.
    """
    ...

def get_scalar_bt(size: int) -> type_t:
    ...

def get_stkarg_area_info(out: stkarg_area_info_t, cc: callcnv_t) -> bool:
    r"""Some calling conventions foresee special areas on the stack for call arguments. This structure lists their sizes. 
            
    """
    ...

def get_stock_tinfo(tif: tinfo_t, id: stock_type_id_t) -> bool:
    ...

def get_tid_name(tid: tid_t) -> str:
    r"""Get a type name for the specified TID 
            
    :param tid: type TID
    :returns: true if there is type with TID
    """
    ...

def get_tid_ordinal(tid: tid_t) -> int:
    r"""Get type ordinal number for TID 
            
    :param tid: type/enum constant/udt member TID
    :returns: type ordinal number or 0
    """
    ...

def get_tinfo_attr(typid: typid_t, key: str, bv: bytevec_t, all_attrs: bool) -> bool:
    ...

def get_tinfo_attrs(typid: typid_t, tav: type_attrs_t, include_ref_attrs: bool) -> bool:
    ...

def get_tinfo_by_edm_name(tif: tinfo_t, til: til_t, mname: str) -> ssize_t:
    ...

def get_tinfo_by_flags(out: tinfo_t, flags: flags64_t) -> bool:
    r"""Get tinfo object that corresponds to data flags 
            
    :param out: type info
    :param flags: simple flags (byte, word, ..., zword)
    """
    ...

def get_tinfo_details(typid: typid_t, bt2: type_t, buf: void) -> bool:
    ...

def get_tinfo_pdata(outptr: void, typid: typid_t, what: int) -> int:
    ...

def get_tinfo_property(typid: typid_t, gta_prop: int) -> int:
    ...

def get_tinfo_property4(typid: typid_t, gta_prop: int, p1: size_t, p2: size_t, p3: size_t, p4: size_t) -> int:
    ...

def get_tinfo_size(p_effalign: uint32, typid: typid_t, gts_code: int) -> int:
    ...

def get_type_flags(t: type_t) -> type_t:
    r"""Get type flags (TYPE_FLAGS_MASK)
    
    """
    ...

def get_type_ordinal(ti: til_t, name: str) -> int:
    r"""Get type ordinal by its name.
    
    """
    ...

def get_udm_by_fullname(udm: udm_t, fullname: str) -> ssize_t:
    r"""Get udt member by full name 
            
    :param udm: member, can be NULL
    :param fullname: udt member name in format <udt name>.<member name>
    :returns: member index into udt_type_data_t or -1
    """
    ...

def get_vftable_ea(ordinal: int) -> ida_idaapi.ea_t:
    r"""Get address of a virtual function table. 
            
    :param ordinal: ordinal number of a vftable type.
    :returns: address of the corresponding virtual function table in the current database.
    """
    ...

def get_vftable_ordinal(vftable_ea: ida_idaapi.ea_t) -> int:
    r"""Get ordinal number of the virtual function table. 
            
    :param vftable_ea: address of a virtual function table.
    :returns: ordinal number of the corresponding vftable type. 0 - failure.
    """
    ...

def guess_func_cc(fti: func_type_data_t, npurged: int, cc_flags: int) -> callcnv_t:
    r"""Use func_type_data_t::guess_cc()
    
    """
    ...

def guess_tinfo(out: tinfo_t, id: tid_t) -> int:
    r"""Generate a type information about the id from the disassembly. id can be a structure/union/enum id or an address. 
            
    :returns: one of Guess tinfo codes
    """
    ...

def idc_get_local_type(ordinal: int, flags: int) -> str:
    ...

def idc_get_local_type_name(ordinal: int) -> str:
    ...

def idc_get_local_type_raw(ordinal: Any) -> Any:
    r"""    """
    ...

def idc_get_type(ea: ida_idaapi.ea_t) -> str:
    ...

def idc_get_type_raw(ea: ida_idaapi.ea_t) -> Any:
    ...

def idc_guess_type(ea: ida_idaapi.ea_t) -> str:
    ...

def idc_parse_decl(til: til_t, decl: str, flags: int) -> Any:
    r"""    """
    ...

def idc_parse_types(input: str, flags: int) -> int:
    ...

def idc_print_type(type: bytes, fields: bytes, name: str, flags: int) -> str:
    r"""    """
    ...

def idc_set_local_type(ordinal: int, dcl: str, flags: int) -> int:
    ...

def inf_big_arg_align(args: Any) -> bool:
    ...

def inf_huge_arg_align(args: Any) -> bool:
    ...

def inf_pack_stkargs(args: Any) -> bool:
    ...

def is_code_far(cm: cm_t) -> bool:
    r"""Does the given model specify far code?.
    
    """
    ...

def is_comp_unsure(comp: comp_t) -> comp_t:
    r"""See COMP_UNSURE.
    
    """
    ...

def is_custom_callcnv(cc: callcnv_t) -> bool:
    r"""Is custom calling convention?
    
    """
    ...

def is_data_far(cm: cm_t) -> bool:
    r"""Does the given model specify far data?.
    
    """
    ...

def is_gcc() -> bool:
    r"""Is the target compiler COMP_GNU?
    
    """
    ...

def is_gcc32() -> bool:
    r"""Is the target compiler 32 bit gcc?
    
    """
    ...

def is_gcc64() -> bool:
    r"""Is the target compiler 64 bit gcc?
    
    """
    ...

def is_golang_cc(cc: callcnv_t) -> bool:
    r"""GO language calling convention (return value in stack)?
    
    """
    ...

def is_one_bit_mask(mask: int) -> bool:
    r"""Is bitmask one bit?
    
    """
    ...

def is_ordinal_name(name: str, ord: uint32 = None) -> bool:
    r"""Check if the name is an ordinal name. Ordinal names have the following format: '#' + set_de(ord) 
            
    """
    ...

def is_purging_cc(cc: callcnv_t) -> bool:
    r"""Does the calling convention clean the stack arguments upon return?. 
            
    """
    ...

def is_restype_enum(til: til_t, type: type_t) -> bool:
    ...

def is_restype_struct(til: til_t, type: type_t) -> bool:
    ...

def is_restype_struni(til: til_t, type: type_t) -> bool:
    ...

def is_restype_void(til: til_t, type: type_t) -> bool:
    ...

def is_sdacl_byte(t: type_t) -> bool:
    r"""Identify an sdacl byte. The first sdacl byte has the following format: 11xx000x. The sdacl bytes are appended to udt fields. They indicate the start of type attributes (as the tah-bytes do). The sdacl bytes are used in the udt headers instead of the tah-byte. This is done for compatibility with old databases, they were already using sdacl bytes in udt headers and as udt field postfixes. (see "sdacl-typeattrs" in the type bit definitions) 
            
    """
    ...

def is_swift_cc(cc: callcnv_t) -> bool:
    r"""Swift calling convention (arguments and return values in registers)?
    
    """
    ...

def is_tah_byte(t: type_t) -> bool:
    r"""The TAH byte (type attribute header byte) denotes the start of type attributes. (see "tah-typeattrs" in the type bit definitions) 
            
    """
    ...

def is_type_arithmetic(t: type_t) -> bool:
    r"""Is the type an arithmetic type? (floating or integral)
    
    """
    ...

def is_type_array(t: type_t) -> bool:
    r"""See BT_ARRAY.
    
    """
    ...

def is_type_bitfld(t: type_t) -> bool:
    r"""See BT_BITFIELD.
    
    """
    ...

def is_type_bool(t: type_t) -> bool:
    r"""See BTF_BOOL.
    
    """
    ...

def is_type_char(t: type_t) -> bool:
    r"""Does the type specify a char value? (signed or unsigned, see Basic type: integer)
    
    """
    ...

def is_type_choosable(ti: til_t, ordinal: int) -> bool:
    r"""Check if a struct/union type is choosable 
            
    :param ti: type library
    :param ordinal: ordinal number of a UDT type
    """
    ...

def is_type_complex(t: type_t) -> bool:
    r"""See BT_COMPLEX.
    
    """
    ...

def is_type_const(t: type_t) -> bool:
    r"""See BTM_CONST.
    
    """
    ...

def is_type_double(t: type_t) -> bool:
    r"""See BTF_DOUBLE.
    
    """
    ...

def is_type_enum(t: type_t) -> bool:
    r"""See BTF_ENUM.
    
    """
    ...

def is_type_ext_arithmetic(t: type_t) -> bool:
    r"""Is the type an extended arithmetic type? (arithmetic or enum)
    
    """
    ...

def is_type_ext_integral(t: type_t) -> bool:
    r"""Is the type an extended integral type? (integral or enum)
    
    """
    ...

def is_type_float(t: type_t) -> bool:
    r"""See BTF_FLOAT.
    
    """
    ...

def is_type_floating(t: type_t) -> bool:
    r"""Is the type a floating point type?
    
    """
    ...

def is_type_func(t: type_t) -> bool:
    r"""See BT_FUNC.
    
    """
    ...

def is_type_int(bt: type_t) -> bool:
    r"""Does the type_t specify one of the basic types in Basic type: integer?
    
    """
    ...

def is_type_int128(t: type_t) -> bool:
    r"""Does the type specify a 128-bit value? (signed or unsigned, see Basic type: integer)
    
    """
    ...

def is_type_int16(t: type_t) -> bool:
    r"""Does the type specify a 16-bit value? (signed or unsigned, see Basic type: integer)
    
    """
    ...

def is_type_int32(t: type_t) -> bool:
    r"""Does the type specify a 32-bit value? (signed or unsigned, see Basic type: integer)
    
    """
    ...

def is_type_int64(t: type_t) -> bool:
    r"""Does the type specify a 64-bit value? (signed or unsigned, see Basic type: integer)
    
    """
    ...

def is_type_integral(t: type_t) -> bool:
    r"""Is the type an integral type (char/short/int/long/bool)?
    
    """
    ...

def is_type_ldouble(t: type_t) -> bool:
    r"""See BTF_LDOUBLE.
    
    """
    ...

def is_type_paf(t: type_t) -> bool:
    r"""Is the type a pointer, array, or function type?
    
    """
    ...

def is_type_partial(t: type_t) -> bool:
    r"""Identifies an unknown or void type with a known size (see Basic type: unknown & void)
    
    """
    ...

def is_type_ptr(t: type_t) -> bool:
    r"""See BT_PTR.
    
    """
    ...

def is_type_ptr_or_array(t: type_t) -> bool:
    r"""Is the type a pointer or array type?
    
    """
    ...

def is_type_struct(t: type_t) -> bool:
    r"""See BTF_STRUCT.
    
    """
    ...

def is_type_struni(t: type_t) -> bool:
    r"""Is the type a struct or union?
    
    """
    ...

def is_type_sue(t: type_t) -> bool:
    r"""Is the type a struct/union/enum?
    
    """
    ...

def is_type_tbyte(t: type_t) -> bool:
    r"""See BTF_FLOAT.
    
    """
    ...

def is_type_typedef(t: type_t) -> bool:
    r"""See BTF_TYPEDEF.
    
    """
    ...

def is_type_uchar(t: type_t) -> bool:
    r"""See BTF_UCHAR.
    
    """
    ...

def is_type_uint(t: type_t) -> bool:
    r"""See BTF_UINT.
    
    """
    ...

def is_type_uint128(t: type_t) -> bool:
    r"""See BTF_UINT128.
    
    """
    ...

def is_type_uint16(t: type_t) -> bool:
    r"""See BTF_UINT16.
    
    """
    ...

def is_type_uint32(t: type_t) -> bool:
    r"""See BTF_UINT32.
    
    """
    ...

def is_type_uint64(t: type_t) -> bool:
    r"""See BTF_UINT64.
    
    """
    ...

def is_type_union(t: type_t) -> bool:
    r"""See BTF_UNION.
    
    """
    ...

def is_type_unknown(t: type_t) -> bool:
    r"""See BT_UNKNOWN.
    
    """
    ...

def is_type_void(t: type_t) -> bool:
    r"""See BTF_VOID.
    
    """
    ...

def is_type_volatile(t: type_t) -> bool:
    r"""See BTM_VOLATILE.
    
    """
    ...

def is_typeid_last(t: type_t) -> bool:
    r"""Is the type_t the last byte of type declaration? (there are no additional bytes after a basic type, see _BT_LAST_BASIC) 
            
    """
    ...

def is_user_cc(cc: callcnv_t) -> bool:
    r"""Does the calling convention specify argument locations explicitly?
    
    """
    ...

def is_vararg_cc(cc: callcnv_t) -> bool:
    r"""Does the calling convention use ellipsis?
    
    """
    ...

def lexcompare_tinfo(t1: typid_t, t2: typid_t, arg3: int) -> int:
    ...

def load_til(name: str, tildir: str = None) -> str:
    r"""Load til from a file without adding it to the database list (see also add_til). Failure to load base tils are reported into 'errbuf'. They do not prevent loading of the main til. 
            
    :param name: filename of the til. If it's an absolute path, tildir is ignored.
    * NB: the file extension is forced to .til
    :param tildir: directory where to load the til from. nullptr means default til subdirectories.
    :returns: pointer to resulting til, nullptr if failed and error message is in errbuf
    """
    ...

def load_til_header(tildir: str, name: str) -> str:
    r"""Get human-readable til description.
    
    """
    ...

def lower_type(til: til_t, tif: tinfo_t, name: str = None, _helper: lowertype_helper_t = None) -> int:
    r"""Lower type. Inspect the type and lower all function subtypes using lower_func_type(). 
    We call the prototypes usually encountered in source files "high level" 
    They may have implicit arguments, array arguments, big structure retvals, etc 
    We introduce explicit arguments (i.e. 'this' pointer) and call the result 
    "low level prototype". See FTI_HIGH.
    In order to improve heuristics for recognition of big structure retvals, 
    it is recommended to pass a helper that will be used to make decisions. 
    That helper will be used only for lowering 'tif', and not for the children 
    types walked through by recursion. 
            
    :returns: 1: removed FTI_HIGH,
    :returns: 2: made substantial changes
    :returns: -1: failure
    """
    ...

def new_til(name: str, desc: str) -> til_t:
    r"""Initialize a til.
    
    """
    ...

def next_named_type(ti: til_t, name: str, ntf_flags: int) -> str:
    r"""Enumerate types. 
            
    :param ti: type library. nullptr means the local type library for the current database.
    :param name: the current name. the name that follows this one will be returned.
    :param ntf_flags: combination of Flags for named types
    :returns: Type or symbol names, depending of ntf_flags. Returns mangled names. Never returns anonymous types. To include them, enumerate types by ordinals.
    """
    ...

def optimize_argloc(vloc: argloc_t, size: int, gaps: rangeset_t) -> bool:
    r"""Verify and optimize scattered argloc into simple form. All new arglocs must be processed by this function. 
            
    :returns: true: success
    :returns: false: the input argloc was illegal
    """
    ...

def pack_idcobj_to_bv(obj: idc_value_t, tif: tinfo_t, bytes: relobj_t, objoff: void, pio_flags: int = 0) -> error_t:
    r"""Write a typed idc object to the byte vector. Byte vector may be non-empty, this function will append data to it 
            
    """
    ...

def pack_idcobj_to_idb(obj: idc_value_t, tif: tinfo_t, ea: ida_idaapi.ea_t, pio_flags: int = 0) -> error_t:
    r"""Write a typed idc object to the database.
    
    """
    ...

def pack_object_to_bv(obj: Any, til: til_t, type: bytes, fields: bytes, base_ea: ida_idaapi.ea_t, pio_flags: int = 0) -> Any:
    r"""Packs a typed object to a string
    
    :param til: Type library. 'None' can be passed.
    :param type: type string
    :param fields: fields string (may be empty or None)
    :param base_ea: base ea used to relocate the pointers in the packed object
    :param pio_flags: flags used while unpacking
    :returns: tuple(1, packed_buf) on success, or tuple(0, err_code) on failure
    """
    ...

def pack_object_to_idb(obj: Any, til: til_t, type: bytes, fields: bytes, ea: ida_idaapi.ea_t, pio_flags: int = 0) -> Any:
    r"""Write a typed object to the database.
    Raises an exception if wrong parameters were passed or conversion fails
    Returns the error_t returned by idaapi.pack_object_to_idb
    
    :param til: Type library. 'None' can be passed.
    :param type: type string
    :param fields: fields string (may be empty or None)
    :param ea: ea to be used while packing
    :param pio_flags: flags used while unpacking
    """
    ...

def parse_decl(out_tif: tinfo_t, til: til_t, decl: str, pt_flags: int) -> str:
    r"""Parse ONE declaration. If the input string contains more than one declaration, the first complete type declaration (PT_TYP) or the last variable declaration (PT_VAR) will be used. 
            
    :param out_tif: type info
    :param til: type library to use. may be nullptr
    :param decl: C declaration to parse
    :param pt_flags: combination of Type parsing flags bits
    :returns: true: ok
    :returns: false: declaration is bad, the error message is displayed if !PT_SIL
    """
    ...

def parse_decls(til: til_t, input: str, printer: printer_t, hti_flags: int) -> int:
    r"""Parse many declarations and store them in a til. If there are any errors, they will be printed using 'printer'. This function uses default include path and predefined macros from the database settings. It always uses the HTI_DCL bit. 
            
    :param til: type library to store the result
    :param input: input string or file name (see hti_flags)
    :param printer: function to output error messages (use msg or nullptr or your own callback)
    :param hti_flags: combination of Type formatting flags
    :returns: number of errors, 0 means ok.
    """
    ...

def print_argloc(vloc: argloc_t, size: int = 0, vflags: int = 0) -> int:
    r"""Convert an argloc to human readable form.
    
    """
    ...

def print_decls(printer: text_sink_t, til: til_t, ordinals: List[int], flags: int) -> int:
    r"""Print types (and possibly their dependencies) in a format suitable for using in
    a header file. This is the reverse parse_decls().
    
    :param printer: a handler for printing text
    :param til: the type library holding the ordinals
    :param ordinals: a list of ordinals corresponding to the types to print
    :param flags: a combination of PDF_ constants
    :returns:  >0: the number of types exported
    :returns:   0: an error occurred
    :returns:  <0: the negated number of types exported. There were minor errors and
                   the resulting output might not be compilable.
    """
    ...

def print_tinfo(prefix: str, indent: int, cmtindent: int, flags: int, tif: tinfo_t, name: str, cmt: str) -> str:
    ...

def print_type(ea: ida_idaapi.ea_t, prtype_flags: int) -> str:
    r"""Get type declaration for the specified address. 
            
    :param ea: address
    :param prtype_flags: combination of Type printing flags
    :returns: success
    """
    ...

def read_tinfo_bitfield_value(typid: typid_t, v: uint64, bitoff: int) -> uint64:
    ...

def register_custom_callcnv(cnv_incref: custom_callcnv_t) -> custom_callcnv_t:
    r"""Register a calling convention 
            
    :returns: CM_CC_INVALID means failure:
    * bad ccinf.name
    * ccinf.name already exists
    * the calling convention is special (usercall, purging, vararg) and there are too many of them already
    """
    ...

def remove_abi_opts(abi_opts: str, user_level: bool = False) -> bool:
    ...

def remove_pointer(tif: tinfo_t) -> tinfo_t:
    r"""BT_PTR: If the current type is a pointer, return the pointed object. If the current type is not a pointer, return the current type. See also get_ptrarr_object() and get_pointed_object() 
            
    """
    ...

def remove_tinfo_pointer(tif: tinfo_t, name: str, til: til_t) -> Any:
    r"""Remove pointer of a type. (i.e. convert "char *" into "char"). Optionally remove
    the "lp" (or similar) prefix of the input name. If the input type is not a
    pointer, then fail.
    
    :param tif: the type info
    :param name: the name of the type to "unpointerify"
    :param til: the type library
    :returns: a tuple (success, new-name)
    """
    ...

def replace_ordinal_typerefs(til: til_t, tif: tinfo_t) -> int:
    r"""Replace references to ordinal types by name references. This function 'unties' the type from the current local type library and makes it easier to export it. 
            
    :param til: type library to use. may be nullptr.
    :param tif: type to modify (in/out)
    :returns: number: of replaced subtypes, -1 on failure
    """
    ...

def resolve_typedef(til: til_t, type: type_t) -> type_t:
    ...

def save_tinfo(tif: tinfo_t, til: til_t, ord: size_t, name: str, ntf_flags: int) -> tinfo_code_t:
    ...

def score_tinfo(tif: tinfo_t) -> int:
    ...

def serialize_tinfo(type: qtype, fields: qtype, fldcmts: qtype, tif: tinfo_t, sudt_flags: int) -> bool:
    ...

def set_abi_name(abiname: str, user_level: bool = False) -> bool:
    r"""Set abi name (see Compiler IDs)
    
    """
    ...

def set_c_header_path(incdir: str) -> None:
    r"""Set include directory path the target compiler.
    
    """
    ...

def set_c_macros(macros: str) -> None:
    r"""Set predefined macros for the target compiler.
    
    """
    ...

def set_compiler(cc: compiler_info_t, flags: int, abiname: str = None) -> bool:
    r"""Change current compiler. 
            
    :param cc: compiler to switch to
    :param flags: Set compiler flags
    :param abiname: ABI name
    :returns: success
    """
    ...

def set_compiler_id(id: comp_t, abiname: str = None) -> bool:
    r"""Set the compiler id (see Compiler IDs)
    
    """
    ...

def set_compiler_string(compstr: str, user_level: bool) -> bool:
    r""":param compstr: - compiler description in form <abbr>:<abiname>
    :param user_level: - initiated by user if TRUE
    :returns: success
    """
    ...

def set_numbered_type(ti: til_t, ordinal: int, ntf_flags: int, name: str, type: type_t, fields: p_list = None, cmt: str = None, fldcmts: p_list = None, sclass: sclass_t = None) -> tinfo_code_t:
    ...

def set_tinfo_attr(tif: tinfo_t, ta: type_attr_t, may_overwrite: bool) -> bool:
    ...

def set_tinfo_attrs(tif: tinfo_t, ta: type_attrs_t) -> bool:
    ...

def set_tinfo_property(tif: tinfo_t, sta_prop: int, x: size_t) -> int:
    ...

def set_tinfo_property4(tif: tinfo_t, sta_prop: int, p1: size_t, p2: size_t, p3: size_t, p4: size_t) -> int:
    ...

def set_type_alias(ti: til_t, src_ordinal: int, dst_ordinal: int) -> bool:
    r"""Create a type alias. Redirects all references to source type to the destination type. This is equivalent to instantaneous replacement all references to srctype by dsttype. 
            
    """
    ...

def set_type_choosable(ti: til_t, ordinal: int, value: bool) -> None:
    r"""Enable/disable 'choosability' flag for a struct/union type 
            
    :param ti: type library
    :param ordinal: ordinal number of a UDT type
    :param value: flag value
    """
    ...

def set_vftable_ea(ordinal: int, vftable_ea: ida_idaapi.ea_t) -> bool:
    r"""Set the address of a vftable instance for a vftable type. 
            
    :param ordinal: ordinal number of the corresponding vftable type.
    :param vftable_ea: address of a virtual function table.
    :returns: success
    """
    ...

def store_til(ti: til_t, tildir: str, name: str) -> bool:
    r"""Store til to a file. If the til contains garbage, it will be collected before storing the til. Your plugin should call compact_til() before calling store_til(). 
            
    :param ti: type library to store
    :param tildir: directory where to store the til. nullptr means current directory.
    :param name: filename of the til. If it's an absolute path, tildir is ignored.
    * NB: the file extension is forced to .til
    :returns: success
    """
    ...

def stroff_as_size(plen: int, tif: tinfo_t, value: asize_t) -> bool:
    r"""Should display a structure offset expression as the structure size?
    
    """
    ...

def switch_to_golang() -> None:
    r"""switch to GOLANG calling convention (to be used as default CC)
    
    """
    ...

def tinfo_errstr(code: tinfo_code_t) -> str:
    r"""Helper function to convert an error code into a printable string. Additional arguments are handled using the functions from err.h 
            
    """
    ...

def udt_type_data_t__find_member(_this: udt_type_data_t, udm: udm_t, strmem_flags: int) -> ssize_t:
    ...

def udt_type_data_t__get_best_fit_member(_this: udt_type_data_t, disp: asize_t) -> ssize_t:
    ...

def unpack_idcobj_from_bv(obj: idc_value_t, tif: tinfo_t, bytes: bytevec_t, pio_flags: int = 0) -> error_t:
    r"""Read a typed idc object from the byte vector.
    
    """
    ...

def unpack_idcobj_from_idb(obj: idc_value_t, tif: tinfo_t, ea: ida_idaapi.ea_t, off0: bytevec_t, pio_flags: int = 0) -> error_t:
    r"""Collection of register objects.
    
    Read a typed idc object from the database 
            
    """
    ...

def unpack_object_from_bv(til: til_t, type: bytes, fields: bytes, bytes: Any, pio_flags: int = 0) -> Any:
    r"""Unpacks a buffer into an object.
    Returns the error_t returned by idaapi.pack_object_to_idb
    
    :param til: Type library. 'None' can be passed.
    :param type: type string
    :param fields: fields string (may be empty or None)
    :param bytes: the bytes to unpack
    :param pio_flags: flags used while unpacking
    :returns: tuple(1, obj) on success, or tuple(0, err) on failure
    """
    ...

def unpack_object_from_idb(til: til_t, type: bytes, fields: bytes, ea: ida_idaapi.ea_t, pio_flags: int = 0) -> Any:
    r"""Unpacks from the database at 'ea' to an object.
    Please refer to unpack_object_from_bv()
    """
    ...

def unregister_custom_callcnv(cnv_decref: custom_callcnv_t) -> custom_callcnv_t:
    r"""Unregister a calling convention 
            
    :returns: true if successfully unregistered the custom calling convention
    """
    ...

def use_golang_cc() -> bool:
    r"""is GOLANG calling convention used by default?
    
    """
    ...

def value_repr_t__from_opinfo(_this: value_repr_t, flags: flags64_t, afl: aflags_t, opinfo: opinfo_t, ap: array_parameters_t) -> bool:
    ...

def value_repr_t__print_(_this: value_repr_t, colored: bool) -> str:
    ...

def verify_argloc(vloc: argloc_t, size: int, gaps: rangeset_t) -> int:
    r"""Verify argloc_t. 
            
    :param vloc: argloc to verify
    :param size: total size of the variable
    :param gaps: if not nullptr, specifies gaps in structure definition. these gaps should not map to any argloc, but everything else must be covered
    :returns: 0 if ok, otherwise an interr code.
    """
    ...

def verify_tinfo(typid: typid_t) -> int:
    ...

def visit_stroff_udms(sfv: udm_visitor_t, path: tid_t, disp: adiff_t, appzero: bool) -> adiff_t:
    r"""Visit structure fields in a stroff expression or in a reference to a struct data variable. This function can be used to enumerate all components of an expression like 'a.b.c'. 
            
    :param sfv: visitor object
    :param path: struct path (path[0] contains the initial struct id)
    :param disp: offset into structure
    :param appzero: should visit field at offset zero?
    :returns: visitor result
    """
    ...

def visit_subtypes(visitor: tinfo_visitor_t, out: type_mods_t, tif: tinfo_t, name: str, cmt: str) -> int:
    ...

def write_tinfo_bitfield_value(typid: typid_t, dst: uint64, v: uint64, bitoff: int) -> uint64:
    ...

ABS_NO: int  # 1
ABS_UNK: int  # 0
ABS_YES: int  # 2
ADDTIL_ABORTED: int  # 3
ADDTIL_COMP: int  # 2
ADDTIL_DEFAULT: int  # 0
ADDTIL_FAILED: int  # 0
ADDTIL_INCOMP: int  # 1
ADDTIL_OK: int  # 1
ADDTIL_SILENT: int  # 2
ALOC_CUSTOM: int  # 7
ALOC_DIST: int  # 2
ALOC_NONE: int  # 0
ALOC_REG1: int  # 3
ALOC_REG2: int  # 4
ALOC_RREL: int  # 5
ALOC_STACK: int  # 1
ALOC_STATIC: int  # 6
ARGREGS_BY_SLOTS: int  # 3
ARGREGS_FP_MASKS_GP: int  # 4
ARGREGS_GP_ONLY: int  # 1
ARGREGS_INDEPENDENT: int  # 2
ARGREGS_MIPS_O32: int  # 5
ARGREGS_POLICY_UNDEFINED: int  # 0
ARGREGS_RISCV: int  # 6
BADORD: int  # 4294967295
BADSIZE: int  # 18446744073709551615
BFA_FUNC_EXT_FORMAT: int  # 128
BFA_FUNC_MARKER: int  # 15
BFA_HIGH: int  # 4
BFA_NORET: int  # 1
BFA_PURE: int  # 2
BFA_STATIC: int  # 8
BFA_VIRTUAL: int  # 16
BTE_ALWAYS: int  # 128
BTE_BITMASK: int  # 16
BTE_CHAR: int  # 32
BTE_HEX: int  # 0
BTE_OUT_MASK: int  # 96
BTE_RESERVED: int  # 8
BTE_SDEC: int  # 64
BTE_SIZE_MASK: int  # 7
BTE_UDEC: int  # 96
BTF_BOOL: int  # 8
BTF_BYTE: int  # 17
BTF_CHAR: int  # 50
BTF_DOUBLE: int  # 25
BTF_ENUM: int  # 45
BTF_FLOAT: int  # 9
BTF_INT: int  # 7
BTF_INT128: int  # 22
BTF_INT16: int  # 19
BTF_INT32: int  # 20
BTF_INT64: int  # 21
BTF_INT8: int  # 18
BTF_LDOUBLE: int  # 41
BTF_SINT: int  # 23
BTF_STRUCT: int  # 13
BTF_TBYTE: int  # 57
BTF_TYPEDEF: int  # 61
BTF_UCHAR: int  # 34
BTF_UINT: int  # 39
BTF_UINT128: int  # 38
BTF_UINT16: int  # 35
BTF_UINT32: int  # 36
BTF_UINT64: int  # 37
BTF_UINT8: int  # 34
BTF_UNION: int  # 29
BTF_UNK: int  # 48
BTF_VOID: int  # 1
BTMT_ARRESERV: int  # 32
BTMT_BFLDI16: int  # 16
BTMT_BFLDI32: int  # 32
BTMT_BFLDI64: int  # 48
BTMT_BFLDI8: int  # 0
BTMT_BOOL1: int  # 16
BTMT_BOOL2: int  # 32
BTMT_BOOL4: int  # 48
BTMT_BOOL8: int  # 32
BTMT_CHAR: int  # 48
BTMT_CLOSURE: int  # 48
BTMT_DEFBOOL: int  # 0
BTMT_DEFCALL: int  # 0
BTMT_DEFPTR: int  # 0
BTMT_DOUBLE: int  # 16
BTMT_ENUM: int  # 32
BTMT_FAR: int  # 32
BTMT_FARCALL: int  # 32
BTMT_FLOAT: int  # 0
BTMT_INTCALL: int  # 48
BTMT_LNGDBL: int  # 32
BTMT_NEAR: int  # 16
BTMT_NEARCALL: int  # 16
BTMT_NONBASED: int  # 16
BTMT_SIGNED: int  # 16
BTMT_SIZE0: int  # 0
BTMT_SIZE12: int  # 16
BTMT_SIZE128: int  # 48
BTMT_SIZE48: int  # 32
BTMT_SPECFLT: int  # 48
BTMT_STRUCT: int  # 0
BTMT_TYPEDEF: int  # 48
BTMT_UNION: int  # 16
BTMT_UNKSIGN: int  # 0
BTMT_UNSIGNED: int  # 32
BTMT_USIGNED: int  # 32
BTM_CONST: int  # 64
BTM_VOLATILE: int  # 128
BT_ARRAY: int  # 11
BT_BITFIELD: int  # 14
BT_BOOL: int  # 8
BT_COMPLEX: int  # 13
BT_FLOAT: int  # 9
BT_FUNC: int  # 12
BT_INT: int  # 7
BT_INT128: int  # 6
BT_INT16: int  # 3
BT_INT32: int  # 4
BT_INT64: int  # 5
BT_INT8: int  # 2
BT_PTR: int  # 10
BT_RESERVED: int  # 15
BT_SEGREG: int  # 55
BT_UNK: int  # 0
BT_UNKNOWN: int  # 48
BT_UNK_BYTE: int  # 17
BT_UNK_DWORD: int  # 33
BT_UNK_OWORD: int  # 49
BT_UNK_QWORD: int  # 32
BT_UNK_WORD: int  # 16
BT_VOID: int  # 1
CCI_PURGE: int  # 2
CCI_USER: int  # 4
CCI_VARARG: int  # 1
CCN_C: int  # 0
CCN_CPP: int  # 1
CC_ALLOW_ARGPERM: int  # 2
CC_ALLOW_REGHOLES: int  # 4
CC_CDECL_OK: int  # 1
CC_GOLANG_OK: int  # 16
CC_HAS_ELLIPSIS: int  # 8
CM_CC_CDECL: int  # 48
CM_CC_ELLIPSIS: int  # 64
CM_CC_FASTCALL: int  # 112
CM_CC_FIRST_PLAIN_CUSTOM: int  # 512
CM_CC_GOLANG: int  # 176
CM_CC_GOSTK: int  # 256
CM_CC_INVALID: int  # 0
CM_CC_LAST_USERCALL: int  # 255
CM_CC_MASK: int  # 240
CM_CC_PASCAL: int  # 96
CM_CC_RESERVE3: int  # 192
CM_CC_SPECIAL: int  # 240
CM_CC_SPECIALE: int  # 208
CM_CC_SPECIALP: int  # 224
CM_CC_SPOILED: int  # 160
CM_CC_STDCALL: int  # 80
CM_CC_SWIFT: int  # 144
CM_CC_THISCALL: int  # 128
CM_CC_UNKNOWN: int  # 16
CM_CC_VOIDARG: int  # 32
CM_MASK: int  # 3
CM_M_FF: int  # 4
CM_M_FN: int  # 12
CM_M_MASK: int  # 12
CM_M_NF: int  # 8
CM_M_NN: int  # 0
CM_N16_F32: int  # 2
CM_N32_F48: int  # 3
CM_N64: int  # 1
CM_N8_F16: int  # 1
CM_UNKNOWN: int  # 0
COMP_BC: int  # 2
COMP_BP: int  # 8
COMP_GNU: int  # 6
COMP_MASK: int  # 15
COMP_MS: int  # 1
COMP_UNK: int  # 0
COMP_UNSURE: int  # 128
COMP_VISAGE: int  # 7
COMP_WATCOM: int  # 3
C_PC_COMPACT: int  # 10
C_PC_FLAT: int  # 3
C_PC_HUGE: int  # 6
C_PC_LARGE: int  # 6
C_PC_MEDIUM: int  # 14
C_PC_SMALL: int  # 2
C_PC_TINY: int  # 2
DEFMASK64: int  # 18446744073709551615
ETF_AUTONAME: int  # 64
ETF_BYTIL: int  # 128
ETF_COMPATIBLE: int  # 8
ETF_FORCENAME: int  # 32
ETF_FUNCARG: int  # 16
ETF_MAY_DESTROY: int  # 4
ETF_NO_ARRAY: int  # 256
ETF_NO_LAYOUT: int  # 2
ETF_NO_SAVE: int  # 1
FAH_BYTE: int  # 255
FAI_ARRAY: int  # 8
FAI_HIDDEN: int  # 1
FAI_RETPTR: int  # 2
FAI_STRUCT: int  # 4
FAI_UNUSED: int  # 16
FIRST_NONTRIVIAL_TYPID: int  # 256
FMTFUNC_PRINTF: int  # 0
FMTFUNC_SCANF: int  # 1
FMTFUNC_STRFMON: int  # 3
FMTFUNC_STRFTIME: int  # 2
FRB_CHAR: int  # 6
FRB_CUSTOM: int  # 12
FRB_ENUM: int  # 8
FRB_FLOAT: int  # 5
FRB_INVBITS: int  # 512
FRB_INVSIGN: int  # 256
FRB_LZERO: int  # 2048
FRB_MASK: int  # 15
FRB_NUMB: int  # 1
FRB_NUMD: int  # 4
FRB_NUMH: int  # 3
FRB_NUMO: int  # 2
FRB_OFFSET: int  # 9
FRB_SEG: int  # 7
FRB_SIGNED: int  # 1024
FRB_STRLIT: int  # 10
FRB_STROFF: int  # 11
FRB_TABFORM: int  # 4096
FRB_UNK: int  # 0
FTI_ALL: int  # 8191
FTI_ARGLOCS: int  # 256
FTI_CALLTYPE: int  # 192
FTI_CONST: int  # 1024
FTI_CTOR: int  # 2048
FTI_DEFCALL: int  # 0
FTI_DTOR: int  # 4096
FTI_EXPLOCS: int  # 512
FTI_FARCALL: int  # 128
FTI_HIGH: int  # 8
FTI_INTCALL: int  # 192
FTI_NEARCALL: int  # 64
FTI_NORET: int  # 2
FTI_PURE: int  # 4
FTI_SPOILED: int  # 1
FTI_STATIC: int  # 16
FTI_VIRTUAL: int  # 32
GTD_CALC_ARGLOCS: int  # 0
GTD_CALC_LAYOUT: int  # 0
GTD_DEL_BITFLDS: int  # 64
GTD_NO_ARGLOCS: int  # 128
GTD_NO_LAYOUT: int  # 128
GTS_BASECLASS: int  # 2
GTS_NESTED: int  # 1
GUESS_FUNC_FAILED: int  # 0
GUESS_FUNC_OK: int  # 2
GUESS_FUNC_TRIVIAL: int  # 1
HTI_CPP: int  # 1
HTI_DCL: int  # 1024
HTI_EXT: int  # 4
HTI_FIL: int  # 64
HTI_HIGH: int  # 32768
HTI_INT: int  # 2
HTI_LEX: int  # 8
HTI_LOWER: int  # 65536
HTI_MAC: int  # 128
HTI_NDC: int  # 2048
HTI_NER: int  # 512
HTI_NOBASE: int  # 1048576
HTI_NWR: int  # 256
HTI_PAK: int  # 28672
HTI_PAK1: int  # 4096
HTI_PAK16: int  # 20480
HTI_PAK2: int  # 8192
HTI_PAK4: int  # 12288
HTI_PAK8: int  # 16384
HTI_PAKDEF: int  # 0
HTI_PAK_SHIFT: int  # 12
HTI_RAWARGS: int  # 131072
HTI_RELAXED: int  # 524288
HTI_SEMICOLON: int  # 2097152
HTI_STANDALONE: int  # 4194304
HTI_TST: int  # 32
HTI_UNP: int  # 16
MAX_DECL_ALIGN: int  # 15
MAX_ENUM_SERIAL: int  # 255
MAX_FUNC_ARGS: int  # 256
NTF_64BIT: int  # 64
NTF_CHKSYNC: int  # 512
NTF_COPY: int  # 4096
NTF_FIXNAME: int  # 128
NTF_IDBENC: int  # 256
NTF_NOBASE: int  # 2
NTF_NOCUR: int  # 32
NTF_NO_NAMECHK: int  # 1024
NTF_REPLACE: int  # 4
NTF_SYMM: int  # 0
NTF_SYMU: int  # 8
NTF_TYPE: int  # 1
NTF_UMANGLED: int  # 8
PDF_DEF_BASE: int  # 4
PDF_DEF_FWD: int  # 2
PDF_HEADER_CMT: int  # 8
PDF_INCL_DEPS: int  # 1
PIO_IGNORE_PTRS: int  # 8
PIO_NOATTR_FAIL: int  # 4
PRALOC_STKOFF: int  # 2
PRALOC_VERIFY: int  # 1
PRTYPE_1LINCMT: int  # 8192
PRTYPE_1LINE: int  # 0
PRTYPE_ARGLOCS: int  # 262144
PRTYPE_COLORED: int  # 2048
PRTYPE_CPP: int  # 16
PRTYPE_DEF: int  # 32
PRTYPE_HEADER: int  # 16384
PRTYPE_MAXSTR: int  # 65536
PRTYPE_METHODS: int  # 4096
PRTYPE_MULTI: int  # 1
PRTYPE_NOARGS: int  # 64
PRTYPE_NOARRS: int  # 128
PRTYPE_NOREGEX: int  # 1024
PRTYPE_NORES: int  # 256
PRTYPE_OFFSETS: int  # 32768
PRTYPE_PRAGMA: int  # 4
PRTYPE_RESTORE: int  # 512
PRTYPE_SEMI: int  # 8
PRTYPE_TAIL: int  # 131072
PRTYPE_TYPE: int  # 2
PT_EMPTY: int  # 8192
PT_FILE: int  # 65536
PT_HIGH: int  # 128
PT_LOWER: int  # 256
PT_NDC: int  # 2
PT_PACKMASK: int  # 112
PT_RAWARGS: int  # 1024
PT_RELAXED: int  # 4096
PT_REPLACE: int  # 512
PT_SEMICOLON: int  # 16384
PT_SIL: int  # 1
PT_STANDALONE: int  # 4194304
PT_SYMBOL: int  # 32768
PT_TYP: int  # 4
PT_VAR: int  # 8
RESERVED_BYTE: int  # 255
SC_AUTO: int  # 5
SC_EXT: int  # 2
SC_FRIEND: int  # 6
SC_REG: int  # 4
SC_STAT: int  # 3
SC_TYPE: int  # 1
SC_UNK: int  # 0
SC_VIRT: int  # 7
SETCOMP_BY_USER: int  # 8
SETCOMP_ONLY_ABI: int  # 4
SETCOMP_ONLY_ID: int  # 2
SETCOMP_OVERRIDE: int  # 1
STI_ACCHAR: int  # 12
STI_ACHAR: int  # 10
STI_ACUCHAR: int  # 13
STI_AEABI_LCMP: int  # 17
STI_AEABI_MEMCLR: int  # 24
STI_AEABI_MEMCPY: int  # 22
STI_AEABI_MEMSET: int  # 23
STI_AEABI_ULCMP: int  # 18
STI_AUCHAR: int  # 11
STI_COMPLEX128: int  # 29
STI_COMPLEX64: int  # 28
STI_DONT_USE: int  # 19
STI_FDELOP: int  # 15
STI_FPURGING: int  # 14
STI_LAST: int  # 31
STI_MSGSEND: int  # 16
STI_PBYTE: int  # 4
STI_PCCHAR: int  # 2
STI_PCHAR: int  # 0
STI_PCUCHAR: int  # 3
STI_PCVOID: int  # 9
STI_PINT: int  # 5
STI_PPVOID: int  # 8
STI_PUCHAR: int  # 1
STI_PUINT: int  # 6
STI_PUNKNOWN: int  # 30
STI_PVOID: int  # 7
STI_RTC_CHECK_2: int  # 25
STI_RTC_CHECK_4: int  # 26
STI_RTC_CHECK_8: int  # 27
STI_SIZE_T: int  # 20
STI_SSIZE_T: int  # 21
STRMEM_ANON: int  # -2147483648
STRMEM_AUTO: int  # 2
STRMEM_CASTABLE_TO: int  # 1073741824
STRMEM_INDEX: int  # 1
STRMEM_LOWBND: int  # 8
STRMEM_MASK: int  # 15
STRMEM_MAXS: int  # 7
STRMEM_MINS: int  # 6
STRMEM_NAME: int  # 3
STRMEM_NEXT: int  # 9
STRMEM_OFFSET: int  # 0
STRMEM_SIZE: int  # 5
STRMEM_SKIP_EMPTY: int  # 536870912
STRMEM_SKIP_GAPS: int  # 16777216
STRMEM_TYPE: int  # 4
STRMEM_VFTABLE: int  # 268435456
STRUC_SEPARATOR: str  # .
SUDT_ALIGN: int  # 2
SUDT_CONST: int  # 64
SUDT_FAST: int  # 16
SUDT_GAPS: int  # 4
SUDT_SERDEF: int  # 512
SUDT_SORT: int  # 1
SUDT_TRUNC: int  # 256
SUDT_UNEX: int  # 8
SUDT_VOLATILE: int  # 128
SWIG_PYTHON_LEGACY_BOOL: int  # 1
TAENUM_64BIT: int  # 32
TAENUM_BIN: int  # 512
TAENUM_LZERO: int  # 2048
TAENUM_NUMSIGN: int  # 1024
TAENUM_OCT: int  # 256
TAENUM_SIGNED: int  # 128
TAENUM_UNSIGNED: int  # 64
TAFLD_BASECLASS: int  # 32
TAFLD_BYTIL: int  # 16384
TAFLD_FRAME_R: int  # 4096
TAFLD_FRAME_S: int  # 8192
TAFLD_GAP: int  # 1024
TAFLD_METHOD: int  # 512
TAFLD_REGCMT: int  # 2048
TAFLD_UNALIGNED: int  # 64
TAFLD_VFTABLE: int  # 256
TAFLD_VIRTBASE: int  # 128
TAH_ALL: int  # 32752
TAH_BYTE: int  # 254
TAH_HASATTRS: int  # 16
TAPTR_PTR32: int  # 32
TAPTR_PTR64: int  # 64
TAPTR_RESTRICT: int  # 96
TAPTR_SHIFTED: int  # 128
TAUDT_CPPOBJ: int  # 128
TAUDT_FIXED: int  # 1024
TAUDT_MSSTRUCT: int  # 32
TAUDT_TUPLE: int  # 2048
TAUDT_UNALIGNED: int  # 64
TAUDT_VFTABLE: int  # 256
TA_FORMAT: str  # format
TA_ORG_ARRDIM: str  # __org_arrdim
TA_ORG_TYPEDEF: str  # __org_typedef
TA_VALUE_REPR: str  # 
TCMP_ANYBASE: int  # 64
TCMP_AUTOCAST: int  # 2
TCMP_CALL: int  # 8
TCMP_DECL: int  # 32
TCMP_DEEP_UDT: int  # 256
TCMP_DELPTR: int  # 16
TCMP_EQUAL: int  # 0
TCMP_IGNMODS: int  # 1
TCMP_MANCAST: int  # 4
TCMP_SKIPTHIS: int  # 128
TERR_ALIEN_NAME: int  # -31
TERR_BAD_ARG: int  # -4
TERR_BAD_ARRAY: int  # -8
TERR_BAD_BASE: int  # -24
TERR_BAD_BF: int  # -9
TERR_BAD_BMASK: int  # -17
TERR_BAD_FX_SIZE: int  # -36
TERR_BAD_GAP: int  # -25
TERR_BAD_GROUPS: int  # -29
TERR_BAD_INDEX: int  # -7
TERR_BAD_LAYOUT: int  # -28
TERR_BAD_MSKVAL: int  # -18
TERR_BAD_NAME: int  # -3
TERR_BAD_OFFSET: int  # -10
TERR_BAD_REPR: int  # -19
TERR_BAD_SERIAL: int  # -30
TERR_BAD_SIZE: int  # -6
TERR_BAD_SUBTYPE: int  # -14
TERR_BAD_TAH: int  # -23
TERR_BAD_TYPE: int  # -5
TERR_BAD_UNIVAR: int  # -11
TERR_BAD_VALUE: int  # -15
TERR_BAD_VARLAST: int  # -12
TERR_COUNT: int  # 39
TERR_DUPNAME: int  # -21
TERR_ENUM_SIZE: int  # -33
TERR_GRP_NOEMPTY: int  # -20
TERR_NESTED: int  # -26
TERR_NOT_COMPAT: int  # -27
TERR_NOT_FOUND: int  # -38
TERR_NOT_IMPL: int  # -34
TERR_NO_BMASK: int  # -16
TERR_OK: int  # 0
TERR_OVERLAP: int  # -13
TERR_SAVE: int  # -1
TERR_SAVE_ERROR: int  # -1
TERR_SERIALIZE: int  # -2
TERR_STOCK: int  # -32
TERR_STRUCT_SIZE: int  # -37
TERR_TYPE_WORSE: int  # -35
TERR_UNION_BF: int  # -22
TERR_WRONGNAME: int  # -3
TIL_ADD_ALREADY: int  # 2
TIL_ADD_FAILED: int  # 0
TIL_ADD_OK: int  # 1
TIL_ALI: int  # 32
TIL_ECC: int  # 512
TIL_ESI: int  # 4
TIL_MAC: int  # 2
TIL_MOD: int  # 64
TIL_ORD: int  # 16
TIL_SLD: int  # 256
TIL_STM: int  # 128
TIL_UNI: int  # 8
TIL_ZIP: int  # 1
TINFO_DEFINITE: int  # 1
TINFO_DELAYFUNC: int  # 2
TINFO_GUESSED: int  # 0
TINFO_STRICT: int  # 4
TPOS_LNNUM: str  # 
TPOS_REGCMT: str  # 
TVIS_CMT: int  # 4
TVIS_NAME: int  # 2
TVIS_RPTCMT: int  # 8
TVIS_TYPE: int  # 1
TVST_DEF: int  # 2
TVST_LEVEL: int  # 4
TVST_PRUNE: int  # 1
TYPE_BASE_MASK: int  # 15
TYPE_FLAGS_MASK: int  # 48
TYPE_FULL_MASK: int  # 63
TYPE_MODIF_MASK: int  # 192
TYPID_ISREF: int  # 256
TYPID_SHIFT: int  # 9
UTP_ENUM: int  # 0
UTP_STRUCT: int  # 1
VALSTR_OPEN: int  # 1
VTBL_MEMNAME: str  # __vftable
VTBL_SUFFIX: str  # _vtbl
annotations: _Feature
cvar: _wrap_cvar  # <ida_typeinf._wrap_cvar object at 0x717729c18fd0>
ida_idaapi: module
ida_idp: module
no_sign: int  # 0
sc_auto: int  # 5
sc_ext: int  # 2
sc_friend: int  # 6
sc_reg: int  # 4
sc_stat: int  # 3
sc_type: int  # 1
sc_unk: int  # 0
sc_virt: int  # 7
type_signed: int  # 1
type_unsigned: int  # 2
weakref: module