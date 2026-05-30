from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with segments.

IDA requires that all program addresses belong to segments (each address must belong to exactly one segment). The situation when an address doesn't belong to any segment is allowed as a temporary situation only when the user changes program segmentation. Bytes outside a segment can't be converted to instructions, have names, comments, etc. Each segment has its start address, ending address and represents a contiguous range of addresses. There might be unused holes between segments.
Each segment has its unique segment selector. This selector is used to distinguish the segment from other segments. For 16-bit programs the selector is equal to the segment base paragraph. For 32-bit programs there is special array to translate the selectors to the segment base paragraphs. A selector is a 32/64 bit value.
The segment base paragraph determines the offsets in the segment. If the start address of the segment == (base << 4) then the first offset in the segment will be 0. The start address should be higher or equal to (base << 4). We will call the offsets in the segment 'virtual addresses'. So, the virtual address of the first byte of the segment is
(start address of segment - segment base linear address)
For IBM PC, the virtual address corresponds to the offset part of the address. For other processors (Z80, for example), virtual addresses correspond to Z80 addresses and linear addresses are used only internally. For MS Windows programs the segment base paragraph is 0 and therefore the segment virtual addresses are equal to linear addresses. 
    
"""

class lock_segment:
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
    def __init__(self, _segm: segment_t) -> Any:
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

class segment_defsr_array:
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

class segment_t:
    @property
    def align(self) -> Any: ...
    @property
    def bitness(self) -> Any: ...
    @property
    def color(self) -> Any: ...
    @property
    def comb(self) -> Any: ...
    @property
    def defsr(self) -> Any: ...
    @property
    def end_ea(self) -> Any: ...
    @property
    def flags(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def orgbase(self) -> Any: ...
    @property
    def perm(self) -> Any: ...
    @property
    def sclass(self) -> Any: ...
    @property
    def sel(self) -> Any: ...
    @property
    def start_ea(self) -> Any: ...
    @property
    def type(self) -> Any: ...
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
    def abits(self) -> int:
        r"""Get number of address bits.
        
        """
        ...
    def abytes(self) -> int:
        r"""Get number of address bytes.
        
        """
        ...
    def clear(self) -> None:
        r"""Set start_ea, end_ea to 0.
        
        """
        ...
    def clr_comorg(self) -> None:
        ...
    def clr_ob_ok(self) -> None:
        ...
    def comorg(self) -> bool:
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
    def is_16bit(self) -> bool:
        r"""Is a 16-bit segment?
        
        """
        ...
    def is_32bit(self) -> bool:
        r"""Is a 32-bit segment?
        
        """
        ...
    def is_64bit(self) -> bool:
        r"""Is a 64-bit segment?
        
        """
        ...
    def is_header_segm(self) -> bool:
        ...
    def is_hidden_segtype(self) -> bool:
        ...
    def is_loader_segm(self) -> bool:
        ...
    def is_visible_segm(self) -> bool:
        ...
    def ob_ok(self) -> bool:
        ...
    def overlaps(self, r: range_t) -> bool:
        r"""Is there an ea in 'r' that is also in this range_t?
        
        """
        ...
    def set_comorg(self) -> None:
        ...
    def set_debugger_segm(self, debseg: bool) -> None:
        ...
    def set_header_segm(self, on: bool) -> None:
        ...
    def set_hidden_segtype(self, hide: bool) -> None:
        ...
    def set_loader_segm(self, ldrseg: bool) -> None:
        ...
    def set_ob_ok(self) -> None:
        ...
    def set_visible_segm(self, visible: bool) -> None:
        ...
    def size(self) -> int:
        r"""Get end_ea - start_ea.
        
        """
        ...
    def update(self) -> bool:
        r"""Update segment information. You must call this function after modification of segment characteristics. Note that not all fields of segment structure may be modified directly, there are special functions to modify some fields. 
                
        :returns: success
        """
        ...
    def use64(self) -> bool:
        r"""Is a 64-bit segment?
        
        """
        ...

def add_segm(para: ida_idaapi.ea_t, start: ida_idaapi.ea_t, end: ida_idaapi.ea_t, name: str, sclass: str, flags: int = 0) -> bool:
    r"""Add a new segment, second form. Segment alignment is set to saRelByte. Segment combination is "public" or "stack" (if segment class is "STACK"). Addressing mode of segment is taken as default (16bit or 32bit). Default segment registers are set to BADSEL. If a segment already exists at the specified range of addresses, this segment will be truncated. Instructions and data in the old segment will be deleted if the new segment has another addressing mode or another segment base address. 
            
    :param para: segment base paragraph. if paragraph can't fit in 16bit, then a new selector is allocated and mapped to the paragraph.
    :param start: start address of the segment. if start==BADADDR then start <- to_ea(para,0).
    :param end: end address of the segment. end address should be higher than start address. For emulate empty segments, use SEG_NULL segment type. If the end address is lower than start address, then fail. If end==BADADDR, then a segment up to the next segment will be created (if the next segment doesn't exist, then 1 byte segment will be created). If 'end' is too high and the new segment would overlap the next segment, 'end' is adjusted properly.
    :param name: name of new segment. may be nullptr
    :param sclass: class of the segment. may be nullptr. type of the new segment is modified if class is one of predefined names:
    * "CODE" -> SEG_CODE
    * "DATA" -> SEG_DATA
    * "CONST" -> SEG_DATA
    * "STACK" -> SEG_BSS
    * "BSS" -> SEG_BSS
    * "XTRN" -> SEG_XTRN
    * "COMM" -> SEG_COMM
    * "ABS" -> SEG_ABSSYM
    :param flags: Add segment flags
    :returns: 1: ok
    :returns: 0: failed, a warning message is displayed
    """
    ...

def add_segm_ex(NONNULL_s: segment_t, name: str, sclass: str, flags: int) -> bool:
    r"""Add a new segment. If a segment already exists at the specified range of addresses, this segment will be truncated. Instructions and data in the old segment will be deleted if the new segment has another addressing mode or another segment base address. 
            
    :param name: name of new segment. may be nullptr. if specified, the segment is immediately renamed
    :param sclass: class of the segment. may be nullptr. if specified, the segment class is immediately changed
    :param flags: Add segment flags
    :returns: 1: ok
    :returns: 0: failed, a warning message is displayed
    """
    ...

def add_segment_translation(segstart: ida_idaapi.ea_t, mappedseg: ida_idaapi.ea_t) -> bool:
    r"""Add segment translation. 
            
    :param segstart: start address of the segment to add translation to
    :param mappedseg: start address of the overlayed segment
    :returns: 1: ok
    :returns: 0: too many translations or bad segstart
    """
    ...

def allocate_selector(segbase: ida_idaapi.ea_t) -> sel_t:
    r"""Allocate a selector for a segment unconditionally. You must call this function before calling add_segm_ex(). add_segm() calls this function itself, so you don't need to allocate a selector. This function will allocate a new free selector and setup its mapping using find_free_selector() and set_selector() functions. 
            
    :param segbase: a new segment base paragraph
    :returns: the allocated selector number
    """
    ...

def change_segment_status(s: segment_t, is_deb_segm: bool) -> int:
    r"""Convert a debugger segment to a regular segment and vice versa. When converting debug->regular, the memory contents will be copied to the database. 
            
    :param s: segment to modify
    :param is_deb_segm: new status of the segment
    :returns: Change segment status result codes
    """
    ...

def del_segm(ea: ida_idaapi.ea_t, flags: int) -> bool:
    r"""Delete a segment. 
            
    :param ea: any address belonging to the segment
    :param flags: Segment modification flags
    :returns: 1: ok
    :returns: 0: failed, no segment at 'ea'.
    """
    ...

def del_segment_translations(segstart: ida_idaapi.ea_t) -> None:
    r"""Delete the translation list 
            
    :param segstart: start address of the segment to delete translation list
    """
    ...

def del_selector(selector: sel_t) -> None:
    r"""Delete mapping of a selector. Be wary of deleting selectors that are being used in the program, this can make a mess in the segments. 
            
    :param selector: number of selector to remove from the translation table
    """
    ...

def find_free_selector() -> sel_t:
    r"""Find first unused selector. 
            
    :returns: a number >= 1
    """
    ...

def find_selector(base: ida_idaapi.ea_t) -> sel_t:
    r"""Find a selector that has mapping to the specified paragraph. 
            
    :param base: paragraph to search in the translation table
    :returns: selector value or base
    """
    ...

def get_defsr(s: Any, reg: Any) -> Any:
    r"""Deprecated, use instead:
        value = s.defsr[reg]
    """
    ...

def get_first_seg() -> Optional[segment_t]:
    r"""Get pointer to the first segment.
    
    """
    ...

def get_group_selector(grpsel: sel_t) -> sel_t:
    r"""Get common selector for a group of segments. 
            
    :param grpsel: selector of group segment
    :returns: common selector of the group or 'grpsel' if no such group is found
    """
    ...

def get_last_seg() -> Optional[segment_t]:
    r"""Get pointer to the last segment.
    
    """
    ...

def get_next_seg(ea: ida_idaapi.ea_t) -> Optional[segment_t]:
    r"""Get pointer to the next segment.
    
    """
    ...

def get_prev_seg(ea: ida_idaapi.ea_t) -> Optional[segment_t]:
    r"""Get pointer to the previous segment.
    
    """
    ...

def get_segm_base(s: segment_t) -> ida_idaapi.ea_t:
    r"""Get segment base linear address. Segment base linear address is used to calculate virtual addresses. The virtual address of the first byte of the segment will be (start address of segment - segment base linear address) 
            
    :param s: pointer to segment
    :returns: 0 if s == nullptr, otherwise segment base linear address
    """
    ...

def get_segm_by_name(name: str) -> Optional[segment_t]:
    r"""Get pointer to segment by its name. If there are several segments with the same name, returns the first of them. 
            
    :param name: segment name. may be nullptr.
    :returns: nullptr or pointer to segment structure
    """
    ...

def get_segm_by_sel(selector: sel_t) -> Optional[segment_t]:
    r"""Get pointer to segment structure. This function finds a segment by its selector. If there are several segments with the same selectors, the last one will be returned. 
            
    :param selector: a segment with the specified selector will be returned
    :returns: pointer to segment or nullptr
    """
    ...

def get_segm_class(s: segment_t) -> str:
    r"""Get segment class. Segment class is arbitrary text (max 8 characters). 
            
    :param s: pointer to segment
    :returns: size of segment class (-1 if s==nullptr or bufsize<=0)
    """
    ...

def get_segm_name(s: segment_t, flags: int = 0) -> str:
    r"""Get true segment name by pointer to segment. 
            
    :param s: pointer to segment
    :param flags: 0-return name as is; 1-substitute bad symbols with _ 1 corresponds to GN_VISIBLE
    :returns: size of segment name (-1 if s==nullptr)
    """
    ...

def get_segm_num(ea: ida_idaapi.ea_t) -> int:
    r"""Get number of segment by address. 
            
    :param ea: linear address belonging to the segment
    :returns: -1 if no segment occupies the specified address. otherwise returns number of the specified segment (0..get_segm_qty()-1)
    """
    ...

def get_segm_para(s: segment_t) -> ida_idaapi.ea_t:
    r"""Get segment base paragraph. Segment base paragraph may be converted to segment base linear address using to_ea() function. In fact, to_ea(get_segm_para(s), 0) == get_segm_base(s). 
            
    :param s: pointer to segment
    :returns: 0 if s == nullptr, the segment base paragraph
    """
    ...

def get_segm_qty() -> int:
    r"""Get number of segments.
    
    """
    ...

def get_segment_alignment(align: uchar) -> str:
    r"""Get text representation of segment alignment code. 
            
    :returns: text digestable by IBM PC assembler.
    """
    ...

def get_segment_cmt(s: segment_t, repeatable: bool) -> str:
    r"""Get segment comment. 
            
    :param s: pointer to segment structure
    :param repeatable: 0: get regular comment. 1: get repeatable comment.
    :returns: size of comment or -1
    """
    ...

def get_segment_combination(comb: uchar) -> str:
    r"""Get text representation of segment combination code. 
            
    :returns: text digestable by IBM PC assembler.
    """
    ...

def get_segment_translations(transmap: eavec_t, segstart: ida_idaapi.ea_t) -> ssize_t:
    r"""Get segment translation list. 
            
    :param transmap: vector of segment start addresses for the translation list
    :param segstart: start address of the segment to get information about
    :returns: -1 if no translation list or bad segstart. otherwise returns size of translation list.
    """
    ...

def get_selector_qty() -> int:
    r"""Get number of defined selectors.
    
    """
    ...

def get_visible_segm_name(s: segment_t) -> str:
    r"""Get segment name by pointer to segment. 
            
    :param s: pointer to segment
    :returns: size of segment name (-1 if s==nullptr)
    """
    ...

def getn_selector(n: int) -> Any:
    r"""Get description of selector (0..get_selector_qty()-1)
    
    """
    ...

def getnseg(n: int) -> Optional[segment_t]:
    r"""Get pointer to segment by its number. 
            
    :param n: segment number in the range (0..get_segm_qty()-1)
    :returns: nullptr or pointer to segment structure
    """
    ...

def getseg(ea: ida_idaapi.ea_t) -> Optional[segment_t]:
    r"""Get pointer to segment by linear address. 
            
    :param ea: linear address belonging to the segment
    :returns: nullptr or pointer to segment structure
    """
    ...

def is_finally_visible_segm(s: segment_t) -> bool:
    r"""See SFL_HIDDEN, SCF_SHHID_SEGM.
    
    """
    ...

def is_miniidb() -> bool:
    r"""Is the database a miniidb created by the debugger?. 
            
    :returns: true if the database contains no segments or only debugger segments
    """
    ...

def is_segm_locked(segm: segment_t) -> bool:
    r"""Is a segment pointer locked?
    
    """
    ...

def is_spec_ea(ea: ida_idaapi.ea_t) -> bool:
    r"""Does the address belong to a segment with a special type?. (SEG_XTRN, SEG_GRP, SEG_ABSSYM, SEG_COMM) 
            
    :param ea: linear address
    """
    ...

def is_spec_segm(seg_type: uchar) -> bool:
    r"""Has segment a special type?. (SEG_XTRN, SEG_GRP, SEG_ABSSYM, SEG_COMM) 
            
    """
    ...

def is_visible_segm(s: segment_t) -> bool:
    r"""See SFL_HIDDEN.
    
    """
    ...

def lock_segm(segm: segment_t, lock: bool) -> None:
    r"""Lock segment pointer Locked pointers are guaranteed to remain valid until they are unlocked. Ranges with locked pointers cannot be deleted or moved. 
            
    """
    ...

def move_segm(s: segment_t, to: ida_idaapi.ea_t, flags: int = 0) -> move_segm_code_t:
    r"""This function moves all information to the new address. It fixes up address sensitive information in the kernel. The total effect is equal to reloading the segment to the target address. For the file format dependent address sensitive information, loader_t::move_segm is called. Also IDB notification event idb_event::segm_moved is called. 
            
    :param s: segment to move
    :param to: new segment start address
    :param flags: Move segment flags
    :returns: Move segment result codes
    """
    ...

def move_segm_start(ea: ida_idaapi.ea_t, newstart: ida_idaapi.ea_t, mode: int) -> bool:
    r"""Move segment start. The main difference between this function and set_segm_start() is that this function may expand the previous segment while set_segm_start() never does it. So, this function allows to change bounds of two segments simultaneously. If the previous segment and the specified segment have the same addressing mode and segment base, then instructions and data are not destroyed - they simply move from one segment to another. Otherwise all instructions/data which migrate from one segment to another are destroyed. 
            
    :param ea: any address belonging to the segment
    :param newstart: new start address of the segment note that segment start address should be higher than segment base linear address.
    :param mode: policy for destroying defined items
    * 0: if it is necessary to destroy defined items, display a dialog box and ask confirmation
    * 1: if it is necessary to destroy defined items, just destroy them without asking the user
    * -1: if it is necessary to destroy defined items, don't destroy them (i.e. function will fail)
    * -2: don't destroy defined items (function will succeed)
    :returns: 1: ok
    :returns: 0: failed, a warning message is displayed
    """
    ...

def move_segm_strerror(code: move_segm_code_t) -> str:
    r"""Return string describing error MOVE_SEGM_... code.
    
    """
    ...

def rebase_program(delta: Any, flags: int) -> int:
    r"""Rebase the whole program by 'delta' bytes. 
            
    :param delta: number of bytes to move the program
    :param flags: Move segment flags it is recommended to use MSF_FIXONCE so that the loader takes care of global variables it stored in the database
    :returns: Move segment result codes
    """
    ...

def segm_adjust_diff(s: segment_t, delta: adiff_t) -> adiff_t:
    r"""Truncate and sign extend a delta depending on the segment.
    
    """
    ...

def segm_adjust_ea(s: segment_t, ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Truncate an address depending on the segment.
    
    """
    ...

def segtype(ea: ida_idaapi.ea_t) -> uchar:
    r"""Get segment type. 
            
    :param ea: any linear address within the segment
    :returns: Segment types, SEG_UNDF if no segment found at 'ea'
    """
    ...

def sel2ea(selector: sel_t) -> ida_idaapi.ea_t:
    r"""Get mapping of a selector as a linear address. 
            
    :param selector: number of selector to translate to linear address
    :returns: linear address the specified selector is mapped to. if there is no mapping, returns to_ea(selector,0);
    """
    ...

def sel2para(selector: sel_t) -> ida_idaapi.ea_t:
    r"""Get mapping of a selector. 
            
    :param selector: number of selector to translate
    :returns: paragraph the specified selector is mapped to. if there is no mapping, returns 'selector'.
    """
    ...

def set_defsr(s: Any, reg: Any, value: Any) -> Any:
    r"""Deprecated, use instead:
        s.defsr[reg] = value
    """
    ...

def set_group_selector(grp: sel_t, sel: sel_t) -> int:
    r"""Create a new group of segments (used OMF files). 
            
    :param grp: selector of group segment (segment type is SEG_GRP) You should create an 'empty' (1 byte) group segment It won't contain anything and will be used to redirect references to the group of segments to the common selector.
    :param sel: common selector of all segments belonging to the segment You should create all segments within the group with the same selector value.
    :returns: 1: ok
    :returns: 0: too many groups (see MAX_GROUPS)
    """
    ...

def set_segm_addressing(s: segment_t, bitness: size_t) -> bool:
    r"""Change segment addressing mode (16, 32, 64 bits). You must use this function to change segment addressing, never change the 'bitness' field directly. This function will delete all instructions, comments and names in the segment 
            
    :param s: pointer to segment
    :param bitness: new addressing mode of segment
    * 2: 64bit segment
    * 1: 32bit segment
    * 0: 16bit segment
    :returns: success
    """
    ...

def set_segm_base(s: segment_t, newbase: ida_idaapi.ea_t) -> bool:
    r"""Internal function.
    
    """
    ...

def set_segm_class(s: segment_t, sclass: str, flags: int = 0) -> int:
    r"""Set segment class. 
            
    :param s: pointer to segment (may be nullptr)
    :param sclass: segment class (may be nullptr). If segment type is SEG_NORM and segment class is one of predefined names, then segment type is changed to:
    * "CODE" -> SEG_CODE
    * "DATA" -> SEG_DATA
    * "STACK" -> SEG_BSS
    * "BSS" -> SEG_BSS
    * if "UNK" then segment type is reset to SEG_NORM.
    :param flags: Add segment flags
    :returns: 1: ok, name is good and segment is renamed
    :returns: 0: failure, name is nullptr or bad or segment is nullptr
    """
    ...

def set_segm_end(ea: ida_idaapi.ea_t, newend: ida_idaapi.ea_t, flags: int) -> bool:
    r"""Set segment end address. The next segment is shrinked to allow expansion of the specified segment. The kernel might even delete the next segment if necessary. The kernel will ask the user for a permission to destroy instructions or data going out of segment scope if such instructions exist. 
            
    :param ea: any address belonging to the segment
    :param newend: new end address of the segment
    :param flags: Segment modification flags
    :returns: 1: ok
    :returns: 0: failed, a warning message is displayed
    """
    ...

def set_segm_name(s: segment_t, name: str, flags: int = 0) -> int:
    r"""Rename segment. The new name is validated (see validate_name). A segment always has a name. If you hadn't specified a name, the kernel will assign it "seg###" name where ### is segment number. 
            
    :param s: pointer to segment (may be nullptr)
    :param name: new segment name
    :param flags: ADDSEG_IDBENC or 0
    :returns: 1: ok, name is good and segment is renamed
    :returns: 0: failure, name is bad or segment is nullptr
    """
    ...

def set_segm_start(ea: ida_idaapi.ea_t, newstart: ida_idaapi.ea_t, flags: int) -> bool:
    r"""Set segment start address. The previous segment is trimmed to allow expansion of the specified segment. The kernel might even delete the previous segment if necessary. The kernel will ask the user for a permission to destroy instructions or data going out of segment scope if such instructions exist. 
            
    :param ea: any address belonging to the segment
    :param newstart: new start address of the segment note that segment start address should be higher than segment base linear address.
    :param flags: Segment modification flags
    :returns: 1: ok
    :returns: 0: failed, a warning message is displayed
    """
    ...

def set_segment_cmt(s: segment_t, cmt: str, repeatable: bool) -> None:
    r"""Set segment comment. 
            
    :param s: pointer to segment structure
    :param cmt: comment string, may be multiline (with '
    '). maximal size is 4096 bytes. Use empty str ("") to delete comment
    :param repeatable: 0: set regular comment. 1: set repeatable comment.
    """
    ...

def set_segment_translations(segstart: ida_idaapi.ea_t, transmap: eavec_t) -> bool:
    r"""Set new translation list. 
            
    :param segstart: start address of the segment to add translation to
    :param transmap: vector of segment start addresses for the translation list. If transmap is empty, the translation list is deleted.
    :returns: 1: ok
    :returns: 0: too many translations or bad segstart
    """
    ...

def set_selector(selector: sel_t, paragraph: ida_idaapi.ea_t) -> int:
    r"""Set mapping of selector to a paragraph. You should call this function _before_ creating a segment which uses the selector, otherwise the creation of the segment will fail. 
            
    :param selector: number of selector to map
    * if selector == BADSEL, then return 0 (fail)
    * if the selector has had a mapping, old mapping is destroyed
    * if the selector number is equal to paragraph value, then the mapping is destroyed because we don't need to keep trivial mappings.
    :param paragraph: paragraph to map selector
    :returns: 1: ok
    :returns: 0: failure (bad selector or too many mappings)
    """
    ...

def set_visible_segm(s: segment_t, visible: bool) -> None:
    r"""See SFL_HIDDEN.
    
    """
    ...

def setup_selector(segbase: ida_idaapi.ea_t) -> sel_t:
    r"""Allocate a selector for a segment if necessary. You must call this function before calling add_segm_ex(). add_segm() calls this function itself, so you don't need to allocate a selector. This function will allocate a selector if 'segbase' requires more than 16 bits and the current processor is IBM PC. Otherwise it will return the segbase value. 
            
    :param segbase: a new segment base paragraph
    :returns: the allocated selector number
    """
    ...

def std_out_segm_footer(ctx: outctx_t, seg: segment_t) -> None:
    r"""Generate segment footer line as a comment line. This function may be used in IDP modules to generate segment footer if the target assembler doesn't have 'ends' directive. 
            
    """
    ...

def take_memory_snapshot(type: int) -> bool:
    r"""Take a memory snapshot of the running process. 
            
    :param type: specifies which snapshot we want (see SNAP_ Snapshot types)
    :returns: success
    """
    ...

def update_segm(s: segment_t) -> bool:
    ...

ADDSEG_FILLGAP: int  # 16
ADDSEG_IDBENC: int  # 128
ADDSEG_NOAA: int  # 64
ADDSEG_NOSREG: int  # 1
ADDSEG_NOTRUNC: int  # 4
ADDSEG_OR_DIE: int  # 2
ADDSEG_QUIET: int  # 8
ADDSEG_SPARSE: int  # 32
CSS_BREAK: int  # -4
CSS_NODBG: int  # -1
CSS_NOMEM: int  # -3
CSS_NORANGE: int  # -2
CSS_OK: int  # 0
MAX_GROUPS: int  # 8
MAX_SEGM_TRANSLATIONS: int  # 64
MOVE_SEGM_CHUNK: int  # -4
MOVE_SEGM_DEBUG: int  # -8
MOVE_SEGM_IDP: int  # -3
MOVE_SEGM_INVAL: int  # -11
MOVE_SEGM_LOADER: int  # -5
MOVE_SEGM_MAPPING: int  # -10
MOVE_SEGM_ODD: int  # -6
MOVE_SEGM_OK: int  # 0
MOVE_SEGM_ORPHAN: int  # -7
MOVE_SEGM_PARAM: int  # -1
MOVE_SEGM_ROOM: int  # -2
MOVE_SEGM_SOURCEFILES: int  # -9
MSF_FIXONCE: int  # 8
MSF_LDKEEP: int  # 4
MSF_NETNODES: int  # 128
MSF_NOFIX: int  # 2
MSF_PRIORITY: int  # 32
MSF_SILENT: int  # 1
SEGMOD_KEEP: int  # 2
SEGMOD_KEEP0: int  # 8
SEGMOD_KEEPSEL: int  # 16
SEGMOD_KILL: int  # 1
SEGMOD_NOMOVE: int  # 32
SEGMOD_SILENT: int  # 4
SEGMOD_SPARSE: int  # 64
SEGPERM_EXEC: int  # 1
SEGPERM_MAXVAL: int  # 7
SEGPERM_READ: int  # 4
SEGPERM_WRITE: int  # 2
SEG_ABSSYM: int  # 10
SEG_BSS: int  # 9
SEG_CODE: int  # 2
SEG_COMM: int  # 11
SEG_DATA: int  # 3
SEG_GRP: int  # 6
SEG_IMEM: int  # 12
SEG_IMP: int  # 4
SEG_MAX_BITNESS_CODE: int  # 2
SEG_MAX_SEGTYPE_CODE: int  # 12
SEG_NORM: int  # 0
SEG_NULL: int  # 7
SEG_UNDF: int  # 8
SEG_XTRN: int  # 1
SFL_COMORG: int  # 1
SFL_DEBUG: int  # 8
SFL_HEADER: int  # 64
SFL_HIDDEN: int  # 4
SFL_HIDETYPE: int  # 32
SFL_LOADER: int  # 16
SFL_OBOK: int  # 2
SNAP_ALL_SEG: int  # 0
SNAP_CUR_SEG: int  # 2
SNAP_LOAD_SEG: int  # 1
SREG_NUM: int  # 16
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
ida_range: module
saAbs: int  # 0
saGroup: int  # 7
saRel1024Bytes: int  # 13
saRel128Bytes: int  # 11
saRel2048Bytes: int  # 14
saRel32Bytes: int  # 8
saRel4K: int  # 6
saRel512Bytes: int  # 12
saRel64Bytes: int  # 9
saRelByte: int  # 1
saRelDble: int  # 5
saRelPage: int  # 4
saRelPara: int  # 3
saRelQword: int  # 10
saRelWord: int  # 2
saRel_MAX_ALIGN_CODE: int  # 14
scCommon: int  # 6
scGroup: int  # 1
scPriv: int  # 0
scPub: int  # 2
scPub2: int  # 4
scPub3: int  # 7
scStack: int  # 5
sc_MAX_COMB_CODE: int  # 7
weakref: module