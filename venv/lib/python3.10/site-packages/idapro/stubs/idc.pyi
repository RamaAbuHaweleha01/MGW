from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""
IDC compatibility module

This file contains IDA built-in function declarations and internal bit
definitions.  Each byte of the program has 32-bit flags (low 8 bits keep
the byte value). These 32 bits are used in get_full_flags/get_flags functions.

This file is subject to change without any notice.
Future versions of IDA may use other definitions.

"""

class DeprecatedIDCError(Exception):
    r"""
        Exception for deprecated function calls
        
    """
    args: getset_descriptor  # <attribute 'args' of 'BaseException' objects>
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
        r"""Initialize self.  See help(type(self)) for accurate signature."""
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
    def __reduce__(self, *args: Any, **kwargs: Any) -> Any:
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
    def __setstate__(self, *args: Any, **kwargs: Any) -> Any:
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
    def with_traceback(self, *args: Any, **kwargs: Any) -> Any:
        r"""Exception.with_traceback(tb) --
            set self.__traceback__ to tb and return self.
        """
        ...

def AddSeg(startea: Any, endea: Any, base: Any, use32: Any, align: Any, comb: Any) -> Any:
    ...

def AutoMark(ea: Any, qtype: Any) -> Any:
    r"""
        Plan to analyze an address
        
    """
    ...

def EVAL_FAILURE(code: Any) -> Any:
    r"""
        Check the result of eval_idc() for evaluation failures
    
        :param code: result of eval_idc()
    
        :returns: True if there was an evaluation error
        
    """
    ...

def GetCommentEx(ea: ida_idaapi.ea_t, rptble: bool) -> str:
    r"""Get an indented comment. 
            
    :param ea: linear address. may point to tail byte, the function will find start of the item
    :param rptble: get repeatable comment?
    :returns: size of comment or -1
    """
    ...

def GetDisasm(ea: Any) -> Any:
    r"""
        Get disassembly line
    
        :param ea: linear address of instruction
    
        :returns: "" - could not decode instruction at the specified location
    
        NOTE: this function may not return exactly the same mnemonics
               as you see on the screen.
        
    """
    ...

def GetDouble(ea: Any) -> Any:
    r"""
        Get value of a floating point number (8 bytes)
        This function assumes number stored using IEEE format
        and in the same endianness as integers.
    
        :param ea: linear address
    
        :returns: double
        
    """
    ...

def GetFloat(ea: Any) -> Any:
    r"""
        Get value of a floating point number (4 bytes)
        This function assumes number stored using IEEE format
        and in the same endianness as integers.
    
        :param ea: linear address
    
        :returns: float
        
    """
    ...

def GetLocalType(ordinal: Any, flags: Any) -> Any:
    r"""
        Retrieve a local type declaration
        :param flags: any of PRTYPE_* constants
        :returns: local type as a C declaration or ""
        
    """
    ...

def LoadFile(filepath: Any, pos: Any, ea: Any, size: Any) -> Any:
    r"""
        Load file into IDA database
    
        :param filepath: path to input file
        :param pos: position in the file
        :param ea: linear address to load
        :param size: number of bytes to load
    
        :returns: 0 - error, 1 - ok
        
    """
    ...

def MakeVar(ea: Any) -> Any:
    ...

def SaveFile(filepath: Any, pos: Any, ea: Any, size: Any) -> Any:
    r"""
        Save from IDA database to file
    
        :param filepath: path to output file
        :param pos: position in the file
        :param ea: linear address to save from
        :param size: number of bytes to save
    
        :returns: 0 - error, 1 - ok
        
    """
    ...

def SetPrcsr(processor: Any) -> Any:
    ...

def SetType(ea: Any, newtype: Any) -> Any:
    r"""
        Set type of function/variable
    
        :param ea: the address of the object
        :param newtype: the type string in C declaration form.
                    Must contain the closing ';'
                    if specified as an empty string, then the
                    item associated with 'ea' will be deleted.
    
        :returns: 1-ok, 0-failed.
        
    """
    ...

def SizeOf(typestr: Any) -> Any:
    r"""
        Returns the size of the type. It is equivalent to IDC's sizeof().
        :param typestr: can be specified as a typeinfo tuple (e.g. the result of get_tinfo()),
                serialized type byte string,
                or a string with C declaration (e.g. "int")
        :returns: -1 if typestring is not valid or has no size. otherwise size of the type
        
    """
    ...

def add_auto_stkpnt(func_ea: Any, ea: Any, delta: Any) -> Any:
    r"""
        Add automatic SP register change point
        :param func_ea: function start
        :param ea: linear address where SP changes
                   usually this is the end of the instruction which
                   modifies the stack pointer (insn.ea+insn.size)
        :param delta: difference between old and new values of SP
        :returns: 1-ok, 0-failed
        
    """
    ...

def add_bpt(args: Any) -> bool:
    r"""This function has the following signatures:
    
        0. add_bpt(ea: ida_idaapi.ea_t, size: asize_t=0, type: bpttype_t=BPT_DEFAULT) -> bool
        1. add_bpt(bpt: const bpt_t &) -> bool
    
    # 0: add_bpt(ea: ida_idaapi.ea_t, size: asize_t=0, type: bpttype_t=BPT_DEFAULT) -> bool
    
    Add a new breakpoint in the debugged process. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    
    # 1: add_bpt(bpt: const bpt_t &) -> bool
    
    Add a new breakpoint in the debugged process. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    
    """
    ...

def add_cref(frm: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: cref_t) -> bool:
    r"""Create a code cross-reference. 
            
    :param to: linear address of referenced instruction
    :param type: cross-reference type
    :returns: success
    """
    ...

def add_default_til(name: Any) -> Any:
    r"""
        Load a type library
    
        :param name: name of type library.
        :returns: 1-ok, 0-failed.
        
    """
    ...

def add_dref(frm: ida_idaapi.ea_t, to: ida_idaapi.ea_t, type: dref_t) -> bool:
    r"""Create a data cross-reference. 
            
    :param to: linear address of referenced data
    :param type: cross-reference type
    :returns: success (may fail if user-defined xref exists from->to)
    """
    ...

def add_entry(ord: int, ea: ida_idaapi.ea_t, name: str, makecode: bool, flags: int = 0) -> bool:
    r"""Add an entry point to the list of entry points. 
            
    :param ord: ordinal number if ordinal number is equal to 'ea' then ordinal is not used
    :param ea: linear address
    :param name: name of entry point. If the specified location already has a name, the old name will be appended to the regular comment.
    :param makecode: should the kernel convert bytes at the entry point to instruction(s)
    :param flags: See AEF_*
    :returns: success (currently always true)
    """
    ...

def add_enum(idx: Any, name: Any, flag: Any) -> Any:
    r"""
        Add a new enum type
    
        :param idx: is not used anymore
        :param name: name of the enum.
        :param flag: flags for representation of numeric constants
                     in the definition of enum.
    
        :returns: id of new enum or BADADDR
        
    """
    ...

def add_enum_member(enum_id: Any, name: Any, value: Any, bmask: Any = -1) -> Any:
    r"""
        Add a member of enum - a symbolic constant
    
        :param enum_id: id of enum
        :param name: name of symbolic constant. Must be unique in the program.
        :param value: value of symbolic constant.
        :param bmask: bitmask of the constant
            ordinary enums accept only -1 as a bitmask
            all bits set in value should be set in bmask too
    
        :returns: 0-ok, otherwise error code (one of ENUM_MEMBER_ERROR_*)
        
    """
    ...

def add_func(args: Any) -> bool:
    r"""Add a new function. If the function end address is BADADDR, then IDA will try to determine the function bounds by calling find_func_bounds(..., FIND_FUNC_DEFINE). 
            
    :param ea1: start address
    :param ea2: end address
    :returns: success
    """
    ...

def add_hidden_range(args: Any) -> bool:
    r"""Mark a range of addresses as hidden. The range will be created in the invisible state with the default color 
            
    :param ea1: linear address of start of the address range
    :param ea2: linear address of end of the address range
    :param description: range parameters
    :param header: range parameters
    :param footer: range parameters
    :param color: the range color
    :returns: success
    """
    ...

def add_idc_hotkey(hotkey: str, idcfunc: str) -> int:
    r"""Add hotkey for IDC function (ui_add_idckey). 
            
    :param hotkey: hotkey name
    :param idcfunc: IDC function name
    :returns: IDC hotkey error codes
    """
    ...

def add_segm_ex(startea: Any, endea: Any, base: Any, use32: Any, align: Any, comb: Any, flags: Any) -> Any:
    r"""
        Create a new segment
    
        :param startea: linear address of the start of the segment
        :param endea: linear address of the end of the segment
                   this address will not belong to the segment
                   'endea' should be higher than 'startea'
        :param base: base paragraph or selector of the segment.
                   a paragraph is 16byte memory chunk.
                   If a selector value is specified, the selector should be
                   already defined.
        :param use32: 0: 16bit segment, 1: 32bit segment, 2: 64bit segment
        :param align: segment alignment. see below for alignment values
        :param comb: segment combination. see below for combination values.
        :param flags: combination of ADDSEG_... bits
    
        :returns: 0-failed, 1-ok
        
    """
    ...

def add_sourcefile(ea1: ida_idaapi.ea_t, ea2: ida_idaapi.ea_t, filename: str) -> bool:
    ...

def add_struc(index: Any, name: Any, is_union: Any) -> Any:
    r"""
        Define a new structure type
    
        :param index: -1
        :param name: name of the new structure type.
        :param is_union: 0: structure
                         1: union
    
        :returns: -1 if can't define structure type because of
                 bad structure name: the name is ill-formed or is
                 already used in the program.
                 otherwise returns ID of the new structure type
        
    """
    ...

def add_struc_member(sid: Any, name: Any, offset: Any, flag: Any, typeid: Any, nbytes: Any, target: Any = -1, tdelta: Any = 0, reftype: Any = 2) -> Any:
    r"""
        Add structure member
    
        :param sid: structure type ID
        :param name: name of the new member
        :param offset: offset of the new member
                       -1 means to add at the end of the structure
        :param flag: type of the new member. Should be one of
                     FF_BYTE..FF_PACKREAL (see above) combined with FF_DATA
        :param typeid: if is_struct(flag) then typeid specifies the structure id for the member
                       if is_off0(flag) then typeid specifies the offset base.
                       if is_strlit(flag) then typeid specifies the string type (STRTYPE_...).
                       if is_stroff(flag) then typeid specifies the structure id
                       if is_enum(flag) then typeid specifies the enum id
                       if is_custom(flags) then typeid specifies the dtid and fid: dtid|(fid<<16)
                       Otherwise typeid should be -1.
        :param nbytes: number of bytes in the new member
    
        :param target: target address of the offset expr. You may specify it as
                       -1, ida will calculate it itself
        :param tdelta: offset target delta. usually 0
        :param reftype: see REF_... definitions
    
        NOTE: The remaining arguments are allowed only if is_off0(flag) and you want
               to specify a complex offset expression
    
        :returns: 0 - ok, otherwise error code (one of typeinf.TERR_*)
    
        
    """
    ...

def add_user_stkpnt(ea: ida_idaapi.ea_t, delta: int) -> bool:
    r"""Add user-defined SP register change point. 
            
    :param ea: linear address where SP changes
    :param delta: difference between old and new values of SP
    :returns: success
    """
    ...

def append_func_tail(funcea: Any, ea1: Any, ea2: Any) -> Any:
    r"""
        Append a function chunk to the function
    
        :param funcea: any address in the function
        :param ea1: start of function tail
        :param ea2: end of function tail
        :returns: 0 if failed, 1 if success
    
        NOTE: If a chunk exists at the specified addresses, it must have exactly
               the specified boundaries
        
    """
    ...

def apply_type(ea: Any, py_type: Any, flags: Any = 1) -> Any:
    r"""
        Apply the specified type to the address
    
        :param ea: the address of the object
        :param py_type: typeinfo tuple (type, fields) as get_tinfo() returns
                     or tuple (name, type, fields) as parse_decl() returns
                     or None
                    if specified as None, then the
                    item associated with 'ea' will be deleted.
        :param flags: combination of TINFO_... constants or 0
        :returns: Boolean
        
    """
    ...

def ask_seg(defval: int, prompt: str) -> Any:
    r"""Display a dialog box and wait for the user to input an segment name.
    This function allows to enter segment register names, segment base
    paragraphs, segment names to denote a segment.
    
    :param defval: The placeholder value
    :param prompt: The prompt to show
    :returns: the selector of the segment entered by the user, or None if the dialog was canceled
    """
    ...

def ask_yn(args: Any) -> int:
    r"""Display a dialog box and get choice from "Yes", "No", "Cancel". 
            
    :param deflt: default choice: one of Button IDs
    :param format: The question in printf() style format
    :returns: the selected button (one of Button IDs). Esc key returns ASKBTN_CANCEL.
    """
    ...

def atoa(ea: Any) -> Any:
    r"""
        Convert address value to a string
        Return address in the form 'seg000:1234'
        (the same as in line prefixes)
    
        :param ea: address to format
        
    """
    ...

def atol(s: Any) -> Any:
    ...

def attach_process(args: Any) -> int:
    r"""Attach the debugger to a running process. \sq{Type, Asynchronous function - available as Request, Notification, dbg_process_attach} 
            
    :param pid: PID of the process to attach to. If NO_PROCESS, a dialog box will interactively ask the user for the process to attach to.
    :param event_id: event to trigger upon attaching
    :returns: -4: debugger was not inited
    :returns: -3: the attaching is not supported
    :returns: -2: impossible to find a compatible process
    :returns: -1: impossible to attach to the given process (process died, privilege needed, not supported by the debugger plugin, ...)
    :returns: 0: the user cancelled the attaching to the process
    :returns: 1: the debugger properly attached to the process
    """
    ...

def auto_mark_range(start: ida_idaapi.ea_t, end: ida_idaapi.ea_t, type: atype_t) -> None:
    r"""Put range of addresses into a queue. 'start' may be higher than 'end', the kernel will swap them in this case. 'end' doesn't belong to the range. 
            
    """
    ...

def auto_unmark(start: ida_idaapi.ea_t, end: ida_idaapi.ea_t, type: atype_t) -> None:
    r"""Remove range of addresses from a queue. 'start' may be higher than 'end', the kernel will swap them in this case. 'end' doesn't belong to the range. 
            
    """
    ...

def auto_wait() -> bool:
    r"""Process everything in the queues and return true. 
            
    :returns: false if the user clicked cancel. (the wait box must be displayed by the caller if desired)
    """
    ...

def batch(batch: Any) -> Any:
    r"""
        Enable/disable batch mode of operation
    
        :param batch: batch mode
                0 - ida will display dialog boxes and wait for the user input
                1 - ida will not display dialog boxes, warnings, etc.
    
        :returns: old balue of batch flag
        
    """
    ...

def byte_value(F: Any) -> Any:
    r"""
        Get byte value from flags
        Get value of byte provided that the byte is initialized.
        This macro works ok only for 8-bit byte machines.
        
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

def call_system(command: Any) -> Any:
    r"""
        Execute an OS command.
    
        :param command: command line to execute
    
        :returns: error code from OS
    
        NOTE: IDA will wait for the started program to finish.
        In order to start the command in parallel, use OS methods.
        For example, you may start another program in parallel using
        "start" command.
        
    """
    ...

def can_exc_continue() -> Any:
    r"""
        Can it continue after EXCEPTION event?
    
        :returns: boolean
        
    """
    ...

def check_bpt(ea: ida_idaapi.ea_t) -> int:
    r"""Check the breakpoint at the specified address. 
            
    :returns: one of Breakpoint status codes
    """
    ...

def choose_func(title: Any) -> Any:
    r"""
        Ask the user to select a function
    
        Arguments:
    
        :param title: title of the dialog box
    
        :returns: -1 - user refused to select a function
                 otherwise returns the selected function start address
        
    """
    ...

def clear_trace(filename: Any) -> Any:
    r"""
        Clear the current trace buffer
        
    """
    ...

def create_align(ea: ida_idaapi.ea_t, length: asize_t, alignment: int) -> bool:
    r"""Create an alignment item. 
            
    :param ea: linear address
    :param length: size of the item in bytes. 0 means to infer from ALIGNMENT
    :param alignment: alignment exponent. Example: 3 means align to 8 bytes. 0 means to infer from LENGTH It is forbidden to specify both LENGTH and ALIGNMENT as 0.
    :returns: success
    """
    ...

def create_array(name: Any) -> Any:
    r"""
        Create array.
    
        :param name: The array name.
    
        :returns: -1 in case of failure, a valid array_id otherwise.
        
    """
    ...

def create_byte(ea: Any) -> Any:
    r"""
        Convert the current item to a byte
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_custom_data(ea: ida_idaapi.ea_t, length: asize_t, dtid: int, fid: int, force: bool = False) -> bool:
    r"""Convert to custom data type.
    
    """
    ...

def create_data(ea: ida_idaapi.ea_t, dataflag: flags64_t, size: asize_t, tid: tid_t) -> bool:
    r"""Convert to data (byte, word, dword, etc). This function may be used to create arrays. 
            
    :param ea: linear address
    :param dataflag: type of data. Value of function byte_flag(), word_flag(), etc.
    :param size: size of array in bytes. should be divisible by the size of one item of the specified type. for variable sized items it can be specified as 0, and the kernel will try to calculate the size.
    :param tid: type id. If the specified type is a structure, then tid is structure id. Otherwise should be BADNODE.
    :returns: success
    """
    ...

def create_double(ea: Any) -> Any:
    r"""
        Convert the current item to a double floating point (8 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_dword(ea: Any) -> Any:
    r"""
        Convert the current item to a double word (4 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_float(ea: Any) -> Any:
    r"""
        Convert the current item to a floating point (4 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_insn(ea: ida_idaapi.ea_t, out: insn_t = None) -> int:
    r"""Create an instruction at the specified address. This function checks if an instruction is present at the specified address and will try to create one if there is none. It will fail if there is a data item or other items hindering the creation of the new instruction. This function will also fill the 'out' structure. 
            
    :param ea: linear address
    :param out: the resulting instruction
    :returns: the length of the instruction or 0
    """
    ...

def create_oword(ea: Any) -> Any:
    r"""
        Convert the current item to an octa word (16 bytes/128 bits)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_pack_real(ea: Any) -> Any:
    r"""
        Convert the current item to a packed real (10 or 12 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_qword(ea: Any) -> Any:
    r"""
        Convert the current item to a quadro word (8 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_strlit(ea: Any, endea: Any) -> Any:
    r"""
        Create a string.
    
        This function creates a string (the string type is determined by the
        value of get_inf_attr(INF_STRTYPE))
    
        :param ea: linear address
        :param endea: ending address of the string (excluded)
            if endea == BADADDR, then length of string will be calculated
            by the kernel
    
        :returns: 1-ok, 0-failure
    
        NOTE: The type of an existing string is returned by get_str_type()
        
    """
    ...

def create_struct(ea: Any, size: Any, strname: Any) -> Any:
    r"""
        Convert the current item to a structure instance
    
        :param ea: linear address
        :param size: structure size in bytes. -1 means that the size
            will be calculated automatically
        :param strname: name of a structure type
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_tbyte(ea: Any) -> Any:
    r"""
        Convert the current item to a tbyte (10 or 12 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_word(ea: Any) -> Any:
    r"""
        Convert the current item to a word (2 bytes)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def create_yword(ea: Any) -> Any:
    r"""
        Convert the current item to a ymm word (32 bytes/256 bits)
    
        :param ea: linear address
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def define_exception(code: uint, name: str, desc: str, flags: int) -> str:
    r"""Convenience function: define new exception code. 
            
    :param code: exception code (cannot be 0)
    :param name: exception name (cannot be empty or nullptr)
    :param desc: exception description (maybe nullptr)
    :param flags: combination of Exception info flags
    :returns: failure message or nullptr. You must call store_exceptions() if this function succeeds
    """
    ...

def define_local_var(start: Any, end: Any, location: Any, name: Any) -> Any:
    r"""
        Create a local variable
    
        :param start: start of address range for the local variable
        :param end: end of address range for the local variable
        :param location: the variable location in the "[bp+xx]" form where xx is
                         a number. The location can also be specified as a
                         register name.
        :param name: name of the local variable
    
        :returns: 1-ok, 0-failure
    
        NOTE: For the stack variables the end address is ignored.
              If there is no function at 'start' then this function will fail.
        
    """
    ...

def del_array_element(tag: Any, array_id: Any, idx: Any) -> Any:
    r"""
        Delete an array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
        :param idx: Index of an element.
    
        :returns: 1 in case of success, 0 otherwise.
        
    """
    ...

def del_bpt(args: Any) -> bool:
    r"""This function has the following signatures:
    
        0. del_bpt(ea: ida_idaapi.ea_t) -> bool
        1. del_bpt(bptloc: const bpt_location_t &) -> bool
    
    # 0: del_bpt(ea: ida_idaapi.ea_t) -> bool
    
    Delete an existing breakpoint in the debugged process. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    
    # 1: del_bpt(bptloc: const bpt_location_t &) -> bool
    
    Delete an existing breakpoint in the debugged process. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    
    """
    ...

def del_cref(frm: ida_idaapi.ea_t, to: ida_idaapi.ea_t, expand: bool) -> bool:
    r"""Delete a code cross-reference. 
            
    :param to: linear address of referenced instruction
    :param expand: policy to delete the referenced instruction
    * 1: plan to delete the referenced instruction if it has no more references.
    * 0: don't delete the referenced instruction even if no more cross-references point to it
    :returns: true: if the referenced instruction will be deleted
    """
    ...

def del_dref(frm: ida_idaapi.ea_t, to: ida_idaapi.ea_t) -> None:
    r"""Delete a data cross-reference. 
            
    :param to: linear address of referenced data
    """
    ...

def del_enum(enum_id: Any) -> Any:
    r"""
        Delete an enum type
    
        :param enum_id: id of enum
    
        :returns: success
        
    """
    ...

def del_enum_member(enum_id: Any, value: Any, serial: Any, bmask: Any = -1) -> Any:
    r"""
        Delete a member of enum - a symbolic constant
    
        :param enum_id: id of enum
        :param value: value of symbolic constant.
        :param serial: serial number of the constant in the
            enumeration. See op_enum() for for details.
        :param bmask: bitmask of the constant ordinary enums accept
            only -1 as a bitmask
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def del_extra_cmt(ea: ida_idaapi.ea_t, what: int) -> bool:
    ...

def del_fixup(source: ida_idaapi.ea_t) -> None:
    r"""Delete fixup information.
    
    """
    ...

def del_func(ea: ida_idaapi.ea_t) -> bool:
    r"""Delete a function. 
            
    :param ea: any address in the function entry chunk
    :returns: success
    """
    ...

def del_hash_string(hash_id: Any, key: Any) -> Any:
    r"""
        Delete a hash element.
    
        :param hash_id: The hash ID.
        :param key: Key of an element
    
        :returns: 1 upon success, 0 otherwise.
        
    """
    ...

def del_hidden_range(ea: ida_idaapi.ea_t) -> bool:
    r"""Delete hidden range. 
            
    :param ea: any address in the hidden range
    :returns: success
    """
    ...

def del_idc_hotkey(hotkey: str) -> bool:
    ...

def del_items(ea: ida_idaapi.ea_t, flags: int = 0, nbytes: asize_t = 1, may_destroy: may_destroy_cb_t = None) -> bool:
    r"""Convert item (instruction/data) to unexplored bytes. The whole item (including the head and tail bytes) will be destroyed. It is allowed to pass any address in the item to this function 
            
    :param ea: any address within the first item to delete
    :param flags: combination of Unexplored byte conversion flags
    :param nbytes: number of bytes in the range to be undefined
    :param may_destroy: optional routine invoked before deleting a head item. If callback returns false then item is not to be deleted and operation fails
    :returns: true on sucessful operation, otherwise false
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

def del_selector(selector: sel_t) -> None:
    r"""Delete mapping of a selector. Be wary of deleting selectors that are being used in the program, this can make a mess in the segments. 
            
    :param selector: number of selector to remove from the translation table
    """
    ...

def del_source_linnum(ea: ida_idaapi.ea_t) -> None:
    ...

def del_sourcefile(ea: ida_idaapi.ea_t) -> bool:
    ...

def del_stkpnt(func_ea: Any, ea: Any) -> Any:
    r"""
        Delete SP register change point
    
        :param func_ea: function start
        :param ea: linear address
        :returns: 1-ok, 0-failed
        
    """
    ...

def del_struc(sid: Any) -> Any:
    r"""
        Delete a structure type
    
        :param sid: structure type ID
    
        :returns: 0 if bad structure type ID is passed
                 1 otherwise the structure type is deleted. All data
                 and other structure types referencing to the
                 deleted structure type will be displayed as array
                 of bytes.
        
    """
    ...

def del_struc_member(sid: Any, member_offset: Any) -> Any:
    r"""
        Delete structure member
    
        :param sid: structure type ID
        :param member_offset: offset of the member
    
        :returns: != 0 - ok.
    
        NOTE: IDA allows 'holes' between members of a
               structure. It treats these 'holes'
               as unnamed arrays of bytes.
        
    """
    ...

def delete_all_segments() -> Any:
    r"""
        Delete all segments, instructions, comments, i.e. everything
        except values of bytes.
        
    """
    ...

def delete_array(array_id: Any) -> Any:
    r"""
        Delete array, by its ID.
    
        :param array_id: The ID of the array to delete.
        
    """
    ...

def demangle_name(name: Any, disable_mask: Any) -> Any:
    r"""
        demangle_name a name
    
        :param name: name to demangle
        :param disable_mask: a mask that tells how to demangle the name
                it is a good idea to get this mask using
                get_inf_attr(INF_SHORT_DN) or get_inf_attr(INF_LONG_DN)
    
        :returns: a demangled name
            If the input name cannot be demangled, returns None
        
    """
    ...

def detach_process() -> bool:
    r"""Detach the debugger from the debugged process. \sq{Type, Asynchronous function - available as Request, Notification, dbg_process_detach} 
            
    """
    ...

def diff_trace_file(NONNULL_filename: str) -> bool:
    r"""Show difference between the current trace and the one from 'filename'.
    
    """
    ...

def enable_bpt(args: Any) -> bool:
    ...

def enable_tracing(trace_level: Any, enable: Any) -> Any:
    r"""
        Enable step tracing
    
        :param trace_level:  what kind of trace to modify
        :param enable: 0: turn off, 1: turn on
    
        :returns: success
        
    """
    ...

def error(message: Any) -> Any:
    r"""Display a fatal message in a message box and quit IDA
    
    :param format: message to print
    """
    ...

def eval_idc(expr: Any) -> Any:
    r"""
        Evaluate an IDC expression
    
        :param expr: an expression
    
        :returns: the expression value. If there are problems, the returned value will be "IDC_FAILURE: xxx"
                 where xxx is the error description
    
        NOTE: Python implementation evaluates IDC only, while IDC can call other registered languages
        
    """
    ...

def exit_process() -> bool:
    r"""Terminate the debugging of the current process. \sq{Type, Asynchronous function - available as Request, Notification, dbg_process_exit} 
            
    """
    ...

def expand_struc(sid: Any, offset: Any, delta: Any, recalc: Any = True) -> Any:
    r"""
        Expand or shrink a structure type
        :param id: structure type ID
        :param offset: offset in the structure
        :param delta: how many bytes to add or remove
        :param recalc: is not used anymore
        :returns: True if ok, False on error
        
    """
    ...

def fclose(handle: Any) -> Any:
    ...

def fgetc(handle: Any) -> Any:
    ...

def filelength(handle: Any) -> Any:
    ...

def find_bytes(bs: Any, range_start: int, range_size: typing.Optional[int] = None, range_end: typing.Optional[int] = 18446744073709551615, mask: Any = None, flags: typing.Optional[int] = 8, radix: typing.Optional[int] = 16, strlit_encoding: Any = 0) -> int:
    ...

def find_code(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_data(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_defined(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_func_end(ea: Any) -> Any:
    r"""
        Determine a new function boundaries
    
        :param ea: starting address of a new function
    
        :returns: if a function already exists, then return its end address.
                If a function end cannot be determined, the return BADADDR
                otherwise return the end address of the new function
        
    """
    ...

def find_imm(ea: ida_idaapi.ea_t, sflag: int, search_value: int) -> int:
    ...

def find_selector(val: Any) -> Any:
    r"""
        Find a selector which has the specified value
    
        :param val: value to search for
    
        :returns: the selector number if found,
                 otherwise the input value (val & 0xFFFF)
    
        NOTE: selector values are always in paragraphs
        
    """
    ...

def find_suspop(ea: ida_idaapi.ea_t, sflag: int) -> int:
    ...

def find_text(start_ea: ida_idaapi.ea_t, y: int, x: int, ustr: str, sflag: int) -> ida_idaapi.ea_t:
    ...

def find_unknown(ea: ida_idaapi.ea_t, sflag: int) -> ida_idaapi.ea_t:
    ...

def first_func_chunk(funcea: Any) -> Any:
    r"""
        Get the first function chunk of the specified function
    
        :param funcea: any address in the function
    
        :returns: the function entry point or BADADDR
    
        NOTE: This function returns the first (main) chunk of the specified function
        
    """
    ...

def fopen(f: Any, mode: Any) -> Any:
    ...

def force_bl_call(ea: Any) -> Any:
    r"""
        Force BL instruction to be a call
    
        :param ea: address of the BL instruction
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def force_bl_jump(ea: Any) -> Any:
    r"""
        Some ARM compilers in Thumb mode use BL (branch-and-link)
        instead of B (branch) for long jumps, since BL has more range.
        By default, IDA tries to determine if BL is a jump or a call.
        You can override IDA's decision using commands in Edit/Other menu
        (Force BL call/Force BL jump) or the following two functions.
    
        Force BL instruction to be a jump
    
        :param ea: address of the BL instruction
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def form(format: Any, args: Any) -> Any:
    ...

def fprintf(handle: Any, format: Any, args: Any) -> Any:
    ...

def fputc(byte: Any, handle: Any) -> Any:
    ...

def fseek(handle: Any, offset: Any, origin: Any) -> Any:
    ...

def ftell(handle: Any) -> Any:
    ...

def func_contains(func_ea: Any, ea: Any) -> Any:
    r"""
        Does the given function contain the given address?
    
        :param func_ea: any address belonging to the function
        :param ea: linear address
    
        :returns:  success
        
    """
    ...

def gen_file(filetype: Any, path: Any, ea1: Any, ea2: Any, flags: Any) -> Any:
    r"""
        Generate an output file
    
        :param filetype:  type of output file. One of OFILE_... symbols. See below.
        :param path:  the output file path (will be overwritten!)
        :param ea1:   start address. For some file types this argument is ignored
        :param ea2:   end address. For some file types this argument is ignored
        :param flags: bit combination of GENFLG_...
    
        :returns: number of the generated lines.
                    -1 if an error occurred
                    OFILE_EXE: 0-can't generate exe file, 1-ok
        
    """
    ...

def gen_flow_graph(outfile: Any, title: Any, ea1: Any, ea2: Any, flags: Any) -> Any:
    r"""
        Generate a flow chart GDL file
    
        :param outfile: output file name. GDL extension will be used
        :param title: graph title
        :param ea1: beginning of the range to flow chart
        :param ea2: end of the range to flow chart.
        :param flags: combination of CHART_... constants
    
        NOTE: If ea2 == BADADDR then ea1 is treated as an address within a function.
               That function will be flow charted.
        
    """
    ...

def gen_simple_call_chart(outfile: Any, title: Any, flags: Any) -> Any:
    r"""
        Generate a function call graph GDL file
    
        :param outfile: output file name. GDL extension will be used
        :param title:   graph title
        :param flags:   combination of CHART_GEN_GDL, CHART_WINGRAPH, CHART_NOLIBFUNCS
        
    """
    ...

def generate_disasm_line(ea: Any, flags: Any) -> Any:
    r"""
        Get disassembly line
    
        :param ea: linear address of instruction
    
        :param flags: combination of the GENDSM_ flags, or 0
    
        :returns: "" - could not decode instruction at the specified location
    
        NOTE: this function may not return exactly the same mnemonics
               as you see on the screen.
        
    """
    ...

def get_array_element(tag: Any, array_id: Any, idx: Any) -> Any:
    r"""
        Get value of array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
        :param idx: Index of an element.
    
        :returns: Value of the specified array element. Note that
                 this function may return char or long result. Unexistent
                 array elements give zero as a result.
        
    """
    ...

def get_array_id(name: Any) -> Any:
    r"""
        Get array array_id, by name.
    
        :param name: The array name.
    
        :returns: -1 in case of failure (i.e., no array with that
                 name exists), a valid array_id otherwise.
        
    """
    ...

def get_bmask_cmt(enum_id: Any, bmask: Any, repeatable: Any) -> Any:
    r"""
        Get bitmask comment (only for bitfields)
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant
        :param repeatable: type of comment, 0-regular, 1-repeatable
    
        :returns: comment attached to bitmask or None
        
    """
    ...

def get_bmask_name(enum_id: Any, bmask: Any) -> Any:
    r"""
        Get bitmask name (only for bitfields)
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant
    
        :returns: name of bitmask or None
        
    """
    ...

def get_bookmark(slot: int) -> ida_idaapi.ea_t:
    ...

def get_bookmark_desc(slot: int) -> Any:
    ...

def get_bpt_attr(ea: Any, bptattr: Any) -> Any:
    r"""
        Get the characteristics of a breakpoint
    
        :param ea: any address in the breakpoint range
        :param bptattr: the desired attribute code, one of BPTATTR_... constants
    
        :returns: the desired attribute value or -1
        
    """
    ...

def get_bpt_ea(n: Any) -> Any:
    r"""
        Get breakpoint address
    
        :param n: number of breakpoint, is in range 0..get_bpt_qty()-1
    
        :returns: address of the breakpoint or BADADDR
        
    """
    ...

def get_bpt_qty() -> int:
    r"""Get number of breakpoints. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    """
    ...

def get_bpt_tev_ea(n: int) -> ida_idaapi.ea_t:
    r"""Get the address associated to a read, read/write or execution trace event. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param n: number of trace event, is in range 0..get_tev_qty()-1. 0 represents the latest added trace event.
    :returns: BADADDR if not a read, read/write or execution trace event.
    """
    ...

def get_bytes(ea: Any, size: Any, use_dbg: Any = False) -> Any:
    r"""
        Return the specified number of bytes of the program
    
        :param ea: linear address
    
        :param size: size of buffer in normal 8-bit bytes
    
        :param use_dbg: if True, use debugger memory, otherwise just the database
    
        :returns: None on failure
                 otherwise a string containing the read bytes
        
    """
    ...

def get_call_tev_callee(n: int) -> ida_idaapi.ea_t:
    r"""Get the called function from a function call trace event. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param n: number of trace event, is in range 0..get_tev_qty()-1. 0 represents the latest added trace event.
    :returns: BADADDR if not a function call event.
    """
    ...

def get_cmt(ea: ida_idaapi.ea_t, rptble: bool) -> str:
    r"""Get an indented comment. 
            
    :param ea: linear address. may point to tail byte, the function will find start of the item
    :param rptble: get repeatable comment?
    :returns: size of comment or -1
    """
    ...

def get_color(ea: Any, what: Any) -> Any:
    r"""
        Get item color
    
        :param ea: address of the item
        :param what: type of the item (one of  CIC_* constants)
    
        :returns: color code in RGB (hex 0xBBGGRR)
        
    """
    ...

def get_curline() -> Any:
    r"""
        Get the disassembly line at the cursor
    
        :returns: string
        
    """
    ...

def get_current_thread() -> thid_t:
    r"""Get current thread ID. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    """
    ...

def get_db_byte(ea: ida_idaapi.ea_t) -> uchar:
    r"""Get one byte (8-bit) of the program at 'ea' from the database. Works even if the debugger is active. See also get_dbg_byte() to read the process memory directly. This function works only for 8bit byte processors. 
            
    """
    ...

def get_debugger_event_cond() -> str:
    ...

def get_entry(ord: int) -> ida_idaapi.ea_t:
    r"""Get entry point address by its ordinal 
            
    :param ord: ordinal number of entry point
    :returns: address or BADADDR
    """
    ...

def get_entry_name(ord: int) -> str:
    r"""Get name of the entry point by its ordinal. 
            
    :param ord: ordinal number of entry point
    :returns: size of entry name or -1
    """
    ...

def get_entry_ordinal(idx: size_t) -> int:
    r"""Get ordinal number of an entry point. 
            
    :param idx: internal number of entry point. Should be in the range 0..get_entry_qty()-1
    :returns: ordinal number or 0.
    """
    ...

def get_entry_qty() -> int:
    r"""Get number of entry points.
    
    """
    ...

def get_enum(name: Any) -> Any:
    r"""
        Get enum by name
    
        :param name: enum type name
    
        :returns: enum type TID or BADADDR
        
    """
    ...

def get_enum_cmt(enum_id: Any) -> Any:
    r"""
        Get enum comment
    
        :param enum_id: enum TID
    
        :returns: enum comment
        
    """
    ...

def get_enum_flag(enum_id: Any) -> Any:
    r"""
        Get flags determining the representation of the enum.
        (currently they define the numeric base: octal, decimal, hex, bin) and signness.
    
        :param enum_id: enum TID
    
        :returns: flag of 0
        
    """
    ...

def get_enum_member(enum_id: Any, value: Any, serial: Any, bmask: Any) -> Any:
    r"""
        Get id of constant
    
        :param enum_id: id of enum
        :param value: value of constant
        :param serial: serial number of the constant in the
                  enumeration. See op_enum() for details.
        :param bmask: bitmask of the constant
                  ordinary enums accept only -1 as a bitmask
    
        :returns: id of constant or -1 if error
        
    """
    ...

def get_enum_member_bmask(const_id: Any) -> Any:
    r"""
        Get bitmask of an enum member
    
        :param const_id: id of const
    
        :returns: member value or None
        
    """
    ...

def get_enum_member_by_name(name: Any) -> Any:
    r"""
        Get a reference to an enum member by its name
    
        :param name: enum member name
    
        :returns: enum member TID or BADADDR
        
    """
    ...

def get_enum_member_cmt(const_id: Any, repeatable: Any = True) -> Any:
    r"""
        Get comment of a constant
    
        :param const_id: id of const
        :param repeatable: not used anymore
    
        :returns: comment string
        
    """
    ...

def get_enum_member_enum(const_id: Any) -> Any:
    r"""
        Get the parent enum of an enum member
    
        :param const_id: id of const
    
        :returns: enum TID or BADADDR
        
    """
    ...

def get_enum_member_name(const_id: Any) -> Any:
    r"""
        Get name of a constant
    
        :param const_id: id of const
    
        Returns: name of constant
        
    """
    ...

def get_enum_member_value(const_id: Any) -> Any:
    r"""
        Get value of an enum member
    
        :param const_id: id of const
    
        :returns: member value or None
        
    """
    ...

def get_enum_name(enum_id: Any, flags: Any = 0) -> Any:
    r"""
        Get name of enum
    
        :param enum_id: enum TID
        :param flags: use ENFL_REGEX to beautify the name
    
        :returns: enum name or None
        
    """
    ...

def get_enum_size(enum_id: Any) -> Any:
    r"""
        Get the number of the members of the enum
    
        :param enum_id: enum TID
    
        :returns: number of members
        
    """
    ...

def get_enum_width(enum_id: Any) -> Any:
    r"""
        Get the width of a enum element
        allowed values: 0 (unspecified),1,2,4,8,16,32,64
    
        :param enum_id: enum TID
    
        :returns: enum width or -1 in case of error
        
    """
    ...

def get_event_bpt_hea() -> Any:
    r"""
        Get hardware address for BREAKPOINT event
    
        :returns: hardware address
        
    """
    ...

def get_event_ea() -> Any:
    r"""
        Get ea for debug event
    
        :returns: ea
        
    """
    ...

def get_event_exc_code() -> Any:
    r"""
        Get exception code for EXCEPTION event
    
        :returns: exception code
        
    """
    ...

def get_event_exc_ea() -> Any:
    r"""
        Get address for EXCEPTION event
    
        :returns: adress of exception
        
    """
    ...

def get_event_exc_info() -> Any:
    r"""
        Get info for EXCEPTION event
    
        :returns: info string
        
    """
    ...

def get_event_exit_code() -> Any:
    r"""
        Get exit code for debug event
    
        :returns: exit code for PROCESS_EXITED, THREAD_EXITED events
        
    """
    ...

def get_event_id() -> Any:
    r"""
        Get ID of debug event
    
        :returns: event ID
        
    """
    ...

def get_event_info() -> Any:
    r"""
        Get debug event info
    
        :returns: event info: for THREAD_STARTED (thread name)
                             for LIB_UNLOADED (unloaded library name)
                             for INFORMATION (message to display)
        
    """
    ...

def get_event_module_base() -> Any:
    r"""
        Get module base for debug event
    
        :returns: module base
        
    """
    ...

def get_event_module_name() -> Any:
    r"""
        Get module name for debug event
    
        :returns: module name
        
    """
    ...

def get_event_module_size() -> Any:
    r"""
        Get module size for debug event
    
        :returns: module size
        
    """
    ...

def get_event_pid() -> Any:
    r"""
        Get process ID for debug event
    
        :returns: process ID
        
    """
    ...

def get_event_tid() -> Any:
    r"""
        Get type ID for debug event
    
        :returns: type ID
        
    """
    ...

def get_extra_cmt(ea: ida_idaapi.ea_t, what: int) -> int:
    ...

def get_fchunk_attr(ea: Any, attr: Any) -> Any:
    r"""
        Get a function chunk attribute
    
        :param ea: any address in the chunk
        :param attr: one of: FUNCATTR_START, FUNCATTR_END, FUNCATTR_OWNER, FUNCATTR_REFQTY
    
        :returns: desired attribute or -1
        
    """
    ...

def get_fchunk_referer(ea: int, idx: Any) -> Any:
    ...

def get_first_bmask(enum_id: Any) -> Any:
    r"""
        Get first bitmask in the enum
    
        :param enum_id: id of enum
    
        :returns: id of constant or -1 if error
        
    """
    ...

def get_first_cref_from(frm: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get first instruction referenced from the specified instruction. If the specified instruction passes execution to the next instruction then the next instruction is returned. Otherwise the lowest referenced address is returned (remember that xrefs are kept sorted!). 
            
    :returns: first referenced address. If the specified instruction doesn't reference to other instructions then returns BADADDR.
    """
    ...

def get_first_cref_to(to: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get first instruction referencing to the specified instruction. If the specified instruction may be executed immediately after its previous instruction then the previous instruction is returned. Otherwise the lowest referencing address is returned. (remember that xrefs are kept sorted!). 
            
    :param to: linear address of referenced instruction
    :returns: linear address of the first referencing instruction or BADADDR.
    """
    ...

def get_first_dref_from(frm: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get first data referenced from the specified address. 
            
    :returns: linear address of first (lowest) data referenced from the specified address. Return BADADDR if the specified instruction/data doesn't reference to anything.
    """
    ...

def get_first_dref_to(to: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of instruction/data referencing to the specified data. 
            
    :param to: linear address of referencing instruction or data
    :returns: BADADDR if nobody refers to the specified data.
    """
    ...

def get_first_enum_member(enum_id: Any, bmask: Any = -1) -> Any:
    r"""
        Get first constant in the enum
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant (ordinary enums accept only -1 as a bitmask)
    
        :returns: value of constant or -1 if no constants are defined
                 All constants are sorted by their values as unsigned longs.
        
    """
    ...

def get_first_fcref_from(frm: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_first_fcref_to(to: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_first_hash_key(hash_id: Any) -> Any:
    r"""
        Get the first key in the hash.
    
        :param hash_id: The hash ID.
    
        :returns: the key, 0 otherwise.
        
    """
    ...

def get_first_index(tag: Any, array_id: Any) -> Any:
    r"""
        Get index of the first existing array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
    
        :returns: -1 if the array is empty, otherwise index of first array
                 element of given type.
        
    """
    ...

def get_first_module() -> Any:
    r"""
        Enumerate process modules
    
        :returns: first module's base address or None on failure
        
    """
    ...

def get_first_seg() -> Any:
    r"""
        Get first segment
    
        :returns: address of the start of the first segment
            BADADDR - no segments are defined
        
    """
    ...

def get_fixup_target_dis(ea: Any) -> Any:
    r"""
        Get fixup target displacement
    
        :param ea: address to get information about
    
        :returns: 0 - no fixup at the specified address
                     otherwise returns fixup target displacement
        
    """
    ...

def get_fixup_target_flags(ea: Any) -> Any:
    r"""
        Get fixup target flags
    
        :param ea: address to get information about
    
        :returns: 0 - no fixup at the specified address
                     otherwise returns fixup target flags
        
    """
    ...

def get_fixup_target_off(ea: Any) -> Any:
    r"""
        Get fixup target offset
    
        :param ea: address to get information about
    
        :returns: BADADDR - no fixup at the specified address
                           otherwise returns fixup target offset
        
    """
    ...

def get_fixup_target_sel(ea: Any) -> Any:
    r"""
        Get fixup target selector
    
        :param ea: address to get information about
    
        :returns: BADSEL - no fixup at the specified address
                          otherwise returns fixup target selector
        
    """
    ...

def get_fixup_target_type(ea: Any) -> Any:
    r"""
        Get fixup target type
    
        :param ea: address to get information about
    
        :returns: 0 - no fixup at the specified address
                     otherwise returns fixup type
        
    """
    ...

def get_forced_operand(ea: ida_idaapi.ea_t, n: int) -> str:
    r"""Get forced operand. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number
    :returns: size of forced operand or -1
    """
    ...

def get_frame_args_size(ea: Any) -> Any:
    r"""
        Get size of arguments in function frame which are purged upon return
    
        :param ea: any address belonging to the function
    
        :returns: Size of function arguments in bytes.
                 If the function doesn't have a frame, return 0
                 If the function doesn't exist, return -1
        
    """
    ...

def get_frame_id(ea: Any) -> Any:
    r"""
        Get ID of function frame structure
    
        :param ea: any address belonging to the function
    
        :returns: ID of function frame or None In order to access stack variables
                 you need to use structure member manipulaion functions with the
                 obtained ID.
        
    """
    ...

def get_frame_lvar_size(ea: Any) -> Any:
    r"""
        Get size of local variables in function frame
    
        :param ea: any address belonging to the function
    
        :returns: Size of local variables in bytes.
                 If the function doesn't have a frame, return 0
                 If the function doesn't exist, return None
        
    """
    ...

def get_frame_regs_size(ea: Any) -> Any:
    r"""
        Get size of saved registers in function frame
    
        :param ea: any address belonging to the function
    
        :returns: Size of saved registers in bytes.
                 If the function doesn't have a frame, return 0
                 This value is used as offset for BP (if FUNC_FRAME is set)
                 If the function doesn't exist, return None
        
    """
    ...

def get_frame_size(ea: Any) -> Any:
    r"""
        Get full size of function frame
    
        :param ea: any address belonging to the function
        :returns: Size of function frame in bytes.
                    This function takes into account size of local
                    variables + size of saved registers + size of
                    return address + size of function arguments
                    If the function doesn't have a frame, return size of
                    function return address in the stack.
                    If the function doesn't exist, return 0
        
    """
    ...

def get_full_flags(ea: ida_idaapi.ea_t) -> flags64_t:
    r"""Get full flags value for address 'ea'. This function returns the byte value in the flags as well. See FF_IVL and MS_VAL. This function is more expensive to use than get_flags() 
            
    :returns: 0 if address is not present in the program
    """
    ...

def get_func_attr(ea: Any, attr: Any) -> Any:
    r"""
        Get a function attribute
    
        :param ea: any address belonging to the function
        :param attr: one of FUNCATTR_... constants
    
        :returns: BADADDR - error otherwise returns the attribute value
        
    """
    ...

def get_func_cmt(ea: Any, repeatable: Any) -> Any:
    r"""
        Retrieve function comment
    
        :param ea: any address belonging to the function
        :param repeatable: 1: get repeatable comment
                0: get regular comment
    
        :returns: function comment string
        
    """
    ...

def get_func_flags(ea: Any) -> Any:
    r"""
        Retrieve function flags
    
        :param ea: any address belonging to the function
    
        :returns: -1 - function doesn't exist otherwise returns the flags
        
    """
    ...

def get_func_name(ea: Any) -> Any:
    r"""
        Retrieve function name
    
        :param ea: any address belonging to the function
    
        :returns: null string - function doesn't exist
                otherwise returns function name
        
    """
    ...

def get_func_off_str(ea: Any) -> Any:
    r"""
        Convert address to 'funcname+offset' string
    
        :param ea: address to convert
    
        :returns: if the address belongs to a function then return a string
                 formed as 'name+offset' where 'name' is a function name
                 'offset' is offset within the function else return null string
        
    """
    ...

def get_hash_long(hash_id: Any, key: Any) -> Any:
    r"""
        Gets the long value of a hash element.
    
        :param hash_id: The hash ID.
        :param key: Key of an element.
    
        :returns: the 32bit or 64bit value of the element, or 0 if no such
                 element.
        
    """
    ...

def get_hash_string(hash_id: Any, key: Any) -> Any:
    r"""
        Gets the string value of a hash element.
    
        :param hash_id: The hash ID.
        :param key: Key of an element.
    
        :returns: the string value of the element, or None if no such
                 element.
        
    """
    ...

def get_idb_path() -> Any:
    r"""
        Get IDB full path
    
        This function returns full path of the current IDB database
        
    """
    ...

def get_inf_attr(attr: Any) -> Any:
    r"""
        Deprecated. Please ida_ida.inf_get_* instead.
        
    """
    ...

def get_input_file_path() -> str:
    r"""Get full path of the input file.
    
    """
    ...

def get_item_end(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the end address of the item at 'ea'. The returned address doesn't belong to the current item. Unexplored bytes are counted as 1 byte entities. 
            
    """
    ...

def get_item_head(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get the start address of the item at 'ea'. If there is no current item, then 'ea' will be returned (see definition at the end of bytes.hpp source) 
            
    """
    ...

def get_item_size(ea: Any) -> Any:
    r"""
        Get size of instruction or data item in bytes
    
        :param ea: linear address
    
        :returns: 1..n
        
    """
    ...

def get_last_bmask(enum_id: Any) -> Any:
    r"""
        Get last bitmask in the enum
    
        :param enum_id: id of enum
    
        :returns: id of constant or -1 if error
        
    """
    ...

def get_last_enum_member(enum_id: Any, bmask: Any = -1) -> Any:
    r"""
        Get last constant in the enum
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant (ordinary enums accept only -1 as a bitmask)
    
        :returns: value of constant or -1 if no constants are defined
                 All constants are sorted by their values
                 as unsigned longs.
        
    """
    ...

def get_last_hash_key(hash_id: Any) -> Any:
    r"""
        Get the last key in the hash.
    
        :param hash_id: The hash ID.
    
        :returns: the key, 0 otherwise.
        
    """
    ...

def get_last_index(tag: Any, array_id: Any) -> Any:
    r"""
        Get index of last existing array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
    
        :returns: -1 if the array is empty, otherwise index of first array
                 element of given type.
        
    """
    ...

def get_local_tinfo(ordinal: Any) -> Any:
    r"""
        Get local type information as 'typeinfo' object
    
        :param ordinal:  slot number (1...NumberOfLocalTypes)
        :returns: None on failure, or (type, fields) tuple.
        
    """
    ...

def get_manual_insn(ea: ida_idaapi.ea_t) -> str:
    r"""Retrieve the user-specified string for the manual instruction. 
            
    :param ea: linear address of the instruction or data item
    :returns: size of manual instruction or -1
    """
    ...

def get_member_by_idx(sid: Any, idx: Any) -> Any:
    r"""
        Get member ID by member ordinal number
    
        :param sid: structure type ID
        :param idx: member ordinal number
    
        :returns: -1 if bad structure type ID is passed or there is
                 no member with the specified index
                 otherwise returns the member ID.
        
    """
    ...

def get_member_cmt(sid: Any, member_offset: Any, repeatable: Any = True) -> Any:
    r"""
        Get comment of a member
    
        :param sid: structure type ID
        :param member_offset: member offset. The offset can be
                              any offset in the member. For example,
                              is a member is 4 bytes long and starts
                              at offset 2, then 2,3,4,5 denote
                              the same structure member.
        :param repeatable: is not used anymore
    
        :returns: None if bad structure type ID is passed
                 or no such member in the structure
                 otherwise returns comment of the specified member.
        
    """
    ...

def get_member_id(sid: Any, member_offset: Any) -> Any:
    r"""
    
        :param sid: structure type ID
        :param member_offset:. The offset can be
        any offset in the member. For example,
        is a member is 4 bytes long and starts
        at offset 2, then 2,3,4,5 denote
        the same structure member.
    
        :returns: -1 if bad structure type ID is passed or there is
        no member at the specified offset.
        otherwise returns the member id.
        
    """
    ...

def get_member_name(sid: Any, member_offset: Any) -> Any:
    r"""
        Get name of a member of a structure
    
        :param sid: structure type ID
        :param member_offset: member offset. The offset can be
                              any offset in the member. For example,
                              is a member is 4 bytes long and starts
                              at offset 2, then 2,3,4,5 denote
                              the same structure member.
    
        :returns: None if bad structure type ID is passed
                 or no such member in the structure
                 otherwise returns name of the specified member.
        
    """
    ...

def get_member_offset(sid: Any, member_name: Any) -> Any:
    r"""
        Get offset of a member of a structure by the member name
    
        :param sid: structure type ID
        :param member_name: name of structure member
    
        :returns: -1 if bad structure type ID is passed
                 or no such member in the structure
                 otherwise returns offset of the specified member.
    
        NOTE: Union members are, in IDA's internals, located
               at subsequent byte offsets: member 0 -> offset 0x0,
               member 1 -> offset 0x1, etc...
        
    """
    ...

def get_member_qty(sid: Any) -> Any:
    r"""
        Get number of members of a structure
    
        :param sid: structure type ID
    
        :returns: -1 if bad structure type ID is passed otherwise
                 returns number of members.
        
    """
    ...

def get_member_size(sid: Any, member_offset: Any) -> Any:
    r"""
        Get size of a member
    
        :param sid: structure type ID
        :param member_offset: member offset. The offset can be
                              any offset in the member. For example,
                              is a member is 4 bytes long and starts
                              at offset 2, then 2,3,4,5 denote
                              the same structure member.
    
        :returns: None if bad structure type ID is passed,
                 or no such member in the structure
                 otherwise returns size of the specified
                 member in bytes.
        
    """
    ...

def get_member_strid(sid: Any, member_offset: Any) -> Any:
    r"""
        Get structure id of a member
    
        :param sid: structure type ID
        :param member_offset: member offset. The offset can be
                              any offset in the member. For example,
                              is a member is 4 bytes long and starts
                              at offset 2, then 2,3,4,5 denote
                              the same structure member.
        :returns: -1 if bad structure type ID is passed
                 or no such member in the structure
                 otherwise returns structure id of the member.
                 If the current member is not a structure, returns -1.
        
    """
    ...

def get_min_spd_ea(func_ea: Any) -> Any:
    r"""
        Return the address with the minimal spd (stack pointer delta)
        If there are no SP change points, then return BADADDR.
    
        :param func_ea: function start
        :returns: BADDADDR - no such function
        
    """
    ...

def get_module_name(base: Any) -> Any:
    r"""
        Get process module name
    
        :param base: the base address of the module
    
        :returns: required info or None
        
    """
    ...

def get_module_size(base: Any) -> Any:
    r"""
        Get process module size
    
        :param base: the base address of the module
    
        :returns: required info or -1
        
    """
    ...

def get_name(ea: Any, gtn_flags: Any = 0) -> Any:
    r"""
        Get name at the specified address
    
        :param ea: linear address
        :param gtn_flags: how exactly the name should be retrieved.
                          combination of GN_ bits
    
        :returns: "" - byte has no name
        
    """
    ...

def get_name_ea(_from: ida_idaapi.ea_t, name: str) -> ida_idaapi.ea_t:
    r"""Get the address of a name. This function resolves a name into an address. It can handle regular global and local names, as well as debugger names. 
            
    :param name: any name in the program or nullptr
    :returns: address of the name or BADADDR
    """
    ...

def get_name_ea_simple(name: Any) -> Any:
    r"""
        Get linear address of a name
    
        :param name: name of program byte
    
        :returns: address of the name
                 BADADDR - No such name
        
    """
    ...

def get_next_bmask(enum_id: Any, bmask: Any) -> Any:
    r"""
        Get next bitmask in the enum
    
        :param enum_id: id of enum
        :param bmask
    
        :returns: id of constant or -1 if error
        
    """
    ...

def get_next_cref_from(frm: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next instruction referenced from the specified instruction. 
            
    :param current: linear address of current referenced instruction This value is returned by get_first_cref_from() or previous call to get_next_cref_from() functions.
    :returns: next referenced address or BADADDR.
    """
    ...

def get_next_cref_to(to: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next instruction referencing to the specified instruction. 
            
    :param to: linear address of referenced instruction
    :param current: linear address of current referenced instruction This value is returned by get_first_cref_to() or previous call to get_next_cref_to() functions.
    :returns: linear address of the next referencing instruction or BADADDR.
    """
    ...

def get_next_dref_from(frm: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next data referenced from the specified address. 
            
    :param current: linear address of current referenced data. This value is returned by get_first_dref_from() or previous call to get_next_dref_from() functions.
    :returns: linear address of next data or BADADDR.
    """
    ...

def get_next_dref_to(to: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of instruction/data referencing to the specified data 
            
    :param to: linear address of referencing instruction or data
    :param current: current linear address. This value is returned by get_first_dref_to() or previous call to get_next_dref_to() functions.
    :returns: BADADDR if nobody refers to the specified data.
    """
    ...

def get_next_enum_member(enum_id: Any, value: Any, bmask: Any = -1) -> Any:
    r"""
        Get next constant in the enum
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant ordinary enums accept only -1 as a bitmask
        :param value: value of the current constant
    
        :returns: value of a constant with value higher than the specified
                 value. -1 if no such constants exist.
                 All constants are sorted by their values as unsigned longs.
        
    """
    ...

def get_next_fchunk(ea: Any) -> Any:
    r"""
        Get next function chunk
    
        :param ea: any address
    
        :returns:  the starting address of the next function chunk or BADADDR
    
        NOTE: This function enumerates all chunks of all functions in the database
        
    """
    ...

def get_next_fcref_from(frm: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_next_fcref_to(to: ida_idaapi.ea_t, current: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_next_fixup_ea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_next_func(ea: Any) -> Any:
    r"""
        Find next function
    
        :param ea: any address belonging to the function
    
        :returns:        BADADDR - no more functions
                otherwise returns the next function start address
        
    """
    ...

def get_next_hash_key(hash_id: Any, key: Any) -> Any:
    r"""
        Get the next key in the hash.
    
        :param hash_id: The hash ID.
        :param key: The current key.
    
        :returns: the next key, 0 otherwise
        
    """
    ...

def get_next_index(tag: Any, array_id: Any, idx: Any) -> Any:
    r"""
        Get index of the next existing array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
        :param idx: Index of the current element.
    
        :returns: -1 if no more elements, otherwise returns index of the
                 next array element of given type.
        
    """
    ...

def get_next_module(base: Any) -> Any:
    r"""
        Enumerate process modules
    
        :param base: previous module's base address
    
        :returns: next module's base address or None on failure
        
    """
    ...

def get_next_seg(ea: Any) -> Any:
    r"""
        Get next segment
    
        :param ea: linear address
    
        :returns: start of the next segment
                 BADADDR - no next segment
        
    """
    ...

def get_numbered_type_name(ordinal: Any) -> Any:
    r"""
        Retrieve a local type name
    
        :param ordinal:  slot number (1...NumberOfLocalTypes)
    
        returns: local type name or None
        
    """
    ...

def get_operand_type(ea: Any, n: Any) -> Any:
    r"""
        Get type of instruction operand
    
        :param ea: linear address of instruction
        :param n: number of operand:
            0 - the first operand
            1 - the second operand
    
        :returns: any of o_* constants or -1 on error
        
    """
    ...

def get_operand_value(ea: Any, n: Any) -> Any:
    r"""
        Get number used in the operand
    
        This function returns an immediate number used in the operand
    
        :param ea: linear address of instruction
        :param n: the operand number
    
        :returns: value
            operand is an immediate value  => immediate value
            operand has a displacement     => displacement
            operand is a direct memory ref => memory address
            operand is a register          => register number
            operand is a register phrase   => phrase number
            otherwise                      => -1
        
    """
    ...

def get_ordinal_limit() -> Any:
    r"""
        Get number of local types + 1
    
        :returns: value >= 1. 1 means that there are no local types.
        
    """
    ...

def get_original_byte(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get original byte value (that was before patching). This function works for wide byte processors too. 
            
    """
    ...

def get_prev_bmask(enum_id: Any, bmask: Any) -> Any:
    r"""
        Get prev bitmask in the enum
    
        :param enum_id: id of enum
        :param bmask
    
        :returns: id of constant or -1 if error
        
    """
    ...

def get_prev_enum_member(enum_id: Any, value: Any, bmask: Any = -1) -> Any:
    r"""
        Get prev constant in the enum
    
        :param enum_id: id of enum
        :param bmask  : bitmask of the constant
                  ordinary enums accept only -1 as a bitmask
        :param value: value of the current constant
    
        :returns: value of a constant with value lower than the specified
            value. -1 if no such constants exist.
            All constants are sorted by their values as unsigned longs.
        
    """
    ...

def get_prev_fchunk(ea: Any) -> Any:
    r"""
        Get previous function chunk
    
        :param ea: any address
    
        :returns: the starting address of the function chunk or BADADDR
    
        NOTE: This function enumerates all chunks of all functions in the database
        
    """
    ...

def get_prev_fixup_ea(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    ...

def get_prev_func(ea: Any) -> Any:
    r"""
        Find previous function
    
        :param ea: any address belonging to the function
    
        :returns: BADADDR - no more functions
                otherwise returns the previous function start address
        
    """
    ...

def get_prev_hash_key(hash_id: Any, key: Any) -> Any:
    r"""
        Get the previous key in the hash.
    
        :param hash_id: The hash ID.
        :param key: The current key.
    
        :returns: the previous key, 0 otherwise
        
    """
    ...

def get_prev_index(tag: Any, array_id: Any, idx: Any) -> Any:
    r"""
        Get index of the previous existing array element.
    
        :param tag: Tag of array, specifies one of two array types: AR_LONG, AR_STR
        :param array_id: The array ID.
        :param idx: Index of the current element.
    
        :returns: -1 if no more elements, otherwise returns index of the
                 previous array element of given type.
        
    """
    ...

def get_process_state() -> int:
    r"""Return the state of the currently debugged process. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :returns: one of Debugged process states
    """
    ...

def get_processes(proclist: procinfo_vec_t) -> ssize_t:
    r"""Take a snapshot of running processes and return their description. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param proclist: array with information about each running process
    :returns: number of processes or -1 on error
    """
    ...

def get_processor_name() -> Any:
    r"""
        Get name of the current processor
        :returns: processor name
        
    """
    ...

def get_qword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one qword (64-bit) of the program at 'ea'. This function takes into account order of bytes specified in idainfo::is_be() This function works only for 8bit byte processors. 
            
    """
    ...

def get_reg_value(args: Any) -> Any:
    r"""Get a register value.
    
    This function has the following signatures:
    
        1. get_reg_val(name: str) -> Union[int, float, bytes]
        2. get_reg_val(name: str, regval: regval_t) -> bool
    
    The first (and most user-friendly) form will return
    a value whose type is related to the register type.
    I.e., either an integer, a float or, in the case of large
    vector registers, a bytes sequence.
    
    :param name: the register name
    :returns: the register value (1st form)
    """
    ...

def get_ret_tev_return(n: int) -> ida_idaapi.ea_t:
    r"""Get the return address from a function return trace event. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param n: number of trace event, is in range 0..get_tev_qty()-1. 0 represents the latest added trace event.
    :returns: BADADDR if not a function return event.
    """
    ...

def get_root_filename() -> str:
    r"""Get file name only of the input file.
    
    """
    ...

def get_screen_ea() -> ida_idaapi.ea_t:
    r"""Get the address at the screen cursor (ui_screenea)
    
    """
    ...

def get_segm_attr(segea: Any, attr: Any) -> Any:
    r"""
        Get segment attribute
    
        :param segea: any address within segment
        :param attr: one of SEGATTR_... constants
        
    """
    ...

def get_segm_by_sel(base: Any) -> Any:
    r"""
        Get segment by segment base
    
        :param base: segment base paragraph or selector
    
        :returns: linear address of the start of the segment or BADADDR
                 if no such segment
        
    """
    ...

def get_segm_end(ea: Any) -> Any:
    r"""
        Get end address of a segment
    
        :param ea: any address in the segment
    
        :returns: end of segment (an address past end of the segment)
                 BADADDR - the specified address doesn't belong to any segment
        
    """
    ...

def get_segm_name(ea: Any) -> Any:
    r"""
        Get name of a segment
    
        :param ea: any address in the segment
    
        :returns: "" - no segment at the specified address
        
    """
    ...

def get_segm_start(ea: Any) -> Any:
    r"""
        Get start address of a segment
    
        :param ea: any address in the segment
    
        :returns: start of segment
                 BADADDR - the specified address doesn't belong to any segment
        
    """
    ...

def get_source_linnum(ea: ida_idaapi.ea_t) -> int:
    ...

def get_sourcefile(ea: ida_idaapi.ea_t, bounds: range_t = None) -> str:
    ...

def get_sp_delta(ea: Any) -> Any:
    r"""
        Get modification of SP made by the instruction
    
        :param ea: end address of the instruction
                   i.e.the last address of the instruction+1
    
        :returns: Get modification of SP made at the specified location
                 If the specified location doesn't contain a SP change point, return 0
                 Otherwise return delta of SP modification
        
    """
    ...

def get_spd(ea: Any) -> Any:
    r"""
        Get current delta for the stack pointer
    
        :param ea: end address of the instruction
                   i.e.the last address of the instruction+1
    
        :returns: The difference between the original SP upon
                 entering the function and SP for the specified address
        
    """
    ...

def get_sreg(ea: Any, reg: Any) -> Any:
    r"""
        Get value of segment register at the specified address
    
        :param ea: linear address
        :param reg: name of segment register
    
        :returns: the value of the segment register or -1 on error
    
        NOTE: The segment registers in 32bit program usually contain selectors,
               so to get paragraph pointed to by the segment register you need to
               call sel2para() function.
        
    """
    ...

def get_step_trace_options() -> int:
    r"""Get current step tracing options. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :returns: Step trace options
    """
    ...

def get_str_type(ea: Any) -> Any:
    r"""
        Get string type
    
        :param ea: linear address
    
        :returns: One of STRTYPE_... constants
        
    """
    ...

def get_strlit_contents(ea: Any, length: Any = -1, strtype: Any = 0) -> Any:
    r"""
        Get string contents
        :param ea: linear address
        :param length: string length. -1 means to calculate the max string length
        :param strtype: the string type (one of STRTYPE_... constants)
    
        :returns: string contents or empty string
        
    """
    ...

def get_struc_cmt(tid: Any) -> Any:
    ...

def get_struc_id(name: Any) -> Any:
    ...

def get_struc_name(tid: Any) -> Any:
    ...

def get_struc_size(tid: Any) -> Any:
    ...

def get_tev_ea(n: int) -> ida_idaapi.ea_t:
    ...

def get_tev_mem(tev: Any, idx: Any) -> Any:
    ...

def get_tev_mem_ea(tev: Any, idx: Any) -> Any:
    ...

def get_tev_mem_qty(tev: Any) -> Any:
    ...

def get_tev_qty() -> int:
    r"""Get number of trace events available in trace buffer. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    """
    ...

def get_tev_reg(tev: Any, reg: Any) -> Any:
    ...

def get_tev_tid(n: int) -> int:
    ...

def get_tev_type(n: int) -> int:
    ...

def get_thread_qty() -> int:
    r"""Get number of threads. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    """
    ...

def get_tinfo(ea: Any) -> Any:
    r"""
        Get type information of function/variable as 'typeinfo' object
    
        :param ea: the address of the object
        :returns: None on failure, or (type, fields) tuple.
        
    """
    ...

def get_trace_file_desc(filename: str) -> str:
    r"""Get the file header of the specified trace file.
    
    """
    ...

def get_type(ea: Any) -> Any:
    r"""
        Get type of function/variable
    
        :param ea: the address of the object
    
        :returns: type string or None if failed
        
    """
    ...

def get_wide_byte(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one wide byte of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. 
            
    """
    ...

def get_wide_dword(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get two wide words (4 'bytes') of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_wide_word(ea: ida_idaapi.ea_t) -> uint64:
    r"""Get one wide word (2 'byte') of the program at 'ea'. Some processors may access more than 8bit quantity at an address. These processors have 32-bit byte organization from the IDA's point of view. This function takes into account order of bytes specified in idainfo::is_be() 
            
    """
    ...

def get_xref_type() -> Any:
    r"""
        Return type of the last xref obtained by
        [RD]first/next[B0] functions.
    
        :returns: constants fl_* or dr_*
        
    """
    ...

def getn_thread(n: int) -> thid_t:
    r"""Get the ID of a thread. \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param n: number of thread, is in range 0..get_thread_qty()-1
    :returns: NO_THREAD if the thread doesn't exist.
    """
    ...

def getn_thread_name(n: int) -> str:
    r"""Get the NAME of a thread \sq{Type, Synchronous function, Notification, none (synchronous function)} 
            
    :param n: number of thread, is in range 0..get_thread_qty()-1 or -1 for the current thread
    :returns: thread name or nullptr if the thread doesn't exist.
    """
    ...

def guess_type(ea: Any) -> Any:
    r"""
        Guess type of function/variable
    
        :param ea: the address of the object, can be the structure member id too
    
        :returns: type string or None if failed
        
    """
    ...

def hasName(F: Any) -> Any:
    ...

def hasUserName(F: Any) -> Any:
    ...

def has_value(F: Any) -> Any:
    ...

def here() -> Any:
    ...

def idadir() -> Any:
    r"""
        Get IDA directory
    
        This function returns the directory where IDA.EXE resides
        
    """
    ...

def import_type(idx: Any, type_name: Any) -> Any:
    r"""
        Copy information from type library to database
        Copy structure, union, or enum definition from the type library
        to the IDA database.
    
        :param idx: -1, ignored
        :param type_name: name of type to copy
    
        :returns: BADNODE-failed, otherwise the type id (structure id or enum id)
        
    """
    ...

def isBin0(F: Any) -> Any:
    ...

def isBin1(F: Any) -> Any:
    ...

def isDec0(F: Any) -> Any:
    ...

def isDec1(F: Any) -> Any:
    ...

def isExtra(F: Any) -> Any:
    ...

def isHex0(F: Any) -> Any:
    ...

def isHex1(F: Any) -> Any:
    ...

def isOct0(F: Any) -> Any:
    ...

def isOct1(F: Any) -> Any:
    ...

def isRef(F: Any) -> Any:
    ...

def is_align(F: Any) -> Any:
    ...

def is_bf(enum_id: Any) -> Any:
    r"""
        Is enum a bitmask ?
    
        :param enum_id: enum TID
    
        :returns: if it is a bitmask enum return True, otherwise False
        
    """
    ...

def is_byte(F: Any) -> Any:
    ...

def is_char0(F: Any) -> Any:
    ...

def is_char1(F: Any) -> Any:
    ...

def is_code(F: Any) -> Any:
    ...

def is_data(F: Any) -> Any:
    ...

def is_defarg0(F: Any) -> Any:
    ...

def is_defarg1(F: Any) -> Any:
    ...

def is_double(F: Any) -> Any:
    ...

def is_dword(F: Any) -> Any:
    ...

def is_enum0(F: Any) -> Any:
    ...

def is_enum1(F: Any) -> Any:
    ...

def is_event_handled() -> Any:
    r"""
        Is the debug event handled?
    
        :returns: boolean
        
    """
    ...

def is_float(F: Any) -> Any:
    ...

def is_flow(F: Any) -> Any:
    ...

def is_head(F: Any) -> Any:
    ...

def is_loaded(ea: Any) -> Any:
    r"""Is the byte initialized?"""
    ...

def is_manual0(F: Any) -> Any:
    ...

def is_manual1(F: Any) -> Any:
    ...

def is_mapped(ea: Any) -> Any:
    ...

def is_member_id(sid: Any) -> Any:
    r"""
        Is a member id?
    
        :param sid: structure type ID
    
        :returns: True there is structure member with the specified ID
                 False otherwise
        
    """
    ...

def is_off0(F: Any) -> Any:
    ...

def is_off1(F: Any) -> Any:
    ...

def is_oword(F: Any) -> Any:
    ...

def is_pack_real(F: Any) -> Any:
    ...

def is_qword(F: Any) -> Any:
    ...

def is_seg0(F: Any) -> Any:
    ...

def is_seg1(F: Any) -> Any:
    ...

def is_stkvar0(F: Any) -> Any:
    ...

def is_stkvar1(F: Any) -> Any:
    ...

def is_strlit(F: Any) -> Any:
    ...

def is_stroff0(F: Any) -> Any:
    ...

def is_stroff1(F: Any) -> Any:
    ...

def is_struct(F: Any) -> Any:
    ...

def is_tail(F: Any) -> Any:
    ...

def is_tbyte(F: Any) -> Any:
    ...

def is_union(sid: Any) -> Any:
    r"""
        Is a structure a union?
    
        :param sid: structure type ID
    
        :returns: True: yes, this is a union id
                 False: no
    
        NOTE: Unions are a special kind of structures
        
    """
    ...

def is_unknown(F: Any) -> Any:
    ...

def is_valid_trace_file(filename: str) -> bool:
    r"""Is the specified file a valid trace file for the current database?
    
    """
    ...

def is_word(F: Any) -> Any:
    ...

def jumpto(args: Any) -> bool:
    r"""This function has the following signatures:
    
        0. jumpto(ea: ida_idaapi.ea_t, opnum: int=-1, uijmp_flags: int=UIJMP_ACTIVATE) -> bool
        1. jumpto(custom_viewer: TWidget *, place: place_t *, x: int, y: int) -> bool
    
    # 0: jumpto(ea: ida_idaapi.ea_t, opnum: int=-1, uijmp_flags: int=UIJMP_ACTIVATE) -> bool
    
    Jump to the specified address (ui_jumpto). 
            
    :returns: success
    
    # 1: jumpto(custom_viewer: TWidget *, place: place_t *, x: int, y: int) -> bool
    
    Set cursor position in custom ida viewer. 
            
    :returns: success
    
    """
    ...

def load_and_run_plugin(name: str, arg: size_t) -> bool:
    r"""Load & run a plugin.
    
    """
    ...

def load_debugger(dbgname: str, use_remote: bool) -> bool:
    ...

def load_trace_file(filename: str) -> str:
    r"""Load a recorded trace file in the 'Tracing' window. If the call succeeds and 'buf' is not null, the description of the trace stored in the binary trace file will be returned in 'buf' 
            
    """
    ...

def loadfile(filepath: Any, pos: Any, ea: Any, size: Any) -> Any:
    ...

def ltoa(n: Any, radix: Any) -> Any:
    ...

def make_array(ea: Any, nitems: Any) -> Any:
    r"""
        Create an array.
    
        :param ea: linear address
        :param nitems: size of array in items
    
        NOTE: This function will create an array of the items with the same type as
        the type of the item at 'ea'. If the byte at 'ea' is undefined, then
        this function will create an array of bytes.
        
    """
    ...

def move_segm(ea: Any, to: Any, flags: Any) -> Any:
    r"""
        Move a segment to a new address
        This function moves all information to the new address
        It fixes up address sensitive information in the kernel
        The total effect is equal to reloading the segment to the target address
    
        :param ea: any address within the segment to move
        :param to: new segment start address
        :param flags: combination MFS_... constants
    
        :returns: MOVE_SEGM_... error code
        
    """
    ...

def msg(message: Any) -> Any:
    r"""Display a message in the message window
    
    :param message: message to print
    """
    ...

def next_addr(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get next address in the program (i.e. next address which has flags). 
            
    :returns: BADADDR if no such address exist.
    """
    ...

def next_func_chunk(funcea: Any, tailea: Any) -> Any:
    r"""
        Get the next function chunk of the specified function
    
        :param funcea: any address in the function
        :param tailea: any address in the current chunk
    
        :returns: the starting address of the next function chunk or BADADDR
    
        NOTE: This function returns the next chunk of the specified function
        
    """
    ...

def next_head(ea: Any, maxea: Any = 18446744073709551615) -> Any:
    r"""
        Get next defined item (instruction or data) in the program
    
        :param ea: linear address to start search from
        :param maxea: the search will stop at the address
            maxea is not included in the search range
    
        :returns: BADADDR - no (more) defined items
        
    """
    ...

def next_not_tail(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of next non-tail byte. 
            
    :returns: BADADDR if none exists.
    """
    ...

def op_bin(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to bin_flag()
    
    """
    ...

def op_chr(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to char_flag()
    
    """
    ...

def op_dec(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to dec_flag()
    
    """
    ...

def op_enum(ea: ida_idaapi.ea_t, n: int, id: tid_t, serial: uchar = 0) -> bool:
    r"""Set operand representation to be enum type If applied to unexplored bytes, converts them to 16/32bit word data 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :param id: id of enum
    :param serial: the serial number of the constant in the enumeration, usually 0. the serial numbers are used if the enumeration contains several constants with the same value
    :returns: success
    """
    ...

def op_flt(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to flt_flag()
    
    """
    ...

def op_hex(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to hex_flag()
    
    """
    ...

def op_man(ea: ida_idaapi.ea_t, n: int, op: str) -> bool:
    r"""Set forced operand. 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number
    :param op: text of operand
    * nullptr: do nothing (return 0)
    * "" : delete forced operand
    :returns: success
    """
    ...

def op_num(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to num_flag()
    
    """
    ...

def op_oct(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""set op type to oct_flag()
    
    """
    ...

def op_offset(args: Any) -> bool:
    r"""See op_offset_ex()
    
    """
    ...

def op_offset_high16(ea: Any, n: Any, target: Any) -> Any:
    r"""
        Convert operand to a high offset
        High offset is the upper 16bits of an offset.
        This type is used by TMS320C6 processors (and probably by other
        RISC processors too)
    
        :param ea: linear address
        :param n: number of operand
            - 0 - the first operand
            - 1 - the second, third and all other operands
            - -1 - all operands
        :param target: the full value (all 32bits) of the offset
        
    """
    ...

def op_plain_offset(ea: Any, n: Any, base: Any) -> Any:
    r"""
        Convert operand to an offset
        (for the explanations of 'ea' and 'n' please see op_bin())
    
        Example:
        ========
    
            seg000:2000 dw      1234h
    
            and there is a segment at paragraph 0x1000 and there is a data item
            within the segment at 0x1234:
    
            seg000:1234 MyString        db 'Hello, world!',0
    
            Then you need to specify a linear address of the segment base to
            create a proper offset:
    
            op_plain_offset(["seg000",0x2000],0,0x10000);
    
            and you will have:
    
            seg000:2000 dw      offset MyString
    
        Motorola 680x0 processor have a concept of "outer offsets".
        If you want to create an outer offset, you need to combine number
        of the operand with the following bit:
    
        Please note that the outer offsets are meaningful only for
        Motorola 680x0.
    
        :param ea: linear address
        :param n: number of operand
            - 0 - the first operand
            - 1 - the second, third and all other operands
            - -1 - all operands
        :param base: base of the offset as a linear address
            If base == BADADDR then the current operand becomes non-offset
        
    """
    ...

def op_seg(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Set operand representation to be 'segment'. If applied to unexplored bytes, converts them to 16/32bit word data 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: success
    """
    ...

def op_stkvar(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Set operand representation to be 'stack variable'. Should be applied to an instruction within a function. Should be applied after creating a stack var using insn_t::create_stkvar(). 
            
    :param ea: linear address
    :param n: 0..UA_MAXOP-1 operand number, OPND_ALL all operands
    :returns: success
    """
    ...

def op_stroff(ea: Any, n: Any, strid: Any, delta: Any) -> Any:
    r"""
        Convert operand to an offset in a structure
    
        :param ea: linear address
        :param n: number of operand
            - 0 - the first operand
            - 1 - the second, third and all other operands
            - -1 - all operands
        :param strid: id of a structure type
        :param delta: struct offset delta. usually 0. denotes the difference
                        between the structure base and the pointer into the structure.
    
        
    """
    ...

def parse_decl(inputtype: Any, flags: Any) -> Any:
    r"""
        Parse type declaration
    
        :param inputtype: file name or C declarations (depending on the flags)
        :param flags: combination of PT_... constants or 0
    
        :returns: None on failure or (name, type, fields) tuple
        
    """
    ...

def parse_decls(inputtype: Any, flags: Any = 0) -> Any:
    r"""
        Parse type declarations
    
        :param inputtype: file name or C declarations (depending on the flags)
        :param flags: combination of PT_... constants or 0
    
        :returns: number of parsing errors (0 no errors)
        
    """
    ...

def patch_byte(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a byte of the program. The original value of the byte is saved and can be obtained by get_original_byte(). This function works for wide byte processors too. 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_dbg_byte(ea: ida_idaapi.ea_t, x: int) -> bool:
    r"""Change one byte of the debugged process memory. 
            
    :param ea: linear address
    :param x: byte value
    :returns: true if the process memory has been modified
    """
    ...

def patch_dword(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a dword of the program. The original value of the dword is saved and can be obtained by get_original_dword(). This function DOESN'T work for wide byte processors. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_qword(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a qword of the program. The original value of the qword is saved and can be obtained by get_original_qword(). This function DOESN'T work for wide byte processors. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def patch_word(ea: ida_idaapi.ea_t, x: uint64) -> bool:
    r"""Patch a word of the program. The original value of the word is saved and can be obtained by get_original_word(). This function works for wide byte processors too. This function takes into account order of bytes specified in idainfo::is_be() 
            
    :returns: true: the database has been modified,
    :returns: false: the debugger is running and the process' memory has value 'x' at address 'ea', or the debugger is not running, and the IDB has value 'x' at address 'ea already.
    """
    ...

def plan_and_wait(sEA: Any, eEA: Any, final_pass: Any = True) -> Any:
    r"""
        Perform full analysis of the range
    
        :param sEA: starting linear address
        :param eEA: ending linear address (excluded)
        :param final_pass: make the final pass over the specified range
    
        :returns: 1-ok, 0-Ctrl-Break was pressed.
        
    """
    ...

def plan_to_apply_idasgn(fname: str) -> int:
    r"""Add a signature file to the list of planned signature files. 
            
    :param fname: file name. should not contain directory part.
    :returns: 0 if failed, otherwise number of planned (and applied) signatures
    """
    ...

def prev_addr(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get previous address in the program. 
            
    :returns: BADADDR if no such address exist.
    """
    ...

def prev_head(ea: Any, minea: Any = 0) -> Any:
    r"""
        Get previous defined item (instruction or data) in the program
    
        :param ea: linear address to start search from
        :param minea: the search will stop at the address
                minea is included in the search range
    
        :returns: BADADDR - no (more) defined items
        
    """
    ...

def prev_not_tail(ea: ida_idaapi.ea_t) -> ida_idaapi.ea_t:
    r"""Get address of previous non-tail byte. 
            
    :returns: BADADDR if none exists.
    """
    ...

def print_decls(ordinals: Any, flags: Any) -> Any:
    r"""
        Print types in a format suitable for use in a header file
    
        :param ordinals: comma-separated list of type ordinals
        :param flags: combination of PDF_... constants or 0
    
        :returns: string containing the type definitions
        
    """
    ...

def print_insn_mnem(ea: Any) -> Any:
    r"""
        Get instruction mnemonics
    
        :param ea: linear address of instruction
    
        :returns: "" - no instruction at the specified location
    
        NOTE: this function may not return exactly the same mnemonics
        as you see on the screen.
        
    """
    ...

def print_operand(ea: Any, n: Any) -> Any:
    r"""
        Get operand of an instruction or data
    
        :param ea: linear address of the item
        :param n: number of operand:
            0 - the first operand
            1 - the second operand
    
        :returns: the current text representation of operand or ""
        
    """
    ...

def process_config_line(directive: Any) -> Any:
    r"""
        Obsolete. Please use ida_idp.process_config_directive().
        
    """
    ...

def process_ui_action(name: Any, flags: Any = 0) -> Any:
    r"""
        Invokes an IDA UI action by name
    
        :param name: Command name
        :param flags: Reserved. Must be zero
        :returns: Boolean
        
    """
    ...

def put_bookmark(ea: ida_idaapi.ea_t, lnnum: int, x: short, y: short, slot: int, comment: str) -> None:
    ...

def qexit(code: int) -> None:
    r"""Call qatexit functions, shut down UI and kernel, and exit. 
            
    :param code: exit code
    """
    ...

def qsleep(milliseconds: Any) -> Any:
    r"""
        qsleep the specified number of milliseconds
        This function suspends IDA for the specified amount of time
    
        :param milliseconds: time to sleep
        
    """
    ...

def read_dbg_byte(ea: Any) -> Any:
    r"""
        Get value of program byte using the debugger memory
    
        :param ea: linear address
        :returns: The value or None on failure.
        
    """
    ...

def read_dbg_dword(ea: Any) -> Any:
    r"""
        Get value of program double-word using the debugger memory
    
        :param ea: linear address
        :returns: The value or None on failure.
        
    """
    ...

def read_dbg_memory(ea: Any, sz: Any) -> Any:
    r"""Reads from the debugee's memory at the specified ea
    
    :param ea: the debuggee's memory address
    :param sz: the amount of data to read
    :returns: The read buffer (as bytes), or None on failure
    """
    ...

def read_dbg_qword(ea: Any) -> Any:
    r"""
        Get value of program quadro-word using the debugger memory
    
        :param ea: linear address
        :returns: The value or None on failure.
        
    """
    ...

def read_dbg_word(ea: Any) -> Any:
    r"""
        Get value of program word using the debugger memory
    
        :param ea: linear address
        :returns: The value or None on failure.
        
    """
    ...

def read_selection_end() -> Any:
    r"""
        Get end address of the selected range
    
        :returns: BADADDR - the user has not selected an range
        
    """
    ...

def read_selection_start() -> Any:
    r"""
        Get start address of the selected range
        returns BADADDR - the user has not selected an range
        
    """
    ...

def readlong(handle: Any, mostfirst: Any) -> Any:
    ...

def readshort(handle: Any, mostfirst: Any) -> Any:
    ...

def readstr(handle: Any) -> Any:
    ...

def rebase_program(delta: Any, flags: int) -> int:
    r"""Rebase the whole program by 'delta' bytes. 
            
    :param delta: number of bytes to move the program
    :param flags: Move segment flags it is recommended to use MSF_FIXONCE so that the loader takes care of global variables it stored in the database
    :returns: Move segment result codes
    """
    ...

def recalc_spd(cur_ea: ida_idaapi.ea_t) -> bool:
    r"""Recalculate SP delta for an instruction that stops execution. The next instruction is not reached from the current instruction. We need to recalculate SP for the next instruction.
    This function will create a new automatic SP register change point if necessary. It should be called from the emulator (emu.cpp) when auto_state == AU_USED if the current instruction doesn't pass the execution flow to the next instruction. 
            
    :param cur_ea: linear address of the current instruction
    :returns: 1: new stkpnt is added
    :returns: 0: nothing is changed
    """
    ...

def refresh_debugger_memory() -> Any:
    r"""Refreshes the debugger memory
    
    :returns: Nothing
    """
    ...

def refresh_idaview_anyway() -> None:
    r"""Refresh all disassembly views (ui_refresh), forces an immediate refresh. Please consider request_refresh() instead 
            
    """
    ...

def refresh_lists() -> None:
    ...

def remove_fchunk(funcea: Any, tailea: Any) -> Any:
    r"""
        Remove a function chunk from the function
    
        :param funcea: any address in the function
        :param tailea: any address in the function chunk to remove
    
        :returns: 0 if failed, 1 if success
        
    """
    ...

def rename_array(array_id: Any, newname: Any) -> Any:
    r"""
        Rename array, by its ID.
    
        :param id: The ID of the array to rename.
        :param newname: The new name of the array.
    
        :returns: 1 in case of success, 0 otherwise
        
    """
    ...

def rename_entry(ord: int, name: str, flags: int = 0) -> bool:
    r"""Rename entry point. 
            
    :param ord: ordinal number of the entry point
    :param name: name of entry point. If the specified location already has a name, the old name will be appended to a repeatable comment.
    :param flags: See AEF_*
    :returns: success
    """
    ...

def resume_process() -> Any:
    ...

def resume_thread(tid: thid_t) -> int:
    r"""Resume thread. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    :param tid: thread id
    :returns: -1: network error
    :returns: 0: failed
    :returns: 1: ok
    """
    ...

def retrieve_input_file_md5() -> bytes:
    r"""Get input file md5.
    
    """
    ...

def rotate_byte(x: Any, count: Any) -> Any:
    ...

def rotate_dword(x: Any, count: Any) -> Any:
    ...

def rotate_left(value: Any, count: Any, nbits: Any, offset: Any) -> Any:
    r"""
        Rotate a value to the left (or right)
    
        :param value: value to rotate
        :param count: number of times to rotate. negative counter means
                      rotate to the right
        :param nbits: number of bits to rotate
        :param offset: offset of the first bit to rotate
    
        :returns: the value with the specified field rotated
                 all other bits are not modified
        
    """
    ...

def rotate_word(x: Any, count: Any) -> Any:
    ...

def run_to(args: Any) -> bool:
    r"""Execute the process until the given address is reached. If no process is active, a new process is started. Technically, the debugger sets up a temporary breakpoint at the given address, and continues (or starts) the execution of the whole process. So, all threads continue their execution! \sq{Type, Asynchronous function - available as Request, Notification, dbg_run_to} 
            
    :param ea: target address
    :param pid: not used yet. please do not specify this parameter.
    :param tid: not used yet. please do not specify this parameter.
    """
    ...

def save_database(idbname: Any, flags: Any = 0) -> Any:
    r"""
        Save current database to the specified idb file
    
        :param idbname: name of the idb file. if empty, the current idb
                        file will be used.
        :param flags: combination of ida_loader.DBFL_... bits or 0
        
    """
    ...

def save_trace_file(filename: str, description: str) -> bool:
    r"""Save the current trace in the specified file.
    
    """
    ...

def savefile(filepath: Any, pos: Any, ea: Any, size: Any) -> Any:
    ...

def sel2para(sel: Any) -> Any:
    r"""
        Get a selector value
    
        :param sel: the selector number
    
        :returns: selector value if found
                 otherwise the input value (sel)
    
        NOTE: selector values are always in paragraphs
        
    """
    ...

def select_thread(tid: thid_t) -> bool:
    r"""Select the given thread as the current debugged thread. All thread related execution functions will work on this thread. The process must be suspended to select a new thread. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    :param tid: ID of the thread to select
    :returns: false if the thread doesn't exist.
    """
    ...

def selector_by_name(segname: Any) -> Any:
    r"""
        Get segment selector by name
    
        :param segname: name of segment
    
        :returns: segment selector or BADADDR
        
    """
    ...

def send_dbg_command(cmd: Any) -> Any:
    r"""Sends a command to the debugger module and returns the output string.
        An exception will be raised if the debugger is not running or the current debugger does not export
        the 'send_dbg_command' IDC command.
        
    """
    ...

def set_array_long(array_id: Any, idx: Any, value: Any) -> Any:
    r"""
        Sets the long value of an array element.
    
        :param array_id: The array ID.
        :param idx: Index of an element.
        :param value: 32bit or 64bit value to store in the array
    
        :returns: 1 in case of success, 0 otherwise
        
    """
    ...

def set_array_params(ea: Any, flags: Any, litems: Any, align: Any) -> Any:
    r"""
        Set array representation format
    
        :param ea: linear address
        :param flags: combination of AP_... constants or 0
        :param litems: number of items per line. 0 means auto
        :param align: element alignment
                      - -1: do not align
                      - 0:  automatic alignment
                      - other values: element width
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def set_array_string(array_id: Any, idx: Any, value: Any) -> Any:
    r"""
        Sets the string value of an array element.
    
        :param array_id: The array ID.
        :param idx: Index of an element.
        :param value: String value to store in the array
    
        :returns: 1 in case of success, 0 otherwise
        
    """
    ...

def set_bmask_cmt(enum_id: Any, bmask: Any, cmt: Any, repeatable: Any) -> Any:
    r"""
        Set bitmask comment (only for bitfields)
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant
        :param cmt: comment
        repeatable - is not used anymore
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_bmask_name(enum_id: Any, bmask: Any, name: Any) -> Any:
    r"""
        Set bitmask name (only for bitfields)
    
        :param enum_id: id of enum
        :param bmask: bitmask of the constant
        :param name: name of bitmask
    
        :returns: True-ok, False-failed
        
    """
    ...

def set_bpt_attr(address: Any, bptattr: Any, value: Any) -> Any:
    r"""
            modifiable characteristics of a breakpoint
    
        :param address: any address in the breakpoint range
        :param bptattr: the attribute code, one of BPTATTR_* constants
                        BPTATTR_CND is not allowed, see set_bpt_cond()
        :param value: the attribute value
    
        :returns: success
        
    """
    ...

def set_bpt_cond(ea: Any, cnd: Any, is_lowcnd: Any = 0) -> Any:
    r"""
        Set breakpoint condition
    
        :param ea: any address in the breakpoint range
        :param cnd: breakpoint condition
        :param is_lowcnd: 0 - regular condition, 1 - low level condition
    
        :returns: success
        
    """
    ...

def set_cmt(ea: ida_idaapi.ea_t, comm: str, rptble: bool) -> bool:
    r"""Set an indented comment. 
            
    :param ea: linear address
    :param comm: comment string
    * nullptr: do nothing (return 0)
    * "" : delete comment
    :param rptble: is repeatable?
    :returns: success
    """
    ...

def set_color(ea: Any, what: Any, color: Any) -> Any:
    r"""
        Set item color
    
        :param ea: address of the item
        :param what: type of the item (one of CIC_* constants)
        :param color: new color code in RGB (hex 0xBBGGRR)
    
        :returns: success (True or False)
        
    """
    ...

def set_debugger_event_cond(NONNULL_evcond: str) -> None:
    ...

def set_debugger_options(options: uint) -> uint:
    r"""Set debugger options. Replaces debugger options with the specification combination Debugger options 
            
    :returns: the old debugger options
    """
    ...

def set_default_sreg_value(ea: Any, reg: Any, value: Any) -> Any:
    r"""
        Set default segment register value for a segment
    
        :param ea: any address in the segment
                   if no segment is present at the specified address
                   then all segments will be affected
        :param reg: name of segment register
        :param value: default value of the segment register. -1-undefined.
        
    """
    ...

def set_enum_bf(enum_id: Any, bf: Any) -> Any:
    r"""
        Set or clear the 'bitmask' attribute of an enum
    
        :param enum_id: enum TID
        :param bf: bitmask enum or not
    
        :returns: success
        
    """
    ...

def set_enum_cmt(enum_id: Any, cmt: Any, repeatable: Any) -> Any:
    r"""
        Set comment for enum type
    
        :param enum_id: enum TID
        :param cmt: comment
        :param repeatable: is comment repeatable ?
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_enum_flag(enum_id: Any, flag: Any) -> Any:
    r"""
        Set enum constant representation flags
    
        :param enum_id: enum TID
        :param flag
    
        :returns: success
        
    """
    ...

def set_enum_member_cmt(const_id: Any, cmt: Any, repeatable: Any = False) -> Any:
    r"""
        Set comment for enum member
    
        :param const_id: enum constant TID
        :param cmt: comment
        :param repeatable: is not used anymore
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_enum_member_name(const_id: Any, name: Any) -> Any:
    r"""
        Set name of enum member
    
        :param const_id: enum constant TID
        :param name: new member name
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_enum_name(enum_id: Any, name: Any) -> Any:
    r"""
        Set name of enum type
    
        :param enum_id: id of enum
        :param name: new enum name
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_enum_width(enum_id: Any, nbytes: Any) -> Any:
    r"""
        Set the width of enum base type
    
        :param enum_id: enum TID
        :param nbytes: width of enum base type, allowed values: 0 (unspecified),1,2,4,8,16,32,64
    
        :returns: success
        
    """
    ...

def set_fchunk_attr(ea: Any, attr: Any, value: Any) -> Any:
    r"""
        Set a function chunk attribute
    
        :param ea: any address in the chunk
        :param attr: only FUNCATTR_START, FUNCATTR_END, FUNCATTR_OWNER
        :param value: desired value
    
        :returns: 0 if failed, 1 if success
        
    """
    ...

def set_fixup(ea: Any, fixuptype: Any, fixupflags: Any, targetsel: Any, targetoff: Any, displ: Any) -> Any:
    r"""
        Set fixup information
    
        :param ea: address to set fixup information about
        :param fixuptype:  fixup type. see get_fixup_target_type()
                           for possible fixup types.
        :param fixupflags: fixup flags. see get_fixup_target_flags()
                           for possible fixup types.
        :param targetsel:  target selector
        :param targetoff:  target offset
        :param displ:      displacement
    
        :returns:        none
        
    """
    ...

def set_flag(off: Any, bit: Any, value: Any) -> Any:
    ...

def set_frame_size(ea: Any, lvsize: Any, frregs: Any, argsize: Any) -> Any:
    r"""
        Make function frame
    
        :param ea: any address belonging to the function
        :param lvsize: size of function local variables
        :param frregs: size of saved registers
        :param argsize: size of function arguments
    
        :returns: ID of function frame or -1
                 If the function did not have a frame, the frame
                 will be created. Otherwise the frame will be modified
        
    """
    ...

def set_func_attr(ea: Any, attr: Any, value: Any) -> Any:
    r"""
        Set a function attribute
    
        :param ea: any address belonging to the function
        :param attr: one of FUNCATTR_... constants
        :param value: new value of the attribute
    
        :returns: 1-ok, 0-failed
        
    """
    ...

def set_func_cmt(ea: Any, cmt: Any, repeatable: Any) -> Any:
    r"""
        Set function comment
    
        :param ea: any address belonging to the function
        :param cmt: a function comment line
        :param repeatable: 1: get repeatable comment
                0: get regular comment
        
    """
    ...

def set_func_end(ea: ida_idaapi.ea_t, newend: ida_idaapi.ea_t) -> bool:
    r"""Move function chunk end address. 
            
    :param ea: any address in the function
    :param newend: new end address of the function
    :returns: success
    """
    ...

def set_func_flags(ea: Any, flags: Any) -> Any:
    r"""
        Change function flags
    
        :param ea: any address belonging to the function
        :param flags: see get_func_flags() for explanations
    
        :returns: !=0 - ok
        
    """
    ...

def set_hash_long(hash_id: Any, key: Any, value: Any) -> Any:
    r"""
        Sets the long value of a hash element.
    
        :param hash_id: The hash ID.
        :param key: Key of an element.
        :param value: 32bit or 64bit value to store in the hash
    
        :returns: 1 in case of success, 0 otherwise
        
    """
    ...

def set_hash_string(hash_id: Any, key: Any, value: Any) -> Any:
    r"""
        Sets the string value of a hash element.
    
        :param hash_id: The hash ID.
        :param key: Key of an element.
        :param value: string value to store in the hash
    
        :returns: 1 in case of success, 0 otherwise
        
    """
    ...

def set_ida_state(st: idastate_t) -> idastate_t:
    r"""Change IDA status indicator value 
            
    :param st: - new indicator status
    :returns: old indicator status
    """
    ...

def set_inf_attr(attr: Any, value: Any) -> Any:
    r"""
        Deprecated. Please ida_ida.inf_set_* instead.
        
    """
    ...

def set_local_type(ordinal: Any, input: Any, flags: Any) -> Any:
    r"""
        Parse one type declaration and store it in the specified slot
    
        :param ordinal:  slot number (1...NumberOfLocalTypes)
                         -1 means allocate new slot or reuse the slot
                         of the existing named type
        :param input:  C declaration. Empty input empties the slot
        :param flags:  combination of PT_... constants or 0
    
        :returns: slot number or 0 if error
        
    """
    ...

def set_manual_insn(ea: ida_idaapi.ea_t, manual_insn: str) -> None:
    r"""Set manual instruction string. 
            
    :param ea: linear address of the instruction or data item
    :param manual_insn: "" - delete manual string. nullptr - do nothing
    """
    ...

def set_member_cmt(sid: Any, member_offset: Any, comment: Any, repeatable: Any) -> Any:
    r"""
        Change structure member comment
    
        :param sid: structure type ID
        :param member_offset: offset of the member
        :param comment: new comment of the structure member
        :param repeatable: 1: change repeatable comment
                           0: change regular comment
    
        :returns: != 0 - ok
        
    """
    ...

def set_member_name(sid: Any, member_offset: Any, name: Any) -> Any:
    r"""
        Change structure member name
    
        :param sid: structure type ID
        :param member_offset: offset of the member
        :param name: new name of the member
    
        :returns: != 0 - ok.
        
    """
    ...

def set_member_type(sid: Any, member_offset: Any, flag: Any, typeid: Any, nitems: Any, target: Any = -1, tdelta: Any = 0, reftype: Any = 2) -> Any:
    r"""
        Change structure member type
    
        :param sid: structure type ID
        :param member_offset: offset of the member
        :param flag: new type of the member. Should be one of
                     FF_BYTE..FF_PACKREAL (see above) combined with FF_DATA
        :param typeid: if is_struct(flag) then typeid specifies the structure id for the member
                       if is_off0(flag) then typeid specifies the offset base.
                       if is_strlit(flag) then typeid specifies the string type (STRTYPE_...).
                       if is_stroff(flag) then typeid specifies the structure id
                       if is_enum(flag) then typeid specifies the enum id
                       if is_custom(flags) then typeid specifies the dtid and fid: dtid|(fid<<16)
                       Otherwise typeid should be -1.
        :param nitems: number of items in the member
    
        :param target: target address of the offset expr. You may specify it as
                       -1, ida will calculate it itself
        :param tdelta: offset target delta. usually 0
        :param reftype: see REF_... definitions
    
        NOTE: The remaining arguments are allowed only if is_off0(flag) and you want
               to specify a complex offset expression
    
        :returns: !=0 - ok.
        
    """
    ...

def set_name(ea: Any, name: Any, flags: Any = 0) -> Any:
    r"""
        Rename an address
    
        :param ea: linear address
        :param name: new name of address. If name == "", then delete old name
        :param flags: combination of SN_... constants
    
        :returns: 1-ok, 0-failure
        
    """
    ...

def set_processor_type(procname: str, level: setproc_level_t) -> bool:
    r"""Set target processor type. Once a processor module is loaded, it cannot be replaced until we close the idb. 
            
    :param procname: name of processor type (one of names present in processor_t::psnames)
    :param level: SETPROC_
    :returns: success
    """
    ...

def set_reg_value(value: Any, name: Any) -> Any:
    r"""
        Set register value
    
        :param name: the register name
        :param value: new register value
    
        NOTE: The debugger should be running
               It is not necessary to use this function to set register values.
               A register name in the left side of an assignment will do too.
        
    """
    ...

def set_remote_debugger(host: str, _pass: str, port: int = -1) -> None:
    r"""Set remote debugging options. Should be used before starting the debugger. 
            
    :param host: If empty, IDA will use local debugger. If nullptr, the host will not be set.
    :param port: If -1, the default port number will be used
    """
    ...

def set_root_filename(file: str) -> None:
    r"""Set full path of the input file.
    
    """
    ...

def set_segm_addressing(ea: Any, bitness: Any) -> Any:
    r"""
        Change segment addressing
    
        :param ea: any address in the segment
        :param bitness: 0: 16bit, 1: 32bit, 2: 64bit
    
        :returns: success (boolean)
        
    """
    ...

def set_segm_alignment(ea: Any, alignment: Any) -> Any:
    r"""
        Change alignment of the segment
    
        :param ea: any address in the segment
        :param alignment: new alignment of the segment (one of the sa... constants)
    
        :returns: success (boolean)
        
    """
    ...

def set_segm_attr(segea: Any, attr: Any, value: Any) -> Any:
    r"""
        Set segment attribute
    
        :param segea: any address within segment
        :param attr: one of SEGATTR_... constants
    
        NOTE: Please note that not all segment attributes are modifiable.
               Also some of them should be modified using special functions
               like set_segm_addressing, etc.
        
    """
    ...

def set_segm_class(ea: Any, segclass: Any) -> Any:
    r"""
        Change class of the segment
    
        :param ea: any address in the segment
        :param segclass: new class of the segment
    
        :returns: success (boolean)
        
    """
    ...

def set_segm_combination(segea: Any, comb: Any) -> Any:
    r"""
        Change combination of the segment
    
        :param segea: any address in the segment
        :param comb: new combination of the segment (one of the sc... constants)
    
        :returns: success (boolean)
        
    """
    ...

def set_segm_name(ea: Any, name: Any) -> Any:
    r"""
        Change name of the segment
    
        :param ea: any address in the segment
        :param name: new name of the segment
    
        :returns: success (boolean)
        
    """
    ...

def set_segm_type(segea: Any, segtype: Any) -> Any:
    r"""
        Set segment type
    
        :param segea: any address within segment
        :param segtype: new segment type:
    
        :returns: !=0 - ok
        
    """
    ...

def set_segment_bounds(ea: Any, startea: Any, endea: Any, flags: Any) -> Any:
    r"""
        Change segment boundaries
    
        :param ea: any address in the segment
        :param startea: new start address of the segment
        :param endea: new end address of the segment
        :param flags: combination of SEGMOD_... flags
    
        :returns: boolean success
        
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

def set_source_linnum(ea: ida_idaapi.ea_t, lnnum: int) -> None:
    ...

def set_step_trace_options(options: int) -> None:
    r"""Modify step tracing options. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    """
    ...

def set_storage_type(start_ea: ida_idaapi.ea_t, end_ea: ida_idaapi.ea_t, stt: storage_type_t) -> error_t:
    r"""Change flag storage type for address range. 
            
    :param start_ea: should be lower than end_ea.
    :param end_ea: does not belong to the range.
    :param stt: storage_type_t
    :returns: error code
    """
    ...

def set_struc_cmt(sid: Any, cmt: Any, repeatable: Any = True) -> Any:
    ...

def set_struc_name(sid: Any, name: Any) -> Any:
    ...

def set_tail_owner(tailea: Any, funcea: Any) -> Any:
    r"""
        Change the function chunk owner
    
        :param tailea: any address in the function chunk
        :param funcea: the starting address of the new owner
    
        :returns: False if failed, True if success
    
        NOTE: The new owner must already have the chunk appended before the call
        
    """
    ...

def set_target_assembler(asmnum: int) -> bool:
    r"""Set target assembler. 
            
    :param asmnum: number of assembler in the current processor module
    :returns: success
    """
    ...

def set_trace_file_desc(filename: str, description: str) -> bool:
    r"""Change the description of the specified trace file.
    
    """
    ...

def sizeof(typestr: Any) -> Any:
    r"""
        Returns the size of the type. It is equivalent to IDC's sizeof().
        :param typestr: can be specified as a typeinfo tuple (e.g. the result of get_tinfo()),
                serialized type byte string,
                or a string with C declaration (e.g. "int")
        :returns: -1 if typestring is not valid or has no size. otherwise size of the type
        
    """
    ...

def split_sreg_range(ea: Any, reg: Any, value: Any, tag: Any = 2) -> Any:
    r"""
        Set value of a segment register.
    
        :param ea: linear address
        :param reg: name of a register, like "cs", "ds", "es", etc.
        :param value: new value of the segment register.
        :param tag: of SR_... constants
    
        NOTE: IDA keeps tracks of all the points where segment register change their
              values. This function allows you to specify the correct value of a segment
              register if IDA is not able to find the correct value.
        
    """
    ...

def start_process(path: str = None, args: str = None, sdir: str = None) -> int:
    r"""Start a process in the debugger. \sq{Type, Asynchronous function - available as Request, Notification, dbg_process_start} 
            
    :param path: path to the executable to start
    :param args: arguments to pass to process
    :param sdir: starting directory for the process
    :returns: -1: impossible to create the process
    :returns: 0: the starting of the process was cancelled by the user
    :returns: 1: the process was properly started
    """
    ...

def step_into() -> bool:
    r"""Execute one instruction in the current thread. Other threads are kept suspended. \sq{Type, Asynchronous function - available as Request, Notification, dbg_step_into} 
            
    """
    ...

def step_over() -> bool:
    r"""Execute one instruction in the current thread, but without entering into functions. Others threads keep suspended. \sq{Type, Asynchronous function - available as Request, Notification, dbg_step_over} 
            
    """
    ...

def step_until_ret() -> bool:
    r"""Execute instructions in the current thread until a function return instruction is executed (aka "step out"). Other threads are kept suspended. \sq{Type, Asynchronous function - available as Request, Notification, dbg_step_until_ret} 
            
    """
    ...

def strlen(s: Any) -> Any:
    ...

def strstr(s1: Any, s2: Any) -> Any:
    ...

def substr(s: Any, x1: Any, x2: Any) -> Any:
    ...

def suspend_process() -> bool:
    r"""Suspend the process in the debugger. \sq{ Type,
    * Synchronous function (if in a notification handler)
    * Asynchronous function (everywhere else)
    * available as Request, Notification,
    * none (if in a notification handler)
    * dbg_suspend_process (everywhere else) }
    
    
    
    """
    ...

def suspend_thread(tid: thid_t) -> int:
    r"""Suspend thread. Suspending a thread may deadlock the whole application if the suspended was owning some synchronization objects. \sq{Type, Synchronous function - available as request, Notification, none (synchronous function)} 
            
    :param tid: thread id
    :returns: -1: network error
    :returns: 0: failed
    :returns: 1: ok
    """
    ...

def take_memory_snapshot(type: int) -> bool:
    r"""Take a memory snapshot of the running process. 
            
    :param type: specifies which snapshot we want (see SNAP_ Snapshot types)
    :returns: success
    """
    ...

def to_ea(seg: Any, off: Any) -> Any:
    r"""
        Return value of expression: ((seg<<4) + off)
        
    """
    ...

def toggle_bnot(ea: Any, n: Any) -> Any:
    r"""
        Toggle the bitwise not operator for the operand
    
        :param ea: linear address
        :param n: number of operand
            - 0 - the first operand
            - 1 - the second, third and all other operands
            - -1 - all operands
        
    """
    ...

def toggle_sign(ea: ida_idaapi.ea_t, n: int) -> bool:
    r"""Toggle sign of n-th operand. allowed values of n: 0-first operand, 1-other operands 
            
    """
    ...

def update_extra_cmt(ea: ida_idaapi.ea_t, what: int, str: str) -> bool:
    ...

def update_hidden_range(ea: Any, visible: Any) -> Any:
    r"""
        Set hidden range state
    
        :param ea:      any address belonging to the hidden range
        :param visible: new state of the range
    
        :returns: != 0 - ok
        
    """
    ...

def validate_idb_names(do_repair: Any = 0) -> Any:
    r"""
        check consistency of IDB name records
        :param do_repair: try to repair netnode header it TRUE
        :returns: number of inconsistent name records
        
    """
    ...

def value_is_float(var: Any) -> Any:
    ...

def value_is_func(var: Any) -> Any:
    ...

def value_is_int64(var: Any) -> Any:
    ...

def value_is_long(var: Any) -> Any:
    ...

def value_is_pvoid(var: Any) -> Any:
    ...

def value_is_string(var: Any) -> Any:
    ...

def wait_for_next_event(wfne: int, timeout: int) -> dbg_event_code_t:
    r"""Wait for the next event.
    This function (optionally) resumes the process execution, and waits for a debugger event until a possible timeout occurs.
    
    :param wfne: combination of Wait for debugger event flags constants
    :param timeout: number of seconds to wait, -1-infinity
    :returns: either an event_id_t (if > 0), or a dbg_event_code_t (if <= 0)
    """
    ...

def warning(message: Any) -> Any:
    r"""Display a message in a warning message box
    
    :param message: message to print
    """
    ...

def write_dbg_memory(ea: Any, data: Any) -> Any:
    r"""
        Write to debugger memory.
    
        :param ea: linear address
        :param data: string to write
        :returns: number of written bytes (-1 - network/debugger error)
    
        Thread-safe function (may be called only from the main thread and debthread)
        
    """
    ...

def writelong(handle: Any, dword: Any, mostfirst: Any) -> Any:
    ...

def writeshort(handle: Any, word: Any, mostfirst: Any) -> Any:
    ...

def writestr(handle: Any, s: Any) -> Any:
    ...

def xtol(s: Any) -> Any:
    ...

ABI_8ALIGN4: int  # 1
ABI_BIGARG_ALIGN: int  # 4
ABI_GCC_LAYOUT: int  # 128
ABI_HARD_FLOAT: int  # 32
ABI_HUGEARG_ALIGN: int  # 512
ABI_MAP_STKARGS: int  # 256
ABI_PACK_STKARGS: int  # 2
ABI_SET_BY_USER: int  # 64
ABI_STACK_LDBL: int  # 8
ABI_STACK_VARARGS: int  # 16
ADDSEG_FILLGAP: int  # 16
ADDSEG_NOSREG: int  # 1
ADDSEG_NOTRUNC: int  # 4
ADDSEG_OR_DIE: int  # 2
ADDSEG_QUIET: int  # 8
ADDSEG_SPARSE: int  # 32
AF2_DOEH: int  # 1
AF2_DORTTI: int  # 2
AF2_MACRO: int  # 4
AF2_MERGESTR: int  # 8
AF_ANORET: int  # 16384
AF_CHKUNI: int  # 262144
AF_CODE: int  # 1
AF_DATOFF: int  # 4194304
AF_DOCODE: int  # 1073741824
AF_DODATA: int  # 536870912
AF_DREFOFF: int  # 1048576
AF_FINAL: int  # 2147483648
AF_FIXUP: int  # 524288
AF_FLIRT: int  # 8388608
AF_FTAIL: int  # 256
AF_HFLIRT: int  # 67108864
AF_IMMOFF: int  # 2097152
AF_JFUNC: int  # 134217728
AF_JUMPTBL: int  # 4
AF_LVAR: int  # 512
AF_MARKCODE: int  # 2
AF_MEMFUNC: int  # 32768
AF_NULLSUB: int  # 268435456
AF_PROC: int  # 128
AF_PROCPTR: int  # 64
AF_PURDAT: int  # 8
AF_REGARG: int  # 2048
AF_SIGCMT: int  # 16777216
AF_SIGMLT: int  # 33554432
AF_STKARG: int  # 1024
AF_STRLIT: int  # 131072
AF_TRACE: int  # 4096
AF_TRFUNC: int  # 65536
AF_UNK: int  # 32
AF_USED: int  # 16
AF_VERSP: int  # 8192
APPT_16BIT: int  # 128
APPT_1THREAD: int  # 32
APPT_32BIT: int  # 256
APPT_CONSOLE: int  # 1
APPT_DRIVER: int  # 16
APPT_GRAPHIC: int  # 2
APPT_LIBRARY: int  # 8
APPT_MTHREAD: int  # 64
APPT_PROGRAM: int  # 4
AP_ALLOWDUPS: int  # 1
AP_ARRAY: int  # 8
AP_IDXBASEMASK: int  # 240
AP_IDXBIN: int  # 48
AP_IDXDEC: int  # 0
AP_IDXHEX: int  # 16
AP_IDXOCT: int  # 32
AP_INDEX: int  # 4
AP_SIGNED: int  # 2
ARGV: list  # []
AR_LONG: int  # 65
AR_STR: int  # 83
AU_CODE: int  # 20
AU_FINAL: int  # 200
AU_LIBF: int  # 60
AU_PROC: int  # 30
AU_UNK: int  # 10
AU_USED: int  # 40
BADADDR: int  # 18446744073709551615
BADSEL: int  # 18446744073709551615
BPLT_ABS: int  # 0
BPLT_REL: int  # 1
BPLT_SYM: int  # 2
BPTATTR_COND: int  # 6
BPTATTR_COUNT: int  # 4
BPTATTR_EA: int  # 1
BPTATTR_FLAGS: int  # 5
BPTATTR_PID: int  # 7
BPTATTR_SIZE: int  # 2
BPTATTR_TID: int  # 8
BPTATTR_TYPE: int  # 3
BPTCK_ACT: int  # 2
BPTCK_NO: int  # 0
BPTCK_NONE: int  # -1
BPTCK_YES: int  # 1
BPT_BRK: int  # 1
BPT_DEFAULT: int  # 12
BPT_ENABLED: int  # 8
BPT_EXEC: int  # 8
BPT_LOWCND: int  # 16
BPT_RDWR: int  # 3
BPT_SOFT: int  # 4
BPT_TRACE: int  # 2
BPT_TRACEON: int  # 32
BPT_TRACE_BBLK: int  # 256
BPT_TRACE_FUNC: int  # 128
BPT_TRACE_INSN: int  # 64
BPT_UPDMEM: int  # 4
BPT_WRITE: int  # 1
BPU_1B: int  # 1
BPU_2B: int  # 2
BPU_4B: int  # 4
BREAKPOINT: int  # 16
CHART_GEN_GDL: int  # 16384
CHART_NOLIBFUNCS: int  # 1024
CHART_PRINT_NAMES: int  # 4096
CHART_WINGRAPH: int  # 32768
CIC_FUNC: int  # 2
CIC_ITEM: int  # 1
CIC_SEGM: int  # 3
COMP_BC: int  # 2
COMP_BP: int  # 8
COMP_GNU: int  # 6
COMP_MASK: int  # 15
COMP_MS: int  # 1
COMP_UNK: int  # 0
COMP_VISAGE: int  # 7
COMP_WATCOM: int  # 3
DBFL_BAK: int  # 4
DBG_ERROR: int  # -1
DBG_TIMEOUT: int  # 0
DEFCOLOR: int  # 4294967295
DELIT_DELNAMES: int  # 2
DELIT_EXPAND: int  # 1
DELIT_SIMPLE: int  # 0
DEMNAM_CMNT: int  # 0
DEMNAM_FIRST: int  # 8
DEMNAM_GCC3: int  # 4
DEMNAM_MASK: int  # 3
DEMNAM_NAME: int  # 1
DEMNAM_NONE: int  # 2
DOPT_BPT_MSGS: int  # 16
DOPT_ENTRY_BPT: int  # 4096
DOPT_EXCDLG: int  # 24576
DOPT_INFO_BPT: int  # 512
DOPT_INFO_MSGS: int  # 256
DOPT_LIB_BPT: int  # 128
DOPT_LIB_MSGS: int  # 64
DOPT_LOAD_DINFO: int  # 32768
DOPT_REAL_MEMORY: int  # 1024
DOPT_REDO_STACK: int  # 2048
DOPT_SEGM_MSGS: int  # 1
DOPT_START_BPT: int  # 2
DOPT_THREAD_BPT: int  # 8
DOPT_THREAD_MSGS: int  # 4
DSTATE_NOTASK: int  # 0
DSTATE_RUN: int  # 1
DSTATE_RUN_WAIT_ATTACH: int  # 2
DSTATE_RUN_WAIT_END: int  # 3
DSTATE_SUSP: int  # -1
DT_TYPE: int  # 4026531840
ENFL_REGEX: int  # 1
EXCDLG_ALWAYS: int  # 24576
EXCDLG_NEVER: int  # 0
EXCDLG_UNKNOWN: int  # 8192
EXCEPTION: int  # 64
EXC_BREAK: int  # 1
EXC_HANDLE: int  # 2
E_NEXT: int  # 2000
E_PREV: int  # 1000
FF_0CHAR: int  # 3145728
FF_0ENUM: int  # 8388608
FF_0FOP: int  # 9437184
FF_0NUMB: int  # 6291456
FF_0NUMD: int  # 2097152
FF_0NUMH: int  # 1048576
FF_0NUMO: int  # 7340032
FF_0OFF: int  # 5242880
FF_0SEG: int  # 4194304
FF_0STK: int  # 11534336
FF_0STRO: int  # 10485760
FF_0VOID: int  # 0
FF_1CHAR: int  # 50331648
FF_1ENUM: int  # 134217728
FF_1FOP: int  # 150994944
FF_1NUMB: int  # 100663296
FF_1NUMD: int  # 33554432
FF_1NUMH: int  # 16777216
FF_1NUMO: int  # 117440512
FF_1OFF: int  # 83886080
FF_1SEG: int  # 67108864
FF_1STK: int  # 184549376
FF_1STRO: int  # 167772160
FF_1VOID: int  # 0
FF_ALIGN: int  # 2952790016
FF_ANYNAME: int  # 49152
FF_BYTE: int  # 0
FF_CODE: int  # 1536
FF_COMM: int  # 2048
FF_DATA: int  # 1024
FF_DOUBLE: int  # 2415919104
FF_DWORD: int  # 536870912
FF_FLOAT: int  # 2147483648
FF_FLOW: int  # 65536
FF_FUNC: int  # 268435456
FF_IMMD: int  # 1073741824
FF_IVL: int  # 256
FF_JUMP: int  # 2147483648
FF_LABL: int  # 32768
FF_LINE: int  # 8192
FF_NAME: int  # 16384
FF_OWORD: int  # 1879048192
FF_PACKREAL: int  # 2684354560
FF_QWORD: int  # 805306368
FF_REF: int  # 4096
FF_STRLIT: int  # 1342177280
FF_STRUCT: int  # 1610612736
FF_TAIL: int  # 512
FF_TBYTE: int  # 1073741824
FF_UNK: int  # 0
FF_WORD: int  # 268435456
FIXUPF_CREATED: int  # 8
FIXUPF_EXTDEF: int  # 2
FIXUPF_REL: int  # 1
FIXUPF_UNUSED: int  # 4
FIXUP_CUSTOM: int  # 32768
FIXUP_HI16: int  # 7
FIXUP_HI8: int  # 6
FIXUP_LOW16: int  # 9
FIXUP_LOW8: int  # 8
FIXUP_OFF16: int  # 1
FIXUP_OFF32: int  # 4
FIXUP_OFF64: int  # 12
FIXUP_OFF8: int  # 13
FIXUP_PTR32: int  # 3
FIXUP_PTR48: int  # 5
FIXUP_SEG16: int  # 2
FT_AIXAR: int  # 24
FT_AOUT: int  # 20
FT_AR: int  # 16
FT_BIN: int  # 2
FT_COFF: int  # 10
FT_COM: int  # 23
FT_COM_OLD: int  # 1
FT_DRV: int  # 3
FT_ELF: int  # 18
FT_EXE: int  # 22
FT_EXE_OLD: int  # 0
FT_HEX: int  # 5
FT_LE: int  # 8
FT_LOADER: int  # 17
FT_LX: int  # 7
FT_MACHO: int  # 25
FT_MEX: int  # 6
FT_NLM: int  # 9
FT_OMF: int  # 12
FT_OMFLIB: int  # 15
FT_PE: int  # 11
FT_PRC: int  # 21
FT_SREC: int  # 13
FT_W32RUN: int  # 19
FT_WIN: int  # 4
FT_ZIP: int  # 14
FUNCATTR_ARGSIZE: int  # 48
FUNCATTR_COLOR: int  # 64
FUNCATTR_END: int  # 8
FUNCATTR_FLAGS: int  # 16
FUNCATTR_FPD: int  # 56
FUNCATTR_FRAME: int  # 24
FUNCATTR_FRREGS: int  # 40
FUNCATTR_FRSIZE: int  # 32
FUNCATTR_OWNER: int  # 24
FUNCATTR_REFQTY: int  # 32
FUNCATTR_START: int  # 0
FUNC_BOTTOMBP: int  # 256
FUNC_FAR: int  # 2
FUNC_FRAME: int  # 16
FUNC_HIDDEN: int  # 64
FUNC_LIB: int  # 4
FUNC_LUMINA: int  # 65536
FUNC_NORET: int  # 1
FUNC_NORET_PENDING: int  # 512
FUNC_OUTLINE: int  # 131072
FUNC_PURGED_OK: int  # 16384
FUNC_SP_READY: int  # 1024
FUNC_STATIC: int  # 8
FUNC_TAIL: int  # 32768
FUNC_THUNK: int  # 128
FUNC_USERFAR: int  # 32
GENDSM_FORCE_CODE: int  # 1
GENDSM_MULTI_LINE: int  # 2
GENFLG_ASMINC: int  # 64
GENFLG_ASMTYPE: int  # 16
GENFLG_GENHTML: int  # 32
GENFLG_IDCTYPE: int  # 8
GENFLG_MAPDMNG: int  # 4
GENFLG_MAPLOC: int  # 8
GENFLG_MAPNAME: int  # 2
GENFLG_MAPSEG: int  # 1
GN_COLORED: int  # 2
GN_DEMANGLED: int  # 4
GN_ISRET: int  # 128
GN_LOCAL: int  # 64
GN_LONG: int  # 32
GN_NOT_ISRET: int  # 256
GN_SHORT: int  # 16
GN_STRICT: int  # 8
GN_VISIBLE: int  # 1
IDA_STATUS_READY: int  # 0
IDA_STATUS_THINKING: int  # 1
IDA_STATUS_WAITING: int  # 2
IDA_STATUS_WORK: int  # 3
IDB_COMPRESSED: int  # 2
IDB_PACKED: int  # 1
IDB_UNPACKED: int  # 0
IDCHK_ARG: int  # -1
IDCHK_KEY: int  # -2
IDCHK_MAX: int  # -3
IDCHK_OK: int  # 0
INFFL_ALLASM: int  # 2
INFFL_AUTO: int  # 1
INFFL_CHKOPS: int  # 32
INFFL_GRAPH_VIEW: int  # 128
INFFL_LOADIDC: int  # 4
INFFL_NMOPS: int  # 64
INFFL_NOUSER: int  # 8
INFFL_READONLY: int  # 16
INFORMATION: int  # 512
INF_ABIBITS: int  # 67
INF_AF: int  # 10
INF_AF2: int  # 11
INF_APPCALL_OPTIONS: int  # 68
INF_APPTYPE: int  # 7
INF_ASMTYPE: int  # 8
INF_BASEADDR: int  # 12
INF_BINPREF: int  # 47
INF_BIN_PREFIX_SIZE: int  # 47
INF_BORDER: int  # 46
INF_CC_CM: int  # 58
INF_CC_DEFALIGN: int  # 62
INF_CC_ID: int  # 57
INF_CC_SIZE_B: int  # 60
INF_CC_SIZE_E: int  # 61
INF_CC_SIZE_I: int  # 59
INF_CC_SIZE_L: int  # 64
INF_CC_SIZE_LDBL: int  # 66
INF_CC_SIZE_LL: int  # 65
INF_CC_SIZE_S: int  # 63
INF_CHANGE_COUNTER: int  # 4
INF_CMTFLAG: int  # 45
INF_CMTFLG: int  # 45
INF_CMT_INDENT: int  # 41
INF_COMMENT: int  # 41
INF_COMPILER: int  # 57
INF_DATABASE_CHANGE_COUNT: int  # 4
INF_DATATYPES: int  # 55
INF_DEMNAMES: int  # 38
INF_END_PRIVRANGE: int  # 28
INF_FILETYPE: int  # 5
INF_GENFLAGS: int  # 2
INF_HIGHOFF: int  # 24
INF_HIGH_OFF: int  # 24
INF_INDENT: int  # 40
INF_LENXREF: int  # 43
INF_LFLAGS: int  # 3
INF_LIMITER: int  # 46
INF_LISTNAMES: int  # 39
INF_LONG_DEMNAMES: int  # 37
INF_LONG_DN: int  # 37
INF_LOWOFF: int  # 23
INF_LOW_OFF: int  # 23
INF_MAIN: int  # 18
INF_MARGIN: int  # 42
INF_MAXREF: int  # 25
INF_MAX_AUTONAME_LEN: int  # 34
INF_MAX_EA: int  # 20
INF_MIN_EA: int  # 19
INF_MODEL: int  # 58
INF_NAMETYPE: int  # 35
INF_NETDELTA: int  # 29
INF_OMAX_EA: int  # 22
INF_OMIN_EA: int  # 21
INF_OSTYPE: int  # 6
INF_OUTFLAGS: int  # 44
INF_PREFFLAG: int  # 48
INF_PRIVRANGE_END_EA: int  # 28
INF_PRIVRANGE_START_EA: int  # 27
INF_PROCNAME: int  # 1
INF_REFCMTNUM: int  # 32
INF_REFCMTS: int  # 32
INF_SHORT_DEMNAMES: int  # 36
INF_SHORT_DN: int  # 36
INF_SIZEOF_ALGN: int  # 62
INF_SIZEOF_BOOL: int  # 60
INF_SIZEOF_ENUM: int  # 61
INF_SIZEOF_INT: int  # 59
INF_SIZEOF_LDBL: int  # 66
INF_SIZEOF_LLONG: int  # 65
INF_SIZEOF_LONG: int  # 64
INF_SIZEOF_SHORT: int  # 63
INF_SPECSEGS: int  # 9
INF_START_CS: int  # 14
INF_START_EA: int  # 16
INF_START_IP: int  # 15
INF_START_PRIVRANGE: int  # 27
INF_START_SP: int  # 17
INF_START_SS: int  # 13
INF_STRLIT_BREAK: int  # 50
INF_STRLIT_FLAGS: int  # 49
INF_STRLIT_PREF: int  # 53
INF_STRLIT_SERNUM: int  # 54
INF_STRLIT_ZEROES: int  # 51
INF_STRTYPE: int  # 52
INF_TYPE_XREFNUM: int  # 31
INF_TYPE_XREFS: int  # 31
INF_VERSION: int  # 0
INF_XREFFLAG: int  # 33
INF_XREFNUM: int  # 30
INF_XREFS: int  # 33
LFLG_64BIT: int  # 4
LFLG_COMPRESS: int  # 1024
LFLG_DBG_NOPATH: int  # 128
LFLG_FLAT_OFF32: int  # 16
LFLG_ILP32: int  # 4096
LFLG_IS_DLL: int  # 8
LFLG_KERNMODE: int  # 2048
LFLG_MSF: int  # 32
LFLG_PACK: int  # 512
LFLG_PC_FLAT: int  # 2
LFLG_PC_FPP: int  # 1
LFLG_SNAPSHOT: int  # 256
LFLG_WIDE_HBF: int  # 64
LIB_LOADED: int  # 128
LIB_UNLOADED: int  # 256
LMT_EMPTY: int  # 4
LMT_THICK: int  # 2
LMT_THIN: int  # 1
LN_AUTO: int  # 4
LN_NORMAL: int  # 1
LN_PUBLIC: int  # 2
LN_WEAK: int  # 8
MAXADDR: int  # 0
MOVE_SEGM_CHUNK: int  # -4
MOVE_SEGM_DEBUG: tuple  # (-8,)
MOVE_SEGM_IDP: int  # -3
MOVE_SEGM_INVAL: tuple  # (-11,)
MOVE_SEGM_LOADER: int  # -5
MOVE_SEGM_MAPPING: tuple  # (-10,)
MOVE_SEGM_ODD: int  # -6
MOVE_SEGM_OK: int  # 0
MOVE_SEGM_ORPHAN: tuple  # (-7,)
MOVE_SEGM_PARAM: int  # -1
MOVE_SEGM_ROOM: int  # -2
MOVE_SEGM_SOURCEFILES: tuple  # (-9,)
MSF_FIXONCE: int  # 8
MSF_LDKEEP: int  # 4
MSF_NOFIX: int  # 2
MSF_SILENT: int  # 1
MS_0TYPE: int  # 15728640
MS_1TYPE: int  # 251658240
MS_CLS: int  # 1536
MS_CODE: int  # 4026531840
MS_COMM: int  # 1046528
MS_VAL: int  # 255
NEF_CODE: int  # 256
NEF_FILL: int  # 16
NEF_FIRST: int  # 128
NEF_FLAT: int  # 1024
NEF_IMPS: int  # 32
NEF_MAN: int  # 8
NEF_NAME: int  # 4
NEF_RELOAD: int  # 512
NEF_RSCS: int  # 2
NEF_SEGS: int  # 1
NM_EA: int  # 6
NM_EA4: int  # 7
NM_EA8: int  # 8
NM_NAM_EA: int  # 5
NM_NAM_OFF: int  # 2
NM_PTR_EA: int  # 4
NM_PTR_OFF: int  # 1
NM_REL_EA: int  # 3
NM_REL_OFF: int  # 0
NM_SERIAL: int  # 10
NM_SHORT: int  # 9
NOTASK: int  # -2
OFILE_ASM: int  # 4
OFILE_DIF: int  # 5
OFILE_EXE: int  # 1
OFILE_IDC: int  # 2
OFILE_LST: int  # 3
OFILE_MAP: int  # 0
OFLG_GEN_ASSUME: int  # 512
OFLG_GEN_NULL: int  # 16
OFLG_GEN_ORG: int  # 256
OFLG_GEN_TRYBLKS: int  # 1024
OFLG_LZERO: int  # 128
OFLG_PREF_SEG: int  # 64
OFLG_SHOW_AUTO: int  # 4
OFLG_SHOW_PREF: int  # 32
OFLG_SHOW_VOID: int  # 2
OPND_OUTER: int  # 128
OSTYPE_MSDOS: int  # 1
OSTYPE_NETW: int  # 8
OSTYPE_OS2: int  # 4
OSTYPE_WIN: int  # 2
PDF_DEF_BASE: int  # 4
PDF_DEF_FWD: int  # 2
PDF_HEADER_CMT: int  # 8
PDF_INCL_DEPS: int  # 1
PREF_FNCOFF: int  # 2
PREF_PFXTRUNC: int  # 8
PREF_SEGADR: int  # 1
PREF_STACK: int  # 4
PROCESS_ATTACHED: int  # 1024
PROCESS_DETACHED: int  # 2048
PROCESS_EXITED: int  # 2
PROCESS_STARTED: int  # 1
PROCESS_SUSPENDED: int  # 4096
PRTYPE_1LINCMT: int  # 8192
PRTYPE_1LINE: int  # 0
PRTYPE_COLORED: int  # 2048
PRTYPE_CPP: int  # 16
PRTYPE_DEF: int  # 32
PRTYPE_METHODS: int  # 4096
PRTYPE_MULTI: int  # 1
PRTYPE_NOARGS: int  # 64
PRTYPE_NOARRS: int  # 128
PRTYPE_NOREGEX: int  # 1024
PRTYPE_NORES: int  # 256
PRTYPE_PRAGMA: int  # 4
PRTYPE_RESTORE: int  # 512
PRTYPE_SEMI: int  # 8
PRTYPE_TYPE: int  # 2
PT_FILE: int  # 65536
PT_HIGH: int  # 128
PT_LOWER: int  # 256
PT_NDC: int  # 2
PT_PACKMASK: int  # 112
PT_PAK1: int  # 16
PT_PAK16: int  # 80
PT_PAK2: int  # 32
PT_PAK4: int  # 48
PT_PAK8: int  # 64
PT_PAKDEF: int  # 0
PT_RAWARGS: int  # 1024
PT_REPLACE: int  # 512
PT_SIL: int  # 1
PT_SILENT: int  # 1
PT_STANDALONE: int  # 4194304
PT_TYP: int  # 4
PT_VAR: int  # 8
REFINFO_NOBASE: int  # 128
REFINFO_PASTEND: int  # 32
REFINFO_RVA: int  # 16
REFINFO_SIGNEDOP: int  # 512
REFINFO_SUBTRACT: int  # 256
REF_HIGH16: int  # 6
REF_HIGH8: int  # 5
REF_LOW16: int  # 4
REF_LOW8: int  # 3
REF_OFF16: int  # 1
REF_OFF32: int  # 2
REF_OFF64: int  # 9
REF_OFF8: int  # 10
SCF_ALLCMT: int  # 2
SCF_LINNUM: int  # 8
SCF_NOCMT: int  # 4
SCF_RPTCMT: int  # 1
SCF_SHHID_FUNC: int  # 64
SCF_SHHID_ITEM: int  # 32
SCF_SHHID_SEGM: int  # 128
SCF_TESTMODE: int  # 16
SEGATTR_ALIGN: int  # 40
SEGATTR_BITNESS: int  # 43
SEGATTR_COLOR: int  # 188
SEGATTR_COMB: int  # 41
SEGATTR_CS: int  # 64
SEGATTR_DS: int  # 80
SEGATTR_END: int  # 8
SEGATTR_ES: int  # 56
SEGATTR_FLAGS: int  # 44
SEGATTR_FS: int  # 88
SEGATTR_GS: int  # 96
SEGATTR_ORGBASE: int  # 32
SEGATTR_PERM: int  # 42
SEGATTR_SEL: int  # 48
SEGATTR_SS: int  # 72
SEGATTR_START: int  # 0
SEGATTR_TYPE: int  # 184
SEGMOD_KEEP: int  # 2
SEGMOD_KILL: int  # 1
SEGMOD_SILENT: int  # 4
SEG_ABSSYM: int  # 10
SEG_BSS: int  # 9
SEG_CODE: int  # 2
SEG_COMM: int  # 11
SEG_DATA: int  # 3
SEG_GRP: int  # 6
SEG_IMEM: int  # 12
SEG_IMP: int  # 4
SEG_NORM: int  # 0
SEG_NULL: int  # 7
SEG_UNDF: int  # 8
SEG_XTRN: int  # 1
SETPROC_IDB: int  # 0
SETPROC_LOADER: int  # 1
SETPROC_LOADER_NON_FATAL: int  # 2
SETPROC_USER: int  # 3
SFL_COMORG: int  # 1
SFL_DEBUG: int  # 8
SFL_HIDDEN: int  # 4
SFL_HIDETYPE: int  # 32
SFL_LOADER: int  # 16
SFL_OBOK: int  # 2
SIZE_MAX: int  # 18446744073709551615
SN_AUTO: int  # 32
SN_CHECK: int  # 0
SN_LOCAL: int  # 512
SN_NOCHECK: int  # 1
SN_NOLIST: int  # 128
SN_NON_AUTO: int  # 64
SN_NON_PUBLIC: int  # 4
SN_NON_WEAK: int  # 16
SN_NOWARN: int  # 256
SN_PUBLIC: int  # 2
SN_WEAK: int  # 8
SR_auto: int  # 3
SR_autostart: int  # 4
SR_inherit: int  # 1
SR_user: int  # 2
STEP: int  # 32
STRF_AUTO: int  # 2
STRF_COMMENT: int  # 16
STRF_GEN: int  # 1
STRF_SAVECASE: int  # 32
STRF_SERIAL: int  # 4
STRF_UNICODE: int  # 8
STRLYT_MASK: int  # 252
STRLYT_PASCAL1: int  # 1
STRLYT_PASCAL2: int  # 2
STRLYT_PASCAL4: int  # 3
STRLYT_SHIFT: int  # 2
STRLYT_TERMCHR: int  # 0
STRTYPE_C: int  # 0
STRTYPE_C16: int  # 1
STRTYPE_C_16: int  # 1
STRTYPE_C_32: int  # 2
STRTYPE_LEN2: int  # 8
STRTYPE_LEN2_16: int  # 9
STRTYPE_LEN4: int  # 12
STRTYPE_LEN4_16: int  # 13
STRTYPE_PASCAL: int  # 4
STRTYPE_PASCAL_16: int  # 5
STRTYPE_TERMCHR: int  # 0
STRWIDTH_1B: int  # 0
STRWIDTH_2B: int  # 1
STRWIDTH_4B: int  # 2
STRWIDTH_MASK: int  # 3
STT_MM: int  # 1
STT_VA: int  # 0
ST_ALREADY_LOGGED: int  # 4
ST_OVER_DEBUG_SEG: int  # 1
ST_OVER_LIB_FUNC: int  # 2
ST_SKIP_LOOPS: int  # 8
SW_SEGXRF: int  # 1
SW_XRFFNC: int  # 4
SW_XRFMRK: int  # 2
SW_XRFVAL: int  # 8
TEV_BPT: int  # 4
TEV_CALL: int  # 2
TEV_EVENT: int  # 6
TEV_INSN: int  # 1
TEV_MEM: int  # 5
TEV_NONE: int  # 0
TEV_RET: int  # 3
THREAD_EXITED: int  # 8
THREAD_STARTED: int  # 4
TINFO_DEFINITE: int  # 1
TINFO_DELAYFUNC: int  # 2
TINFO_GUESSED: int  # 0
TRACE_FUNC: int  # 2
TRACE_INSN: int  # 1
TRACE_STEP: int  # 0
WFNE_ANY: int  # 1
WFNE_CONT: int  # 8
WFNE_NOWAIT: int  # 16
WFNE_SILENT: int  # 4
WFNE_SUSP: int  # 2
WORDMASK: int  # 18446744073709551615
XREF_USER: int  # 32
dr_I: int  # 5
dr_O: int  # 1
dr_R: int  # 3
dr_T: int  # 4
dr_W: int  # 2
fl_CF: int  # 16
fl_CN: int  # 17
fl_F: int  # 21
fl_JF: int  # 18
fl_JN: int  # 19
ida_auto: module
ida_bytes: module
ida_dbg: module
ida_diskio: module
ida_entry: module
ida_expr: module
ida_fixup: module
ida_frame: module
ida_funcs: module
ida_gdl: module
ida_ida: _module_wrapper_t
ida_idaapi: module
ida_idc: module
ida_idd: module
ida_idp: module
ida_kernwin: module
ida_lines: module
ida_loader: module
ida_moves: module
ida_nalt: module
ida_name: module
ida_netnode: module
ida_offset: module
ida_pro: module
ida_search: module
ida_segment: module
ida_segregs: module
ida_typeinf: module
ida_ua: module
ida_xref: module
o_cond: int  # 14
o_crb: int  # 12
o_creg: int  # 11
o_creglist: int  # 10
o_crf: int  # 11
o_crreg: int  # 10
o_dbreg: int  # 9
o_dcr: int  # 13
o_displ: int  # 4
o_far: int  # 6
o_fpreg: int  # 11
o_fpreglist: int  # 12
o_idpspec0: int  # 8
o_idpspec1: int  # 9
o_idpspec2: int  # 10
o_idpspec3: int  # 11
o_idpspec4: int  # 12
o_idpspec5: int  # 13
o_imm: int  # 5
o_mem: int  # 2
o_mmxreg: int  # 12
o_near: int  # 7
o_phrase: int  # 3
o_reg: int  # 1
o_reglist: int  # 9
o_shmbme: int  # 10
o_spr: int  # 8
o_text: int  # 13
o_trreg: int  # 8
o_twofpr: int  # 9
o_void: int  # 0
o_xmmreg: int  # 13
os: module  # <module 'os' from '/usr/lib/python3.10/os.py'>
print_function: _Feature
re: module  # <module 're' from '/usr/lib/python3.10/re.py'>
saAbs: int  # 0
saGroup: int  # 7
saRel32Bytes: int  # 8
saRel4K: int  # 6
saRel64Bytes: int  # 9
saRelByte: int  # 1
saRelDble: int  # 5
saRelPage: int  # 4
saRelPara: int  # 3
saRelQword: int  # 10
saRelWord: int  # 2
scCommon: int  # 6
scPriv: int  # 0
scPub: int  # 2
scPub2: int  # 4
scPub3: int  # 7
scStack: int  # 5
struct: module
sys: module  # <module 'sys' (built-in)>
time: module  # <module 'time' (built-in)>
types: module