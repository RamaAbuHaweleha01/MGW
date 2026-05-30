from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Functions that deal with C-like expressions and built-in IDC language.

Functions marked THREAD_SAFE may be called from any thread. No simultaneous calls should be made for the same variable. We protect only global structures, individual variables must be protected manually. 
    
"""

class highlighter_cbs_t:
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
    def cur_block_state(self) -> int:
        ...
    def prev_block_state(self) -> int:
        ...
    def set_block_state(self, arg0: int) -> None:
        ...
    def set_style(self, arg0: int, arg1: int, arg2: syntax_highlight_style) -> None:
        ...

class idc_global_t:
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

class idc_value_t:
    @property
    def e(self) -> Any: ...
    @property
    def funcidx(self) -> Any: ...
    @property
    def i64(self) -> Any: ...
    @property
    def num(self) -> Any: ...
    @property
    def obj(self) -> Any: ...
    @property
    def pvoid(self) -> Any: ...
    @property
    def reserve(self) -> Any: ...
    @property
    def str(self) -> Any: ...
    @property
    def vtype(self) -> Any: ...
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
    def c_str(self) -> str:
        r"""VT_STR
        
        """
        ...
    def clear(self) -> None:
        r"""See free_idcv()
        
        """
        ...
    def create_empty_string(self) -> None:
        ...
    def is_convertible(self) -> bool:
        r"""Convertible types are VT_LONG, VT_FLOAT, VT_INT64, and VT_STR.
        
        """
        ...
    def is_integral(self) -> bool:
        r"""Does value represent a whole number? 
                
        """
        ...
    def is_zero(self) -> bool:
        r"""Does value represent the integer 0?
        
        """
        ...
    def qstr(self) -> str:
        r"""VT_STR
        
        """
        ...
    def set_float(self, f: fpvalue_t) -> None:
        ...
    def set_int64(self, v: int64) -> None:
        ...
    def set_long(self, v: int) -> None:
        ...
    def set_pvoid(self, p: void) -> None:
        ...
    def set_string(self, args: Any) -> None:
        ...
    def swap(self, v: idc_value_t) -> None:
        r"""Set this = r and v = this.
        
        """
        ...
    def u_str(self) -> uchar:
        r"""VT_STR
        
        """
        ...

class idc_values_t:
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
    def __getitem__(self, i: size_t) -> idc_value_t:
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
    def __setitem__(self, i: size_t, v: idc_value_t) -> None:
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
    def append(self, x: idc_value_t) -> None:
        ...
    def at(self, _idx: size_t) -> idc_value_t:
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
    def extend(self, x: idc_values_t) -> None:
        ...
    def extract(self) -> idc_value_t:
        ...
    def front(self) -> Any:
        ...
    def grow(self, args: Any) -> None:
        ...
    def inject(self, s: idc_value_t, len: size_t) -> None:
        ...
    def insert(self, it: idc_value_t, x: idc_value_t) -> qvector:
        ...
    def pop_back(self) -> None:
        ...
    def push_back(self, args: Any) -> idc_value_t:
        ...
    def qclear(self) -> None:
        ...
    def reserve(self, cnt: size_t) -> None:
        ...
    def resize(self, args: Any) -> None:
        ...
    def size(self) -> int:
        ...
    def swap(self, r: idc_values_t) -> None:
        ...
    def truncate(self) -> None:
        ...

def add_idc_class(name: str, super: idc_class_t = None) -> idc_class_t:
    r"""Create a new IDC class. 
            
    :param name: name of the new class
    :param super: the base class for the new class. if the new class is not based on any other class, pass nullptr
    :returns: pointer to the created class. If such a class already exists, a pointer to it will be returned. Pointers to other existing classes may be invalidated by this call.
    """
    ...

def add_idc_func(name: Any, fp: Any, args: Any, defvals: Any = ..., flags: Any = 0) -> Any:
    r"""Add an IDC function. This function does not modify the predefined kernel functions. Example: 
         error_t idaapi myfunc5(idc_value_t *argv, idc_value_t *res)
        
          msg("myfunc is called with arg0=%a and arg1=%s\n", argv[0].num, argv[1].str);
          res->num = 5;     // let's return 5
          return eOk;
        
         const char myfunc5_args[] = { VT_LONG, VT_STR, 0 };
         const ext_idcfunc_t myfunc_desc = { "MyFunc5", myfunc5, myfunc5_args, nullptr, 0, EXTFUN_BASE };
        
         after this:
        
        
         there is a new IDC function which can be called like this:
         "test");
    
    
            
    :returns: success
    """
    ...

def add_idc_gvar(name: str) -> idc_value_t:
    r"""Add global IDC variable. 
            
    :param name: name of the global variable
    :returns: pointer to the created variable or existing variable. NB: the returned pointer is valid until a new global var is added.
    """
    ...

def call_idc_func__(*args: Any, **kwargs: Any) -> Any:
    ...

def compile_idc_file(nonnul_line: str) -> str:
    ...

def compile_idc_snippet(func: str, text: str, resolver: idc_resolver_t = None, only_safe_funcs: bool = False) -> str:
    r"""Compile text with IDC statements. 
            
    :param func: name of the function to create out of the snippet
    :param text: text to compile
    :param resolver: callback object to get values of undefined variables This object will be called if IDC function contains references to undefined variables. May be nullptr.
    :param only_safe_funcs: if true, any calls to functions without EXTFUN_SAFE flag will lead to a compilation error.
    :returns: true: ok
    :returns: false: error, see errbuf
    """
    ...

def compile_idc_text(nonnul_line: str) -> str:
    ...

def copy_idcv(dst: idc_value_t, src: idc_value_t) -> error_t:
    r"""Copy 'src' to 'dst'. For idc objects only a reference is copied. 
            
    """
    ...

def create_idcv_ref(ref: idc_value_t, v: idc_value_t) -> bool:
    r"""Create a variable reference. Currently only references to global variables can be created. 
            
    :param ref: ptr to the result
    :param v: variable to reference
    :returns: success
    """
    ...

def deep_copy_idcv(dst: idc_value_t, src: idc_value_t) -> error_t:
    r"""Deep copy an IDC object. This function performs deep copy of idc objects. If 'src' is not an object, copy_idcv() will be called 
            
    """
    ...

def del_idc_func(name: Any) -> Any:
    r"""Delete an IDC function 
            
    """
    ...

def del_idcv_attr(obj: idc_value_t, attr: str) -> error_t:
    r"""Delete an object attribute. 
            
    :param obj: variable that holds an object reference
    :param attr: attribute name
    :returns: error code, eOk on success
    """
    ...

def deref_idcv(v: idc_value_t, vref_flags: int) -> idc_value_t:
    r"""Dereference a VT_REF variable. 
            
    :param v: variable to dereference
    :param vref_flags: Dereference IDC variable flags
    :returns: pointer to the dereference result or nullptr. If returns nullptr, qerrno is set to eExecBadRef "Illegal variable reference"
    """
    ...

def eval_expr(rv: idc_value_t, where: ida_idaapi.ea_t, line: str) -> str:
    r"""Compile and calculate an expression. 
            
    :param rv: pointer to the result
    :param where: the current linear address in the addressing space of the program being disassembled. If will be used to resolve names of local variables etc. if not applicable, then should be BADADDR.
    :param line: the expression to evaluate
    :returns: true: ok
    :returns: false: error, see errbuf
    """
    ...

def eval_idc_expr(rv: idc_value_t, where: ida_idaapi.ea_t, line: str) -> str:
    r"""Same as eval_expr(), but will always use the IDC interpreter regardless of the currently installed extlang. 
            
    """
    ...

def exec_idc_script(result: idc_value_t, path: str, func: str, args: idc_value_t, argsnum: size_t) -> str:
    r"""Compile and execute IDC function(s) from file. 
            
    :param result: ptr to idc_value_t to hold result of the function. If execution fails, this variable will contain the exception information. You may pass nullptr if you are not interested in the returned value.
    :param path: text file containing text of IDC functions
    :param func: function name to execute
    :param args: array of parameters
    :param argsnum: number of parameters to pass to 'fname' This number should be equal to number of parameters the function expects.
    :returns: true: ok
    :returns: false: error, see errbuf
    """
    ...

def exec_system_script(file: str, complain_if_no_file: bool = True) -> bool:
    r"""Compile and execute "main" function from system file. 
            
    :param file: file name with IDC function(s). The file will be searched using get_idc_filename().
    :param complain_if_no_file: * 1: display warning if the file is not found
    * 0: don't complain if file doesn't exist
    :returns: 1: ok, file is compiled and executed
    :returns: 0: failure, compilation or execution error, warning is displayed
    """
    ...

def find_idc_class(name: str) -> idc_class_t:
    r"""Find an existing IDC class by its name. 
            
    :param name: name of the class
    :returns: pointer to the class or nullptr. The returned pointer is valid until a new call to add_idc_class()
    """
    ...

def find_idc_func(prefix: str, n: int = 0) -> str:
    ...

def find_idc_gvar(name: str) -> idc_value_t:
    r"""Find an existing global IDC variable by its name. 
            
    :param name: name of the global variable
    :returns: pointer to the variable or nullptr. NB: the returned pointer is valid until a new global var is added. FIXME: it is difficult to use this function in a thread safe manner
    """
    ...

def first_idcv_attr(obj: idc_value_t) -> str:
    ...

def free_idcv(v: idc_value_t) -> None:
    r"""Free storage used by VT_STR/VT_OBJ IDC variables. After this call the variable has a numeric value 0 
            
    """
    ...

def get_idc_filename(file: str) -> str:
    r"""Get full name of IDC file name. Search for file in list of include directories, IDCPATH directory and system directories. 
            
    :param file: file name without full path
    :returns: nullptr is file not found. otherwise returns pointer to buf
    """
    ...

def get_idcv_attr(res: idc_value_t, obj: idc_value_t, attr: str, may_use_getattr: bool = False) -> error_t:
    r"""Get an object attribute. 
            
    :param res: buffer for the attribute value
    :param obj: variable that holds an object reference. if obj is nullptr it searches global variables, then user functions
    :param attr: attribute name
    :param may_use_getattr: may call getattr functions to calculate the attribute if it does not exist
    :returns: error code, eOk on success
    """
    ...

def get_idcv_class_name(obj: idc_value_t) -> str:
    r"""Retrieves the IDC object class name. 
            
    :param obj: class instance variable
    :returns: error code, eOk on success
    """
    ...

def get_idcv_slice(res: idc_value_t, v: idc_value_t, i1: int, i2: int, flags: int = 0) -> error_t:
    r"""Get slice. 
            
    :param res: output variable that will contain the slice
    :param v: input variable (string or object)
    :param i1: slice start index
    :param i2: slice end index (excluded)
    :param flags: IDC variable slice flags or 0
    :returns: eOk if success
    """
    ...

def idcv_float(v: idc_value_t) -> error_t:
    r"""Convert IDC variable to a floating point.
    
    """
    ...

def idcv_int64(v: idc_value_t) -> error_t:
    r"""Convert IDC variable to a 64bit number. 
            
    :returns: v = 0 if impossible to convert to int64
    """
    ...

def idcv_long(v: idc_value_t) -> error_t:
    r"""Convert IDC variable to a long (32/64bit) number. 
            
    :returns: v = 0 if impossible to convert to long
    """
    ...

def idcv_num(v: idc_value_t) -> error_t:
    r"""Convert IDC variable to a long number. 
            
    :returns: * v = 0 if IDC variable = "false" string
    * v = 1 if IDC variable = "true" string
    * v = number if IDC variable is number or string containing a number
    * eTypeConflict if IDC variable = empty string
    """
    ...

def idcv_object(v: idc_value_t, icls: idc_class_t = None) -> error_t:
    r"""Create an IDC object. The original value of 'v' is discarded (freed). 
            
    :param v: variable to hold the object. any previous value will be cleaned
    :param icls: ptr to the desired class. nullptr means "object" class this ptr must be returned by add_idc_class() or find_idc_class()
    :returns: always eOk
    """
    ...

def idcv_string(v: idc_value_t) -> error_t:
    r"""Convert IDC variable to a text string.
    
    """
    ...

def last_idcv_attr(obj: idc_value_t) -> str:
    ...

def move_idcv(dst: idc_value_t, src: idc_value_t) -> error_t:
    r"""Move 'src' to 'dst'. This function is more effective than copy_idcv since it never copies big amounts of data. 
            
    """
    ...

def next_idcv_attr(obj: idc_value_t, attr: str) -> str:
    ...

def prev_idcv_attr(obj: idc_value_t, attr: str) -> str:
    ...

def print_idcv(v: idc_value_t, name: str = None, indent: int = 0) -> str:
    r"""Get text representation of idc_value_t.
    
    """
    ...

def py_add_idc_func(name: str, fp_ptr: size_t, args: str, defvals: idc_values_t, flags: int) -> bool:
    ...

def py_get_call_idc_func() -> int:
    ...

def pyw_convert_defvals(out: idc_values_t, py_seq: Any) -> bool:
    ...

def pyw_register_idc_func(name: str, args: str, py_fp: Any) -> int:
    ...

def pyw_unregister_idc_func(ctxptr: size_t) -> bool:
    ...

def set_header_path(path: str, add: bool) -> bool:
    r"""Set or append a header path. IDA looks for the include files in the appended header paths, then in the ida executable directory. 
            
    :param path: list of directories to add (separated by ';') may be nullptr, in this case nothing is added
    :param add: true: append. false: remove old paths.
    :returns: true: success
    :returns: false: no memory
    """
    ...

def set_idcv_attr(obj: idc_value_t, attr: str, value: idc_value_t, may_use_setattr: bool = False) -> error_t:
    r"""Set an object attribute. 
            
    :param obj: variable that holds an object reference. if obj is nullptr then it tries to modify a global variable with the attribute name
    :param attr: attribute name
    :param value: new attribute value
    :param may_use_setattr: may call setattr functions for the class
    :returns: error code, eOk on success
    """
    ...

def set_idcv_slice(v: idc_value_t, i1: int, i2: int, _in: idc_value_t, flags: int = 0) -> error_t:
    r"""Set slice. 
            
    :param v: variable to modify (string or object)
    :param i1: slice start index
    :param i2: slice end index (excluded)
    :param flags: IDC variable slice flags or 0
    :returns: eOk on success
    """
    ...

def swap_idcvs(v1: idc_value_t, v2: idc_value_t) -> None:
    r"""Swap 2 variables.
    
    """
    ...

def throw_idc_exception(r: idc_value_t, desc: str) -> error_t:
    r"""Create an idc execution exception object. This helper function can be used to return an exception from C++ code to IDC. In other words this function can be called from idc_func_t() callbacks. Sample usage: if ( !ok ) return throw_idc_exception(r, "detailed error msg"); 
            
    :param r: object to hold the exception object
    :param desc: exception description
    :returns: eExecThrow
    """
    ...

CPL_DEL_MACROS: int  # 1
CPL_ONLY_SAFE: int  # 4
CPL_USE_LABELS: int  # 2
EXTFUN_BASE: int  # 1
EXTFUN_NORET: int  # 2
EXTFUN_SAFE: int  # 4
HF_COMMENT: int  # 5
HF_DEFAULT: int  # 0
HF_KEYWORD1: int  # 1
HF_KEYWORD2: int  # 2
HF_KEYWORD3: int  # 3
HF_MAX: int  # 12
HF_NUMBER: int  # 7
HF_PREPROC: int  # 6
HF_STRING: int  # 4
HF_USER1: int  # 8
HF_USER2: int  # 9
HF_USER3: int  # 10
HF_USER4: int  # 11
IDC_LANG_EXT: str  # idc
SWIG_PYTHON_LEGACY_BOOL: int  # 1
VARSLICE_SINGLE: int  # 1
VREF_COPY: int  # 2
VREF_LOOP: int  # 0
VREF_ONCE: int  # 1
VT_FLOAT: int  # 3
VT_FUNC: int  # 6
VT_INT64: int  # 9
VT_LONG: int  # 2
VT_OBJ: int  # 5
VT_PVOID: int  # 8
VT_REF: int  # 10
VT_STR: int  # 7
VT_WILD: int  # 4
annotations: _Feature
ctypes: module
eExecThrow: int  # 90
ida_idaapi: module
types: module
weakref: module