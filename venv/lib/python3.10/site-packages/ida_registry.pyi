from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Registry related functions.

IDA uses the registry to store global configuration options that must persist after IDA has been closed.
On Windows, IDA uses the Windows registry directly. On Unix systems, the registry is stored in a file (typically ~/.idapro/ida.reg).
The root key for accessing IDA settings in the registry is defined by ROOT_KEY_NAME. 
    
"""

def reg_data_type(name: str, subkey: str = None) -> regval_type_t:
    r"""Get data type of a given value. 
            
    :param name: value name
    :param subkey: key name
    :returns: false if the [key+]value doesn't exist
    """
    ...

def reg_delete(name: str, subkey: str = None) -> bool:
    r"""Delete a value from the registry. 
            
    :param name: value name
    :param subkey: parent key
    :returns: success
    """
    ...

def reg_delete_subkey(name: str) -> bool:
    r"""Delete a key from the registry.
    
    """
    ...

def reg_delete_tree(name: str) -> bool:
    r"""Delete a subtree from the registry.
    
    """
    ...

def reg_exists(name: str, subkey: str = None) -> bool:
    r"""Is there already a value with the given name? 
            
    :param name: value name
    :param subkey: parent key
    """
    ...

def reg_read_binary(name: str, subkey: str = None) -> Any:
    r"""Read binary data from the registry. 
            
    :param name: value name
    :param subkey: key name
    :returns: false if 'data' is not large enough to hold all data present. in this case 'data' is left untouched.
    """
    ...

def reg_read_bool(name: str, defval: bool, subkey: str = None) -> bool:
    r"""Read boolean value from the registry. 
            
    :param name: value name
    :param defval: default value
    :param subkey: key name
    :returns: boolean read from registry, or 'defval' if the read failed
    """
    ...

def reg_read_int(name: str, defval: int, subkey: str = None) -> int:
    r"""Read integer value from the registry. 
            
    :param name: value name
    :param defval: default value
    :param subkey: key name
    :returns: the value read from the registry, or 'defval' if the read failed
    """
    ...

def reg_read_string(name: str, subkey: str = None, _def: str = None) -> Any:
    r"""Read a string from the registry. 
            
    :param name: value name
    :param subkey: key name
    :returns: success
    """
    ...

def reg_read_strlist(subkey: str) -> List[str]:
    r"""Retrieve all string values associated with the given key.
    
    :param subkey: a key from which to read the list of items
    :returns: the list of items
    """
    ...

def reg_subkey_exists(name: str) -> bool:
    r"""Is there already a key with the given name?
    
    """
    ...

def reg_subkey_subkeys(name: str) -> Any:
    r"""Get all subkey names of given key.
    
    """
    ...

def reg_subkey_values(name: str) -> Any:
    r"""Get all value names under given key.
    
    """
    ...

def reg_update_filestrlist(subkey: str, add: str, maxrecs: size_t, rem: str = None) -> None:
    r"""Update registry with a file list. Case sensitivity will vary depending on the target OS. 
            
    """
    ...

def reg_update_strlist(subkey: str, add: Any, maxrecs: int, rem: Any = None, ignorecase: bool = False) -> Any:
    r"""Add and/or remove items from the list, and possibly trim that list.
    
    :param subkey: the key under which the list is located
    :param add: an item to add to the list, or None
    :param maxrecs: the maximum number of items the list should hold
    :param rem: an item to remove from the list, or None
    :param ignorecase: ignore case for 'add' and 'rem'
    """
    ...

def reg_write_binary(name: str, py_bytes: Any, subkey: str = None) -> Any:
    r"""Write binary data to the registry. 
            
    :param name: value name
    :param subkey: key name
    """
    ...

def reg_write_bool(name: str, value: int, subkey: str = None) -> None:
    r"""Write boolean value to the registry. 
            
    :param name: value name
    :param value: boolean to write (nonzero = true)
    :param subkey: key name
    """
    ...

def reg_write_int(name: str, value: int, subkey: str = None) -> None:
    r"""Write integer value to the registry. 
            
    :param name: value name
    :param value: value to write
    :param subkey: key name
    """
    ...

def reg_write_string(name: str, utf8: str, subkey: str = None) -> None:
    r"""Write a string to the registry. 
            
    :param name: value name
    :param utf8: utf8-encoded string
    :param subkey: key name
    """
    ...

def reg_write_strlist(items: List[str], subkey: str) -> Any:
    r"""Write string values associated with the given key.
    
    :param items: the list of items to write
    :param subkey: a key under which to write the list of items
    """
    ...

def set_registry_name(name: str) -> bool:
    ...

HVUI_REGISTRY_NAME: str  # hvui
IDA_REGISTRY_NAME: str  # ida
ROOT_KEY_NAME: str  # Software\Hex-Rays\IDA
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
reg_binary: int  # 3
reg_dword: int  # 4
reg_sz: int  # 1
reg_unknown: int  # 0
weakref: module