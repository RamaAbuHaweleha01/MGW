from typing import Any, Optional, List, Dict, Tuple, Callable, Union

def create_undo_point(args: Any) -> bool:
    r"""Create a new restore point. The user can undo to this point in the future. 
            
    :param bytes: body of the record for UNDO_ACTION_START
    :param size: size of the record for UNDO_ACTION_START
    :returns: success; fails if undo is disabled
    """
    ...

def get_redo_action_label() -> str:
    r"""Get the label of the action that will be redone. This function returns the text that can be displayed in the redo menu 
            
    :returns: success
    """
    ...

def get_undo_action_label() -> str:
    r"""Get the label of the action that will be undone. This function returns the text that can be displayed in the undo menu 
            
    :returns: success
    """
    ...

def perform_redo() -> bool:
    r"""Perform redo. 
            
    :returns: success
    """
    ...

def perform_undo() -> bool:
    r"""Perform undo. 
            
    :returns: success
    """
    ...

SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module