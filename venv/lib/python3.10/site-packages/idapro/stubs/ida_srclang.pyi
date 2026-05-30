from typing import Any, Optional, List, Dict, Tuple, Callable, Union

r"""Third-party compiler support.

"""

def get_parser_option(parser_name: str, option_name: str) -> str:
    r"""Get option for the parser with the specified name 
            
    :param parser_name: name of the target parser
    :param option_name: parser option name
    :returns: success
    """
    ...

def get_selected_parser_name() -> str:
    r"""Get current parser name. 
            
    :returns: success
    """
    ...

def parse_decls_for_srclang(lang: srclang_t, til: til_t, input: str, is_path: bool) -> int:
    r"""Parse type declarations in the specified language 
            
    :param lang: the source language(s) expected in the input
    :param til: type library to store the types
    :param input: input source. can be a file path or decl string
    :param is_path: true if input parameter is a path to a source file, false if the input is an in-memory source snippet
    :returns: -1: no parser was found that supports the given source language(s)
    :returns: else: the number of errors encountered in the input source
    """
    ...

def parse_decls_with_parser(parser_name: str, til: til_t, input: str, is_path: bool) -> int:
    r"""Parse type declarations using the parser with the specified name 
            
    :param parser_name: name of the target parser
    :param til: type library to store the types
    :param input: input source. can be a file path or decl string
    :param is_path: true if input parameter is a path to a source file, false if the input is an in-memory source snippet
    :returns: -1: no parser was found with the given name
    :returns: else: the number of errors encountered in the input source
    """
    ...

def parse_decls_with_parser_ext(parser_name: str, til: til_t, input: str, hti_flags: int) -> int:
    r"""Parse type declarations using the parser with the specified name 
            
    :param parser_name: name of the target parser
    :param til: type library to store the types
    :param input: input source. can be a file path or decl string
    :param hti_flags: combination of Type formatting flags
    :returns: -1: no parser was found with the given name
    :returns: else: the number of errors encountered in the input source
    """
    ...

def select_parser_by_name(name: str) -> bool:
    r"""Set the parser with the given name as the current parser. Pass nullptr or an empty string to select the default parser. 
            
    :returns: false if no parser was found with the given name
    """
    ...

def select_parser_by_srclang(lang: srclang_t) -> bool:
    r"""Set the parser that supports the given language(s) as the current parser. The selected parser must support all languages specified by the given srclang_t. 
            
    :returns: false if no such parser was found
    """
    ...

def set_parser_argv(parser_name: str, argv: str) -> int:
    r"""Set the command-line args to use for invocations of the parser with the given name 
            
    :param parser_name: name of the target parser
    :param argv: argument list
    :returns: -1: no parser was found with the given name
    :returns: -2: the operation is not supported by the given parser
    :returns: 0: success
    """
    ...

def set_parser_option(parser_name: str, option_name: str, option_value: str) -> bool:
    r"""Set option for the parser with the specified name 
            
    :param parser_name: name of the target parser
    :param option_name: parser option name
    :param option_value: parser option value
    :returns: success
    """
    ...

SRCLANG_C: int  # 1
SRCLANG_CPP: int  # 2
SRCLANG_GO: int  # 16
SRCLANG_OBJC: int  # 4
SRCLANG_SWIFT: int  # 8
SWIG_PYTHON_LEGACY_BOOL: int  # 1
annotations: _Feature
ida_idaapi: module
weakref: module