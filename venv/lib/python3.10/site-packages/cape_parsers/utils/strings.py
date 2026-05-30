# Copyright (C) 2010-2015 Cuckoo Foundation, Optiv, Inc. (brad.spengler@optiv.com)
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.
import logging
from pathlib import Path

try:
    import re2 as re

    HAVE_RE2 = True
except ImportError:
    import re

    HAVE_RE2 = False


log = logging.getLogger(__name__)


def bytes2str(convert):
    """Converts bytes to string
    @param convert: string as bytes.
    @return: string.
    """
    if isinstance(convert, bytes):
        try:
            convert = convert.decode()
        except UnicodeDecodeError:
            convert = "".join(chr(_) for _ in convert)

        return convert

    if isinstance(convert, bytearray):
        try:
            convert = convert.decode()
        except UnicodeDecodeError:
            convert = "".join(chr(_) for _ in convert)

        return convert

    items = []
    if isinstance(convert, dict):
        tmp_dict = {}
        items = convert.items()
        for k, v in items:
            if isinstance(v, bytes):
                try:
                    tmp_dict[k] = v.decode()
                except UnicodeDecodeError:
                    tmp_dict[k] = "".join(str(ord(_)) for _ in v)
            elif isinstance(v, str):
                tmp_dict[k] = v
        return tmp_dict
    elif isinstance(convert, list):
        converted_list = []
        items = enumerate(convert)
        for k, v in items:
            if isinstance(v, bytes):
                try:
                    converted_list.append(v.decode())
                except UnicodeDecodeError:
                    converted_list.append("".join(str(ord(_)) for _ in v))

        return converted_list

    return convert


def extract_strings(filepath: str = False, data: bytes = False, on_demand: bool = False, dedup: bool = False, minchars: int = 0):
    """Extract strings from analyzed file.
    @return: list of printable strings.
    """

    nulltermonly = False
    if minchars == 0:
        minchars = 5

    if filepath:
        p = Path(filepath)
        if not p.exists():
            log.error("Sample file doesn't exist: %s", filepath)
            return
        try:
            data = p.read_bytes()
        except (IOError, OSError) as e:
            log.error("Error reading file: %s", e)
            return

    if not data:
        return

    endlimit = b"8192" if not HAVE_RE2 else b""
    if nulltermonly:
        apat = b"([\x20-\x7e]{" + str(minchars).encode() + b"," + endlimit + b"})\x00"
        upat = b"((?:[\x20-\x7e][\x00]){" + str(minchars).encode() + b"," + endlimit + b"})\x00\x00"
    else:
        apat = b"[\x20-\x7e]{" + str(minchars).encode() + b"," + endlimit + b"}"
        upat = b"(?:[\x20-\x7e][\x00]){" + str(minchars).encode() + b"," + endlimit + b"}"

    strings = [bytes2str(string) for string in re.findall(apat, data)]
    strings.extend(str(ws.decode("utf-16le")) for ws in re.findall(upat, data))

    if dedup:
        strings = list(set(strings))

    return strings
