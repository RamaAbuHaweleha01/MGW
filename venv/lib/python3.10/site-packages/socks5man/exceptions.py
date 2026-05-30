class Socks5manError(Exception):
    """Error that halts socks5man"""

class Socks5CreationError(Socks5manError):
    """Errors specifically related to the creation of new socks5 servers"""

class Socks5ConfigError(Socks5manError):
    """Error that should be raised when issues occur when reading a config
    file"""

class Socks5manDatabaseError(Socks5manError):
    """Error that should be raised when issue/exceptions occur when
    performing database operations"""
