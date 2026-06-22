from lib.common.abstracts import Package
from lib.common.common import check_file_extension

class Ps1(Package):
    """PowerShell script analysis package."""

    PATHS = [
        ("SystemRoot", "system32\\WindowsPowerShell\\v1.0", "powershell.exe"),
    ]

    def start(self, path):
        path = check_file_extension(path, ".ps1")
        powershell = self.get_path("powershell.exe")
        args = f'-ExecutionPolicy Bypass -NoProfile -File "{path}"'
        return self.execute(powershell, args)
