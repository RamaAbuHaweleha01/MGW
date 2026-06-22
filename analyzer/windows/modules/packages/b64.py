from lib.common.abstracts import Package
from lib.common.common import check_file_extension

class Txt(Package):
    """Plain text file analysis."""
    PATHS = [("SystemRoot", "system32", "notepad.exe")]

    def start(self, path):
        notepad = self.get_path("notepad.exe")
        return self.execute(notepad, f'"{path}"')
