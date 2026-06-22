import os
from lib.common.abstracts import Package
from lib.common.exceptions import CuckooPackageError

class Elf(Package):
    def start(self, path):
        """Run ELF executable under strace."""
        os.chmod(path, 0o755)
        arguments = self.options.get("arguments", "")
        strace_log = os.path.join(self.strace_output, "strace.log") \
            if hasattr(self, "strace_output") and self.strace_output else "/tmp/strace.log"
        cmd = f"/usr/bin/strace -f -e trace=all -o {strace_log} {path}"
        if arguments:
            cmd += f" {arguments}"
        return self.execute(cmd)
