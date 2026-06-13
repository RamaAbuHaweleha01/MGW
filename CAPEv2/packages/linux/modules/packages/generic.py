import os
from lib.common.abstracts import Package

class Generic(Package):
    def start(self, path):
        """Execute the file based on its type."""
        ext = os.path.splitext(path)[1].lower()
        arguments = self.options.get("arguments", "")
        strace_log = os.path.join(self.strace_output, "strace.log") \
            if hasattr(self, "strace_output") and self.strace_output else "/tmp/strace.log"

        def strace_wrap(target, args=""):
            # Build as list — Popen requires list, not shell string
            cmd = ["/usr/bin/strace", "-f", "-e", "trace=all", "-o", strace_log, target]
            if args:
                cmd += args.split()
            return self.execute(" ".join(cmd))

        if ext in (".elf", "") and os.access(path, os.X_OK):
            os.chmod(path, 0o755)
            return strace_wrap(path, arguments)
        elif ext == ".sh":
            return strace_wrap("/bin/bash " + path)
        elif ext == ".py":
            return strace_wrap("/usr/bin/python3 " + path)
        else:
            try:
                os.chmod(path, 0o755)
                return strace_wrap(path, arguments)
            except Exception:
                return self.execute("/usr/bin/strings " + path)
