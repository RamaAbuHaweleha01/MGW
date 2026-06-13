from lib.common.abstracts import Package

class Txt(Package):
    """Plain text / encoded payload static analysis."""
    def start(self, path):
        return self.execute("/usr/bin/strings", path)
