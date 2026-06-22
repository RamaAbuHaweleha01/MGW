from lib.common.abstracts import Package

class Ps1(Package):
    """PowerShell script — static strings analysis on Linux."""
    def start(self, path):
        return self.execute("/usr/bin/strings", path)
