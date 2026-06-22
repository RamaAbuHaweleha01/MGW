from lib.common.abstracts import Package

class Bat(Package):
    """Batch/shell script package."""
    def start(self, path):
        return self.execute("/bin/bash", path)
