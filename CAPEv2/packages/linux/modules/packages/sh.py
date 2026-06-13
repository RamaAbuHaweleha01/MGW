from lib.common.abstracts import Package

class Sh(Package):
    def start(self, path):
        return self.execute("/bin/bash", path)
