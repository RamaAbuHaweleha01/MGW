from lib.common.abstracts import Package

class Command(Package):
    def start(self, path):
        cmd_path = "C:\\Windows\\System32\\cmd.exe"
        cmd_args = f"/c \"{path}\""
        return self.execute(cmd_path, cmd_args, path)
