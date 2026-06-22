
import ast
import json
import logging
import os
import re

from lib.cuckoo.common.abstracts import Processing
from lib.cuckoo.common.path_utils import path_exists

log = logging.getLogger(__name__)

__author__ = "@winson0123"
__version__ = "1.0.0"

fd_syscalls = [
    "read",
    "write",
    "close",
    "newfstat",
    "lseek",
    "ioctl",
    "pread64",
    "pwrite64",
    "readv",
    "writev",
    "fcntl",
    "flock",
    "fsync",
    "fdatasync",
    "ftruncate",
    "getdents",
    "fchdir",
    "fchmod",
    "fchown",
    "fstatfs",
    "readahead",
    "fsetxattr",
    "fgetxattr",
    "flistxattr",
    "fremovexattr",
    "getdents64",
    "newfstatat",
    "sync_file_range",
    "vmsplice",
    "fallocate",
    "preadv",
    "pwritev",
    "syncfs",
    "preadv2",
    "pwritev2",
    "fsconfig",
    "quotactl_fd",
]


class ParseProcessLog(list):
    """Parses the process log file"""

    def __init__(self, process_id, logs, syscalls_info, options):
        """@param log_path: log file path."""
        self.logs = logs
        self.process_id = process_id
        self.children_ids = []
        self.first_seen = None
        self.process_name = None
        self.calls = self
        self.file_descriptors = []
        self.sockets = []
        self.options = options

        self.fetch_calls(syscalls_info)

    def __iter__(self):
        return iter(super().__iter__())

    def __repr__(self):
        return f"<ParseProcessLog for pid: {self.process_id}>"

    def begin_reporting(self):
        pass

    def split_arguments(self, args_str):
        args = []
        current = ""
        brace_level = 0
        for char in args_str:
            if char == "," and brace_level == 0:
                args.append(current)
                current = ""
            else:
                current += char
                if char in ["{", "["]:
                    brace_level += 1
                elif char in ["}", "]"]:
                    brace_level = max(brace_level - 1, 0)
        args.append(current)

        return [x.strip() for x in args if x.strip() != ""]

    def fetch_calls(self, syscalls_info):
        for event in self.logs:
            time = event["time"]
            category = "misc"
            syscall = event["syscall"]
            arguments = []
            args = self.split_arguments(event["args"])
            if syscall_info := syscalls_info.get(syscall, None):
                category = syscall_info.get("category", "misc")
                arg_names = syscall_info.get("signature", None)
                for arg_name, arg in zip(arg_names, args):
                    arguments.append(
                        {
                            "name": arg_name,
                            "value": arg,
                        }
                    )
            else:
                arguments.append(event["args"])
            retval = event["retval"]

            if len(self.calls) == 0:
                self.first_seen = time

            if syscall == "execve":
                try:
                    self.process_name = " ".join(ast.literal_eval(args[1]))
                except Exception:
                    self.process_name = str(args[1])

            if syscall in ["fork", "vfork", "clone", "clone3"]:
                if syscall.startswith("clone"):
                    if "CLONE_THREAD" in event["args"]:
                        self.children_ids.append((int(retval), "(thread)"))
                    elif "flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD" in event["args"]:
                        self.children_ids.append((int(retval), "(fork)"))
                else:
                    self.children_ids.append((int(retval), syscall + "(" + event["args"] + ")"))

            self.calls.append(
                {
                    "timestamp": time,
                    "category": category,
                    "api": syscall,
                    "return": retval,
                    "arguments": arguments,
                }
            )

            # Handle file descriptor tracking (Python 3.8 compatible)
            if retval > "0":
                if syscall in ["open", "creat"]:
                    self.file_descriptors.append(
                        {
                            "time": time,
                            "syscall": syscall,
                            "fd": retval,
                            "filename": ast.literal_eval(args[0]),
                        }
                    )
                elif syscall in ["openat", "openat2"]:
                    self.file_descriptors.append(
                        {
                            "time": time,
                            "syscall": syscall,
                            "fd": retval,
                            "filename": ast.literal_eval(args[1]),
                        }
                    )
                elif syscall in ["dup", "dup2", "dup3"]:
                    orig_fd = ast.literal_eval(args[0]) if args else None
                    self.file_descriptors.append(
                        {
                            "time": time,
                            "syscall": syscall,
                            "fd": retval,
                            "original_fd": orig_fd,
                        }
                    )
                elif syscall in ["socket", "accept", "accept4"]:
                    self.sockets.append(
                        {
                            "time": time,
                            "syscall": syscall,
                            "fd": retval,
                        }
                    )

            # Consider close syscalls for tracking closed file descriptors
            if retval == "0" and syscall == "close":
                self.file_descriptors.append(
                    {
                        "time": time,
                        "syscall": syscall,
                        "fd": args[0],
                    }
                )


class Processes:
    """Processes analyzer."""

    key = "processes"

    def __init__(self, logs_path, options):
        self._logs_path = logs_path
        self.syscalls_info = self.load_syscalls_args()
        self.options = options
        self.results = []

    def load_syscalls_args(self):
        syscalls_json = open("/opt/CAPEv2/data/linux/linux-syscalls.json", "r")
        syscalls_dict = json.load(syscalls_json)
        return {
            syscall["name"]: {
                "signature": syscall["signature"],
                "category": "kernel" if "kernel" in syscall["file"] else syscall["file"].split("/")[0],
            }
            for syscall in syscalls_dict["syscalls"]
        }

    def update_file_descriptors(self, fd_calls):
        if not self.options.get("update_file_descriptors"):
            return
        fd_lookup = {
            "0": [{"filename": "STDIN", "time_opened": "00:00:00.000000", "time_closed": None}],
            "1": [{"filename": "STDOUT", "time_opened": "00:00:00.000000", "time_closed": None}],
            "2": [{"filename": "STDERR", "time_opened": "00:00:00.000000", "time_closed": None}]
        }
        for fd_call in fd_calls:
            if fd_call["syscall"] in ["open", "creat", "openat", "openat2"]:
                fd_lookup.setdefault(fd_call["fd"], []).append(
                    {
                        "filename": fd_call["filename"],
                        "time_opened": fd_call["time"],
                        "time_closed": None,
                    }
                )
            elif fd_call["syscall"] in ["dup", "dup2", "dup3"]:
                for fd in reversed(fd_lookup.get(fd_call["original_fd"], [])):
                    if fd["time_closed"] is None:
                        fd_lookup.setdefault(fd_call["fd"], []).append(
                            {
                                "filename": fd["filename"],
                                "time_opened": fd_call["time"],
                                "time_closed": None,
                            }
                        )
            elif fd_call["syscall"] == "close":
                for fd in reversed(fd_lookup.get(fd_call["fd"], [])):
                    if fd["time_closed"] is None:
                        fd["time_closed"] = fd_call["time"]

        for process in self.results:
            calls = [c for c in process["calls"] if c["api"] in fd_syscalls]
            for call in calls:
                for fd in fd_lookup.get(call["arguments"][0]["value"], []):
                    if (
                        fd["time_opened"] < call["timestamp"]
                        and (fd["time_closed"] is None or call["timestamp"] <= fd["time_closed"])
                    ):
                        call["arguments"][0]["value"] += f' ({fd["filename"]})'
                        break

    def update_parent_ids(self, relations):
        process_dict = {entry["process_id"]: entry for entry in self.results}
        for parent_id, children in relations.items():
            if parent_id in process_dict:
                for child_id, name in children:
                    if child_id in process_dict:
                        process_dict[child_id]["parent_id"] = parent_id
                        if process_dict[child_id]["process_name"] is None:
                            process_dict[child_id]["process_name"] = name
        self.results = list(process_dict.values())

    def log_concat(self, unfinished, resumed):
        data = ""
        for key in unfinished.keys():
            for head in unfinished[key]:
                for tail in resumed[key]:
                    if head["syscall"] != tail["syscall"]:
                        continue
                    data += " ".join([str(key), tail["time"], head["unfinished"] + tail["resumed"] + "\n"])
                    resumed[key].remove(tail)
                    break
        return data

    def extract_logs(self, raw_logs, pattern):
        extracted_logs = dict()
        for match in pattern.finditer(raw_logs):
            match = match.groupdict()
            pid = int(match.pop("pid"))
            if pid not in extracted_logs:
                extracted_logs[pid] = []
            extracted_logs[pid].append(match)
        return extracted_logs

    def normalize_logs(self):
        log_pattern = re.compile(
            r"(?P<pid>\d+)\s+(?P<time>\d+:\d+:\d+\.\d+)\s+(?P<syscall>\w+)\((?P<args>.*)\)\s+=\s(?P<retval>.+)\n"
        )
        unfinished_pattern = re.compile(
            r"(?P<pid>\d+)\s+\d+:\d+:\d+\.\d+\s+(?P<unfinished>(?P<syscall>\w+)\(.*)\s+<unfinished\s...>\n"
        )
        resumed_pattern = re.compile(
            r"(?P<pid>\d+)\s+(?P<time>\d+:\d+:\d+\.\d+)\s+<\.\.\.\s(?P<syscall>\w+)\sresumed>(?P<resumed>.*)\n"
        )
        with open(self._logs_path, "r") as log_file:
            raw_logs = log_file.read()

        normal_logs = self.extract_logs(raw_logs, log_pattern)
        unfinished_logs = self.extract_logs(raw_logs, unfinished_pattern)
        resumed_logs = self.extract_logs(raw_logs, resumed_pattern)

        concat_raw_logs = self.log_concat(unfinished_logs, resumed_logs)
        concat_logs = self.extract_logs(concat_raw_logs, log_pattern)
        for pid in concat_logs.keys():
            if pid not in normal_logs:
                normal_logs[pid] = []
            normal_logs[pid].extend(concat_logs[pid])
            normal_logs[pid].sort(key=lambda d: d["time"])

        return normal_logs

    def run(self):
        parent_child_relation = {}
        fd = []

        if not path_exists(self._logs_path):
            log.warning('Strace logs does not exist at path "%s"', self._logs_path)
            return self.results

        if not os.stat(self._logs_path).st_size > 0:
            log.warning('Strace logs does not contain data at path "%s"', self._logs_path)
            return self.results

        processes = self.normalize_logs()

        for pid in processes.keys():
            current_log = ParseProcessLog(pid, processes[pid], self.syscalls_info, self.options)

            parent_child_relation[current_log.process_id] = current_log.children_ids

            self.results.append(
                {
                    "process_id": current_log.process_id,
                    "process_name": current_log.process_name,
                    "parent_id": None,
                    "first_seen": current_log.first_seen,
                    "calls": current_log.calls,
                }
            )

            fd += current_log.file_descriptors

        self.update_parent_ids(parent_child_relation)
        self.update_file_descriptors(fd)

        self.results.sort(key=lambda process: process["first_seen"])

        return self.results


class ProcessTree:
    """Generates process tree."""

    key = "processtree"

    def __init__(self):
        self.processes = []
        self.tree = []

    def add_node(self, node, tree):
        ret = False
        for process in tree:
            if process["pid"] == node["parent_id"]:
                process["children"].append(node)
                ret = True
                break
            else:
                if self.add_node(node, process["children"]):
                    ret = True
                    break
        return ret

    def event_apicall(self, call, process):
        for entry in self.processes:
            if entry["pid"] == process["process_id"]:
                return

        self.processes.append(
            {
                "name": process["process_name"],
                "pid": process["process_id"],
                "parent_id": process["parent_id"],
                "children": [],
            }
        )

    def run(self):
        children = []

        for process in self.processes:
            has_parent = False
            for process_again in self.processes:
                if process_again == process:
                    continue
                if process_again["pid"] == process["parent_id"]:
                    has_parent = True
                    break

            if has_parent:
                children.append(process)
            else:
                self.tree.append(process)

        for process in children:
            if not self.add_node(process, self.tree):
                self.tree.append(process)

        return self.tree


class StraceAnalysis(Processing):
    """Strace Analyzer."""

    key = "behavior"
    os = "linux"

    def run(self):
        strace = {"processes": Processes(os.path.join(self.logs_path, "strace.log"), self.options).run()}

        instances = [ProcessTree()]
        enabled_instances = [instance for instance in instances if getattr(self.options, instance.key, True)]

        if enabled_instances:
            for process in strace["processes"]:
                for call in process["calls"]:
                    for instance in enabled_instances:
                        try:
                            instance.event_apicall(call, process)
                        except Exception:
                            log.exception('Failure in partial behavior "%s"', instance.key)

        for instance in instances:
            try:
                strace[instance.key] = instance.run()
            except Exception as e:
                log.exception('Failed to run partial behavior class "%s" due to "%s"', instance.key, e)

        return strace

