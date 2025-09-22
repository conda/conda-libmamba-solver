"""
Profiles conda processes using psutil plus a couple of built-in modules

NOTICE: this currently only works on Unix-like systems (macOS and Linux)
"""
import resource
import subprocess
import sys
import time

import psutil

#: Value used to convert bytes to megabytes
BYTES_TO_MB = 1024 * 1024

#: Value used to convert kilobytes to megabytes
KILOBYTES_TO_MB = 1024


def main():
    """
    Entry point for the profiling script
    """

    # Gather the network and time statistics before
    net_io_stats_before = psutil.net_io_counters()
    time_before = time.time()

    command = [
        "python", "-m",
        # conda command
        "conda", "create",
        # settings
        "--name", "profile-env",
        "--channel", "conda-forge",
        "--override-channels",
        "--yes",
        "--quiet",
        # packages
        "python",
    ]
    sub = subprocess.Popen(command)
    sub.wait()

    rusage = resource.getrusage(resource.RUSAGE_CHILDREN)

    # Gather statistics after
    net_io_stats_after = psutil.net_io_counters()
    time_after = time.time()

    bytes_recv = round(
        (net_io_stats_after.bytes_recv - net_io_stats_before.bytes_recv) / BYTES_TO_MB,
        ndigits=2
    )
    bytes_sent = round(
        (net_io_stats_after.bytes_sent - net_io_stats_before.bytes_sent) / BYTES_TO_MB,
        ndigits=2
    )
    total_time = round(time_after - time_before, ndigits=2)
    max_mem_convert = BYTES_TO_MB if sys.platform == "darwin" else KILOBYTES_TO_MB
    max_memory_usage = round(rusage.ru_maxrss / max_mem_convert, ndigits=2)

    command_str = " ".join(command)

    print()

    print("Statistics for:", command_str)

    # Network
    print(f"Received: {bytes_recv}MB")
    print(f"Sent: {bytes_sent}MB")

    # Memory
    print(f"Max memory usage: {max_memory_usage}MB")

    # Time
    print(f"Total time: {total_time}s")
    print(f"User CPU time: {rusage.ru_utime}s")
    print(f"System CPU time: {rusage.ru_stime}s")


if __name__ == "__main__":
    main()
