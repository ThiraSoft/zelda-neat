#!/usr/bin/env python3
"""Thermal Watchdog Daemon for zelda-neat tasks.
Monitors CPU (coretemp) and GPU (amdgpu) temperatures every second.
If temperature reaches or exceeds the threshold (default 75.0°C),
it gracefully sends SIGINT (and SIGKILL if needed) to the specified PID tree.
"""

import argparse
import glob
import os
import signal
import sys
import time


def find_target_pid():
    # Try finding dataset.py
    try:
        out = os.popen("pgrep -f 'dataset.py'").read().strip().split()
        if out:
            # Pick the lowest PID (parent)
            pids = sorted([int(p) for p in out if int(p) != os.getpid()])
            if pids:
                return pids[0]
    except Exception:
        pass
    return None


def get_temperatures():
    cpu_temps = []
    gpu_temps = []
    for p in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(p, "name")) as f:
                name = f.read().strip().lower()
        except Exception:
            continue

        for t in glob.glob(os.path.join(p, "temp*_input")):
            try:
                with open(t) as f:
                    val = float(f.read().strip()) / 1000.0
                if "coretemp" in name or "k10temp" in name or "cpu" in name:
                    cpu_temps.append(val)
                elif "amdgpu" in name or "gpu" in name:
                    gpu_temps.append(val)
            except Exception:
                continue

    max_cpu = max(cpu_temps) if cpu_temps else 0.0
    max_gpu = max(gpu_temps) if gpu_temps else 0.0
    return max_cpu, max_gpu


def kill_process_tree(pid):
    print(f"[THERMAL GUARD] Terminating PID {pid} tree...", flush=True)
    try:
        # Find children
        children = os.popen(f"pgrep -P {pid}").read().strip().split()
        pids_to_kill = [int(p) for p in children] + [pid]
    except Exception:
        pids_to_kill = [pid]

    for p in pids_to_kill:
        try:
            os.kill(p, signal.SIGINT)
        except ProcessLookupError:
            pass

    # Wait up to 5 seconds
    time.sleep(3)
    for p in pids_to_kill:
        try:
            os.kill(p, 0)
            # If still alive, force kill
            os.kill(p, signal.SIGKILL)
            print(f"[THERMAL GUARD] Force-killed PID {p}", flush=True)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None, help="Target PID to watch")
    parser.add_argument("--max-temp", type=float, default=75.0, help="Max temperature in Celsius")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="Poll interval in seconds")
    parser.add_argument("--status-file", type=str, default="/tmp/thermal_guard.status")
    args = parser.parse_args()

    target_pid = args.pid or find_target_pid()
    if not target_pid:
        print("[THERMAL GUARD] No running dataset process found.", flush=True)
        sys.exit(1)

    print(f"[THERMAL GUARD] Monitoring PID {target_pid} (Threshold: {args.max_temp}°C, Interval: {args.poll_sec}s)", flush=True)

    last_heartbeat = 0.0

    while True:
        # Check target alive
        try:
            os.kill(target_pid, 0)
        except ProcessLookupError:
            print(f"[THERMAL GUARD] Target PID {target_pid} exited. Watchdog stopping.", flush=True)
            with open(args.status_file, "w") as f:
                f.write(f"Target PID {target_pid} finished normally.\n")
            sys.exit(0)

        cpu_t, gpu_t = get_temperatures()

        now = time.time()
        if now - last_heartbeat >= 5.0:
            last_heartbeat = now
            status_msg = f"PID {target_pid} OK | CPU: {cpu_t:.1f}°C | GPU: {gpu_t:.1f}°C (Limit: {args.max_temp}°C)"
            with open(args.status_file, "w") as f:
                f.write(status_msg + "\n")

        if cpu_t >= args.max_temp or gpu_t >= args.max_temp:
            alert = (
                f"[THERMAL GUARD EMERGENCY] Overheat detected! "
                f"CPU: {cpu_t:.1f}°C, GPU: {gpu_t:.1f}°C (Threshold: {args.max_temp}°C). "
                f"Cutting process {target_pid} immediately!"
            )
            print(alert, flush=True)
            with open(args.status_file, "w") as f:
                f.write(alert + "\n")
            with open("/mnt/data/datasets/zelda/thermal_alert.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {alert}\n")

            kill_process_tree(target_pid)
            sys.exit(2)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
