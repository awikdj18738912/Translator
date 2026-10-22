#!/usr/bin/env python3
"""Sample physical GPU1 memory, utilization, power, and the active model phase."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


output = Path(sys.argv[1])
phase_file = Path(sys.argv[2])
stop_file = Path(sys.argv[3])
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
output.parent.mkdir(parents=True, exist_ok=True)
write_header = not output.exists() or output.stat().st_size == 0
with output.open("a", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    if write_header:
        writer.writerow([
            "timestamp_utc", "phase", "gpu_index", "memory_used_mib",
            "memory_total_mib", "gpu_utilization_percent", "power_watts",
        ])
        stream.flush()
    while not stop_file.exists():
        phase = phase_file.read_text().strip() if phase_file.exists() else "unknown"
        try:
            values = subprocess.check_output(
                [
                    "nvidia-smi", "-i", "1",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=4,
            ).strip().split(",")
            values = [value.strip() for value in values]
            if len(values) == 5:
                writer.writerow([datetime.now(timezone.utc).isoformat(), phase, *values])
                stream.flush()
        except Exception:
            pass
        time.sleep(interval)
