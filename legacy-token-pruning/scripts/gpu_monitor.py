#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("outputs/gpu.log")

def get_gpu_stats():
    try:
        # Run nvidia-smi
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        stats = result.strip().split(',')
        if len(stats) >= 4:
            util = int(stats[0].strip())
            mem_used = int(stats[1].strip())
            mem_total = int(stats[2].strip())
            temp = int(stats[3].strip())
            
            # Human readable status
            status = "Idle"
            if util > 80:
                status = "🔥 Heavy Training (Tensor Cores Active)"
            elif util > 20:
                status = "⚡ Moderate Compute"
            elif mem_used > 5000:
                status = "💾 Extracting / Loading Weights into VRAM"
                
            return f"[{datetime.now().strftime('%H:%M:%S')}] {status:<40} | GPU: {util:3d}% | VRAM: {mem_used:5d}/{mem_total} MB | Temp: {temp}°C"
    except Exception:
        return f"[{datetime.now().strftime('%H:%M:%S')}] Cannot reach GPU driver."

if __name__ == "__main__":
    print(f"Starting human-readable GPU Monitor...")
    # Initialize the log file with a header
    with open(LOG_FILE, "w") as f:
        f.write("=== A100 GPU Task Manager (Human Readable) ===\n")
        
    while True:
        stat_line = get_gpu_stats()
        
        # We will keep a sliding window of the last 15 updates so the file doesn't get massive
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            
        lines.append(stat_line + "\n")
        if len(lines) > 16:
            lines = [lines[0]] + lines[-15:] # Keep header + last 15
            
        with open(LOG_FILE, "w") as f:
            f.writelines(lines)
            
        time.sleep(2)
