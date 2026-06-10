#!/usr/bin/env python3
import time
import csv
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

METRICS_FILE = Path("outputs/metrics.csv")
LOG_FILE = Path("outputs/dashboard.log")

def get_download_status():
    if not LOG_FILE.exists():
        return "Waiting for logs..."
    
    # Read the last 20 lines to find tqdm progress
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()[-20:]
        for line in reversed(lines):
            if "%|" in line and "M/5.00G" in line:
                return line.strip()
    return "Dataset downloaded. Training in progress."

def generate_dashboard():
    # Metrics Table
    table = Table(title="🔥 SOTA A100 Training Telemetry", style="cyan", border_style="blue")
    table.add_column("Epoch", justify="center", style="magenta", width=8)
    table.add_column("Train Loss", justify="right", style="green", width=12)
    table.add_column("Train Acc", justify="right", style="bold green", width=12)
    table.add_column("Test Loss", justify="right", style="blue", width=12)
    table.add_column("Test Acc", justify="right", style="bold blue", width=12)

    has_metrics = False
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    has_metrics = True
                    for row in rows[-15:]:
                        table.add_row(
                            row.get("Epoch", "-"),
                            f"{float(row.get('Train_Loss', 0)):.4f}",
                            f"{float(row.get('Train_Acc', 0))*100:.2f}%",
                            f"{float(row.get('Test_Loss', 0)):.4f}",
                            f"{float(row.get('Test_Acc', 0))*100:.2f}%"
                        )
        except Exception:
            pass
            
    if not has_metrics:
        dl_status = get_download_status()
        table.add_row("-", "-", "Awaiting", "Data", "-")
        return Panel(Align.center(table), title=f"[yellow]Status: {dl_status}[/yellow]", border_style="yellow")

    return Panel(Align.center(table), title="[bold green]Training Active[/bold green]", border_style="green")

if __name__ == "__main__":
    console = Console()
    console.clear()
    with Live(generate_dashboard(), refresh_per_second=2, console=console) as live:
        try:
            while True:
                time.sleep(1)
                live.update(generate_dashboard())
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard closed.[/yellow]")
