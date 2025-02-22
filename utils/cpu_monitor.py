import psutil
import time
import os
import platform
from datetime import datetime
from collections import deque
from statistics import mean

class CPUMonitor:
    def __init__(self, history_size: int = 5):
        self.history = [deque(maxlen=history_size) for _ in range(psutil.cpu_count())]
        self.cpu_info = self._get_cpu_info()
        self.last_update = None
        
    def _get_cpu_info(self):
        """Get detailed CPU information for each core."""
        info = []
        try:
            # Get CPU brand and model
            if platform.system() == "Darwin":  # macOS
                import subprocess
                cmd = "sysctl -n machdep.cpu.brand_string".split()
                cpu_model = subprocess.check_output(cmd).decode().strip()
                
                # Get core types (efficiency vs performance) on Apple Silicon
                cmd = "sysctl -n hw.perflevel".split()
                try:
                    core_types = subprocess.check_output(cmd).decode().strip().split()
                    is_apple_silicon = "apple" in cpu_model.lower()
                except:
                    core_types = []
                    is_apple_silicon = False
                
                for i in range(psutil.cpu_count()):
                    core_info = {
                        'number': i,
                        'model': cpu_model,
                        'type': "Performance" if (is_apple_silicon and i < 4) else "Efficiency" if is_apple_silicon else "Core",
                        'cluster': "CPU 1" if i < psutil.cpu_count() // 2 else "CPU 2"
                    }
                    info.append(core_info)
        except:
            # Fallback to basic information
            for i in range(psutil.cpu_count()):
                info.append({
                    'number': i,
                    'model': platform.processor(),
                    'type': "Core",
                    'cluster': "CPU"
                })
        return info
        
    def display(self):
        """Display CPU stats with core details."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print("\033[H\033[2J\033[3J")  # Clear screen
        
        print(f"\nCPU Core Usage (Updated: {timestamp})")
        print("=" * 100)
        header = f"{'Core ID':^8} {'Type':^12} {'Cluster':^10} {'Load':^70}"
        print(header)
        print("=" * 100)
        
        # Get current CPU percentages
        percentages = psutil.cpu_percent(percpu=True, interval=2)
        
        # Display each core with its details
        for i, (cpu_info, usage) in enumerate(zip(self.cpu_info, percentages)):
            bar = "█" * int(usage / 1.5)  # Scale to fit in terminal
            core_info = f"CPU {cpu_info['number']:02d}"
            core_type = cpu_info['type']
            cluster = cpu_info['cluster']
            
            print(f"{core_info:8} {core_type:12} {cluster:10} [{usage:5.1f}% |{bar:<60}|]")
        
        print("=" * 100)
        if platform.system() == "Darwin":
            print(f"Processor: {self.cpu_info[0]['model']}")

def monitor_cpu():
    """Run the CPU monitor."""
    try:
        monitor = CPUMonitor()
        print("\033[?25l")  # Hide cursor
        while True:
            monitor.display()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\033[?25h")  # Show cursor
        print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor_cpu()
