import psutil

def get_process_ram_usage(process_name):
    total_ram = 0.0
    for proc in psutil.process_iter(attrs=["name", "memory_info"]):
        try:
            if proc.info["name"] and process_name.lower() in proc.info["name"].lower():
                total_ram += proc.info["memory_info"].rss  # RAM (RSS) in Bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return total_ram / (1024 * 1024)