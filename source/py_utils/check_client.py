import re

def extract_last_block_metrics(logfile):
    with open(logfile, "r") as f:
        log = f.read()
        # 按 block 分割，每个 block 以 'Real-time throughput' 开头
        blocks = re.split(r'(?=Real-time throughput)', log.strip())

        if not blocks:
            return None

        last_block = blocks[-1]

        def get_float(pattern):
            match = re.search(pattern, last_block)
            return float(match.group(1)) if match else None

        return {
            "Average Throughput": get_float(r"Average Throughput.*?: (\d+)"),
            "Average Latency": get_float(r"Average Latency: ([\d.]+)"),
            "99th Latency": get_float(r"99th Latency: ([\d.]+)"),
            "50th Latency": get_float(r"50th Latency: ([\d.]+)")
        }
