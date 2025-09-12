# utils/logger.py
import csv, json, os, time, subprocess, torch

class MetricsLogger:
    def __init__(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, "metrics.csv")
        self.t0 = time.time()
        self._w = None
        self._it_count = 0
        self._samples = 0
        self.meta = {}

    def set_meta(self, **kwargs):
        self.meta.update(kwargs)

    def bump_iterations(self, batch_size):
        self._it_count += 1
        self._samples += int(batch_size)

    def wall_seconds(self):
        return time.time() - self.t0

    def it_per_sec(self):
        s = self.wall_seconds()
        return self._it_count / s if s > 0 else 0.0

    def samples_per_sec(self):
        s = self.wall_seconds()
        return self._samples / s if s > 0 else 0.0

    def log_epoch(self, **row):
        hdr = list(row.keys())
        if self._w is None:
            self._w = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self._w, fieldnames=hdr)
            self.writer.writeheader()
        self.writer.writerow(row); self._w.flush()

    def close(self):
        if self._w: self._w.close()

    @staticmethod
    def param_count(model):
        return sum(p.numel() for p in model.parameters())

def short_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "n/a"
