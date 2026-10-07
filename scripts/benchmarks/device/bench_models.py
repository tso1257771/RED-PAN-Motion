"""On-device efficiency benchmark of the three RED-PAN models (ONNX Runtime CPU).

Per model and thread count: wall latency, process CPU time, peak RSS, board power
(when the INA3221 rails are readable), and the derived real-time factor. Window
lengths differ (60 s vs 90 s), so latency per 1000 samples is reported alongside
latency per window, the same normalization as kMAC per sample.

One process per config (--only) keeps peak RSS free of allocator carry-over.
"""
import argparse, json, os, resource, statistics as st, threading, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # scripts/benchmarks/device -> repo root
ONNX_DIR = Path(os.environ.get("RPM_ONNX_DIR", HERE / "onnx"))
RESULTS_DIR = Path(os.environ.get("RPM_RESULTS_DIR", HERE / "results"))
CKPT_DIR = Path(os.environ.get("RPM_CKPT_DIR", REPO / "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "redpan_60s":    {"path": ONNX_DIR / "redpan_60s.onnx",    "npts": 6000, "precision": "fp32"},
    "redpan_motion": {"path": ONNX_DIR / "redpan_motion.onnx", "npts": 9000, "precision": "fp32"},
    "edge_rp90":     {"path": ONNX_DIR / "edge_rp90.onnx",     "npts": 9000, "precision": "fp32"},
    "edge_rp90_int8": {"path": ONNX_DIR / "edge_rp90_int8.onnx", "npts": 9000, "precision": "int8"},
}
PWR = Path("/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0")
PWR_CH = {"VDD_IN": "in_power0_input", "VDD_GPU": "in_power1_input", "VDD_CPU": "in_power2_input"}
rng = np.random.default_rng(0)


def power_readable():
    try:
        (PWR / PWR_CH["VDD_IN"]).read_text(); return True
    except Exception:
        return False


def read_power():
    out = {}
    for n, f in PWR_CH.items():
        try:
            out[n] = int((PWR / f).read_text().strip())
        except Exception:
            pass
    return out


def cat(p, cast=str):
    try:
        return cast(Path(p).read_text().strip())
    except Exception:
        return None


def sys_state():
    zones = {}
    for z in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        t, v = cat(z / "type"), cat(z / "temp", int)
        if t and v is not None:
            zones[t] = v / 1000.0
    return {"loadavg": os.getloadavg(),
            "cpu_mhz": [(cat(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq", int) or 0) / 1000
                        for c in range(os.cpu_count())],
            "temp_c": zones,
            "mem_available_mb": round(next((int(l.split()[1]) / 1024 for l in open("/proc/meminfo")
                                            if l.startswith("MemAvailable")), 0), 1),
            "power_mw": read_power()}


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0


class Sampler(threading.Thread):
    def __init__(self, period=0.02):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.rss, self.pwr = period, threading.Event(), [], []

    def run(self):
        ok = power_readable()
        while not self.stop_flag.is_set():
            self.rss.append(rss_mb())
            if ok:
                self.pwr.append(read_power())
            time.sleep(self.period)


def make_inputs(sess, npts, n=4):
    """Event-like and noise-like windows matching each graph's declared inputs."""
    out = []
    for i in range(n):
        d = {}
        for inp in sess.get_inputs():
            ch = inp.shape[1]
            a = rng.standard_normal((1, ch, npts)).astype(np.float32)
            if inp.name == "z_raw":
                a = (rng.random((1, ch, npts), dtype=np.float32) * 2 - 1)
            elif i % 2 == 0:  # event: damped oscillation, elicits picker/detector response
                t0 = int(rng.integers(1000, npts - 1200)); t = np.arange(600)
                a[0, :, t0:t0 + 600] += 6.0 * (np.sin(2 * np.pi * 0.05 * t) * np.exp(-t / 150)).astype(np.float32)
            d[inp.name] = a
        out.append(d)
    return out


def bench(name, threads, reps, warmup, idle_power=None):
    spec = MODELS[name]
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3

    base_rss = rss_mb()
    t = time.perf_counter()
    sess = ort.InferenceSession(str(spec["path"]), so, providers=["CPUExecutionProvider"])
    init_s = time.perf_counter() - t
    sess_rss = rss_mb()

    inputs = make_inputs(sess, spec["npts"])
    for i in range(warmup):
        sess.run(None, inputs[i % len(inputs)])

    smp = Sampler(); smp.start()
    lat, cpu = [], []
    for i in range(reps):
        inp = inputs[i % len(inputs)]
        c0, w0 = time.process_time(), time.perf_counter()
        sess.run(None, inp)
        lat.append((time.perf_counter() - w0) * 1000)
        cpu.append((time.process_time() - c0) * 1000)
    smp.stop_flag.set(); smp.join()

    med = st.median(lat)
    ls = sorted(lat)
    window_s = spec["npts"] / 100.0
    r = {"model": name, "precision": spec["precision"], "npts": spec["npts"],
         "window_s": window_s, "threads": threads, "reps": reps,
         "init_s": round(init_s, 3),
         "latency_ms": {"min": round(min(lat), 1), "median": round(med, 1),
                        "mean": round(st.mean(lat), 1),
                        "p90": round(ls[min(int(0.9 * len(ls)), len(ls) - 1)], 1),
                        "max": round(max(lat), 1),
                        "stdev": round(st.stdev(lat), 2) if len(lat) > 1 else 0.0},
         "ms_per_1000_samples": round(med / spec["npts"] * 1000, 2),
         "cpu_ms_median": round(st.median(cpu), 1),
         "core_efficiency": round(st.median(cpu) / med / threads, 3),
         "realtime_factor": round(window_s / (med / 1000), 1),
         "rss_mb": {"baseline": round(base_rss, 1), "after_session": round(sess_rss, 1),
                    "session_delta": round(sess_rss - base_rss, 1),
                    "peak_during_run": round(max(smp.rss) if smp.rss else 0, 1),
                    "peak_rusage": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)},
         "latency_all_ms": [round(v, 1) for v in lat]}

    if smp.pwr:
        pw = {}
        for ch in PWR_CH:
            vals = [s[ch] for s in smp.pwr if ch in s]
            if vals:
                pw[ch] = {"mean": round(st.mean(vals), 1), "max": max(vals), "n": len(vals)}
        r["power_mw"] = pw
        if "VDD_IN" in pw:
            r["energy_per_window_J"] = round(pw["VDD_IN"]["mean"] / 1000 * med / 1000, 4)
            if idle_power:
                dyn = pw["VDD_IN"]["mean"] - idle_power
                r["idle_power_mw"] = idle_power
                r["dynamic_power_mw"] = round(dyn, 1)
                r["dynamic_energy_per_window_J"] = round(dyn / 1000 * med / 1000, 4)
    return r


HDR = f"{'model':16s}{'prec':5s}{'thr':>4s}{'median':>10s}{'min':>8s}{'p90':>8s}{'sd':>7s}{'ms/1ks':>8s}{'eff':>6s}{'RTF':>8s}{'peakRSS':>9s}"


def line(r):
    L = r["latency_ms"]
    s = (f"{r['model']:16s}{r['precision']:5s}{r['threads']:4d}{L['median']:9.1f}ms{L['min']:8.1f}"
         f"{L['p90']:8.1f}{L['stdev']:7.2f}{r['ms_per_1000_samples']:8.2f}{r['core_efficiency']:6.2f}"
         f"{r['realtime_factor']:7.0f}x{r['rss_mb']['peak_during_run']:8.1f}MB")
    if "energy_per_window_J" in r:
        s += f"  {r['energy_per_window_J']*1000:6.0f}mJ"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, help="e.g. edge_rp90_t4")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--idle-power", type=float, default=None, help="idle VDD_IN mW, for dynamic energy")
    ap.add_argument("--header", action="store_true")
    a = ap.parse_args()

    name, th = a.only.rsplit("_t", 1)
    r = bench(name, int(th), a.reps, a.warmup, a.idle_power)
    r["state_after"] = sys_state()
    r["ort_version"] = ort.__version__
    r["tag"] = a.tag
    (RESULTS_DIR / f"results_{a.tag}_{a.only}.json").write_text(json.dumps(r, indent=2))
    if a.header:
        print(HDR)
    print(line(r))


if __name__ == "__main__":
    main()
