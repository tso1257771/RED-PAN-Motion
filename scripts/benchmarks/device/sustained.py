"""Sustained-load run: does this board throttle, and what is steady-state power?

Runs one model back to back for N seconds at a given thread count, sampling
latency, CPU clocks, temperatures and board power each second. Reports drift
between the first and last quarter of the run.
"""
import argparse, json, statistics as st, time
from pathlib import Path
import numpy as np, onnxruntime as ort
import bench_models as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="edge_rp90")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--tag", default="sustained")
    a = ap.parse_args()

    spec = B.MODELS[a.model]
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(spec["path"]), so, providers=["CPUExecutionProvider"])
    inputs = B.make_inputs(sess, spec["npts"])
    for i in range(5):
        sess.run(None, inputs[i % len(inputs)])

    t_end = time.time() + a.seconds
    samples, i = [], 0
    next_probe = 0.0
    while time.time() < t_end:
        inp = inputs[i % len(inputs)]
        w0 = time.perf_counter()
        sess.run(None, inp)
        lat = (time.perf_counter() - w0) * 1000
        rec = {"t": round(time.time(), 2), "lat_ms": round(lat, 1)}
        if time.time() >= next_probe:                 # probe sysfs ~1 Hz, not per window
            s = B.sys_state()
            rec.update(cpu_c=s["temp_c"].get("CPU-therm"), pmic_c=s["temp_c"].get("PMIC-Die"),
                       mhz=max(s["cpu_mhz"]), vdd_in=s["power_mw"].get("VDD_IN"),
                       vdd_cpu=s["power_mw"].get("VDD_CPU"))
            next_probe = time.time() + 1.0
        samples.append(rec); i += 1

    q = max(1, len(samples) // 4)
    first, last = samples[:q], samples[-q:]
    g = lambda rs, k: [r[k] for r in rs if r.get(k) is not None]
    out = {"model": a.model, "threads": a.threads, "seconds": a.seconds,
           "windows_run": len(samples),
           "windows_per_s": round(len(samples) / a.seconds, 2),
           "latency_ms": {"median": round(st.median(g(samples, "lat_ms")), 1),
                          "first_quarter": round(st.median(g(first, "lat_ms")), 1),
                          "last_quarter": round(st.median(g(last, "lat_ms")), 1)},
           "cpu_c": {"start": g(first, "cpu_c")[0] if g(first, "cpu_c") else None,
                     "max": max(g(samples, "cpu_c") or [0]),
                     "end": g(last, "cpu_c")[-1] if g(last, "cpu_c") else None},
           "max_mhz_seen": max(g(samples, "mhz") or [0]),
           "min_mhz_seen": min(g(samples, "mhz") or [0]),
           "vdd_in_mw": {"median": round(st.median(g(samples, "vdd_in")), 0) if g(samples, "vdd_in") else None,
                         "max": max(g(samples, "vdd_in") or [0])},
           "vdd_cpu_mw": {"median": round(st.median(g(samples, "vdd_cpu")), 0) if g(samples, "vdd_cpu") else None},
           "samples": samples}
    L = out["latency_ms"]
    out["throttle_drift_pct"] = round(100 * (L["last_quarter"] - L["first_quarter"]) / L["first_quarter"], 1)
    Path(B.RESULTS_DIR / f"results_{a.tag}_{a.model}_t{a.threads}.json").write_text(json.dumps(out, indent=2))
    print(f"{a.model} t{a.threads} {a.seconds}s: {out['windows_run']} windows "
          f"({out['windows_per_s']}/s) | median {L['median']}ms "
          f"| first-Q {L['first_quarter']} -> last-Q {L['last_quarter']} ({out['throttle_drift_pct']:+.1f}%) "
          f"| CPU {out['cpu_c']['start']}->{out['cpu_c']['max']}C max "
          f"| clk {out['min_mhz_seen']}-{out['max_mhz_seen']}MHz "
          f"| VDD_IN {out['vdd_in_mw']['median']}mW")


if __name__ == "__main__":
    main()
