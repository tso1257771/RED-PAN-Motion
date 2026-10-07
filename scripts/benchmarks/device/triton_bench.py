"""Benchmark models served by Triton over HTTP (binary tensor extension).

Reports client-observed latency and, from Triton's own stats, the server-side
breakdown (queue / input / infer / output), so serving overhead is separated
from raw compute. Verifies the served output against local ORT once per model.
"""
import argparse, json, http.client, statistics as st, struct, time
from pathlib import Path
import numpy as np
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # scripts/benchmarks/device -> repo root
ONNX_DIR = Path(os.environ.get("RPM_ONNX_DIR", HERE / "onnx"))
RESULTS_DIR = Path(os.environ.get("RPM_RESULTS_DIR", HERE / "results"))
CKPT_DIR = Path(os.environ.get("RPM_CKPT_DIR", REPO / "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


HOST = os.environ.get("RPM_TRITON_HOST", "localhost")
PORT = int(os.environ.get("RPM_TRITON_PORT", "7183"))
rng = np.random.default_rng(0)

SPECS = {
    "edge_rp90":      {"npts": 9000, "inputs": {"x": (1, 3, 9000), "z_raw": (1, 1, 9000)},
                       "outputs": ["picker", "polarity", "detector"], "onnx": "edge_rp90.onnx"},
    "edge_rp90_int8": {"npts": 9000, "inputs": {"x": (1, 3, 9000), "z_raw": (1, 1, 9000)},
                       "outputs": ["picker", "polarity", "detector"], "onnx": "edge_rp90_int8.onnx"},
    "onnx_RP60_ref":  {"npts": 6000, "inputs": {"input": (1, 6000, 3)},
                       "outputs": ["picker", "detector"], "onnx": None},
}


def make_arrays(spec, event=True):
    out = {}
    for name, shape in spec["inputs"].items():
        a = rng.standard_normal(shape).astype(np.float32)
        if name == "z_raw":
            a = (rng.random(shape, dtype=np.float32) * 2 - 1)
        elif event:
            npts = spec["npts"]
            t0 = int(rng.integers(1000, npts - 1200)); t = np.arange(600)
            pulse = 6.0 * (np.sin(2 * np.pi * 0.05 * t) * np.exp(-t / 150)).astype(np.float32)
            if a.shape[-1] == npts:            # (B, C, T)
                a[..., t0:t0 + 600] += pulse
            else:                              # (B, T, C)
                a[:, t0:t0 + 600, :] += pulse[None, :, None]
        out[name] = np.ascontiguousarray(a)
    return out


def build_request(model, arrays, spec):
    body, inputs = b"", []
    for name, arr in arrays.items():
        raw = arr.tobytes()
        inputs.append({"name": name, "shape": list(arr.shape), "datatype": "FP32",
                       "parameters": {"binary_data_size": len(raw)}})
        body += raw
    header = json.dumps({"inputs": inputs,
                         "outputs": [{"name": o, "parameters": {"binary_data": True}}
                                     for o in spec["outputs"]]}).encode()
    return header, header + body


def infer(conn, model, header, payload):
    conn.request("POST", f"/v2/models/{model}/infer", body=payload,
                 headers={"Inference-Header-Content-Length": str(len(header)),
                          "Content-Type": "application/octet-stream",
                          "Content-Length": str(len(payload))})
    r = conn.getresponse()
    data = r.read()
    if r.status != 200:
        raise RuntimeError(f"{r.status}: {data[:300]}")
    return r.getheader("Inference-Header-Content-Length"), data


def parse_outputs(hdr_len, data):
    hdr = json.loads(data[:int(hdr_len)])
    buf, off, out = data[int(hdr_len):], 0, {}
    for o in hdr["outputs"]:
        n = int(np.prod(o["shape"])) * 4
        out[o["name"]] = np.frombuffer(buf[off:off + n], dtype=np.float32).reshape(o["shape"])
        off += n
    return out


def server_stats(conn, model):
    conn.request("GET", f"/v2/models/{model}/stats")
    r = conn.getresponse(); d = json.loads(r.read())
    s = d["model_stats"][0]["inference_stats"]
    b = d["model_stats"][0].get("batch_stats", [{}])
    return {"count": s["success"]["count"], "success_ns": s["success"]["ns"],
            "queue_ns": s["queue"]["ns"], "compute_input_ns": s["compute_input"]["ns"],
            "compute_infer_ns": s["compute_infer"]["ns"], "compute_output_ns": s["compute_output"]["ns"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="onnx_RP60_ref,edge_rp90,edge_rp90_int8")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="triton")
    a = ap.parse_args()

    print(f"{'model':16s}{'client med':>12s}{'p90':>9s}{'sd':>8s}{'srv infer':>11s}"
          f"{'queue':>8s}{'in/out':>9s}{'overhead':>10s}{'req/s':>8s}")
    report = {}
    for model in a.models.split(","):
        spec = SPECS[model]
        conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
        batches = [build_request(model, make_arrays(spec, i % 2 == 0), spec) for i in range(4)]
        for i in range(a.warmup):
            infer(conn, model, *batches[i % 4])

        s0 = server_stats(conn, model)
        lat = []
        t_wall = time.perf_counter()
        for i in range(a.reps):
            h, p = batches[i % 4]
            t0 = time.perf_counter()
            infer(conn, model, h, p)
            lat.append((time.perf_counter() - t0) * 1000)
        wall = time.perf_counter() - t_wall
        s1 = server_stats(conn, model)

        n = s1["count"] - s0["count"]
        d = lambda k: (s1[k] - s0[k]) / n / 1e6
        med = st.median(lat)
        ls = sorted(lat)
        srv_infer, q = d("compute_infer_ns"), d("queue_ns")
        io = d("compute_input_ns") + d("compute_output_ns")
        r = {"model": model, "reps": a.reps,
             "client_ms": {"median": round(med, 1), "min": round(min(lat), 1),
                           "p90": round(ls[int(0.9 * len(ls))], 1),
                           "stdev": round(st.stdev(lat), 2)},
             "server_ms": {"success_total": round(d("success_ns"), 1), "queue": round(q, 2),
                           "compute_input": round(d("compute_input_ns"), 2),
                           "compute_infer": round(srv_infer, 1),
                           "compute_output": round(d("compute_output_ns"), 2)},
             "serving_overhead_ms": round(med - srv_infer, 1),
             "requests_per_s": round(a.reps / wall, 2)}
        # correctness: compare the served picker against local ORT for the edge models
        if spec["onnx"]:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.log_severity_level = 3
            sess = ort.InferenceSession(str(ONNX_DIR / spec["onnx"]), so,
                                        providers=["CPUExecutionProvider"])
            arrays = make_arrays(spec, True)
            h, p = build_request(model, arrays, spec)
            served = parse_outputs(*infer(conn, model, h, p))
            local = sess.run(None, {k: v for k, v in arrays.items()})
            r["max_abs_diff_vs_local_ort"] = max(
                float(np.abs(served[n2] - l).max()) for n2, l in zip(spec["outputs"], local))
        report[model] = r
        c = r["client_ms"]; s = r["server_ms"]
        print(f"{model:16s}{c['median']:11.1f}ms{c['p90']:9.1f}{c['stdev']:8.2f}{s['compute_infer']:10.1f}ms"
              f"{s['queue']:8.2f}{r['server_ms']['compute_input']+r['server_ms']['compute_output']:9.2f}"
              f"{r['serving_overhead_ms']:9.1f}ms{r['requests_per_s']:8.2f}")
        conn.close()
    (RESULTS_DIR / f"results_{a.tag}.json").write_text(json.dumps(report, indent=2))
    for m, r in report.items():
        if "max_abs_diff_vs_local_ort" in r:
            print(f"  {m}: served vs local ORT max|diff| = {r['max_abs_diff_vs_local_ort']:.2e}")


if __name__ == "__main__":
    main()
