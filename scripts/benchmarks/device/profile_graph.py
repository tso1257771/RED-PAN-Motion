"""ORT node-level profile: which ops dominate a graph's runtime on this CPU."""
import sys, json, glob, os, collections
import numpy as np, onnxruntime as ort

path = sys.argv[1]; npts = int(sys.argv[2])
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.enable_profiling = True
so.log_severity_level = 3
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
inp = {i.name: rng.standard_normal((1, i.shape[1], npts)).astype(np.float32) for i in sess.get_inputs()}
sess.run(None, inp); sess.run(None, inp)
prof = sess.end_profiling()

ev = json.load(open(prof))
by_op, by_node = collections.Counter(), collections.Counter()
total = 0
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        d = e["dur"]; total += d
        by_op[e["args"].get("op_name", "?")] += d
        by_node[e["name"].replace("_kernel_time", "")] += d
print(f"{os.path.basename(path)}: total kernel time {total/1e6:.2f} s over {len(by_node)} nodes\n")
print("top ops:")
for op, d in by_op.most_common(8):
    print(f"  {op:22s} {d/1e6:8.3f}s  {100*d/total:5.1f}%")
print("\ntop nodes:")
for n, d in by_node.most_common(10):
    print(f"  {n[:58]:58s} {d/1e6:7.3f}s")
os.remove(prof)
