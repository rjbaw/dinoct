import numpy as np
import openvino as ov
import statistics

core = ov.Core()
model = core.read_model("curve_model.onnx")

model.reshape([1, 3, 512, 500])

config = {"PERFORMANCE_HINT": "LATENCY"}
compiled_model = core.compile_model(model, "GPU", config)

ireqs = ov.AsyncInferQueue(compiled_model)

input_tensor = np.random.rand(1, 3, 512, 500).astype(np.float32)

for ireq in ireqs:
    ireq.set_input_tensor(0, ov.Tensor(input_tensor))

print("Warming up...")
for _ in range(5):
    ireqs.start_async()
    ireqs.wait_all()

print("Benchmarking 333 iterations...")
latencies = []

for _ in range(333):
    idle_id = ireqs.get_idle_request_id()
    if ireqs[idle_id].latency > 0:
        latencies.append(ireqs[idle_id].latency)
    ireqs.start_async()

ireqs.wait_all()
for ireq in ireqs:
    latencies.append(ireq.latency)

print(f"Count: {len(latencies)} iterations")
print(f"Average Latency: {statistics.mean(latencies):.2f} ms")
print(f"Median Latency: {statistics.median(latencies):.2f} ms")
print(f"Min Latency: {min(latencies):.2f} ms")
print(f"Max Latency: {max(latencies):.2f} ms")

print(f"Std Deviation: {statistics.stdev(latencies):.2f} ms")
