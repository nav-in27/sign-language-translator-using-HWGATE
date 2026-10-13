import time
import json
from pydantic import BaseModel
from typing import List, Union

class KeypointData(BaseModel):
    keypoints: List[List[float]]
    confidences: Union[List[List[float]], List[float]]

payload = {
    "keypoints": [[0.1234, 0.5678, 0.0] for _ in range(133)],
    "confidences": [[1.0] for _ in range(133)]
}
payload_str = json.dumps(payload)

# Test 1: Pydantic parsing
t0 = time.perf_counter()
for _ in range(100):
    d = json.loads(payload_str)
    obj = KeypointData(**d)
pydantic_time = (time.perf_counter() - t0) * 10

# Test 2: Direct json.loads without Pydantic
t0 = time.perf_counter()
for _ in range(100):
    d = json.loads(payload_str)
    kps = d["keypoints"]
    confs = d["confidences"]
direct_time = (time.perf_counter() - t0) * 10

print(f"Pydantic Validation Latency: {pydantic_time:.2f} ms per frame")
print(f"Direct JSON Parsing Latency: {direct_time:.2f} ms per frame")
print(f"Speedup: {pydantic_time / max(0.001, direct_time):.1f}x faster!")
