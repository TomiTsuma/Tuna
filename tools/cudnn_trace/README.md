# cudnn_trace

Pass-through `libcudnn.so.9` (Linux, x86-64) that forwards every cuDNN export
to the real library and records:
- calls per export,
- for the backend (graph) API: descriptors created and finalized, by
  descriptor type; attributes set and read, by descriptor type, attribute
  name and element type (with the largest element count seen); executions.

It is the coverage tool for growing Tuna's remoted cuDNN set (Phase 3g): run a
workload under it for N and 2N steps, and the difference is what one
steady-state step calls.

```bash
REAL=$(python -c "import nvidia.cudnn as p; print(list(p.__path__)[0])")/lib/libcudnn.so.9
python3 gen_trace.py $REAL gen
gcc -O2 -shared -fPIC -Igen -Wl,-soname,libcudnn.so.9 -Wl,--version-script=gen/libcudnn.map \
    -o bin/libcudnn.so.9 trace.c gen/thunks.S -ldl -lpthread
LD_PRELOAD=bin/libcudnn.so.9 TUNA_REAL_CUDNN=$REAL TUNA_TRACE_LOG=trace.txt python train.py
```

PyTorch finds cuDNN through RPATH, hence `LD_PRELOAD`. The report is written
at exit to `$TUNA_TRACE_LOG` (default stderr). Enum values in the report are
cuDNN's numbers (`cudnnBackendDescriptorType_t`, `cudnnBackendAttributeName_t`,
`cudnnBackendAttributeType_t` in `cudnn_graph.h`). Findings for ResNet-18:
`docs/changelog/*_phase3g_cudnn_remoting_changelog.md`.
