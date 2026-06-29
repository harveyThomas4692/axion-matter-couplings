#!/usr/bin/env python3
"""
Compatibility shim for cymetric's JAX trainer (helper.py train_model).

By default `train_step = _make_train_step(optimizer)` is created INSIDE the per-epoch
loop, so every epoch builds a fresh @eqx.filter_jit function and XLA recompiles the
step from scratch, never freeing the prior compiled code. On some platforms the
mmap'd executable sections accumulate until the process map limit is exhausted,
aborting with "LLVM compilation error: Cannot allocate memory".

This shim hoists the train_step creation OUT of the epoch loop (compile once, reuse).
The optimizer is fixed across epochs, so the change is behaviour-preserving.

Idempotent. Run after (re)installing cymetric[jax]:
    python moduli_scan/patch_cymetric.py
"""
import os, sys
import cymetric.jax.models.helper as H

path = H.__file__
src = open(path).read()
MARK = "[SCAN-PATCH] hoisted"

if MARK in src:
    print(f"already patched: {path}")
    sys.exit(0)

IN_LOOP = "        train_step = _make_train_step(optimizer)\n"
LOOP_HDR = "    for epoch in range(epochs):\n"

assert src.count(IN_LOOP) == 1, f"expected 1 in-loop train_step, got {src.count(IN_LOOP)}"
assert src.count(LOOP_HDR) == 1, f"expected 1 epoch-loop header, got {src.count(LOOP_HDR)}"

# remove the in-loop creation
src = src.replace(IN_LOOP, "")
# insert a single creation just before the epoch loop
src = src.replace(
    LOOP_HDR,
    "    train_step = _make_train_step(optimizer)  # [SCAN-PATCH] hoisted out of epoch "
    "loop (Linux vm.max_map_count fix)\n" + LOOP_HDR,
)

bak = path + ".orig"
if not os.path.exists(bak):
    open(bak, "w").write(open(path).read())
open(path, "w").write(src)
print(f"patched: {path}\nbackup:  {bak}")
