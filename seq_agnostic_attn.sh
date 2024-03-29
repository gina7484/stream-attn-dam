#!/bin/bash

run_single() {
    echo "Running sequence-length-agnostic-attention for $LEN with BOUNDED channel depths"
    cargo test --release --package stream-attn-dam --lib -- $LEN test::seq_agnostic::tests::bounded_seq_agnostic_attn --exact --nocapture > bounded_run_$LEN.txt
    echo "Running sequence-length-agnostic-attention for $LEN with INFINITE channel depths"
    cargo test --release --package stream-attn-dam --lib -- $LEN test::seq_agnostic::tests::unbounded_seq_agnostic_attn --exact --nocapture > unbounded_run_$LEN.txt
    BOUNDED="$(grep Simulated bounded_run_$LEN.txt)"
    INFINITE="$(grep Simulated unbounded_run_$LEN.txt)"
    echo "[BOUNDED]  $BOUNDED"
    echo "[INFINITE] $INFINITE"
    echo ""
    rm bounded_run_$LEN.txt unbounded_run_$LEN.txt
}

cd /home/dam/dam-experiments/stream-attn-dam/

# The total sweep will be 3.5 -4 hr
LEN=512     # 3.5 sec
run_single

LEN=2048    # 45-50 sec
run_single

LEN=8192    # 12-13 min
run_single

LEN=32768   # 3.5 hr
run_single