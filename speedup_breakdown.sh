taskset -c 1 cargo test --release --package stream-attn-dam --lib -- 512 test::naive_attn::tests::qkt_red_div_matvec_test --exact --nocapture