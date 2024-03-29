mod naive_attn;

use clap::Parser;

#[derive(Parser, Debug)]
struct Cli {
    seq_len: usize
}

fn main() {
    let args = Cli::parse();
    assert!(args.seq_len >= 1);
}