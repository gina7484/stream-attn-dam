#[cfg(test)]
mod tests {
    use dam::{
        simulation::{
            InitializationOptionsBuilder, ProgramBuilder,
        },
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::node::{
        streamattn_binary::{Binary, BinaryOpType},
        streamattn_matvec::MatVecProd,
        streamattn_qkt::QKTExp,
        streamattn_reduce::{ReduceOp, ReduceOpType},
    };

    use std::env;

    #[test]
    fn qkt_red_div_matvec_test() {
        let args: Vec<String> = env::args().collect();

        let arg = match args.get(1) {
            Some(val) => val,
            None => {
                println!("Not enough arguments provided!");
                return;
            }
        };

        let seq_len: u64 = match arg.parse::<u64>() {
            Ok(val) => val,
            Err(e) => {
                println!("Unable to parse number from argument: {}", e);
                return;
            }
        };

        const QKT_LATENCY: u64 = 11;
        const REDUCE_LATENCY: u64 = 23;
        const REDUCE_II: u64 = 2;
        const BINARY_LATENCY: u64 = 8;
        const MATVEC_LATENCY: u64 = 13;
        const MATVEC_II: u64 = 2;
        const INIT_INTERVAL: u64 = 1;

        let chan_size = 2; // FIFO Depth

        let mut ctx = ProgramBuilder::default();

        // Generators
        // Q = FIFO[T](N)
        let (q_sender, q_receiver) = ctx.bounded::<f64>(seq_len as usize);
        let q_iter = || (0..(seq_len)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(q_iter, q_sender)); // Q : [1,D] shaped vectors

        // K = SRAM[T](N)-> As this is a SRAM where we read N*N times,
        // this will be a generator with a N*N long iter
        let (kt_sender, kt_receiver) = ctx.bounded::<f64>((seq_len * seq_len) as usize);
        let kt_iter =
            || (0..(seq_len * seq_len)).map(|i| if i % seq_len == 0 { 0.11_f64 } else { 0.1_f64 });
        ctx.add_child(GeneratorContext::new(kt_iter, kt_sender)); // KT: [D,1] shaped vectors

        // V = SRAM[T](N) -> As this is a SRAM where we read N*N times, this will be a generator with a N*N long iter
        let (v_sender, v_receiver) = ctx.bounded::<f64>((seq_len * seq_len) as usize);
        let v_iter =
            || (0..(seq_len * seq_len)).map(|i| if i % seq_len == 0 { 0.11_f64 } else { 0.1_f64 });
        ctx.add_child(GeneratorContext::new(v_iter, v_sender)); // KT: [D,1] shaped vectors

        // ===================== QKT & Exp block =====================
        // QK1 = FIFO[T](3) -> 3 = chan_size + (REDUCE_II - 1)
        // The (QKT_LATENCY - 1) term is to emulate the pipeline bubble squashing behavior in Spatial
        let (qkt_exp_short_sender, qkt_exp_short_receiver) =
            ctx.bounded::<f64>((chan_size + (REDUCE_II - 1) + QKT_LATENCY - 1) as usize);
        // QK2 = FIFO[T](N+24)
        // N+24 = SEQ_LEN + REDUCE_LATENCY + 1
        // The (QKT_LATENCY - 1) term is to emulate the pipeline bubble squashing behavior in Spatial
        let (qkt_exp_long_sender, qkt_exp_long_receiver) =
            ctx.bounded::<f64>((seq_len + REDUCE_LATENCY + 1 + QKT_LATENCY - 1) as usize);

        ctx.add_child(QKTExp::new(
            q_receiver,
            kt_receiver,
            vec![qkt_exp_short_sender, qkt_exp_long_sender],
            QKT_LATENCY,
            INIT_INTERVAL,
            seq_len,
        ));

        // ===================== Row Sum =====================
        // QKRecipSum = FIFO[T](2)
        //  |_ 2 = chan_size
        //         (min channel depth for full-throughput between
        //         two sender and receiver
        //         where (enq rate of sender) < (deq rate of receiver))
        let (rowsum_sender, rowsum_recv) =
            ctx.bounded::<f64>((chan_size + REDUCE_LATENCY - 1) as usize);
        ctx.add_child(ReduceOp::new(
            qkt_exp_short_receiver,
            rowsum_sender,
            REDUCE_LATENCY,
            REDUCE_II,
            seq_len,
            seq_len,
            ReduceOpType::Sum,
        ));

        // ===================== Div =====================
        // QKOut = FIFO[T](2)
        let (div_sender, div_receiver) =
            ctx.bounded::<f64>((chan_size + BINARY_LATENCY - 1) as usize);
        ctx.add_child(Binary::<f64>::new(
            qkt_exp_long_receiver,
            rowsum_recv,
            div_sender,
            BINARY_LATENCY,
            INIT_INTERVAL,
            seq_len,
            seq_len,
            BinaryOpType::Div,
        ));

        // ===================== MatVec =====================
        // output = FIFO[T](N)
        let (matvec_sender, matvec_receiver) = ctx.bounded::<f64>(seq_len as usize);
        ctx.add_child(MatVecProd::new(
            div_receiver,
            v_receiver,
            matvec_sender,
            MATVEC_LATENCY,
            MATVEC_II,
            seq_len,
            seq_len,
        ));

        // Checkers
        let out_iter1 = || (0..(seq_len)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter1,
            matvec_receiver,
            |_a, _b| true,
        ));

        let flavor_inf: bool = false;
        let initialized = ctx
            .initialize(
                InitializationOptionsBuilder::default()
                    .run_flavor_inference(flavor_inf)
                    .build()
                    .unwrap(),
            )
            .unwrap();
        // #[cfg(feature = "dot")]
        // println!("{}", initialized.to_dot_string());

        let summary = initialized.run(Default::default());
        println!("elapsed cycles: {}",summary.elapsed_cycles().unwrap());
        // dbg!(summary.elapsed_cycles());
        // #[cfg(feature = "dot")]
        // {
        //     println!("{}", summary.to_dot_string());
        // }
    }
}
