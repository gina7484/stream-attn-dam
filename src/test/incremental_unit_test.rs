#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::node::{
        streamattn_bin::{Binary, BinaryOpType},
        streamattn_matvec::MatVecProd,
        streamattn_qkt::QKTExp,
        streamattn_reduce::{ReduceOp, ReduceOpType},
    };

    #[test]
    fn qkt_reduce_test() {
        const QKT_LATENCY: u64 = 11;
        const REDUCE_LATENCY: u64 = 23;
        const REDUCE_II: u64 = 2;
        const BINARY_LATENCY: u64 = 8;
        const MATVEC_LATENCY: u64 = 12;
        const INIT_INTERVAL: u64 = 1;

        const SEQ_LEN: u64 = 64;
        const SEQ_LEN_F64: f64 = SEQ_LEN as f64;
        let chan_size_long = (SEQ_LEN as usize) + 2;

        let chan_size = 2; // FIFO Depth

        let mut ctx = ProgramBuilder::default();

        // Generators
        let (q_sender, q_receiver) = ctx.bounded::<f64>(SEQ_LEN as usize);
        let (kt_sender, kt_receiver) = ctx.bounded::<f64>(SEQ_LEN as usize);
        let q_iter = || (0..(SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        let kt_iter =
            || (0..(SEQ_LEN * SEQ_LEN)).map(|i| if i % SEQ_LEN == 0 { 0.11_f64 } else { 0.1_f64 });
        ctx.add_child(GeneratorContext::new(q_iter, q_sender)); // Q : [1,D] shaped vectors
        ctx.add_child(GeneratorContext::new(kt_iter, kt_sender)); // KT: [D,1] shaped vectors

        // QKT & Exp block
        let (qkt_exp_short_sender, qkt_exp_short_receiver) =
            ctx.bounded::<f64>((chan_size + QKT_LATENCY) as usize);
        let (qkt_exp_long_sender, qkt_exp_long_receiver) =
            ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);

        ctx.add_child(QKTExp::new(
            q_receiver,
            kt_receiver,
            vec![qkt_exp_short_sender, qkt_exp_long_sender],
            QKT_LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
        ));

        let (rowsum_sender, rowsum_recv) = ctx.bounded::<f64>((SEQ_LEN) as usize);

        ctx.add_child(ReduceOp::new(
            qkt_exp_short_receiver,
            rowsum_sender,
            REDUCE_LATENCY,
            REDUCE_II,
            SEQ_LEN,
            SEQ_LEN,
            ReduceOpType::Sum,
        ));

        // Checkers
        let out_iter1 = || (0..(SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(out_iter1, rowsum_recv, |a, b| {
            true
        }));

        // Checkers
        let out_iter2 = || (0..(SEQ_LEN * SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter2,
            qkt_exp_long_receiver,
            |a, b| true,
        ));

        let initialized = ctx.initialize(Default::default()).unwrap();
        #[cfg(feature = "dot")]
        println!("{}", initialized.to_dot_string());

        let summary = initialized.run(Default::default());
        dbg!(summary.elapsed_cycles());
        #[cfg(feature = "dot")]
        {
            println!("{}", summary.to_dot_string());
        }
    }
}
