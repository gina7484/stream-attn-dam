#[cfg(test)]
mod tests {
    use dam::{
        simulation::{DotConvertible, ProgramBuilder},
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::node::{
        streamattn_binary::{Binary, BinaryOpType},
        streamattn_matvec::MatVecProd,
        streamattn_qkt::QKTExp,
        streamattn_reduce::{ReduceOp, ReduceOpType},
    };

    #[test]
    fn qkt_test() {
        const QKT_LATENCY: u64 = 11;
        const REDUCE_LATENCY: u64 = 2;
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
            ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
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

        // Checkers
        let out_iter1 = || (0..(SEQ_LEN * SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter1,
            qkt_exp_short_receiver,
            |a, b| true,
        ));

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

    #[test]
    fn reduce_test() {
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
        let (qtk_sender, qtk_receiver) = ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
        let qkt_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(qkt_iter, qtk_sender)); // Q : [1,D] shaped vectors

        // QKT & Exp block
        let (rowsum_sender, rowsum_recv) = ctx.bounded::<f64>((SEQ_LEN) as usize);

        ctx.add_child(ReduceOp::new(
            qtk_receiver,
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

    #[test]
    fn division_test() {
        const QKT_LATENCY: u64 = 11;
        const REDUCE_LATENCY: u64 = 24;
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
        let (qtk_sender, qtk_receiver) = ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
        // In spatial unit test, it reads the whole SRAM[N] N times.
        // To allow the same effecet, we model this through a FIFO[N*N].
        let qkt_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(qkt_iter, qtk_sender)); // Q : [1,D] shaped vectors

        let (rowsum_sender, rowsum_recv) = ctx.bounded::<f64>((SEQ_LEN) as usize);
        let rowsum_iter = || (0..(SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(rowsum_iter, rowsum_sender)); // Q : [1,D] shaped vectors

        let (output_sender, output_receiver) = ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
        ctx.add_child(Binary::<f64>::new(
            qtk_receiver,
            rowsum_recv,
            output_sender,
            BINARY_LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
            BinaryOpType::Div,
        ));
        // Checkers
        let out_iter1 = || (0..(SEQ_LEN * SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter1,
            output_receiver,
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

    #[test]
    fn matvec_test() {
        const QKT_LATENCY: u64 = 11;
        const REDUCE_LATENCY: u64 = 23;
        const REDUCE_II: u64 = 2;
        const BINARY_LATENCY: u64 = 8;
        const MATVEC_LATENCY: u64 = 13;
        const MATVEC_II: u64 = 2;
        const INIT_INTERVAL: u64 = 1;

        const SEQ_LEN: u64 = 64;

        let chan_size = 2; // FIFO Depth

        let mut ctx: ProgramBuilder<'_> = ProgramBuilder::default();

        // Generators
        // QKOut = FIFO[T](N*N)
        let (qtk_sender, qtk_receiver) = ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
        let qkt_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(qkt_iter, qtk_sender)); // Q : [1,D] shaped vectors

        // V = SRAM[T](N) -> As this is a SRAM where we read N*N times, this will be a generator with a N*N long iter
        let (v_sender, v_receiver) = ctx.bounded::<f64>((SEQ_LEN * SEQ_LEN) as usize);
        let v_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        ctx.add_child(GeneratorContext::new(v_iter, v_sender)); // Q : [1,D] shaped vectors

        // QKT & Exp block
        let (matvec_sender, matvec_receiver) = ctx.bounded::<f64>((SEQ_LEN) as usize);

        ctx.add_child(MatVecProd::new(
            qtk_receiver,
            v_receiver,
            matvec_sender,
            MATVEC_LATENCY,
            MATVEC_II,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Checkers
        // output = FIFO[T](N)
        let out_iter1 = || (0..(SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter1,
            matvec_receiver,
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
