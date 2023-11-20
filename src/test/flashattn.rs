#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::node::{
        flashattn_binary_op::BinaryOp, flashattn_running_op::*, streamattn_binary::BinaryOpType,
        streamattn_qkt::QKTExp,
    };

    #[test]
    fn bounded_seq_agnostic_attn() {
        const LATENCY: u64 = 1;
        const INIT_INTERVAL: u64 = 1;

        const SEQ_LEN: u64 = 16;

        let chan_size = 2; // FIFO Depth

        let mut ctx = ProgramBuilder::default();

        // Generators
        let (q_sender, q_receiver) = ctx.bounded::<f64>(chan_size);
        let (kt_sender, kt_receiver) = ctx.bounded::<f64>(chan_size);
        let (v_sender, v_receiver) = ctx.bounded::<f64>(chan_size);

        let q_iter = || (0..(SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        let kt_iter =
            || (0..(SEQ_LEN * SEQ_LEN)).map(|i| if i % SEQ_LEN == 0 { 0.11_f64 } else { 0.1_f64 });
        let v_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|_i| 1_f64);

        ctx.add_child(GeneratorContext::new(q_iter, q_sender)); // Q : [1,D] shaped vectors
        ctx.add_child(GeneratorContext::new(kt_iter, kt_sender)); // KT: [D,1] shaped vectors
        ctx.add_child(GeneratorContext::new(v_iter, v_sender)); // KT: [D,1] shaped vectors

        // QKT & Exp block
        let (qkt_exp_sender, qkt_exp_receiver) = ctx.bounded::<f64>(chan_size);

        ctx.add_child(QKTExp::new(
            q_receiver,
            kt_receiver,
            vec![qkt_exp_sender],
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
        ));

        // Incremental Max
        let (delta_sender1, delta_receiver1) = ctx.bounded::<f64>(chan_size);
        let (delta_sender2, delta_receiver2) = ctx.bounded::<f64>(chan_size);
        let (curr_sender1, curr_receiver1) = ctx.bounded::<f64>(chan_size);
        let (curr_sender2, curr_receiver2) = ctx.bounded::<f64>(chan_size);
        ctx.add_child(IncrMax::new(
            qkt_exp_receiver,
            vec![curr_sender1, curr_sender2],
            vec![delta_sender1, delta_sender2],
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Incremental Sum
        let (rowsum_sender, rowsum_receiver) = ctx.bounded::<f64>(chan_size);
        ctx.add_child(IncrSum::new(
            delta_receiver1,
            curr_receiver1,
            rowsum_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Incremental outer product
        let (matmul_sender, matmul_receiver) = ctx.bounded::<f64>(chan_size);
        ctx.add_child(IncrOutP::new(
            delta_receiver2,
            curr_receiver2,
            v_receiver,
            matmul_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Div
        let (final_sender, final_receiver) = ctx.bounded::<f64>(chan_size);
        ctx.add_child(BinaryOp::new(
            matmul_receiver,
            rowsum_receiver,
            final_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            BinaryOpType::Div,
        ));

        // Checkers
        let out_iter = || (0..(SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter,
            final_receiver,
            |a, b| (a - b).abs() < 0.0001,
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
    fn unbounded_seq_agnostic_attn() {
        const LATENCY: u64 = 1;
        const INIT_INTERVAL: u64 = 1;

        const SEQ_LEN: u64 = 16;

        let chan_size = 2; // FIFO Depth

        let mut ctx = ProgramBuilder::default();

        // Generators
        let (q_sender, q_receiver) = ctx.unbounded::<f64>();
        let (kt_sender, kt_receiver) = ctx.unbounded::<f64>();
        let (v_sender, v_receiver) = ctx.unbounded::<f64>();

        let q_iter = || (0..(SEQ_LEN)).map(|i| (i as f64) * 0.01_f64);
        let kt_iter =
            || (0..(SEQ_LEN * SEQ_LEN)).map(|i| if i % SEQ_LEN == 0 { 0.11_f64 } else { 0.1_f64 });
        let v_iter = || (0..(SEQ_LEN * SEQ_LEN)).map(|_i| 1_f64);

        ctx.add_child(GeneratorContext::new(q_iter, q_sender)); // Q : [1,D] shaped vectors
        ctx.add_child(GeneratorContext::new(kt_iter, kt_sender)); // KT: [D,1] shaped vectors
        ctx.add_child(GeneratorContext::new(v_iter, v_sender)); // KT: [D,1] shaped vectors

        // QKT & Exp block
        let (qkt_exp_sender, qkt_exp_receiver) = ctx.unbounded::<f64>();

        ctx.add_child(QKTExp::new(
            q_receiver,
            kt_receiver,
            vec![qkt_exp_sender],
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
        ));

        // Incremental Max
        let (delta_sender1, delta_receiver1) = ctx.unbounded::<f64>();
        let (delta_sender2, delta_receiver2) = ctx.unbounded::<f64>();
        let (curr_sender1, curr_receiver1) = ctx.unbounded::<f64>();
        let (curr_sender2, curr_receiver2) = ctx.unbounded::<f64>();
        ctx.add_child(IncrMax::new(
            qkt_exp_receiver,
            vec![curr_sender1, curr_sender2],
            vec![delta_sender1, delta_sender2],
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Incremental Sum
        let (rowsum_sender, rowsum_receiver) = ctx.unbounded::<f64>();
        ctx.add_child(IncrSum::new(
            delta_receiver1,
            curr_receiver1,
            rowsum_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Incremental outer product
        let (matmul_sender, matmul_receiver) = ctx.unbounded::<f64>();
        ctx.add_child(IncrOutP::new(
            delta_receiver2,
            curr_receiver2,
            v_receiver,
            matmul_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            SEQ_LEN,
        ));

        // Div
        let (final_sender, final_receiver) = ctx.unbounded::<f64>();
        ctx.add_child(BinaryOp::new(
            matmul_receiver,
            rowsum_receiver,
            final_sender,
            LATENCY,
            INIT_INTERVAL,
            SEQ_LEN,
            BinaryOpType::Div,
        ));

        // Checkers
        let out_iter = || (0..(SEQ_LEN)).map(|_i| (1_f64));
        ctx.add_child(ApproxCheckerContext::new(
            out_iter,
            final_receiver,
            |a, b| (a - b).abs() < 0.0001,
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
