use super::streamattn_binary::BinaryOpType;
use dam::context_tools::*;

#[context_macro]
pub struct BinaryOp<A: Clone> {
    // Performs binary op on two scalars: A @ B (element-wise)
    pub in1_stream: Receiver<A>,
    pub in2_stream: Receiver<A>,
    pub out_stream: Sender<A>,
    pub latency: u64,       // pipeline depth
    pub init_inverval: u64, // initiation interval
    pub loop_bound: u64,
    pub op: BinaryOpType,
}

impl<A: DAMType> BinaryOp<A>
where
    BinaryOp<A>: Context,
{
    pub fn new(
        in1_stream: Receiver<A>,
        in2_stream: Receiver<A>,
        out_stream: Sender<A>,
        latency: u64,       // pipeline depth
        init_inverval: u64, // initiation interval
        loop_bound: u64,
        op: BinaryOpType,
    ) -> Self {
        let binary_op = BinaryOp {
            in1_stream,
            in2_stream,
            out_stream,
            latency,
            init_inverval,
            loop_bound,
            op,
            context_info: Default::default(),
        };
        (binary_op.in1_stream).attach_receiver(&binary_op);
        (binary_op.in2_stream).attach_receiver(&binary_op);
        (binary_op.out_stream).attach_sender(&binary_op);

        binary_op
    }
}

impl<A> Context for BinaryOp<A>
where
    A: DAMType + num::Num,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        for _i in 0..self.loop_bound {
            let in1_deq = self.in1_stream.dequeue(&self.time);
            let in2_deq = self.in2_stream.dequeue(&self.time);

            match (in1_deq, in2_deq) {
                (Ok(in1), Ok(in2)) => {
                    let in1_data = in1.data;
                    let in2_data = in2.data;
                    let out_data: A;
                    match self.op {
                        BinaryOpType::Add => {
                            out_data = in1_data + in2_data;
                        }
                        BinaryOpType::Div => {
                            out_data = in1_data / in2_data;
                        }
                        BinaryOpType::Mul => {
                            out_data = in1_data * in2_data;
                        }
                        BinaryOpType::Sub => {
                            out_data = in1_data - in2_data;
                        }
                    }
                    let curr_time = self.time.tick();
                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement::new(curr_time + self.latency, out_data),
                        )
                        .unwrap();
                }
                (_, _) => {
                    panic!("Reached unhandled case");
                }
            }
            self.time.incr_cycles(self.init_inverval);
        }
    }
}
