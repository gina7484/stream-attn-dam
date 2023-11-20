use dam::context_tools::*;

pub enum BinaryOpType {
    Add,
    Sub,
    Div,
    Mul,
}

#[context_macro]
pub struct Binary<A: Clone> {
    in1_stream: Receiver<A>,     // operand 1: A
    pub in2_stream: Receiver<A>, // operand 2: B
    pub out1_stream: Sender<A>,
    pub latency: u64,       // pipeline depth
    pub init_inverval: u64, // initiation interval
    pub inner_loop_bound: u64,
    pub outer_loop_bound: u64,
    op: BinaryOpType,
}

impl<A: DAMType> Binary<A>
where
    Binary<A>: Context,
{
    pub fn new(
        in1_stream: Receiver<A>, // operand 1: A
        in2_stream: Receiver<A>, // operand 2: B
        out1_stream: Sender<A>,
        latency: u64,       // pipeline depth
        init_inverval: u64, // initiation interval
        inner_loop_bound: u64,
        outer_loop_bound: u64,
        op: BinaryOpType,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out1_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            op,
            context_info: Default::default(),
        };
        ctx.in1_stream.attach_receiver(&ctx);
        ctx.in2_stream.attach_receiver(&ctx);
        ctx.out1_stream.attach_sender(&ctx);

        ctx
    }
}

impl<A: DAMType + num::Num> Context for Binary<A> {
    fn run(&mut self) {
        //self.time.incr_cycles(4);
        for _i in 0..self.outer_loop_bound {
            //self.time.incr_cycles(4);
            let _ = self.in1_stream.peek_next(&self.time);
            let _ = self.in2_stream.peek_next(&self.time);
            let in1_deq = self.in1_stream.dequeue(&self.time);
            let in2_deq = self.in2_stream.dequeue(&self.time);

            match (in1_deq, in2_deq) {
                (Ok(in1), Ok(in2)) => {
                    let in1_data = in1.data;
                    let in2_data = in2.data;
                    let out_data: A;
                    match self.op {
                        BinaryOpType::Add => {
                            out_data = in1_data + in2_data.clone();
                        }
                        BinaryOpType::Div => {
                            out_data = in1_data / in2_data.clone();
                        }
                        BinaryOpType::Mul => {
                            out_data = in1_data * in2_data.clone();
                        }
                        BinaryOpType::Sub => {
                            out_data = in1_data - in2_data.clone();
                        }
                    }
                    let curr_time = self.time.tick();
                    self.out1_stream
                        .enqueue(
                            &self.time,
                            ChannelElement::new(curr_time + self.latency, out_data.clone()),
                        )
                        .unwrap();

                    self.time.incr_cycles(self.init_inverval);

                    for _i in 1..self.inner_loop_bound {
                        let in1_deq = self.in1_stream.dequeue(&self.time);

                        match in1_deq {
                            Ok(in1) => {
                                let in1_data = in1.data;
                                let out_data: A;
                                match self.op {
                                    BinaryOpType::Add => {
                                        out_data = in1_data + in2_data.clone();
                                    }
                                    BinaryOpType::Div => {
                                        out_data = in1_data / in2_data.clone();
                                    }
                                    BinaryOpType::Mul => {
                                        out_data = in1_data * in2_data.clone();
                                    }
                                    BinaryOpType::Sub => {
                                        out_data = in1_data - in2_data.clone();
                                    }
                                }
                                let curr_time = self.time.tick();
                                self.out1_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement::new(
                                            curr_time + self.latency,
                                            out_data.clone(),
                                        ),
                                    )
                                    .unwrap();

                                self.time.incr_cycles(self.init_inverval);
                            }
                            _ => {
                                panic!("Reached unhandled case");
                            }
                        }
                    }
                }
                (_, _) => {
                    panic!("Reached unhandled case");
                }
            }
        }
    }
}
