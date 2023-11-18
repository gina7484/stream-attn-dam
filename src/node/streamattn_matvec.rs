use dam::context_tools::*;

#[context_macro]

pub struct MatVecProd<A: Clone> {
    pub in1_stream: Receiver<A>, // operand 1: A
    pub in2_stream: Receiver<A>, // operand 2: B
    pub out1_stream: Sender<A>,
    pub latency: u64,       // pipeline depth
    pub init_inverval: u64, // initiation interval
    pub inner_loop_bound: u64,
    pub outer_loop_bound: u64,
}

impl<A: DAMType> MatVecProd<A>
where
    MatVecProd<A>: Context,
{
    pub fn new(
        in1_stream: Receiver<A>, // operand 1: A
        in2_stream: Receiver<A>, // operand 2: B
        out1_stream: Sender<A>,
        latency: u64,       // pipeline depth
        init_inverval: u64, // initiation interval
        inner_loop_bound: u64,
        outer_loop_bound: u64,
    ) -> Self {
        let matmul_outer = MatVecProd {
            in1_stream,
            in2_stream,
            out1_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            context_info: Default::default(),
        };
        (matmul_outer.in1_stream).attach_receiver(&matmul_outer);
        (matmul_outer.in2_stream).attach_receiver(&matmul_outer);
        (matmul_outer.out1_stream).attach_sender(&matmul_outer);

        matmul_outer
    }
}

impl<A> Context for MatVecProd<A>
where
    A: DAMType + num::Num + Copy,
{
    fn init(&mut self) {}
    fn run(&mut self) -> () {
        //self.time.incr_cycles(4);
        for _i in 0..self.outer_loop_bound {
            //self.time.incr_cycles(4);
            let s_deq = self.in1_stream.dequeue(&self.time);
            let v_deq = self.in2_stream.dequeue(&self.time);

            match (s_deq, v_deq) {
                (Ok(s_elem), Ok(v_elem)) => {
                    let s_data = s_elem.data;
                    let v_data = v_elem.data;
                    let mut accum_sum = s_data * v_data;

                    self.time.incr_cycles(self.init_inverval);

                    for i in 1..self.inner_loop_bound {
                        let s_deq = self.in1_stream.dequeue(&self.time);
                        let v_deq = self.in2_stream.dequeue(&self.time);

                        match (s_deq, v_deq) {
                            (Ok(s_elem), Ok(v_elem)) => {
                                let s_data = s_elem.data;
                                let v_data = v_elem.data;
                                accum_sum = accum_sum + s_data * v_data;
                            }
                            (_, _) => {
                                panic!("Reached unhandled case");
                            }
                        }
                        if i == self.inner_loop_bound - 1 {
                            let curr_time = self.time.tick();
                            self.out1_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement::new(
                                        curr_time + self.latency,
                                        accum_sum.clone(),
                                    ),
                                )
                                .unwrap();
                        }
                        self.time.incr_cycles(self.init_inverval);
                    }
                }
                (_, _) => {
                    panic!("Reached unhandled case");
                }
            }
        }
    }
}
