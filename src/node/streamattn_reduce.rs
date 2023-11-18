use dam::context_tools::*;

pub trait MinMax {
    fn get_max(self, rhs: Self) -> Self;
    fn get_min_val() -> Self;
    fn get_zero() -> Self;
}
impl MinMax for u8 {
    fn get_max(self, rhs: u8) -> u8 {
        self.max(rhs)
    }

    fn get_min_val() -> u8 {
        u8::MIN
    }

    fn get_zero() -> u8 {
        0u8
    }
}
impl MinMax for u16 {
    fn get_max(self, rhs: u16) -> u16 {
        self.max(rhs)
    }
    fn get_min_val() -> u16 {
        u16::MIN
    }
    fn get_zero() -> u16 {
        0u16
    }
}
impl MinMax for u32 {
    fn get_max(self, rhs: u32) -> u32 {
        self.max(rhs)
    }
    fn get_min_val() -> u32 {
        u32::MIN
    }
    fn get_zero() -> u32 {
        0u32
    }
}
impl MinMax for u64 {
    fn get_max(self, rhs: u64) -> u64 {
        self.max(rhs)
    }
    fn get_min_val() -> u64 {
        u64::MIN
    }
    fn get_zero() -> u64 {
        0u64
    }
}
impl MinMax for i8 {
    fn get_max(self, rhs: i8) -> i8 {
        self.max(rhs)
    }
    fn get_min_val() -> i8 {
        i8::MIN
    }
    fn get_zero() -> i8 {
        0i8
    }
}
impl MinMax for i16 {
    fn get_max(self, rhs: i16) -> i16 {
        self.max(rhs)
    }
    fn get_min_val() -> i16 {
        i16::MIN
    }
    fn get_zero() -> i16 {
        0i16
    }
}
impl MinMax for i32 {
    fn get_max(self, rhs: i32) -> i32 {
        self.max(rhs)
    }
    fn get_min_val() -> i32 {
        i32::MIN
    }
    fn get_zero() -> i32 {
        0i32
    }
}
impl MinMax for i64 {
    fn get_max(self, rhs: i64) -> i64 {
        self.max(rhs)
    }
    fn get_min_val() -> i64 {
        i64::MIN
    }
    fn get_zero() -> i64 {
        0i64
    }
}
impl MinMax for f32 {
    fn get_max(self, rhs: f32) -> f32 {
        self.max(rhs)
    }
    fn get_min_val() -> f32 {
        f32::MIN
    }
    fn get_zero() -> f32 {
        0_f32
    }
}
impl MinMax for f64 {
    fn get_max(self, rhs: f64) -> f64 {
        self.max(rhs)
    }
    fn get_min_val() -> f64 {
        f64::MIN
    }
    fn get_zero() -> f64 {
        0_f64
    }
}

pub enum ReduceOpType {
    Max,
    Sum,
}

#[context_macro]
pub struct ReduceOp<A: Clone> {
    pub in_stream: Receiver<A>, // operand: scalar (element of a 'inner_loop_bound' long vector)
    pub out_stream: Sender<A>,  // output -> scalar FIFO
    pub latency: u64,           // pipeline depth to do a computation on a scalar value
    pub init_inverval: u64,     // initiation interval
    pub inner_loop_bound: u64, // As this is a reduction, we need a inner loop bound to specify how many elements are reduce
    pub outer_loop_bound: u64,
    op: ReduceOpType,
}

impl<A: DAMType> ReduceOp<A>
where
    ReduceOp<A>: Context,
{
    pub fn new(
        in_stream: Receiver<A>,
        out_stream: Sender<A>,
        latency: u64,
        init_inverval: u64,
        inner_loop_bound: u64,
        outer_loop_bound: u64,
        op: ReduceOpType,
    ) -> Self {
        let reduce = ReduceOp {
            in_stream,
            out_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            op,
            context_info: Default::default(),
        };
        (reduce.in_stream).attach_receiver(&reduce);
        (reduce.out_stream).attach_sender(&reduce);

        reduce
    }
}

impl<A> Context for ReduceOp<A>
where
    A: DAMType + num::Num + MinMax + Copy,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        //self.time.incr_cycles(4);
        for _i in 0..self.outer_loop_bound {
            //self.time.incr_cycles(4);
            let first_peek = self.in_stream.dequeue(&self.time);
            match first_peek {
                Ok(first_elem) => {
                    let mut temp_res = first_elem.data;
                    self.time.incr_cycles(self.init_inverval);
                    for i in 1..self.inner_loop_bound {
                        let in_deq = self.in_stream.dequeue(&self.time);
                        match in_deq {
                            Ok(in_elem) => {
                                let in_data = in_elem.data;
                                match self.op {
                                    ReduceOpType::Max => {
                                        temp_res = temp_res.get_max(in_data);
                                    }
                                    ReduceOpType::Sum => {
                                        temp_res = temp_res + in_data;
                                    }
                                }
                            }
                            _ => {
                                panic!("Reached unhandled case");
                            }
                        }
                        if i == self.inner_loop_bound - 1 {
                            let curr_time = self.time.tick();
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement::new(curr_time + self.latency, temp_res),
                                )
                                .unwrap();
                        }
                        self.time.incr_cycles(self.init_inverval);
                    }
                }
                _ => {
                    panic!("Reached unhandled case");
                }
            }
        }
    }
}
