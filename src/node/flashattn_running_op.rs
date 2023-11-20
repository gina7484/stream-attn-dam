use dam::context_tools::*;

use super::streamattn_reduce::MinMax;

#[context_macro]
pub struct IncrMax<A: Clone> {
    pub in_stream: Receiver<A>,
    pub delta_out_stream: Vec<Sender<A>>,
    pub curr_out_stream: Vec<Sender<A>>,
    pub latency: u64,
    pub init_inverval: u64,
    pub inner_loop_bound: u64,
    pub outer_loop_bound: u64,
}

impl<A: DAMType> IncrMax<A>
where
    IncrMax<A>: Context,
{
    pub fn new(
        in_stream: Receiver<A>,
        delta_out_stream: Vec<Sender<A>>,
        curr_out_stream: Vec<Sender<A>>,
        latency: u64,
        init_inverval: u64,
        inner_loop_bound: u64,
        outer_loop_bound: u64,
    ) -> Self {
        let incr_max = IncrMax {
            in_stream,
            delta_out_stream,
            curr_out_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            context_info: Default::default(),
        };
        (incr_max.in_stream).attach_receiver(&incr_max);
        for i in incr_max.delta_out_stream.iter() {
            i.attach_sender(&incr_max);
        }
        for i in incr_max.curr_out_stream.iter() {
            i.attach_sender(&incr_max);
        }

        incr_max
    }
}

impl<A> Context for IncrMax<A>
where
    A: DAMType + num::Float + MinMax + Copy,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        for _i in 0..self.outer_loop_bound {
            let mut temp_res = A::get_min_val();
            for _j in 0..self.inner_loop_bound {
                let in_deq = self.in_stream.dequeue(&self.time);
                match in_deq {
                    Ok(in_elem) => {
                        // First Iteration
                        let in_data = in_elem.data;
                        let new_max = temp_res.get_max(in_data);
                        let delta = (temp_res - new_max).exp();
                        let curr = (in_data - new_max).exp();
                        temp_res = new_max;

                        let curr_time = self.time.tick();
                        for k in self.delta_out_stream.iter() {
                            k.enqueue(
                                &self.time,
                                ChannelElement::new(curr_time + self.latency, delta.clone()),
                            )
                            .unwrap();
                        }
                        for k in self.curr_out_stream.iter() {
                            k.enqueue(
                                &self.time,
                                ChannelElement::new(curr_time + self.latency, curr.clone()),
                            )
                            .unwrap();
                        }

                        self.time.incr_cycles(self.init_inverval);
                        // initiation interval
                    }
                    _ => {
                        panic!("Reached unhandled case");
                    }
                }
            }
        }
    }
}

#[context_macro]
pub struct IncrSum<A: Clone> {
    pub in_delta_stream: Receiver<A>,
    pub in_curr_stream: Receiver<A>,
    pub out_stream: Sender<A>,
    pub latency: u64,
    pub init_inverval: u64,
    pub inner_loop_bound: u64,
    pub outer_loop_bound: u64,
}

impl<A: DAMType> IncrSum<A>
where
    IncrSum<A>: Context,
{
    pub fn new(
        in_delta_stream: Receiver<A>,
        in_curr_stream: Receiver<A>,
        out_stream: Sender<A>,
        latency: u64,
        init_inverval: u64,
        inner_loop_bound: u64,
        outer_loop_bound: u64,
    ) -> Self {
        let incr_sum = IncrSum {
            in_delta_stream,
            in_curr_stream,
            out_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            context_info: Default::default(),
        };
        (incr_sum.in_delta_stream).attach_receiver(&incr_sum);
        (incr_sum.in_curr_stream).attach_receiver(&incr_sum);
        (incr_sum.out_stream).attach_sender(&incr_sum);

        incr_sum
    }
}

impl<A> Context for IncrSum<A>
where
    A: DAMType + num::Num + MinMax + Copy,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        for _i in 0..self.outer_loop_bound {
            let mut temp_res = A::get_zero();
            for j in 0..self.inner_loop_bound {
                let _ = self.in_delta_stream.peek_next(&self.time);
                let _ = self.in_curr_stream.peek_next(&self.time);
                let in_delta_deq = self.in_delta_stream.dequeue(&self.time);
                let in_curr_deq = self.in_curr_stream.dequeue(&self.time);
                match (in_delta_deq, in_curr_deq) {
                    (Ok(in_delta), Ok(in_curr)) => {
                        // First Iteration
                        let in_delta_data = in_delta.data;
                        let in_curr_data = in_curr.data;
                        let new_sum = temp_res * in_delta_data + in_curr_data;
                        temp_res = new_sum;

                        if j == self.inner_loop_bound - 1 {
                            let curr_time = self.time.tick();
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement::new(curr_time + self.latency, temp_res),
                                )
                                .unwrap();
                        }

                        self.time.incr_cycles(self.init_inverval);
                        // initiation interval
                    }
                    (_, _) => {
                        panic!("Reached unhandled case");
                    }
                }
            }
        }
    }
}

#[context_macro]
pub struct IncrOutP<A: Clone> {
    pub in_delta_stream: Receiver<A>,
    pub in_curr_stream: Receiver<A>,
    pub in_v_stream: Receiver<A>, // should be an vector, but we assume d=1 for simplicity
    pub out_stream: Sender<A>,
    pub latency: u64,
    pub init_inverval: u64,
    pub inner_loop_bound: u64,
    pub outer_loop_bound: u64,
}

impl<A: DAMType> IncrOutP<A>
where
    IncrOutP<A>: Context,
{
    pub fn new(
        in_delta_stream: Receiver<A>,
        in_curr_stream: Receiver<A>,
        in_v_stream: Receiver<A>, // should be an vector, but we assume d=1 for simplicity
        out_stream: Sender<A>,
        latency: u64,
        init_inverval: u64,
        inner_loop_bound: u64,
        outer_loop_bound: u64,
    ) -> Self {
        let incr_outer_p = IncrOutP {
            in_delta_stream,
            in_curr_stream,
            in_v_stream,
            out_stream,
            latency,
            init_inverval,
            inner_loop_bound,
            outer_loop_bound,
            context_info: Default::default(),
        };
        (incr_outer_p.in_delta_stream).attach_receiver(&incr_outer_p);
        (incr_outer_p.in_curr_stream).attach_receiver(&incr_outer_p);
        (incr_outer_p.in_v_stream).attach_receiver(&incr_outer_p);
        (incr_outer_p.out_stream).attach_sender(&incr_outer_p);

        incr_outer_p
    }
}

impl<A> Context for IncrOutP<A>
where
    A: DAMType + num::Num + MinMax + Copy,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        for _i in 0..self.outer_loop_bound {
            let mut temp_res = A::get_zero();
            for j in 0..self.inner_loop_bound {
                let _ = self.in_delta_stream.peek_next(&self.time);
                let _ = self.in_curr_stream.peek_next(&self.time);
                let _ = self.in_v_stream.peek_next(&self.time);
                let in_delta_deq = self.in_delta_stream.dequeue(&self.time);
                let in_curr_deq = self.in_curr_stream.dequeue(&self.time);
                let in_v_deq = self.in_v_stream.dequeue(&self.time);
                match (in_delta_deq, in_curr_deq, in_v_deq) {
                    (Ok(in_delta), Ok(in_curr), Ok(in_v)) => {
                        // First Iteration
                        let in_delta_data = in_delta.data;
                        let in_curr_data = in_curr.data;
                        let in_v_data = in_v.data;
                        let new_sum = temp_res * in_delta_data + in_curr_data * in_v_data;
                        temp_res = new_sum;

                        if j == self.inner_loop_bound - 1 {
                            let curr_time = self.time.tick();
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement::new(curr_time + self.latency, temp_res),
                                )
                                .unwrap();
                        }

                        self.time.incr_cycles(self.init_inverval);
                        // initiation interval
                    }
                    (_, _, _) => {
                        panic!("Reached unhandled case");
                    }
                }
            }
        }
    }
}
