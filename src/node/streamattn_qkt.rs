use dam::context_tools::*;

use ndarray::{ArrayBase, Dim, OwnedRepr};

#[context_macro]
pub struct QKTExp<A: Clone> {
    pub q: Receiver<A>,           // operand 1: Vector
    pub kt: Receiver<A>,          // operand 2: Vector
    pub out_fifo: Vec<Sender<A>>, // list of output scalar FIFOs
    pub latency: u64,             // pipeline depth
    pub init_inverval: u64,       // initiation interval
    pub seq_len: u64,
}

impl<A: DAMType> QKTExp<A>
where
    QKTExp<A>: Context,
    ArrayBase<OwnedRepr<A>, Dim<[usize; 1]>>: DAMType,
{
    pub fn new(
        q: Receiver<A>,           // operand 1: Vector
        kt: Receiver<A>,          // operand 2: Vector
        out_fifo: Vec<Sender<A>>, // list of output scalar FIFOs
        latency: u64,             // pipeline depth
        init_inverval: u64,       // initiation interval
        seq_len: u64,
    ) -> Self {
        let qkt_exp = QKTExp {
            q,
            kt,
            out_fifo,
            latency,
            init_inverval,
            seq_len,
            context_info: Default::default(),
        };
        (qkt_exp.q).attach_receiver(&qkt_exp);
        (qkt_exp.kt).attach_receiver(&qkt_exp);
        for i in qkt_exp.out_fifo.iter() {
            i.attach_sender(&qkt_exp);
        }

        qkt_exp
    }
}

impl<A> Context for QKTExp<A>
where
    A: DAMType + num::Float,
{
    fn init(&mut self) {}

    fn run(&mut self) -> () {
        self.time.incr_cycles(4);
        for _i in 0..self.seq_len {
            let _ = self.q.peek_next(&self.time);
            let _ = self.kt.peek_next(&self.time);

            let q_deq = self.q.dequeue(&self.time);
            match q_deq {
                Ok(q) => {
                    self.time.incr_cycles(4);
                    for _i in 0..self.seq_len {
                        let kt_deq = self.kt.dequeue(&self.time);
                        match kt_deq {
                            Ok(kt) => {
                                let qkt_exp_res = (q.data * kt.data).exp();
                                let curr_time = self.time.tick();

                                for k in self.out_fifo.iter() {
                                    let _ = k.wait_until_available(&self.time);
                                }

                                for k in self.out_fifo.iter() {
                                    k.enqueue(
                                        &self.time,
                                        ChannelElement::new(
                                            curr_time + self.latency,
                                            qkt_exp_res.clone(),
                                        ),
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
                _ => {
                    panic!("Reached unhandled case");
                }
            }
        }
    }
}
