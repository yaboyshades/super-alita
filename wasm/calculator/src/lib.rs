wit_bindgen::generate!({
    world: "calculator",
});

struct Calculator;

impl Guest for Calculator {
    fn calc(op: Operation) -> u32 {
        log(&format!("Calculating: {:?}", op));
        match op {
            Operation::Add(ops) => ops.left + ops.right,
            Operation::Sub(ops) => ops.left - ops.right,
            Operation::Mul(ops) => ops.left * ops.right,
            Operation::Div(ops) => ops.left / ops.right,
        }
    }
}

export!(Calculator);
