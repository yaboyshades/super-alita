wit_bindgen::generate!({ world: "calculator" });

struct Impl;

impl Guest for Impl {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

export!(Impl);
