wit_bindgen::generate!({ world: "code-radar" });

struct Impl;

impl Guest for Impl {
    fn analyze(source: String) -> Vec<Diagnostic> {
        // Extremely naive metrics placeholder
        let mut out = Vec::new();
        for (i, line) in source.lines().enumerate() {
            if line.len() > 120 {
                out.push(Diagnostic {
                    line: i as u32,
                    col: 0,
                    severity: 1,
                    code: "line-too-long".into(),
                    message: format!("Line length {} exceeds 120", line.len()),
                });
            }
        }
        out
    }
}

export!(Impl);