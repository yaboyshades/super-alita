wit_bindgen::generate!({ world: "code-radar" });

struct Radar;

impl Guest for Radar {
    fn analyze(source: String) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        let mut complexity: u32 = 1;
        let mut nesting: u32 = 0;
        let mut max_nesting: u32 = 0;
        let mut prev: &str = "";
        let mut dup_lines: u32 = 0;
        for (idx, raw) in source.lines().enumerate() {
            let line = raw.trim();
            if line.len() > 120 {
                diags.push(Diagnostic { line: (idx + 1) as u32, col: 1, severity: 1, code: "LINE_LEN".into(), message: format!("Len {} > 120", line.len()) });
            }
            if line.contains(" if ") || line.starts_with("if ") || line.contains(" for ") || line.starts_with("for ") || line.contains(" while ") {
                complexity += 1;
            }
            if line.contains('{') { nesting += 1; if nesting > max_nesting { max_nesting = nesting; } }
            if line.contains('}') { if nesting > 0 { nesting -= 1; } }
            if line == prev && line.len() > 8 { dup_lines += 1; }
            prev = line;
        }
        if complexity > 12 {
            diags.push(Diagnostic { line: 1, col: 1, severity: 2, code: "COMPLEXITY".into(), message: format!("Complexity {} > 12", complexity) });
        }
        if max_nesting > 5 {
            diags.push(Diagnostic { line: 1, col: 1, severity: 2, code: "NESTING".into(), message: format!("Nesting {} > 5", max_nesting) });
        }
        if dup_lines > 0 {
            diags.push(Diagnostic { line: 1, col: 1, severity: 1, code: "DUPLICATION".into(), message: format!("Adjacent repetition count {}", dup_lines) });
        }
        diags
    }
}

export!(Radar);