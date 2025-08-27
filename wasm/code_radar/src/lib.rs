wit_bindgen::generate!({ path: "radar.wit", world: "code-radar" });

use crate::vscode::example::host_api;
use std::time::{SystemTime, UNIX_EPOCH};

static mut METRICS: Vec<PerformanceMetric> = Vec::new();

struct Radar;

impl Guest for Radar {
    fn analyze(source: String) -> Vec<Diagnostic> {
        let start = SystemTime::now();
        let result = analyze_source(&source);
        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;

        record_metric("analyze", duration);
        result
    }

    fn analyze_file(path: String) -> Result<Vec<Diagnostic>, String> {
        let start = SystemTime::now();

        // Use host API to get file info
        match host_api::get_file_info(&path) {
            Ok(_file_info) => {
                // Read file content via host API
                match host_api::read_file_snippet(&path, 1, 1000) {
                    Ok(content) => {
                        let result = analyze_source(&content);
                        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;
                        record_metric("analyze_file", duration);
                        Ok(result)
                    }
                    Err(e) => Err(format!("Failed to read file: {}", e)),
                }
            }
            Err(e) => Err(format!("File not found: {}", e)),
        }
    }

    fn detect_smells(source: String) -> SmellAnalysis {
        let start = SystemTime::now();

        let complexity = calculate_complexity(&source);
        let maintainability = calculate_maintainability(&source);
        let debt = estimate_tech_debt(&source);
        let smells = detect_smell_types(&source);

        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;
        record_metric("detect_smells", duration);

        SmellAnalysis {
            complexity_score: complexity,
            maintainability_index: maintainability,
            debt_minutes: debt,
            smell_types: smells,
        }
    }

    fn predict_issues(source: String, history: Vec<String>) -> Vec<Diagnostic> {
        let start = SystemTime::now();

        let mut predictions = Vec::new();
        let current_smells = detect_smell_types(&source);

        // Predict based on historical patterns
        if history.len() > 3 {
            let historical_complexity: f32 = history
                .iter()
                .map(|h| calculate_complexity(h) as f32)
                .sum::<f32>()
                / history.len() as f32;

            let current_complexity = calculate_complexity(&source) as f32;

            if current_complexity > historical_complexity * 1.5 {
                predictions.push(Diagnostic {
                    line: 1,
                    col: 1,
                    severity: 2,
                    code: "PREDICTED_COMPLEXITY".into(),
                    message: "Complexity trend suggests future maintenance issues".into(),
                    suggestion: Some("Consider refactoring into smaller functions".into()),
                });
            }
        }

        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;
        record_metric("predict_issues", duration);

        predictions
    }

    fn get_performance_stats() -> Vec<PerformanceMetric> {
        unsafe { METRICS.clone() }
    }
}

fn analyze_source(source: &str) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let mut complexity: u32 = 1;
    let mut nesting: u32 = 0;
    let mut max_nesting: u32 = 0;
    let mut prev: &str = "";
    let mut dup_lines: u32 = 0;

    for (idx, raw) in source.lines().enumerate() {
        let line = raw.trim();
        if line.len() > 120 {
            diags.push(Diagnostic {
                line: (idx + 1) as u32,
                col: 1,
                severity: 1,
                code: "LINE_LEN".into(),
                message: format!("Line length {} > 120", line.len()),
                suggestion: Some("Break long lines for better readability".into()),
            });
        }
        if line.contains(" if ")
            || line.starts_with("if ")
            || line.contains(" for ")
            || line.starts_with("for ")
            || line.contains(" while ")
        {
            complexity += 1;
        }
        if line.contains('{') {
            nesting += 1;
            if nesting > max_nesting {
                max_nesting = nesting;
            }
        }
        if line.contains('}') {
            if nesting > 0 {
                nesting -= 1;
            }
        }
        if line == prev && line.len() > 8 {
            dup_lines += 1;
        }
        prev = line;
    }

    if complexity > 12 {
        diags.push(Diagnostic {
            line: 1,
            col: 1,
            severity: 2,
            code: "COMPLEXITY".into(),
            message: format!("Cyclomatic complexity {} > 12", complexity),
            suggestion: Some("Extract methods to reduce complexity".into()),
        });
    }
    if max_nesting > 5 {
        diags.push(Diagnostic {
            line: 1,
            col: 1,
            severity: 2,
            code: "NESTING".into(),
            message: format!("Max nesting depth {} > 5", max_nesting),
            suggestion: Some("Use early returns or extract methods".into()),
        });
    }
    if dup_lines > 0 {
        diags.push(Diagnostic {
            line: 1,
            col: 1,
            severity: 1,
            code: "DUPLICATION".into(),
            message: format!("Adjacent duplicate lines: {}", dup_lines),
            suggestion: Some("Extract common logic into functions".into()),
        });
    }

    diags
}

fn calculate_complexity(source: &str) -> u32 {
    let mut complexity = 1;
    for line in source.lines() {
        let line = line.trim();
        if line.contains(" if ")
            || line.starts_with("if ")
            || line.contains(" for ")
            || line.starts_with("for ")
            || line.contains(" while ")
            || line.contains(" match ")
            || line.contains(" catch ")
            || line.contains(" case ")
        {
            complexity += 1;
        }
    }
    complexity
}

fn calculate_maintainability(source: &str) -> u32 {
    let lines = source.lines().count() as u32;
    let complexity = calculate_complexity(source);
    let avg_line_length: f32 = source.lines().map(|l| l.len() as f32).sum::<f32>() / lines as f32;

    // Simple maintainability index (simplified version)
    let base_score: u32 = 100;
    let complexity_penalty = complexity * 2;
    let length_penalty = if avg_line_length > 80.0 { 10 } else { 0 };
    let size_penalty = if lines > 500 { 20 } else { 0 };

    base_score.saturating_sub(complexity_penalty + length_penalty + size_penalty)
}

fn estimate_tech_debt(source: &str) -> u32 {
    let complexity = calculate_complexity(source);
    let lines = source.lines().count() as u32;

    // Estimate in minutes to refactor
    let base_debt = if complexity > 15 { 30 } else { 0 };
    let size_debt = lines / 50; // 1 minute per 50 lines for large files

    base_debt + size_debt
}

fn detect_smell_types(source: &str) -> Vec<String> {
    let mut smells = Vec::new();
    let complexity = calculate_complexity(source);
    let lines = source.lines().count();

    if complexity > 12 {
        smells.push("High Complexity".into());
    }
    if lines > 500 {
        smells.push("Large Class".into());
    }

    let mut long_methods = 0;
    let mut current_method_lines = 0;
    for line in source.lines() {
        let line = line.trim();
        if line.contains("fn ") || line.contains("function ") || line.contains("def ") {
            if current_method_lines > 50 {
                long_methods += 1;
            }
            current_method_lines = 0;
        }
        current_method_lines += 1;
    }
    if long_methods > 0 {
        smells.push("Long Method".into());
    }

    // Check for duplication
    let unique_lines: std::collections::HashSet<&str> = source.lines().collect();
    if unique_lines.len() < source.lines().count() * 3 / 4 {
        smells.push("Duplicated Code".into());
    }

    smells
}

fn record_metric(operation: &str, duration_ms: u32) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let metric = PerformanceMetric {
        operation: operation.to_string(),
        duration_ms,
        memory_used: current_memory_usage(),
        timestamp,
    };

    // Emit to telemetry interface
    #[cfg(not(test))]
    {
        exports::telemetry::emit_metric(&metric);
    }

    unsafe {
        METRICS.push(metric);
        if METRICS.len() > 100 {
            METRICS.remove(0);
        }
    }
}

export!(Radar);

fn current_memory_usage() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { core::arch::wasm32::memory_size(0) * 64 * 1024 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::fs;

        fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
            .map(|pages| pages * 4096)
            .unwrap_or(0) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_memory_usage() {
        let _ = Radar::analyze("fn main() {}".into());
        let metrics = Radar::get_performance_stats();
        let metric = metrics.last().expect("metric");
        println!("memory_used_code_radar={}", metric.memory_used);
        assert!(metric.memory_used > 0);
    }
}
