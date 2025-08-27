wit_bindgen::generate!({ path: "calculator.wit", world: "calculator" });

use std::time::{SystemTime, UNIX_EPOCH};

static mut METRICS: Vec<PerformanceMetric> = Vec::new();

struct Impl;

impl Guest for Impl {
    fn add(a: u32, b: u32) -> u32 {
        let start = SystemTime::now();
        let result = a + b;
        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;

        record_metric("add", duration);
        result
    }

    fn multiply(a: u32, b: u32) -> u32 {
        let start = SystemTime::now();
        let result = a * b;
        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;

        record_metric("multiply", duration);
        result
    }

    fn divide(a: u32, b: u32) -> Result<u32, String> {
        let start = SystemTime::now();

        if b == 0 {
            return Err("Division by zero".to_string());
        }

        let result = a / b;
        let duration = start.elapsed().unwrap_or_default().as_millis() as u32;

        record_metric("divide", duration);
        Ok(result)
    }

    fn get_performance_stats() -> Vec<PerformanceMetric> {
        unsafe { METRICS.clone() }
    }
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

    // Store locally
    unsafe {
        METRICS.push(metric);
        // Keep only last 100 metrics
        if METRICS.len() > 100 {
            METRICS.remove(0);
        }
    }
}

fn current_memory_usage() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        // memory_size returns number of 64KiB pages
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
        let _ = Impl::add(1, 1);
        let metrics = Impl::get_performance_stats();
        let metric = metrics.last().expect("metric");
        println!("memory_used_calculator={}", metric.memory_used);
        assert!(metric.memory_used > 0);
    }
}

export!(Impl);
