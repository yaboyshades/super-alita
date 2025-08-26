wit_bindgen::generate!({ world: "calculator" });

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
        memory_used: 0, // TODO: implement memory tracking
        timestamp,
    };
    
    // Emit to telemetry interface
    exports::telemetry::emit_metric(&metric);
    
    // Store locally
    unsafe {
        METRICS.push(metric);
        // Keep only last 100 metrics
        if METRICS.len() > 100 {
            METRICS.remove(0);
        }
    }
}

export!(Impl);
