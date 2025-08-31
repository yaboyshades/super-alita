package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// VerificationRequest represents a request for Mangle verification
type VerificationRequest struct {
	Expression string            `json:"expression"`
	Variables  map[string]string `json:"variables"`
	Timeout    int               `json:"timeout,omitempty"`
	EngineID   string            `json:"engine_id,omitempty"`
}

// VerificationResponse represents the response from Mangle verification
type VerificationResponse struct {
	Result      bool              `json:"result"`
	Satisfiable bool              `json:"satisfiable"`
	Model       map[string]string `json:"model,omitempty"`
	Error       string            `json:"error,omitempty"`
	Duration    float64           `json:"duration"`
	EngineID    string            `json:"engine_id"`
	Timestamp   string            `json:"timestamp"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp string            `json:"timestamp"`
	Version   string            `json:"version"`
	Stats     map[string]interface{} `json:"stats"`
}

// Engine represents a Mangle verification engine
type Engine struct {
	ID        string
	Active    bool
	LastUsed  time.Time
	UseCount  int64
	mutex     sync.RWMutex
}

// NewEngine creates a new Mangle engine instance
func NewEngine(id string) *Engine {
	return &Engine{
		ID:       id,
		Active:   true,
		LastUsed: time.Now(),
		UseCount: 0,
	}
}

// Use marks the engine as used and increments usage counter
func (e *Engine) Use() {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.LastUsed = time.Now()
	e.UseCount++
}

// IsActive returns whether the engine is currently active
func (e *Engine) IsActive() bool {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return e.Active
}

// SetActive sets the active state of the engine
func (e *Engine) SetActive(active bool) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.Active = active
}

// GetStats returns engine statistics
func (e *Engine) GetStats() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return map[string]interface{}{
		"id":        e.ID,
		"active":    e.Active,
		"last_used": e.LastUsed.Format(time.RFC3339),
		"use_count": e.UseCount,
	}
}

// EnginePool manages a pool of Mangle verification engines
type EnginePool struct {
	engines []*Engine
	current int
	mutex   sync.RWMutex
}

// NewEnginePool creates a new engine pool with specified size
func NewEnginePool(size int) *EnginePool {
	engines := make([]*Engine, size)
	for i := 0; i < size; i++ {
		engines[i] = NewEngine(fmt.Sprintf("engine_%d", i))
	}
	
	return &EnginePool{
		engines: engines,
		current: 0,
	}
}

// GetEngine returns the next available engine (round-robin)
func (p *EnginePool) GetEngine() *Engine {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Find next active engine
	for attempts := 0; attempts < len(p.engines); attempts++ {
		engine := p.engines[p.current]
		p.current = (p.current + 1) % len(p.engines)
		
		if engine.IsActive() {
			return engine
		}
	}
	
	// No active engines found, return first one anyway
	return p.engines[0]
}

// GetStats returns pool statistics
func (p *EnginePool) GetStats() map[string]interface{} {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	activeCount := 0
	totalUseCount := int64(0)
	engineStats := make([]map[string]interface{}, len(p.engines))
	
	for i, engine := range p.engines {
		if engine.IsActive() {
			activeCount++
		}
		totalUseCount += engine.UseCount
		engineStats[i] = engine.GetStats()
	}
	
	return map[string]interface{}{
		"total_engines":  len(p.engines),
		"active_engines": activeCount,
		"total_use_count": totalUseCount,
		"engines":        engineStats,
	}
}

// CircuitBreaker provides circuit breaker functionality for resilience
type CircuitBreaker struct {
	maxFailures   int
	resetTimeout  time.Duration
	failureCount  int
	lastFailTime  time.Time
	state         string // "closed", "open", "half-open"
	mutex         sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        "closed",
	}
}

// Call executes a function through the circuit breaker
func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	// Check if circuit should be reset
	if cb.state == "open" && time.Since(cb.lastFailTime) > cb.resetTimeout {
		cb.state = "half-open"
		cb.failureCount = 0
	}
	
	// If circuit is open, fail fast
	if cb.state == "open" {
		return fmt.Errorf("circuit breaker is open")
	}
	
	// Execute function
	err := fn()
	
	if err != nil {
		cb.failureCount++
		cb.lastFailTime = time.Now()
		
		if cb.failureCount >= cb.maxFailures {
			cb.state = "open"
		}
		return err
	}
	
	// Success - reset failure count and close circuit
	cb.failureCount = 0
	cb.state = "closed"
	return nil
}

// GetState returns the current circuit breaker state
func (cb *CircuitBreaker) GetState() string {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.state
}

// GetStats returns circuit breaker statistics
func (cb *CircuitBreaker) GetStats() map[string]interface{} {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	
	return map[string]interface{}{
		"state":         cb.state,
		"failure_count": cb.failureCount,
		"max_failures":  cb.maxFailures,
		"last_fail_time": cb.lastFailTime.Format(time.RFC3339),
	}
}

// VerificationService provides Mangle verification services
type VerificationService struct {
	enginePool     *EnginePool
	circuitBreaker *CircuitBreaker
	cache          sync.Map // Simple in-memory cache
	
	// Prometheus metrics
	requestsTotal     prometheus.Counter
	requestDuration   prometheus.Histogram
	activeEngines     prometheus.Gauge
	cacheHits         prometheus.Counter
	cacheMisses       prometheus.Counter
}

// NewVerificationService creates a new verification service
func NewVerificationService(enginePoolSize int) *VerificationService {
	service := &VerificationService{
		enginePool:     NewEnginePool(enginePoolSize),
		circuitBreaker: NewCircuitBreaker(5, 30*time.Second),
	}
	
	// Initialize Prometheus metrics
	service.requestsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "mangle_verification_requests_total",
		Help: "Total number of verification requests",
	})
	
	service.requestDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "mangle_verification_duration_seconds",
		Help:    "Duration of verification requests",
		Buckets: prometheus.DefBuckets,
	})
	
	service.activeEngines = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "mangle_active_engines",
		Help: "Number of active Mangle engines",
	})
	
	service.cacheHits = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "mangle_cache_hits_total",
		Help: "Total number of cache hits",
	})
	
	service.cacheMisses = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "mangle_cache_misses_total",
		Help: "Total number of cache misses",
	})
	
	// Register metrics
	prometheus.MustRegister(
		service.requestsTotal,
		service.requestDuration,
		service.activeEngines,
		service.cacheHits,
		service.cacheMisses,
	)
	
	return service
}

// generateCacheKey creates a cache key from the verification request
func (vs *VerificationService) generateCacheKey(req *VerificationRequest) string {
	data, _ := json.Marshal(map[string]interface{}{
		"expression": req.Expression,
		"variables":  req.Variables,
	})
	return fmt.Sprintf("%x", data)
}

// Verify performs Mangle verification with caching and circuit breaking
func (vs *VerificationService) Verify(req *VerificationRequest) *VerificationResponse {
	start := time.Now()
	defer func() {
		vs.requestsTotal.Inc()
		vs.requestDuration.Observe(time.Since(start).Seconds())
	}()
	
	// Check cache first
	cacheKey := vs.generateCacheKey(req)
	if cached, ok := vs.cache.Load(cacheKey); ok {
		vs.cacheHits.Inc()
		if response, ok := cached.(*VerificationResponse); ok {
			// Update timestamp for cached response
			response.Timestamp = time.Now().Format(time.RFC3339)
			return response
		}
	}
	vs.cacheMisses.Inc()
	
	// Get engine from pool
	engine := vs.enginePool.GetEngine()
	
	// Create response
	response := &VerificationResponse{
		EngineID:  engine.ID,
		Timestamp: time.Now().Format(time.RFC3339),
	}
	
	// Perform verification through circuit breaker
	err := vs.circuitBreaker.Call(func() error {
		return vs.performVerification(engine, req, response)
	})
	
	if err != nil {
		response.Error = err.Error()
		response.Result = false
		response.Satisfiable = false
	}
	
	response.Duration = time.Since(start).Seconds()
	
	// Cache successful results
	if response.Error == "" {
		vs.cache.Store(cacheKey, response)
	}
	
	return response
}

// performVerification executes the actual Mangle verification
func (vs *VerificationService) performVerification(engine *Engine, req *VerificationRequest, resp *VerificationResponse) error {
	engine.Use()
	
	// Simulate Mangle verification (replace with actual Mangle integration)
	// This is a placeholder implementation
	
	// Set default timeout
	timeout := 10
	if req.Timeout > 0 {
		timeout = req.Timeout
	}
	
	// Simulate processing time
	time.Sleep(time.Duration(timeout/10) * time.Millisecond)
	
	// Simple expression evaluation for demonstration
	// In real implementation, this would use the Mangle library
	if req.Expression == "" {
		return fmt.Errorf("empty expression")
	}
	
	// Mock verification logic
	if len(req.Expression) > 100 {
		return fmt.Errorf("expression too complex")
	}
	
	// Simulate successful verification
	resp.Result = true
	resp.Satisfiable = true
	resp.Model = map[string]string{
		"status": "verified",
		"method": "mangle_smt",
	}
	
	// Add variables to model if provided
	for k, v := range req.Variables {
		resp.Model[k] = v
	}
	
	return nil
}

// GetStats returns comprehensive service statistics
func (vs *VerificationService) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"engine_pool":      vs.enginePool.GetStats(),
		"circuit_breaker":  vs.circuitBreaker.GetStats(),
		"cache_size":       vs.getCacheSize(),
		"service_status":   "active",
	}
}

// getCacheSize returns the current cache size
func (vs *VerificationService) getCacheSize() int {
	count := 0
	vs.cache.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

// HTTP Handlers

func (vs *VerificationService) verifyHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req VerificationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	response := vs.Verify(&req)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (vs *VerificationService) healthHandler(w http.ResponseWriter, r *http.Request) {
	stats := vs.GetStats()
	
	response := HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now().Format(time.RFC3339),
		Version:   "1.0.0",
		Stats:     stats,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (vs *VerificationService) statsHandler(w http.ResponseWriter, r *http.Request) {
	stats := vs.GetStats()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func main() {
	// Configuration
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	enginePoolSize := 4
	if sizeStr := os.Getenv("ENGINE_POOL_SIZE"); sizeStr != "" {
		if size, err := strconv.Atoi(sizeStr); err == nil {
			enginePoolSize = size
		}
	}
	
	// Create verification service
	service := NewVerificationService(enginePoolSize)
	
	// Create router
	r := mux.NewRouter()
	
	// API routes
	r.HandleFunc("/verify", service.verifyHandler).Methods("POST")
	r.HandleFunc("/health", service.healthHandler).Methods("GET")
	r.HandleFunc("/stats", service.statsHandler).Methods("GET")
	
	// Prometheus metrics endpoint
	r.Handle("/metrics", promhttp.Handler())
	
	// Create server
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	
	// Start server in goroutine
	go func() {
		log.Printf("Mangle verification service starting on port %s", port)
		log.Printf("Engine pool size: %d", enginePoolSize)
		
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()
	
	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	
	log.Println("Shutting down server...")
	
	// Create context with timeout for graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	// Shutdown server
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}
	
	log.Println("Server exited")
}
