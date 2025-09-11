# Constitutional gRPC Development Kit v1# Constitutional Compliance Report0 - Usage Guide

Quick start guide for using the Constitutional gRPC Development Kit to create compliant gRPC services.

## Prerequisites

- Python 3.11+
- gRPC libraries (`grpcio`, `grpcio-tools`)
- Constitutional framework understanding

## Quick Start (5 minutes)

### 1. Generate a New Service

```bash
# Generate a constitutional gRPC service
python generate_service.py \
  --service-na**Constitutional Framework**: All generated services comply with the 14-article constitutional framework, ensuring production-ready, maintainable, and principled gRPC development.e MyService \
  --proto-file ./examples/example_service.proto \
  --output-dir ./my_service

# Output:
# Generating constitutional gRPC service: MyService
# ✅ Generated servicer: ./my_service/my_service_servicer.py
# ✅ Generated server: ./my_service/server.py
# ✅ Generated __init__.py: ./my_service/__init__.py
# ✅ Generated test suite: ./my_service/test_my_service_servicer.py
# ✅ Generated documentation: ./my_service/README.md
# 🎉 Constitutional MyService gRPC service generated successfully!
```

### 2. Validate Constitutional Compliance

```bash
# Check constitutional compliance
python validate_compliance.py \
  --service-dir ./my_service \
  --config compliance_config.yaml

# Output:
# 🏛️ Constitutional Compliance Report
# Service: my_service
# Overall Score: 0.87
# Compliance Status: ✅ COMPLIANT
#
# 📊 Constitutional Article Scores:
#   ✅ article_i: 0.90
#   ✅ article_ii: 0.85
#   ✅ article_iii: 0.88
#   ...
```

### 3. Run Your Service

```bash
cd my_service

# Install dependencies
pip install grpcio grpcio-tools protobuf

# Generate protobuf code
python -m grpc_tools.protoc \
  --python_out=. \
  --grpc_python_out=. \
  --proto_path=../examples \
  ../examples/example_service.proto

# Run the constitutional gRPC server
python server.py

# Output:
# Constitutional MyService gRPC server starting...
# ✅ Server running on port 50051
# ⚖️ Constitutional compliance: ACTIVE
```

## Development Workflow

### Step 1: Design Your Service

1. **Define your protobuf** (`.proto` file):
   - Follow constitutional Article V (Clarity)
   - Use descriptive message and service names
   - Keep message structures simple (Article III)

2. **Configure service generation** (optional):
   ```bash
   cp examples/service_config.yaml my_service_config.yaml
   # Edit configuration as needed
   ```

### Step 2: Generate Constitutional Service

```bash
# Basic generation
python generate_service.py \
  --service-name YourService \
  --proto-file your_service.proto \
  --output-dir ./generated_service

# With custom configuration
python generate_service.py \
  --service-name YourService \
  --proto-file your_service.proto \
  --output-dir ./generated_service \
  --config my_service_config.yaml

# Advanced options
python generate_service.py \
  --service-name YourService \
  --proto-file your_service.proto \
  --output-dir ./generated_service \
  --unified-integration \
  --constitutional-threshold 0.80
```

### Step 3: Customize Implementation

The generated service provides constitutional-compliant scaffolding. Customize as needed:

1. **Servicer methods**: Implement your business logic in the generated servicer
2. **Constitutional validation**: Adjust thresholds and validation logic
3. **Integration**: Configure unified intelligence integration
4. **Testing**: Extend the generated test suite

### Step 4: Validate and Test

```bash
# Run constitutional compliance check
python ../validate_compliance.py \
  --service-dir . \
  --report-format json \
  --threshold 0.75 \
  --strict

# Run the test suite
pytest -v --cov=. --cov-report=term-missing --cov-fail-under=80

# Run specific test categories
pytest -v -m "not integration"  # Unit tests only
pytest -v -m integration        # Integration tests only
pytest -v -m constitutional     # Constitutional tests only
```

### Step 5: Deploy

```bash
# Build Docker image (if Dockerfile was generated)
docker build -t your-service:latest .

# Run with Docker
docker run -p 50051:50051 your-service:latest

# Deploy to Kubernetes (if manifests were generated)
kubectl apply -f k8s/
```

## Configuration Options

### Service Generator Options

| Option | Description | Example |
|--------|-------------|---------|
| `--service-name` | Name of the service | `MyService` |
| `--proto-file` | Protobuf definition | `./service.proto` |
| `--output-dir` | Output directory | `./my_service` |
| `--config` | YAML configuration | `./config.yaml` |
| `--unified-integration` | Enable unified integration | (flag) |
| `--constitutional-threshold` | Compliance threshold | `0.80` |

### Compliance Checker Options

| Option | Description | Example |
|--------|-------------|---------|
| `--service-dir` | Service directory | `./my_service` |
| `--config` | Compliance config | `./compliance.yaml` |
| `--report-format` | Output format | `json`, `yaml`, `text` |
| `--output-file` | Report file | `./report.json` |
| `--threshold` | Compliance threshold | `0.75` |
| `--strict` | Exit on non-compliance | (flag) |

## Customization Examples

### Custom Service Configuration

```yaml
# my_service_config.yaml
service:
  name: "CustomService"
  version: "2.0.0"

server:
  default_port: 9090
  max_workers: 20

constitutional:
  compliance_threshold: 0.85
  strict_mode: true

methods:
  - name: "ProcessData"
    request_type: "DataRequest"
    response_type: "DataResponse"
    implementation_type: "custom"
```

### Custom Compliance Configuration

```yaml
# custom_compliance.yaml
thresholds:
  max_function_lines: 40  # Stricter than default (50)
  min_test_coverage: 85   # Higher than default (80)

analysis:
  enabled_checks:
    library_usage: true
    test_coverage: true
    complexity_analysis: true
    documentation: true
```

### Custom Templates

You can override the default templates by creating custom Jinja2 templates:

```bash
# Copy default templates
cp templates/servicer_template.py.j2 my_templates/
cp templates/server_template.py.j2 my_templates/

# Edit templates as needed
# Use with --template-dir option (future enhancement)
```

## Integration Patterns

### Unified Intelligence Integration

Generated services automatically integrate with the unified intelligence layer:

```python
# Automatically generated in servicer
class ConstitutionalMyServiceServicer:
    def __init__(self):
        self.unified_integration = ConstitutionalUnifiedIntegration()

    async def ProcessRequest(self, request, context):
        # Constitutional validation
        if not self._validate_constitutional_request(request):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                               "Constitutional validation failed")

        # Process through unified intelligence
        result = await self.unified_integration.process_request({
            "content": request.content,
            "method": "ProcessRequest",
            "constitutional_context": {...}
        })

        return ProcessRequestResponse(...)
```

### External Service Integration

```python
# Add to servicer for external integrations
class ConstitutionalMyServiceServicer:
    def __init__(self):
        super().__init__()
        self.external_client = ExternalServiceClient()

    async def ProcessRequest(self, request, context):
        # Constitutional validation + external processing
        result = await self.external_client.process(request.content)
        return self._create_response(result)
```

## Monitoring and Observability

### Metrics

Generated services expose constitutional compliance metrics:

```python
# Automatically available metrics
- constitutional_requests_total
- constitutional_violations_total
- constitutional_score_histogram
- response_time_histogram
- active_connections
```

### Health Checks

```bash
# gRPC health check
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Constitutional compliance status
grpcurl -plaintext localhost:50051 \
  constitutional.v1.Constitutional/GetComplianceStatus
```

### Logging

Constitutional context automatically included in logs:

```json
{
  "timestamp": "2025-01-09T10:30:00Z",
  "level": "INFO",
  "service": "MyService",
  "method": "ProcessRequest",
  "constitutional_score": 0.87,
  "compliance_status": "COMPLIANT",
  "response_time_ms": 45
}
```

## Troubleshooting

### Common Issues

#### 1. Constitutional Violations

```
Error: Constitutional validation failed (score: 0.65 < 0.75)
```

**Solution**: Check compliance report for specific violations:
```bash
python validate_compliance.py --service-dir . --report-format text
```

#### 2. Generation Failures

```
Error: Proto file not found: service.proto
```

**Solution**: Verify proto file path and existence:
```bash
ls -la *.proto
python generate_service.py --proto-file ./path/to/service.proto ...
```

#### 3. Test Failures

```
AssertionError: Constitutional score below threshold
```

**Solution**: Review constitutional implementation:
```bash
pytest -v -k constitutional --tb=long
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug logging level
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run your commands
"

# Or use verbose flags
python generate_service.py --verbose ...
python validate_compliance.py --verbose ...
```

## Advanced Usage

### Batch Generation

Generate multiple services from a configuration directory:

```bash
# Create multiple service configurations
mkdir configs
echo "service: {name: ServiceA}" > configs/service_a.yaml
echo "service: {name: ServiceB}" > configs/service_b.yaml

# Batch generate (future enhancement)
for config in configs/*.yaml; do
  python generate_service.py --config $config --output-dir ./services/
done
```

### CI/CD Integration

```yaml
# .github/workflows/constitutional-compliance.yml
name: Constitutional Compliance
on: [push, pull_request]

jobs:
  compliance-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Check Constitutional Compliance
      run: |
        python validate_compliance.py \
          --service-dir ./my_service \
          --report-format json \
          --output-file compliance-report.json \
          --strict
    - name: Upload Compliance Report
      uses: actions/upload-artifact@v2
      with:
        name: compliance-report
        path: compliance-report.json
```

### Performance Optimization

For high-performance constitutional validation:

```python
# Optimize constitutional scoring
constitutional_config = {
    "cache_scores": True,
    "batch_validation": True,
    "threshold": 0.75,
    "fast_mode": True,  # Skip detailed analysis
}
```

## Next Steps

1. **Explore Examples**: Study the `examples/` directory
2. **Read Documentation**: Review generated service README files
3. **Customize Templates**: Adapt templates for your specific needs
4. **Integrate Monitoring**: Set up constitutional compliance dashboards
5. **Scale Deployment**: Use Kubernetes manifests for production

## Support

- **Documentation**: See individual generated service README files
- **Templates**: Check `templates/` directory for customization
- **Examples**: Review `examples/` for common patterns
- **Compliance**: Use `validate_compliance.py --help` for options

**🏛️ Constitutional Framework**: All generated services comply with the 14-article constitutional framework, ensuring production-ready, maintainable, and principled gRPC development.
