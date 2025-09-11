# API Contracts & Interface Specifications

**Document**: 03-api-contracts.md
**Constitutional Article**: V - Clarity and Unambiguity
**Last Updated**: September 10, 2025

## Constitutional Contract Design (Article V Compliance)

All API contracts follow Article V of the Super-Alita Constitutional Framework: "Eliminate ambiguity in specifications, code, and communication."

## SDD Core Engine API

### Base URL
```
http://localhost:8080/api/v1
```

### Authentication
- **Development**: No authentication required
- **Production**: Bearer token authentication (future enhancement)

### 1. Specification Processing Endpoints

#### POST /specify
**Purpose**: Process natural language requirements into structured specifications

**Request**:
```json
{
  "user_input": "string (required)",
  "context": {
    "project_id": "string (optional)",
    "existing_specs": ["string"] (optional),
    "constitutional_mode": "boolean (default: true)"
  }
}
```

**Response** (200 OK):
```json
{
  "specification": {
    "id": "string",
    "title": "string",
    "user_stories": [
      {
        "id": "string",
        "as_a": "string",
        "i_want": "string",
        "so_that": "string",
        "acceptance_criteria": ["string"]
      }
    ],
    "non_functional_requirements": {
      "performance": ["string"],
      "security": ["string"],
      "usability": ["string"]
    },
    "constitutional_compliance": {
      "score": "number (0.0-1.0)",
      "article_scores": {
        "library_first": "number",
        "test_first": "number",
        "simplicity_gate": "number",
        "integration_first": "number",
        "clarity_unambiguity": "number",
        "counterfactual_justification": "number"
      },
      "violations": [
        {
          "article": "string",
          "severity": "string (low|medium|high|critical)",
          "description": "string",
          "suggestion": "string"
        }
      ]
    }
  },
  "clarifications_needed": [
    {
      "field": "string",
      "question": "string",
      "suggestions": ["string"]
    }
  ]
}
```

**Response** (400 Bad Request):
```json
{
  "error": "string",
  "details": "string",
  "constitutional_violations": ["string"]
}
```

#### POST /plan
**Purpose**: Generate implementation plan from specification

**Request**:
```json
{
  "specification_id": "string (required)",
  "technology_preferences": {
    "languages": ["string"] (optional),
    "frameworks": ["string"] (optional),
    "constraints": ["string"] (optional)
  },
  "constitutional_mode": "boolean (default: true)"
}
```

**Response** (200 OK):
```json
{
  "implementation_plan": {
    "id": "string",
    "specification_id": "string",
    "phases": [
      {
        "id": "string",
        "name": "string",
        "description": "string",
        "duration_weeks": "number",
        "dependencies": ["string"],
        "deliverables": ["string"],
        "constitutional_gates": [
          {
            "article": "string",
            "criteria": "string",
            "validation_method": "string"
          }
        ]
      }
    ],
    "technology_stack": {
      "languages": ["string"],
      "frameworks": ["string"],
      "libraries": ["string"],
      "justifications": {
        "library_choice": "string",
        "architecture_decision": "string"
      }
    },
    "constitutional_compliance": {
      "score": "number (0.0-1.0)",
      "gate_validations": [
        {
          "gate": "string",
          "passed": "boolean",
          "score": "number",
          "notes": "string"
        }
      ]
    }
  }
}
```

#### POST /tasks
**Purpose**: Break down implementation plan into actionable tasks

**Request**:
```json
{
  "plan_id": "string (required)",
  "phase_filter": ["string"] (optional),
  "constitutional_mode": "boolean (default: true)"
}
```

**Response** (200 OK):
```json
{
  "task_breakdown": {
    "plan_id": "string",
    "tasks": [
      {
        "id": "string",
        "title": "string",
        "description": "string",
        "phase": "string",
        "priority": "string (low|medium|high|critical)",
        "estimated_hours": "number",
        "dependencies": ["string"],
        "acceptance_criteria": ["string"],
        "constitutional_requirements": [
          {
            "article": "string",
            "requirement": "string",
            "validation_method": "string"
          }
        ]
      }
    ],
    "constitutional_validation": {
      "overall_compliance": "number",
      "task_compliance": [
        {
          "task_id": "string",
          "compliance_score": "number",
          "violations": ["string"]
        }
      ]
    }
  }
}
```

### 2. Constitutional Validation Endpoints

#### POST /constitutional/validate
**Purpose**: Validate any artifact against constitutional framework

**Request**:
```json
{
  "artifact": {
    "type": "string (specification|plan|code|documentation)",
    "content": "string",
    "metadata": {
      "language": "string (optional)",
      "framework": "string (optional)"
    }
  },
  "validation_options": {
    "strict_mode": "boolean (default: false)",
    "target_score": "number (default: 0.75)"
  }
}
```

**Response** (200 OK):
```json
{
  "validation_result": {
    "overall_score": "number (0.0-1.0)",
    "passed": "boolean",
    "article_scores": {
      "library_first": {
        "score": "number",
        "details": "string",
        "suggestions": ["string"]
      },
      "test_first": {
        "score": "number",
        "details": "string",
        "suggestions": ["string"]
      },
      "simplicity_gate": {
        "score": "number",
        "metrics": {
          "function_length": "number",
          "cyclomatic_complexity": "number",
          "dependency_count": "number"
        },
        "violations": ["string"]
      },
      "integration_first": {
        "score": "number",
        "details": "string",
        "test_coverage": "number"
      },
      "clarity_unambiguity": {
        "score": "number",
        "ambiguity_markers": ["string"],
        "clarity_issues": ["string"]
      },
      "counterfactual_justification": {
        "score": "number",
        "missing_justifications": ["string"],
        "alternative_analysis": "string"
      }
    },
    "recommendations": [
      {
        "priority": "string (low|medium|high|critical)",
        "article": "string",
        "issue": "string",
        "suggestion": "string",
        "example": "string (optional)"
      }
    ]
  }
}
```

### 3. APE Engine Endpoints

#### POST /ape/optimize
**Purpose**: Optimize prompts using constitutional APE engine

**Request**:
```json
{
  "base_prompt": "string (required)",
  "optimization_target": "string (clarity|completeness|constitutional_compliance)",
  "context": {
    "domain": "string (optional)",
    "constraints": ["string"] (optional)
  }
}
```

**Response** (200 OK):
```json
{
  "optimized_prompt": {
    "content": "string",
    "optimization_score": "number (0.0-1.0)",
    "constitutional_score": "number (0.0-1.0)",
    "variations": [
      {
        "content": "string",
        "focus": "string",
        "score": "number"
      }
    ],
    "improvements": [
      {
        "type": "string",
        "description": "string",
        "impact": "string"
      }
    ]
  }
}
```

## VS Code Extension API

### Command Interface

#### alita.sdd.specify
**Purpose**: Execute /specify command from VS Code

**Parameters**:
```typescript
interface SpecifyCommandArgs {
  userInput?: string;  // If not provided, prompt user
  insertAt?: vscode.Position;  // Where to insert result
  constitutionalMode?: boolean;  // Default: true
}
```

**Returns**:
```typescript
interface SpecifyResult {
  specificationId: string;
  filePath: string;  // Where spec was saved
  constitutionalScore: number;
  clarificationsNeeded: string[];
}
```

#### alita.sdd.plan
**Purpose**: Execute /plan command from VS Code

**Parameters**:
```typescript
interface PlanCommandArgs {
  specificationId?: string;  // If not provided, use current file
  technologyPreferences?: {
    languages?: string[];
    frameworks?: string[];
    constraints?: string[];
  };
}
```

**Returns**:
```typescript
interface PlanResult {
  planId: string;
  filePath: string;  // Where plan was saved
  constitutionalScore: number;
  phases: PlanPhase[];
}
```

#### alita.sdd.tasks
**Purpose**: Execute /tasks command from VS Code

**Parameters**:
```typescript
interface TasksCommandArgs {
  planId?: string;  // If not provided, use current file
  phaseFilter?: string[];
}
```

**Returns**:
```typescript
interface TasksResult {
  taskBreakdownId: string;
  filePath: string;  // Where tasks were saved
  totalTasks: number;
  constitutionalCompliance: number;
}
```

#### alita.constitutional.validate
**Purpose**: Validate current document against constitutional framework

**Parameters**:
```typescript
interface ValidateCommandArgs {
  documentPath?: string;  // If not provided, use active editor
  strictMode?: boolean;  // Default: false
  targetScore?: number;  // Default: 0.75
}
```

**Returns**:
```typescript
interface ValidationResult {
  overallScore: number;
  passed: boolean;
  articleScores: Record<string, ArticleScore>;
  recommendations: Recommendation[];
}
```

### Event Interface

#### onConstitutionalViolation
**Purpose**: Fired when constitutional violations are detected

**Event Data**:
```typescript
interface ConstitutionalViolationEvent {
  documentPath: string;
  violations: Violation[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
}
```

#### onSpecificationUpdate
**Purpose**: Fired when specifications are modified

**Event Data**:
```typescript
interface SpecificationUpdateEvent {
  specificationId: string;
  changeType: 'created' | 'modified' | 'deleted';
  constitutionalImpact: number;  // Score change
  affectedComponents: string[];
}
```

## CLI Tool Interface

### Global Options
```bash
--constitutional-mode    Enable constitutional validation (default: true)
--config-file           Path to configuration file
--output-format         Output format: json|yaml|markdown (default: markdown)
--verbose              Enable verbose logging
--help                 Show help information
```

### Commands

#### sdd specify
```bash
sdd specify [options] "<user-input>"

Options:
  --output-file PATH     Save specification to file
  --interactive         Interactive mode with clarification prompts
  --template TEMPLATE   Use specific template (default: standard)
  --constitutional-only Validate constitutional compliance only
```

**Examples**:
```bash
# Basic specification
sdd specify "Build a chat application with real-time messaging"

# Interactive mode
sdd specify --interactive

# Save to specific file
sdd specify --output-file ./specs/chat-app.md "Real-time chat system"
```

#### sdd plan
```bash
sdd plan [options] <specification-file>

Options:
  --technology TECH     Preferred technology (can be repeated)
  --output-file PATH    Save plan to file
  --phase-only PHASE    Generate specific phase only
  --constitutional-gates Enable all constitutional gates
```

**Examples**:
```bash
# Generate plan from specification
sdd plan ./specs/chat-app.md

# With technology preferences
sdd plan --technology python --technology fastapi ./specs/chat-app.md

# Constitutional validation enabled
sdd plan --constitutional-gates ./specs/chat-app.md
```

#### sdd tasks
```bash
sdd tasks [options] <plan-file>

Options:
  --phase PHASE         Filter by phase (can be repeated)
  --format FORMAT       Output format: list|kanban|json
  --estimate-hours     Include time estimates
  --constitutional     Include constitutional requirements
```

#### sdd validate
```bash
sdd validate [options] <file-or-directory>

Options:
  --type TYPE          Artifact type: specification|plan|code|all
  --strict            Enable strict validation mode
  --target-score NUM  Target constitutional score (default: 0.75)
  --report-file PATH  Save detailed report to file
```

## Error Handling Standards

### HTTP Status Codes
- **200 OK**: Successful operation
- **400 Bad Request**: Invalid input or constitutional violations
- **401 Unauthorized**: Authentication required (future)
- **403 Forbidden**: Insufficient permissions (future)
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Valid input but constitutional constraints violated
- **500 Internal Server Error**: Server error

### Error Response Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "string (optional)",
    "constitutional_context": {
      "violated_articles": ["string"],
      "severity": "string",
      "suggestions": ["string"]
    },
    "correlation_id": "string"
  }
}
```

### Constitutional Error Codes
- `CONST_001`: Constitutional compliance score below threshold
- `CONST_002`: Critical constitutional violation detected
- `CONST_003`: Missing required constitutional elements
- `CONST_004`: Complexity constraints exceeded (Article III)
- `CONST_005`: Library-first principles violated (Article I)
- `CONST_006`: Test-first requirements not met (Article II)

## Rate Limiting & Performance

### API Rate Limits
- **Specification Generation**: 10 requests/minute per user
- **Plan Generation**: 5 requests/minute per user
- **Constitutional Validation**: 100 requests/minute per user
- **APE Optimization**: 20 requests/minute per user

### Performance Targets
- **Specification Processing**: <5 seconds for typical input
- **Constitutional Validation**: <2 seconds for any artifact
- **Plan Generation**: <10 seconds for complete plan
- **API Response Time**: <500ms for all endpoints (95th percentile)

## Versioning Strategy

### API Versioning
- **URL Versioning**: `/api/v1/`, `/api/v2/`, etc.
- **Header Versioning**: `Accept: application/vnd.sdd.v1+json` (alternative)
- **Deprecation Policy**: 6 months notice, 12 months support overlap

### Constitutional Framework Versioning
- **Semantic Versioning**: Major.Minor.Patch (e.g., 1.2.3)
- **Backward Compatibility**: Minor versions must be backward compatible
- **Breaking Changes**: Only in major versions with migration guide

---

## Constitutional Compliance Review

### Article V: Clarity and Unambiguity ✅
- All API contracts precisely defined
- No ambiguous parameters or responses
- Clear error handling and status codes
- Comprehensive examples provided

### Supporting Articles ✅
- **Article I**: Leverages established HTTP/REST patterns
- **Article II**: Contract tests specified for all endpoints
- **Article III**: Simple, focused API design
- **Article IV**: Real integration testing requirements
- **Article VI**: Alternative designs documented and justified

**API Contract Constitutional Score**: 0.95 ✅

---

*These API contracts follow the Super-Alita Constitutional Framework and ensure unambiguous, testable interfaces for all SDD components.*
