# Tribal Knowledge Extractor Kit v1.0 - Constitutional Specification

**Development Kit Version**: 1.0.0
**Constitutional Authority**: Constitutional Mastery Architect v5.0
**Specification Date**: 2025-09-07
**Target Phase**: Phase 1 (Domain Immersion & Specification Drafting)

## **Executive Summary**

The Tribal Knowledge Extractor Kit (TKE) is a critical constitutional engine that automatically analyzes git diffs from resolved Socratic Testing Engine challenges to extract and codify implicit architectural decisions into a permanent, machine-readable Decision Registry. This engine enables the constitutional learning loop by capturing "tribal knowledge" - the unwritten rules, critical workarounds, and architectural insights that emerge during specification hardening.

## **Constitutional Mandate**

### **Primary Constitutional Articles Addressed:**
- **Article VI (Implicit Knowledge Codification)**: Unwritten rules and critical workarounds must be actively sought out and made explicit
- **Article VIII (Automation of Expertise)**: Any repeatable engineering process must be automated into a Development Kit
- **Article IX (Constitutional Self-Review)**: The system must analyze and improve its own patterns
- **Article XIII (Cross-Ecosystem Learning)**: Universal patterns must be synthesized from experiences across projects

### **Secondary Constitutional Compliance:**
- **Article I (Library-First)**: TKE designed as standalone, reusable component
- **Article II (Test-First)**: Comprehensive test suite before implementation
- **Article III (Simplicity Gate)**: Minimal complexity with explicit justification

## **Functional Requirements**

### **Core Capabilities**

#### **1. Git Diff Analysis Engine**
- **Input**: Git commit hash of a resolved specification challenge
- **Process**: Parse git diff to identify:
  - Added specification clauses addressing ambiguities
  - Modified acceptance criteria resolving edge cases
  - New architectural constraints introduced
  - Test case additions that reveal implicit requirements
- **Output**: Structured diff analysis with categorized changes

#### **2. Decision Pattern Recognition**
- **Ambiguity Resolution Patterns**: Detect when vague requirements become specific (e.g., "gracefully handle requests" → "handle up to 1000 concurrent requests with <100ms response time")
- **Constraint Addition Patterns**: Identify new architectural constraints (e.g., "must support OAuth 2.0 authentication")
- **Edge Case Codification**: Extract boundary conditions and error handling requirements
- **Test-First Evidence**: Correlate specification changes with corresponding test additions

#### **3. Architectural Decision Registry (ADR) Generation**
- **Decision ID**: Generate unique, sequential decision identifiers
- **Decision Context**: Extract the original ambiguous requirement
- **Decision Made**: Document the specific resolution chosen
- **Alternatives Considered**: Analyze commit messages and code comments for rejected alternatives
- **Rationale**: Extract justification from commit messages, PR descriptions, or inline comments
- **Consequences**: Document implications and constraints introduced
- **Constitutional Article**: Link decision to relevant constitutional principle

#### **4. Machine-Readable Registry Format**
```yaml
architectural_decision_registry:
  version: "1.0.0"
  last_updated: "2025-09-07T18:52:00Z"
  project: "super-alita"

decisions:
  - id: "ADR-001"
    date: "2025-09-07"
    title: "Specification Ambiguity Resolution for Request Handling"
    status: "accepted"
    context: |
      Original specification: "A sample feature that demonstrates how gracefully the system handles many requests"
      Socratic challenge: "What specific, measurable behavior is expected?"
    decision: |
      System must handle up to 1000 concurrent requests with <100ms average response time
      and graceful degradation above 1000 requests with appropriate error responses.
    alternatives:
      - "No specific limits (rejected due to untestable requirement)"
      - "Fixed 500 request limit (rejected as potentially too restrictive)"
    rationale: |
      Measurable performance criteria enable proper testing and system validation.
      1000 request threshold provides reasonable production capacity with room for growth.
    consequences:
      - "Requires load testing infrastructure"
      - "Necessitates performance monitoring"
      - "Impacts infrastructure sizing decisions"
    constitutional_articles: ["V", "II"]
    source_commit: "abc123def"
    extracted_by: "tribal_knowledge_extractor_v1.0.0"
```

### **Integration Requirements**

#### **CLI Interface**
```bash
python tools/tribal_knowledge_extractor.py \
  --commit abc123def \
  --output architectural_decisions.yaml \
  --project super-alita \
  --format yaml
```

#### **Launcher Mode Integration**
```bash
python start.py --mode tribal-extractor --commit abc123def
```

#### **Library Interface**
```python
from tools.tribal_knowledge_extractor import TribalKnowledgeExtractor

extractor = TribalKnowledgeExtractor(
    repo_path=".",
    output_format="yaml"
)

decisions = extractor.extract_from_commit("abc123def")
extractor.append_to_registry(decisions, "architectural_decisions.yaml")
```

## **Technical Architecture**

### **Core Components**

#### **1. GitDiffParser**
- **Responsibility**: Parse git diff output and categorize changes
- **Dependencies**: `gitpython`, `difflib`
- **Output**: `DiffAnalysis` object with categorized changes

#### **2. PatternRecognizer**
- **Responsibility**: Apply pattern matching to identify decision types
- **Patterns**:
  - Ambiguity resolution (vague → specific)
  - Constraint addition (new requirements)
  - Edge case handling (boundary conditions)
  - Test-first evidence (spec + test correlation)
- **Output**: `DecisionPattern` objects with confidence scores

#### **3. DecisionExtractor**
- **Responsibility**: Convert recognized patterns into structured ADR entries
- **Dependencies**: `pyyaml`, `datetime`
- **Output**: `ArchitecturalDecision` objects

#### **4. RegistryManager**
- **Responsibility**: Maintain and update the ADR registry file
- **Features**:
  - Append new decisions
  - Version management
  - Duplicate detection
  - Validation against schema
- **Output**: Updated `architectural_decisions.yaml`

### **Data Models**

```python
@dataclass
class DiffAnalysis:
    commit_hash: str
    author: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    added_lines: List[str]
    removed_lines: List[str]
    modified_sections: List[DiffSection]

@dataclass
class DecisionPattern:
    pattern_type: str  # "ambiguity_resolution", "constraint_addition", etc.
    confidence: float  # 0.0 to 1.0
    original_text: str
    resolved_text: str
    context: str

@dataclass
class ArchitecturalDecision:
    id: str
    title: str
    context: str
    decision: str
    alternatives: List[str]
    rationale: str
    consequences: List[str]
    constitutional_articles: List[str]
    source_commit: str
```

## **User Stories & Acceptance Criteria**

### **Story 1: Automated Decision Extraction**
**As a** constitutional architect
**I want** the TKE to automatically extract architectural decisions from spec resolution commits
**So that** implicit knowledge becomes explicit and searchable

**Acceptance Criteria:**
- **Given** a git commit that resolves Socratic Testing Engine challenges
- **When** I run `python tools/tribal_knowledge_extractor.py --commit abc123def`
- **Then** the tool extracts at least one architectural decision
- **And** the decision includes context, resolution, and rationale
- **And** the decision is linked to relevant constitutional articles

### **Story 2: Registry Persistence**
**As a** development team member
**I want** extracted decisions to be persistently stored in a searchable registry
**So that** architectural knowledge accumulates over time

**Acceptance Criteria:**
- **Given** extracted architectural decisions
- **When** I specify an output file `--output decisions.yaml`
- **Then** decisions are appended to the registry without duplication
- **And** the registry maintains version information and timestamps
- **And** the registry format is valid YAML with proper schema

### **Story 3: Pattern Recognition Accuracy**
**As a** constitutional architect
**I want** the TKE to accurately distinguish between different types of decisions
**So that** the registry provides meaningful categorization

**Acceptance Criteria:**
- **Given** a commit with ambiguity resolution changes
- **When** the TKE analyzes the diff
- **Then** it correctly identifies "ambiguity_resolution" patterns with >80% confidence
- **And** it extracts the original vague text and specific resolution
- **And** it links the decision to Constitutional Article V (Clarity and Unambiguity)

## **Non-Functional Requirements**

### **Performance**
- **Analysis Time**: Process typical commit diffs in <5 seconds
- **Memory Usage**: <100MB for commits with <1000 changed lines
- **Scalability**: Handle repositories with >10,000 commits

### **Reliability**
- **Error Handling**: Graceful degradation for malformed commits
- **Validation**: Schema validation for all generated YAML
- **Logging**: Comprehensive logging for debugging and audit

### **Maintainability**
- **Code Coverage**: >85% test coverage for all pattern recognition logic
- **Documentation**: Inline docstrings for all public methods
- **Constitutional Compliance**: Adherent to all 13 constitutional articles

## **Security & Safety**

### **Git Repository Safety**
- **Read-Only Access**: Never modify git repository state
- **Path Validation**: Validate all file paths to prevent directory traversal
- **Resource Limits**: Timeout protection for large diff analysis

### **Output Safety**
- **YAML Injection Prevention**: Escape all user-provided content
- **File Permission Validation**: Verify write permissions before registry updates
- **Backup Strategy**: Create registry backups before modifications

## **Dependencies**

### **Required Dependencies**
```yaml
dependencies:
  - gitpython>=3.1.0  # Git repository interaction
  - pyyaml>=6.0       # YAML parsing and generation
  - click>=8.0        # CLI interface
  - pydantic>=2.0     # Data validation and serialization
```

### **Optional Dependencies**
```yaml
optional_dependencies:
  - python-markdown>=3.4  # Enhanced commit message parsing
  - nltk>=3.8             # Natural language processing for pattern recognition
```

## **Implementation Phases**

### **Phase 1: Core Infrastructure (MVP)**
1. GitDiffParser implementation
2. Basic pattern recognition for ambiguity resolution
3. Simple ADR generation
4. CLI interface with basic options
5. Test suite covering core functionality

### **Phase 2: Advanced Pattern Recognition**
1. Multiple pattern types (constraint addition, edge cases)
2. Confidence scoring and validation
3. Enhanced commit message analysis
4. Integration with Socratic Testing Engine output

### **Phase 3: Registry Management & Integration**
1. Full RegistryManager implementation
2. Launcher mode integration
3. Library interface for programmatic use
4. Performance optimization and error handling

## **Testing Strategy**

### **Unit Tests**
- **GitDiffParser**: Mock git diff outputs with known patterns
- **PatternRecognizer**: Test pattern matching with curated examples
- **DecisionExtractor**: Validate ADR generation logic
- **RegistryManager**: Test YAML operations and validation

### **Integration Tests**
- **End-to-End Workflows**: Full commit → registry pipeline
- **Git Repository Integration**: Real repository testing
- **Launcher Integration**: Mode dispatch and argument handling
- **Constitutional Compliance**: Verify adherence to all 13 articles

### **Edge Case Testing**
- **Empty Commits**: Commits with no relevant changes
- **Malformed Diffs**: Invalid or corrupted git diff output
- **Large Commits**: Performance testing with massive diffs
- **Merge Commits**: Complex multi-parent commit scenarios

## **Success Metrics**

### **Quantitative Metrics**
- **Extraction Accuracy**: >90% of manually verified decisions correctly extracted
- **Pattern Recognition Precision**: >80% confidence for primary pattern types
- **Registry Quality**: 0 schema validation errors across 100+ decisions
- **Performance**: <5 second analysis time for 95% of commits

### **Qualitative Metrics**
- **Constitutional Compliance**: 100% adherence to all 13 articles
- **Developer Adoption**: TKE integrated into >3 Development Kit workflows
- **Knowledge Quality**: Extracted decisions provide actionable architectural guidance
- **Ecosystem Integration**: Seamless operation with existing constitutional engines

## **Future Enhancements**

### **Advanced Analytics**
- **Decision Trend Analysis**: Identify patterns in architectural evolution
- **Constitutional Article Usage**: Track which articles are most frequently applied
- **Decision Impact Assessment**: Correlate decisions with project health metrics

### **Cross-Project Intelligence**
- **Universal Pattern Recognition**: Identify patterns common across multiple projects
- **Best Practice Synthesis**: Generate recommendations based on successful decisions
- **Constitutional Amendment Proposals**: Suggest new articles based on recurring patterns

## **Constitutional Compliance Statement**

This specification explicitly addresses all 13 Constitutional Articles:

- **[I, VIII]**: TKE designed as standalone, automated Development Kit
- **[II]**: Comprehensive test-first approach with G/W/T scenarios
- **[III]**: Minimal complexity with explicit architectural justifications
- **[V]**: All ambiguities resolved through specific acceptance criteria
- **[VI]**: Primary purpose is codifying implicit knowledge into explicit rules
- **[IX]**: Tool analyzes its own extraction patterns for continuous improvement
- **[XIII]**: Enables cross-project pattern synthesis and universal rule discovery

**Constitutional Readiness**: This specification meets all constitutional requirements for immediate implementation.

---

**Next Phase**: Execute Socratic Testing Engine analysis to identify and resolve any remaining ambiguities before implementation.
