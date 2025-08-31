# Deep Code Analysis - Agent Instructions

## Overview
The `src/deepcode/` directory contains deep code analysis and understanding capabilities:
- **Code Intelligence** - Advanced code understanding and analysis
- **Pattern Recognition** - Code pattern detection and classification
- **Semantic Analysis** - Deep semantic understanding of code structures
- **Code Generation** - AI-powered code generation and refactoring

## Key Files & Responsibilities

### Deep Code Components
- Code analysis engines and parsers
- Semantic understanding modules
- Pattern recognition algorithms
- Code generation utilities

## Development Guidelines

### Code Analysis Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import ast
import libcst as cst
from pathlib import Path

class AnalysisLevel(Enum):
    LEXICAL = "lexical"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"

@dataclass
class CodeElement:
    """Representation of a code element"""
    element_type: str
    name: str
    location: Dict[str, int]  # line, column info
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnalysisResult:
    """Result of code analysis"""
    file_path: str
    analysis_level: AnalysisLevel
    elements: List[CodeElement]
    patterns: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

class CodeAnalyzer(ABC):
    """Base class for code analyzers"""

    def __init__(self, analysis_level: AnalysisLevel = AnalysisLevel.SEMANTIC):
        self.analysis_level = analysis_level
        self.patterns: Dict[str, Any] = {}

    @abstractmethod
    async def analyze_code(self, code: str, file_path: str = None) -> AnalysisResult:
        """Analyze code and return analysis result"""
        pass

    @abstractmethod
    def extract_elements(self, code: str) -> List[CodeElement]:
        """Extract code elements from source"""
        pass

    def register_pattern(self, pattern_name: str, pattern_config: Dict[str, Any]):
        """Register code pattern for detection"""
        self.patterns[pattern_name] = pattern_config
```

### Python Code Analyzer
```python
class PythonCodeAnalyzer(CodeAnalyzer):
    """Python-specific code analyzer"""

    def __init__(self, analysis_level: AnalysisLevel = AnalysisLevel.SEMANTIC):
        super().__init__(analysis_level)
        self.ast_visitor = PythonASTVisitor()
        self.cst_transformer = PythonCSTTransformer()

    async def analyze_code(self, code: str, file_path: str = None) -> AnalysisResult:
        """Analyze Python code"""
        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Extract elements
            elements = self.extract_elements(code)

            # Detect patterns
            patterns = self._detect_patterns(tree, code)

            # Calculate metrics
            metrics = self._calculate_metrics(tree, code)

            # Generate suggestions
            suggestions = await self._generate_suggestions(tree, elements, patterns)

            return AnalysisResult(
                file_path=file_path or "<unknown>",
                analysis_level=self.analysis_level,
                elements=elements,
                patterns=patterns,
                metrics=metrics,
                suggestions=suggestions
            )

        except SyntaxError as e:
            return AnalysisResult(
                file_path=file_path or "<unknown>",
                analysis_level=self.analysis_level,
                elements=[],
                patterns=[],
                metrics={'syntax_error': str(e)},
                suggestions=[f"Fix syntax error: {e}"]
            )

    def extract_elements(self, code: str) -> List[CodeElement]:
        """Extract Python code elements"""
        elements = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    elements.append(CodeElement(
                        element_type="function",
                        name=node.name,
                        location={"line": node.lineno, "column": node.col_offset},
                        content=ast.get_source_segment(code, node) or "",
                        metadata={
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                            "is_async": isinstance(node, ast.AsyncFunctionDef)
                        }
                    ))

                elif isinstance(node, ast.ClassDef):
                    elements.append(CodeElement(
                        element_type="class",
                        name=node.name,
                        location={"line": node.lineno, "column": node.col_offset},
                        content=ast.get_source_segment(code, node) or "",
                        metadata={
                            "bases": [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases],
                            "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                        }
                    ))

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        elements.append(CodeElement(
                            element_type="import",
                            name=alias.name,
                            location={"line": node.lineno, "column": node.col_offset},
                            content=f"import {alias.name}",
                            metadata={"alias": alias.asname}
                        ))

                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        elements.append(CodeElement(
                            element_type="import_from",
                            name=alias.name,
                            location={"line": node.lineno, "column": node.col_offset},
                            content=f"from {node.module} import {alias.name}",
                            metadata={"module": node.module, "alias": alias.asname}
                        ))

        except Exception as e:
            logger.error(f"Error extracting elements: {e}")

        return elements

    def _detect_patterns(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect code patterns in AST"""
        patterns = []

        # Design pattern detection
        patterns.extend(self._detect_design_patterns(tree))

        # Anti-pattern detection
        patterns.extend(self._detect_anti_patterns(tree))

        # Code smell detection
        patterns.extend(self._detect_code_smells(tree, code))

        return patterns

    def _detect_design_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect common design patterns"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Singleton pattern detection
                if self._is_singleton_pattern(node):
                    patterns.append({
                        "type": "design_pattern",
                        "pattern": "singleton",
                        "location": {"line": node.lineno},
                        "confidence": 0.8
                    })

                # Factory pattern detection
                if self._is_factory_pattern(node):
                    patterns.append({
                        "type": "design_pattern",
                        "pattern": "factory",
                        "location": {"line": node.lineno},
                        "confidence": 0.7
                    })

        return patterns

    def _detect_anti_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect anti-patterns and code issues"""
        patterns = []

        for node in ast.walk(tree):
            # God class detection (too many methods)
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                if method_count > 20:
                    patterns.append({
                        "type": "anti_pattern",
                        "pattern": "god_class",
                        "location": {"line": node.lineno},
                        "severity": "high",
                        "details": f"Class has {method_count} methods"
                    })

            # Long method detection
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    line_count = node.end_lineno - node.lineno
                    if line_count > 50:
                        patterns.append({
                            "type": "anti_pattern",
                            "pattern": "long_method",
                            "location": {"line": node.lineno},
                            "severity": "medium",
                            "details": f"Method has {line_count} lines"
                        })

        return patterns

    def _calculate_metrics(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Calculate code metrics"""
        metrics = {
            "lines_of_code": len(code.splitlines()),
            "blank_lines": len([line for line in code.splitlines() if not line.strip()]),
            "comment_lines": len([line for line in code.splitlines() if line.strip().startswith('#')]),
            "classes": 0,
            "functions": 0,
            "imports": 0,
            "complexity": 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics["functions"] += 1
                metrics["complexity"] += self._calculate_cyclomatic_complexity(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["imports"] += 1

        return metrics

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity
```

### Semantic Code Understanding
```python
import openai
from typing import Dict, Any, List

class SemanticCodeAnalyzer:
    """AI-powered semantic code analysis"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def analyze_semantic_meaning(self, code: str, context: str = None) -> Dict[str, Any]:
        """Analyze semantic meaning of code"""
        prompt = self._build_analysis_prompt(code, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code analyst. Analyze the provided code and return structured insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            analysis = self._parse_ai_response(response.choices[0].message.content)
            return analysis

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {"error": str(e)}

    async def suggest_improvements(self, code: str, analysis_result: AnalysisResult) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        prompt = self._build_improvement_prompt(code, analysis_result)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer. Provide specific, actionable improvement suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            suggestions = self._parse_suggestions(response.choices[0].message.content)
            return suggestions

        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return [f"Error generating suggestions: {e}"]

    def _build_analysis_prompt(self, code: str, context: str = None) -> str:
        """Build prompt for semantic analysis"""
        prompt = f"""
Analyze the following code for semantic meaning, purpose, and functionality:

```python
{code}
```
"""

        if context:
            prompt += f"\nContext: {context}"

        prompt += """

Please provide analysis in the following format:
1. Purpose: What does this code do?
2. Key Components: What are the main parts?
3. Dependencies: What external dependencies does it have?
4. Complexity: How complex is this code?
5. Maintainability: How maintainable is this code?
6. Potential Issues: Any potential problems?
"""

        return prompt

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        # Simple parsing - could be enhanced with more sophisticated NLP
        sections = {}
        current_section = None

        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and any(keyword in line.lower() for keyword in ['purpose', 'components', 'dependencies', 'complexity', 'maintainability', 'issues']):
                parts = line.split(':', 1)
                current_section = parts[0].strip().lower()
                sections[current_section] = parts[1].strip() if len(parts) > 1 else ""
            elif current_section and line:
                sections[current_section] += " " + line

        return sections
```

### Code Generation Engine
```python
class CodeGenerationEngine:
    """AI-powered code generation"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.templates: Dict[str, str] = {}

    async def generate_code(
        self,
        specification: str,
        language: str = "python",
        style_guide: str = None,
        existing_code: str = None
    ) -> Dict[str, Any]:
        """Generate code from specification"""
        prompt = self._build_generation_prompt(specification, language, style_guide, existing_code)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert {language} developer. Generate high-quality, well-documented code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            generated_code = self._extract_code_from_response(response.choices[0].message.content)

            return {
                "success": True,
                "code": generated_code,
                "language": language,
                "metadata": {
                    "model": self.model,
                    "specification": specification
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def refactor_code(self, code: str, refactoring_goals: List[str]) -> Dict[str, Any]:
        """Refactor existing code based on goals"""
        prompt = f"""
Refactor the following code to achieve these goals:
{', '.join(refactoring_goals)}

Original code:
```python
{code}
```

Please provide:
1. Refactored code
2. Explanation of changes
3. Benefits of the refactoring
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code refactoring specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            result = self._parse_refactoring_response(response.choices[0].message.content)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def register_template(self, template_name: str, template_code: str):
        """Register code template"""
        self.templates[template_name] = template_code

    async def generate_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code from template"""
        if template_name not in self.templates:
            return {
                "success": False,
                "error": f"Template not found: {template_name}"
            }

        template = self.templates[template_name]

        try:
            # Simple template substitution (could be enhanced with Jinja2)
            generated_code = template
            for key, value in parameters.items():
                generated_code = generated_code.replace(f"{{{key}}}", str(value))

            return {
                "success": True,
                "code": generated_code,
                "template": template_name,
                "parameters": parameters
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Code Pattern Library
```python
class CodePatternLibrary:
    """Library of code patterns and best practices"""

    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """Initialize default code patterns"""

        # Singleton pattern
        self.patterns["singleton"] = {
            "name": "Singleton Pattern",
            "category": "creational",
            "description": "Ensure a class has only one instance",
            "template": '''
class {class_name}:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super({class_name}, cls).__new__(cls)
        return cls._instance
''',
            "indicators": [
                "single instance",
                "__new__ method",
                "class variable _instance"
            ]
        }

        # Factory pattern
        self.patterns["factory"] = {
            "name": "Factory Pattern",
            "category": "creational",
            "description": "Create objects without specifying exact classes",
            "template": '''
class {factory_name}:
    @staticmethod
    def create_{product_type}(product_type: str) -> {base_class}:
        if product_type == "type_a":
            return {type_a_class}()
        elif product_type == "type_b":
            return {type_b_class}()
        else:
            raise ValueError(f"Unknown product type: {{product_type}}")
''',
            "indicators": [
                "create method",
                "factory",
                "object creation"
            ]
        }

        # Observer pattern
        self.patterns["observer"] = {
            "name": "Observer Pattern",
            "category": "behavioral",
            "description": "Define subscription mechanism for event notifications",
            "template": '''
class {subject_name}:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update(event)
''',
            "indicators": [
                "observers",
                "attach/detach",
                "notify",
                "event subscription"
            ]
        }

    def detect_pattern(self, code: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Detect patterns in code"""
        detected_patterns = []

        for pattern_name, pattern_info in self.patterns.items():
            confidence = self._calculate_pattern_confidence(code, pattern_info)

            if confidence >= threshold:
                detected_patterns.append({
                    "pattern": pattern_name,
                    "name": pattern_info["name"],
                    "category": pattern_info["category"],
                    "confidence": confidence,
                    "description": pattern_info["description"]
                })

        return detected_patterns

    def _calculate_pattern_confidence(self, code: str, pattern_info: Dict[str, Any]) -> float:
        """Calculate confidence that pattern exists in code"""
        indicators = pattern_info.get("indicators", [])
        matches = 0

        code_lower = code.lower()

        for indicator in indicators:
            if indicator.lower() in code_lower:
                matches += 1

        return matches / len(indicators) if indicators else 0.0

    def suggest_pattern(self, code_description: str) -> List[Dict[str, Any]]:
        """Suggest appropriate patterns for given description"""
        suggestions = []
        description_lower = code_description.lower()

        for pattern_name, pattern_info in self.patterns.items():
            # Simple keyword matching for suggestions
            pattern_keywords = pattern_info.get("indicators", [])

            relevance = sum(
                1 for keyword in pattern_keywords
                if keyword.lower() in description_lower
            )

            if relevance > 0:
                suggestions.append({
                    "pattern": pattern_name,
                    "name": pattern_info["name"],
                    "relevance": relevance,
                    "description": pattern_info["description"],
                    "category": pattern_info["category"]
                })

        # Sort by relevance
        suggestions.sort(key=lambda x: x["relevance"], reverse=True)

        return suggestions

    def generate_pattern_implementation(
        self,
        pattern_name: str,
        parameters: Dict[str, str]
    ) -> str:
        """Generate pattern implementation with parameters"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        pattern = self.patterns[pattern_name]
        template = pattern["template"]

        # Simple template substitution
        for key, value in parameters.items():
            template = template.replace(f"{{{key}}}", value)

        return template.strip()
```

## Testing Guidelines

### Deep Code Analysis Testing
```python
import pytest
from unittest.mock import patch, AsyncMock
from src.deepcode.python_analyzer import PythonCodeAnalyzer, AnalysisLevel

@pytest.mark.asyncio
async def test_python_code_analysis():
    """Test Python code analysis"""
    analyzer = PythonCodeAnalyzer(AnalysisLevel.SEMANTIC)

    test_code = '''
class TestClass:
    def __init__(self):
        self.value = 0

    def method1(self):
        return self.value * 2

    def method2(self, x):
        if x > 0:
            return x + 1
        else:
            return x - 1
'''

    result = await analyzer.analyze_code(test_code, "test.py")

    assert result.file_path == "test.py"
    assert len(result.elements) > 0

    # Check for class detection
    class_elements = [e for e in result.elements if e.element_type == "class"]
    assert len(class_elements) == 1
    assert class_elements[0].name == "TestClass"

    # Check for method detection
    function_elements = [e for e in result.elements if e.element_type == "function"]
    assert len(function_elements) >= 3  # __init__, method1, method2

def test_code_element_extraction():
    """Test code element extraction"""
    analyzer = PythonCodeAnalyzer()

    test_code = '''
import os
from typing import Dict

def test_function(x: int) -> int:
    return x * 2

class TestClass:
    pass
'''

    elements = analyzer.extract_elements(test_code)

    # Check imports
    import_elements = [e for e in elements if e.element_type in ["import", "import_from"]]
    assert len(import_elements) == 2

    # Check function
    function_elements = [e for e in elements if e.element_type == "function"]
    assert len(function_elements) == 1
    assert function_elements[0].name == "test_function"

    # Check class
    class_elements = [e for e in elements if e.element_type == "class"]
    assert len(class_elements) == 1
    assert class_elements[0].name == "TestClass"

@pytest.mark.asyncio
async def test_semantic_analysis():
    """Test semantic code analysis"""
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.choices[0].message.content = '''
Purpose: This function calculates the factorial of a number
Key Components: Recursive function with base case
Dependencies: None
Complexity: Low to medium
Maintainability: Good
Potential Issues: No input validation
'''
        mock_client.chat.completions.create.return_value = mock_response

        analyzer = SemanticCodeAnalyzer("test_key")

        test_code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

        result = await analyzer.analyze_semantic_meaning(test_code)

        assert "purpose" in result
        assert "complexity" in result

def test_pattern_detection():
    """Test code pattern detection"""
    library = CodePatternLibrary()

    singleton_code = '''
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
'''

    detected_patterns = library.detect_pattern(singleton_code)

    assert len(detected_patterns) > 0
    singleton_patterns = [p for p in detected_patterns if p["pattern"] == "singleton"]
    assert len(singleton_patterns) == 1
    assert singleton_patterns[0]["confidence"] > 0.5

@pytest.mark.asyncio
async def test_code_generation():
    """Test AI-powered code generation"""
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.choices[0].message.content = '''
Here's the generated code:

```python
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

This function calculates the average of a list of numbers.
'''
        mock_client.chat.completions.create.return_value = mock_response

        generator = CodeGenerationEngine("test_key")

        result = await generator.generate_code(
            "Create a function that calculates the average of a list of numbers",
            "python"
        )

        assert result["success"] is True
        assert "def calculate_average" in result["code"]
```

### Performance Testing
```python
@pytest.mark.performance
async def test_analysis_performance():
    """Test code analysis performance"""
    analyzer = PythonCodeAnalyzer()

    # Generate large test code
    test_code = "\n".join([
        f"def function_{i}():\n    return {i}"
        for i in range(100)
    ])

    start_time = time.time()
    result = await analyzer.analyze_code(test_code)
    analysis_time = time.time() - start_time

    # Should analyze 100 functions quickly
    assert analysis_time < 5.0  # Less than 5 seconds
    assert len(result.elements) >= 100

@pytest.mark.benchmark
def test_pattern_detection_performance(benchmark):
    """Benchmark pattern detection performance"""
    library = CodePatternLibrary()

    large_code = """
class LargeClass:
    def __init__(self):
        self._observers = []

    def method1(self): pass
    def method2(self): pass
    # ... many more methods
""" + "\n".join([f"    def method_{i}(self): pass" for i in range(50)])

    result = benchmark(library.detect_pattern, large_code)
    assert isinstance(result, list)
```

## Security Guidelines

### Code Analysis Security
```python
class SecureCodeAnalyzer:
    """Security-focused code analyzer"""

    def __init__(self):
        self.security_patterns = {
            "sql_injection": [
                r"execute\s*\(\s*[\"'].*%.*[\"']\s*%",
                r"cursor\.execute\s*\(\s*f[\"']",
                r"query\s*=.*\+.*input"
            ],
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"subprocess\.call.*shell=True"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*[\"'][^\"']+[\"']",
                r"api_key\s*=\s*[\"'][^\"']+[\"']",
                r"secret\s*=\s*[\"'][^\"']+[\"']"
            ]
        }

    def scan_security_vulnerabilities(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for security vulnerabilities"""
        vulnerabilities = []

        for vuln_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)

                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    vulnerabilities.append({
                        "type": vuln_type,
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(),
                        "severity": self._get_severity(vuln_type)
                    })

        return vulnerabilities

    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type"""
        severity_map = {
            "sql_injection": "high",
            "code_injection": "critical",
            "hardcoded_secrets": "medium"
        }
        return severity_map.get(vuln_type, "low")
```

### Safe Code Execution
```python
class SafeCodeExecutor:
    """Execute generated code safely"""

    def __init__(self):
        self.allowed_imports = {
            'math', 'datetime', 'json', 'typing', 're', 'collections'
        }

    async def execute_generated_code(
        self,
        code: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Safely execute generated code"""
        # Validate code before execution
        violations = self._validate_code_safety(code)

        if violations:
            return {
                "success": False,
                "error": f"Security violations: {violations}"
            }

        try:
            # Use restricted execution environment
            from src.sandbox.exec_sandbox import execute_in_sandbox

            result = await execute_in_sandbox(
                code=code,
                timeout=timeout,
                allowed_imports=self.allowed_imports
            )

            return {
                "success": True,
                "result": result,
                "output": result.get("stdout", ""),
                "errors": result.get("stderr", "")
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _validate_code_safety(self, code: str) -> List[str]:
        """Validate code for safety before execution"""
        violations = []

        # Check for dangerous imports
        dangerous_imports = ['os', 'subprocess', 'sys', 'importlib', '__builtin__']
        for dangerous in dangerous_imports:
            if f"import {dangerous}" in code or f"from {dangerous}" in code:
                violations.append(f"Dangerous import: {dangerous}")

        # Check for dangerous functions
        dangerous_functions = ['eval', 'exec', 'compile', 'open', '__import__']
        for dangerous in dangerous_functions:
            if f"{dangerous}(" in code:
                violations.append(f"Dangerous function: {dangerous}")

        return violations
```

## Performance Guidelines

### Optimized Analysis Pipeline
```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

class ParallelCodeAnalyzer:
    """Parallel code analysis for performance"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)

    async def analyze_multiple_files(
        self,
        file_paths: List[str]
    ) -> Dict[str, AnalysisResult]:
        """Analyze multiple files in parallel"""
        loop = asyncio.get_event_loop()

        tasks = []
        for file_path in file_paths:
            task = loop.run_in_executor(
                self.process_pool,
                self._analyze_single_file,
                file_path
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            file_paths[i]: result
            for i, result in enumerate(results)
            if not isinstance(result, Exception)
        }

    def _analyze_single_file(self, file_path: str) -> AnalysisResult:
        """Analyze single file (runs in separate process)"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()

            analyzer = PythonCodeAnalyzer()
            # Note: This needs to be sync for process pool
            return asyncio.run(analyzer.analyze_code(code, file_path))

        except Exception as e:
            return AnalysisResult(
                file_path=file_path,
                analysis_level=AnalysisLevel.LEXICAL,
                elements=[],
                patterns=[],
                metrics={"error": str(e)}
            )
```

## Common Patterns

### Analysis Pipeline
```python
class CodeAnalysisPipeline:
    """Pipeline for comprehensive code analysis"""

    def __init__(self):
        self.stages: List[Callable] = []

    def add_stage(self, stage_func: Callable):
        """Add analysis stage to pipeline"""
        self.stages.append(stage_func)

    async def run_pipeline(self, code: str) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        results = {}

        for i, stage in enumerate(self.stages):
            stage_name = stage.__name__

            try:
                stage_result = await stage(code, results)
                results[stage_name] = stage_result
            except Exception as e:
                logger.error(f"Pipeline stage {stage_name} failed: {e}")
                results[stage_name] = {"error": str(e)}

        return results

# Example pipeline stages
async def lexical_analysis_stage(code: str, previous_results: Dict) -> Dict[str, Any]:
    """Lexical analysis stage"""
    analyzer = PythonCodeAnalyzer(AnalysisLevel.LEXICAL)
    result = await analyzer.analyze_code(code)
    return {"elements": len(result.elements), "metrics": result.metrics}

async def pattern_detection_stage(code: str, previous_results: Dict) -> Dict[str, Any]:
    """Pattern detection stage"""
    library = CodePatternLibrary()
    patterns = library.detect_pattern(code)
    return {"patterns": patterns}

async def security_scan_stage(code: str, previous_results: Dict) -> Dict[str, Any]:
    """Security scanning stage"""
    scanner = SecureCodeAnalyzer()
    vulnerabilities = scanner.scan_security_vulnerabilities(code)
    return {"vulnerabilities": vulnerabilities}
```

## Debugging Tips
- **AST visualization** - Visualize AST structures for debugging analysis
- **Pattern matching logs** - Log pattern detection results for tuning
- **AI prompt debugging** - Debug AI prompts and responses
- **Performance profiling** - Profile analysis performance on large codebases
- **Security validation** - Validate security scanning effectiveness
