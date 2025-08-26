#!/usr/bin/env python3
"""
Dynamic Tool Generator: Secure Meta-programming for on-the-fly tool creation

This module enables the agent to dynamically generate and compile tools
based on user requests, with security-first execution and full audit trails.

Key Features:
- Secure parameter sanitization and validation
- Security-first code generation and execution
- Full audit trails and provenance tracking
- Tool library with usage statistics
- Automated testing and validation
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .atom_tool import AtomTool, ToolResult, ToolSignature
from .secure_executor import (
    CodeExecutionError,
    get_secure_executor,
    get_tool_registry,
)
from .tool_memory import format_tool_response, get_memory_manager, prompt_save_tool

logger = logging.getLogger(__name__)


@dataclass
class ToolTemplate:
    """Template for generating dynamic tools."""

    name: str
    description: str
    parameters: dict[str, str]
    code_template: str
    test_cases: list[dict[str, Any]]


class DynamicToolGenerator:
    """Generates secure dynamic tools based on requirements"""

    def __init__(self):
        self.secure_executor = get_secure_executor()
        self.tool_registry = get_tool_registry()
        self.memory_manager = get_memory_manager()

        self.tool_templates = {
            "quantum_circuit": self._generate_quantum_circuit_tool,
            "data_analysis": self._generate_data_analysis_tool,
            "math_computation": self._generate_math_tool,
            "text_processing": self._generate_text_processing_tool,
            "simulation": self._generate_simulation_tool,
        }

    def generate_dynamic_tool(
        self,
        tool_type: str,
        params: dict[str, Any],
        user_id: str = "system",
        context_id: str = "default",
    ) -> tuple[str, dict[str, Any]]:
        """Generate secure code for a dynamic tool with full audit trail"""
        try:
            # Check if tool exists in registry first
            tool_hash = hashlib.md5(
                f"{tool_type}_{sorted(params.items())}".encode()
            ).hexdigest()[:8]
            tool_name = f"{tool_type}_{tool_hash}"

            existing_tool = self.tool_registry.get_tool(tool_name)
            if existing_tool:
                logger.info(f"Using existing tool from registry: {tool_name}")
                return existing_tool["code"], existing_tool

            # Generate new tool code
            if tool_type in self.tool_templates:
                code = self.tool_templates[tool_type](params)
            else:
                code = self._generate_generic_tool(tool_type, params)

            # Validate and test the generated code
            test_result = self.secure_executor.run_unit_test(code, params)

            if test_result["status"] == "failure":
                logger.error(f"Generated tool failed unit test: {test_result['error']}")
                raise CodeExecutionError(
                    f"Tool validation failed: {test_result['error']}"
                )

            # Register the tool
            self.tool_registry.register_tool(
                tool_name=tool_name,
                code=code,
                params_schema=params,
                description=f"Dynamic {tool_type} tool",
                author=user_id,
            )

            # Store in memory with provenance
            memory_id = self.memory_manager.store_dynamic_tool_atom(
                tool_name=tool_name,
                code=code,
                params=params,
                result=test_result.get("result"),
                error=None,
                user_id=user_id,
                context_id=context_id,
                tags=[tool_type, "generated", "validated"],
            )

            logger.info(f"Generated and validated dynamic tool: {tool_name}")

            return code, {
                "tool_name": tool_name,
                "memory_id": memory_id,
                "test_result": test_result,
                "params_schema": params,
            }

        except Exception as e:
            logger.error(f"Failed to generate dynamic tool: {e}")
            raise

    def execute_dynamic_tool(
        self,
        code: str,
        params: dict[str, Any],
        user_id: str = "system",
        context_id: str = "default",
        tool_name: str = "dynamic_tool",
    ) -> dict[str, Any]:
        """Execute a dynamic tool with security and audit trail"""
        start_time = datetime.now()

        try:
            # Execute with security and audit
            func, audit_log = self.secure_executor.execute_with_audit(
                code=code, params=params, user_id=user_id, context_id=context_id
            )

            # Run the function
            result = func(**params)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Store execution in memory
            memory_id = self.memory_manager.store_dynamic_tool_atom(
                tool_name=tool_name,
                code=code,
                params=params,
                result=result,
                error=None,
                user_id=user_id,
                context_id=context_id,
                execution_time=execution_time,
                tags=["executed", "success"],
            )

            # Update usage statistics
            self.memory_manager.update_tool_usage(tool_name, success=True)

            # Format response
            response = format_tool_response(result, code, params, tool_name=tool_name)
            response.update(
                {
                    "execution_time": execution_time,
                    "audit_id": audit_log.uuid,
                    "memory_id": memory_id,
                    "save_prompt": prompt_save_tool(user_id, tool_name),
                }
            )

            return response

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            # Store failed execution
            memory_id = self.memory_manager.store_dynamic_tool_atom(
                tool_name=tool_name,
                code=code,
                params=params,
                result=None,
                error=error_msg,
                user_id=user_id,
                context_id=context_id,
                execution_time=execution_time,
                tags=["executed", "failed"],
            )

            # Update usage statistics
            self.memory_manager.update_tool_usage(tool_name, success=False)

            # Format error response
            response = format_tool_response(
                None, code, params, error=error_msg, tool_name=tool_name
            )
            response.update({"execution_time": execution_time, "memory_id": memory_id})

            return response

    def _extract_parameters(self, text: str) -> dict[str, Any]:
        """Extract parameters from natural language text with security validation"""
        params = {}

        # Common patterns for parameter extraction
        patterns = [
            (r"depth[=:\s]+(\d+)", "depth", int),
            (r"qubits?[=:\s]+(\d+)", "qubits", int),
            (r"size[=:\s]+(\d+)", "size", int),
            (r"length[=:\s]+(\d+)", "length", int),
            (r"iterations?[=:\s]+(\d+)", "iterations", int),
            (r"steps?[=:\s]+(\d+)", "steps", int),
            (r"samples?[=:\s]+(\d+)", "samples", int),
            (r"precision[=:\s]+(\d+)", "precision", int),
            (r"threshold[=:\s]+([\d.]+)", "threshold", float),
            (r"rate[=:\s]+([\d.]+)", "rate", float),
        ]

        for pattern, param_name, param_type in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = param_type(match.group(1))
                    # Add reasonable bounds for security
                    if (param_type == int and 0 <= value <= 1000) or (
                        param_type == float and 0.0 <= value <= 1.0
                    ):
                        params[param_name] = value
                except ValueError:
                    continue

        return params

    def _generate_quantum_circuit_tool(self, params: dict[str, Any]) -> str:
        """Generate secure quantum circuit simulation tool"""
        depth = params.get("depth", 3)
        qubits = params.get("qubits", 2)

        # Validate parameters for security
        if not (1 <= depth <= 20) or not (1 <= qubits <= 10):
            raise ValueError("Invalid quantum circuit parameters")

        code = f'''
def quantum_circuit_sim(depth={depth}, qubits={qubits}):
    """Secure quantum circuit simulator with bounded parameters"""
    import math
    import random

    class QuantumState:
        def __init__(self, n_qubits):
            self.n_qubits = min(n_qubits, 10)  # Security bound
            self.n_states = 2 ** self.n_qubits
            self.amplitudes = [0.0] * self.n_states
            self.amplitudes[0] = 1.0  # |00...0⟩ state

        def apply_hadamard(self, qubit):
            if 0 <= qubit < self.n_qubits:
                new_amplitudes = [0.0] * self.n_states
                for i in range(self.n_states):
                    if (i >> qubit) & 1 == 0:  # qubit is 0
                        new_amplitudes[i] += self.amplitudes[i] / math.sqrt(2)
                        new_amplitudes[i | (1 << qubit)] += self.amplitudes[i] / math.sqrt(2)
                    else:  # qubit is 1
                        new_amplitudes[i] += self.amplitudes[i] / math.sqrt(2)
                        new_amplitudes[i & ~(1 << qubit)] -= self.amplitudes[i] / math.sqrt(2)
                self.amplitudes = new_amplitudes

        def apply_cnot(self, control, target):
            if 0 <= control < self.n_qubits and 0 <= target < self.n_qubits and control != target:
                new_amplitudes = [0.0] * self.n_states
                for i in range(self.n_states):
                    if (i >> control) & 1 == 1:  # control is 1
                        new_amplitudes[i ^ (1 << target)] = self.amplitudes[i]
                    else:  # control is 0
                        new_amplitudes[i] = self.amplitudes[i]
                self.amplitudes = new_amplitudes

        def measure(self):
            probabilities = [abs(amp) ** 2 for amp in self.amplitudes]
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]

            rand = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if rand <= cumulative:
                    return i
            return 0

        def get_state_vector(self):
            return self.amplitudes.copy()

        def draw(self):
            result = f"Quantum Circuit ({{qubits}} qubits, {{depth}} depth)\\n"
            result += "State vector:\\n"
            for i, amp in enumerate(self.amplitudes):
                if abs(amp) > 1e-6:
                    binary = format(i, f'0{{self.n_qubits}}b')
                    result += f"|{{binary}}⟩: {{amp:.3f}}\\n"
            return result

    # Create and simulate circuit
    qc = QuantumState(qubits)

    # Apply random gates with security bounds
    safe_depth = min(depth, 20)
    for layer in range(safe_depth):
        # Apply Hadamard gates
        if random.random() < 0.5:
            qubit = random.randint(0, qubits - 1)
            qc.apply_hadamard(qubit)

        # Apply CNOT gates
        if qubits > 1 and random.random() < 0.3:
            control = random.randint(0, qubits - 1)
            target = random.randint(0, qubits - 1)
            if control != target:
                qc.apply_cnot(control, target)

    return qc
'''
        return code.strip()

    def _generate_data_analysis_tool(self, params: dict[str, Any]) -> str:
        """Generate secure data analysis tool"""
        size = params.get("size", 100)

        if not (10 <= size <= 10000):
            raise ValueError("Invalid data size parameters")

        code = f'''
def data_analysis_tool(size={size}):
    """Secure data analysis with bounded parameters"""
    import math
    import random

    # Generate safe synthetic data
    safe_size = min(size, 10000)  # Security bound
    data = [random.gauss(0, 1) for _ in range(safe_size)]

    # Basic statistics
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)

    # Find min/max
    min_val = min(data)
    max_val = max(data)

    # Simple histogram (10 bins)
    bins = [0] * 10
    range_size = (max_val - min_val) / 10
    for value in data:
        bin_index = min(int((value - min_val) / range_size), 9)
        bins[bin_index] += 1

    result = {{
        "size": n,
        "mean": round(mean, 3),
        "std_dev": round(std_dev, 3),
        "min": round(min_val, 3),
        "max": round(max_val, 3),
        "histogram": bins
    }}

    return result
'''
        return code.strip()

    def _generate_math_tool(self, params: dict[str, Any]) -> str:
        """Generate secure mathematical computation tool"""
        precision = params.get("precision", 6)

        if not (1 <= precision <= 15):
            raise ValueError("Invalid precision parameters")

        code = f'''
def math_computation_tool(precision={precision}):
    """Secure mathematical computation with bounded precision"""
    import math

    safe_precision = min(precision, 15)  # Security bound

    # Mathematical constants and computations
    pi_approx = 0
    for n in range(100):  # Limited iterations for security
        pi_approx += ((-1) ** n) / (2 * n + 1)
    pi_approx *= 4

    # Fibonacci sequence
    fib = [0, 1]
    for i in range(2, min(20, safe_precision + 5)):  # Security bound
        fib.append(fib[i-1] + fib[i-2])

    # Prime numbers using sieve
    def sieve_of_eratosthenes(limit):
        limit = min(limit, 1000)  # Security bound
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = sieve_of_eratosthenes(100)

    result = {{
        "pi_approximation": round(pi_approx, safe_precision),
        "euler_number": round(math.e, safe_precision),
        "golden_ratio": round((1 + math.sqrt(5)) / 2, safe_precision),
        "fibonacci_sequence": fib[:safe_precision],
        "first_primes": primes[:safe_precision]
    }}

    return result
'''
        return code.strip()

    def _generate_text_processing_tool(self, params: dict[str, Any]) -> str:
        """Generate secure text processing tool"""
        length = params.get("length", 100)

        if not (10 <= length <= 10000):
            raise ValueError("Invalid text length parameters")

        code = f'''
def text_processing_tool(length={length}):
    """Secure text processing with bounded parameters"""
    import re

    # Generate sample text
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "and", "runs", "through", "forest", "with", "great", "speed"]

    safe_length = min(length, 10000)  # Security bound
    text_words = []
    word_count = 0

    while len(" ".join(text_words)) < safe_length and word_count < 1000:
        import random
        text_words.append(random.choice(words))
        word_count += 1

    text = " ".join(text_words)[:safe_length]

    # Text analysis
    word_freq = {{}}
    for word in text.split():
        clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
        if clean_word:
            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

    # Statistics
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))

    result = {{
        "text_length": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(sum(len(word) for word in text.split()) / word_count, 2) if word_count > 0 else 0,
        "most_common_words": sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    }}

    return result
'''
        return code.strip()

    def _generate_simulation_tool(self, params: dict[str, Any]) -> str:
        """Generate secure simulation tool"""
        steps = params.get("steps", 100)

        if not (10 <= steps <= 10000):
            raise ValueError("Invalid simulation steps parameters")

        code = f'''
def simulation_tool(steps={steps}):
    """Secure random walk simulation with bounded parameters"""
    import random
    import math

    safe_steps = min(steps, 10000)  # Security bound

    # 2D Random Walk
    x, y = 0, 0
    positions = [(x, y)]

    for _ in range(safe_steps):
        direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        x += direction[0]
        y += direction[1]
        positions.append((x, y))

    # Calculate statistics
    distances = [math.sqrt(pos[0]**2 + pos[1]**2) for pos in positions]
    max_distance = max(distances)
    final_distance = distances[-1]

    # Displacement analysis
    displacement_x = positions[-1][0]
    displacement_y = positions[-1][1]

    result = {{
        "steps": safe_steps,
        "final_position": positions[-1],
        "max_distance_from_origin": round(max_distance, 3),
        "final_distance_from_origin": round(final_distance, 3),
        "displacement_x": displacement_x,
        "displacement_y": displacement_y,
        "path_efficiency": round(final_distance / safe_steps, 3) if safe_steps > 0 else 0
    }}

    return result
'''
        return code.strip()

    def _generate_generic_tool(self, tool_type: str, params: dict[str, Any]) -> str:
        """Generate secure generic tool for unknown types"""
        code = f'''
def generic_tool_{tool_type}(**params):
    """Secure generic tool for {tool_type}"""
    import json

    # Safe parameter processing
    safe_params = {{}}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool)) and len(str(key)) < 100:
            safe_params[key] = value

    result = {{
        "tool_type": "{tool_type}",
        "parameters": safe_params,
        "message": "Generic tool executed successfully",
        "status": "completed"
    }}

    return result
'''
        return code.strip()


# Dynamic AtomTool wrapper
class DynamicAtomTool(AtomTool):
    """AtomTool wrapper for dynamically generated tools"""

    def __init__(
        self,
        tool_name: str,
        code: str,
        params_schema: dict[str, Any],
        description: str = None,
    ):
        self.tool_name = tool_name
        self.code = code
        self.params_schema = params_schema
        self.generator = DynamicToolGenerator()

        super().__init__(
            signature=ToolSignature(
                name=tool_name,
                description=description or f"Dynamic tool: {tool_name}",
                parameters=params_schema,
            )
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the dynamic tool"""
        try:
            response = self.generator.execute_dynamic_tool(
                code=self.code, params=kwargs, tool_name=self.tool_name
            )

            return ToolResult(
                success=True,
                result=response.get("Result"),
                message=response.get("Status", "Success"),
                metadata=response,
            )

        except Exception as e:
            return ToolResult(
                success=False, error=str(e), message="Dynamic tool execution failed"
            )


# Global instance
_dynamic_generator = DynamicToolGenerator()


def get_dynamic_generator() -> DynamicToolGenerator:
    """Get global dynamic tool generator instance"""
    return _dynamic_generator
    """
    Generates tools dynamically based on user requests.

    This is the core meta-programming engine that transforms natural language
    requests into executable code tools.
    """

    def __init__(self):
        self._templates = {}
        self._compiled_cache = {}
        self._setup_built_in_templates()

    def _setup_built_in_templates(self):
        """Setup built-in tool templates for common tasks."""

        # Quantum Circuit Template
        self._templates["quantum_circuit"] = ToolTemplate(
            name="Quantum Circuit Generator",
            description="Generate and simulate quantum circuits",
            parameter_patterns=[
                r"qubits?=(\d+)",
                r"depth=(\d+)",
                r"gates?=(\w+)",
                r"backend=(\w+)",
            ],
            code_template="""
def quantum_circuit_sim(qubits={qubits}, depth={depth}, gates='{gates}', backend='{backend}'):
    '''Dynamically generated quantum circuit simulator'''
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.primitives import Sampler
        from qiskit.visualization import circuit_drawer
        import numpy as np

        # Create quantum circuit
        circuit = QuantumCircuit(qubits, qubits)

        # Apply gates based on depth and pattern
        gate_types = gates.split(',') if ',' in gates else [gates]

        for layer in range(depth):
            for qubit in range(qubits):
                if 'h' in gate_types or 'hadamard' in gate_types:
                    circuit.h(qubit)
                if 'x' in gate_types and layer % 2 == 0:
                    circuit.x(qubit)
                if 'cnot' in gate_types and qubit < qubits - 1:
                    circuit.cx(qubit, qubit + 1)

        # Add measurements
        circuit.measure_all()

        # Get circuit diagram
        circuit_text = str(circuit_drawer(circuit, output='text'))

        # Simulate (mock for now)
        result = {{
            'circuit_diagram': circuit_text,
            'qubits': qubits,
            'depth': depth,
            'gates_used': gate_types,
            'success': True
        }}

        return result

    except ImportError:
        return {{
            'error': 'Qiskit not available - install with: pip install qiskit',
            'mock_result': 'Quantum circuit with {{}} qubits, depth {{}}'.format(qubits, depth),
            'success': False
        }}
    except Exception as e:
        return {{
            'error': str(e),
            'success': False
        }}
""",
            required_imports=["qiskit"],
            category="quantum",
        )

        # Mathematical Analysis Template
        self._templates["math_analysis"] = ToolTemplate(
            name="Mathematical Analysis Generator",
            description="Generate mathematical analysis and computation tools",
            parameter_patterns=[
                r"function=([^\\s]+)",
                r"range=\\[([-\\d.]+),([-\\d.]+)\\]",
                r"points=(\d+)",
                r"method=(\w+)",
            ],
            code_template="""
def math_analysis(function='{function}', range_start={range_start}, range_end={range_end}, points={points}, method='{method}'):
    '''Dynamically generated mathematical analysis tool'''
    try:
        import numpy as np
        import sympy as sp
        from sympy import symbols, lambdify, diff, integrate, limit, series

        # Parse the function
        x = symbols('x')
        try:
            expr = sp.sympify(function)
        except:
            expr = sp.sympify(function.replace('^', '**'))

        # Create numerical function
        f = lambdify(x, expr, 'numpy')

        # Generate analysis based on method
        results = {{
            'function': str(expr),
            'method': method,
            'range': [range_start, range_end]
        }}

        if method in ['plot', 'analyze', 'all']:
            # Generate points for plotting
            x_vals = np.linspace(range_start, range_end, points)
            y_vals = f(x_vals)

            results['x_values'] = x_vals.tolist()[:10]  # Limit output size
            results['y_values'] = y_vals.tolist()[:10]
            results['min_value'] = float(np.min(y_vals))
            results['max_value'] = float(np.max(y_vals))

        if method in ['derivative', 'analyze', 'all']:
            # Calculate derivative
            derivative = diff(expr, x)
            results['derivative'] = str(derivative)

        if method in ['integral', 'analyze', 'all']:
            # Calculate integral
            integral = integrate(expr, x)
            results['integral'] = str(integral)

        if method in ['zeros', 'analyze', 'all']:
            # Find zeros/roots
            try:
                zeros = sp.solve(expr, x)
                results['zeros'] = [str(z) for z in zeros[:5]]  # Limit to 5 zeros
            except:
                results['zeros'] = 'Could not solve analytically'

        results['success'] = True
        return results

    except ImportError as e:
        return {{
            'error': f'Required library not available: {{e}}',
            'suggestion': 'Install with: pip install numpy sympy matplotlib',
            'success': False
        }}
    except Exception as e:
        return {{
            'error': str(e),
            'success': False
        }}
""",
            required_imports=["numpy", "sympy"],
            category="mathematics",
        )

        # Data Processing Template
        self._templates["data_processor"] = ToolTemplate(
            name="Data Processing Generator",
            description="Generate data processing and analysis tools",
            parameter_patterns=[
                r"operation=(\w+)",
                r"format=(csv|json|xml)",
                r"columns?=([^\\s]+)",
                r"filter=([^\\s]+)",
            ],
            code_template="""
def data_processor(operation='{operation}', format='{format}', columns='{columns}', filter_expr='{filter}'):
    '''Dynamically generated data processing tool'''
    try:
        import pandas as pd
        import numpy as np
        import json

        # Mock data generation for demonstration
        if format == 'csv':
            data = pd.DataFrame({{
                'id': range(1, 11),
                'value': np.random.randn(10),
                'category': ['A', 'B'] * 5,
                'timestamp': pd.date_range('2024-01-01', periods=10)
            }})
        else:
            data = pd.DataFrame({{'sample': [1, 2, 3, 4, 5]}})

        results = {{
            'operation': operation,
            'format': format,
            'input_shape': data.shape
        }}

        # Apply operation
        if operation == 'summary':
            results['summary'] = data.describe().to_dict()
            results['columns'] = list(data.columns)

        elif operation == 'filter':
            if filter_expr and filter_expr != 'None':
                try:
                    filtered_data = data.query(filter_expr)
                    results['filtered_shape'] = filtered_data.shape
                    results['sample_data'] = filtered_data.head().to_dict()
                except:
                    results['error'] = f'Invalid filter expression: {{filter_expr}}'

        elif operation == 'aggregate':
            if columns and columns != 'None':
                col_list = columns.split(',')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if numeric_cols.any():
                    results['aggregation'] = data[numeric_cols].agg(['mean', 'sum', 'count']).to_dict()

        elif operation == 'transform':
            # Basic transformation example
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_transformed = data.copy()
                data_transformed[numeric_cols] = data_transformed[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
                results['transformation'] = 'standardization'
                results['sample_transformed'] = data_transformed.head().to_dict()

        results['success'] = True
        return results

    except ImportError:
        return {{
            'error': 'Pandas not available - install with: pip install pandas numpy',
            'success': False
        }}
    except Exception as e:
        return {{
            'error': str(e),
            'success': False
        }}
""",
            required_imports=["pandas", "numpy"],
            category="data",
        )

        logger.info(f"Initialized {len(self._templates)} dynamic tool templates")

    def extract_parameters(self, text: str, template: ToolTemplate) -> dict[str, Any]:
        """Extract parameters from natural language text using template patterns."""
        params = {}

        for pattern in template.parameter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                param_name = (
                    pattern.split("=")[0]
                    .strip("(")
                    .replace("?", "")
                    .replace("\\\\", "")
                )
                if param_name.endswith("s"):  # Handle plural forms
                    param_name = param_name[:-1]

                # Convert types
                value = matches[0]
                if value.isdigit():
                    params[param_name] = int(value)
                elif value.replace(".", "").replace("-", "").isdigit():
                    params[param_name] = float(value)
                else:
                    params[param_name] = value

        return params

    def detect_tool_type(self, request: str) -> str | None:
        """Detect what type of tool is needed based on the request."""
        request_lower = request.lower()

        # Quantum computing keywords
        if any(
            keyword in request_lower
            for keyword in ["quantum", "qubit", "circuit", "gate", "superposition"]
        ):
            return "quantum_circuit"

        # Mathematical analysis keywords
        if any(
            keyword in request_lower
            for keyword in [
                "function",
                "derivative",
                "integral",
                "plot",
                "analyze",
                "math",
            ]
        ):
            return "math_analysis"

        # Data processing keywords
        if any(
            keyword in request_lower
            for keyword in [
                "data",
                "csv",
                "process",
                "filter",
                "aggregate",
                "transform",
            ]
        ):
            return "data_processor"

        return None

    def generate_tool(self, request: str) -> Optional["DynamicAtomTool"]:
        """
        Generate a dynamic tool based on the request.

        Args:
            request: Natural language description of desired tool

        Returns:
            DynamicAtomTool instance or None if generation fails
        """
        try:
            # Detect tool type
            tool_type = self.detect_tool_type(request)
            if not tool_type or tool_type not in self._templates:
                logger.warning(f"Could not detect tool type for request: {request}")
                return None

            template = self._templates[tool_type]

            # Extract parameters
            params = self.extract_parameters(request, template)
            logger.info(f"Extracted parameters: {params}")

            # Set defaults for missing parameters
            defaults = self._get_defaults(tool_type)
            for key, value in defaults.items():
                if key not in params:
                    params[key] = value

            # Generate unique tool key
            request_hash = hashlib.md5(request.encode()).hexdigest()[:8]
            tool_key = f"dynamic_{tool_type}_{request_hash}"

            # Check cache first
            if tool_key in self._compiled_cache:
                logger.info(f"Using cached tool: {tool_key}")
                return self._compiled_cache[tool_key]

            # Generate code
            code = template.code_template.format(**params)

            # Compile tool
            tool = DynamicAtomTool(
                key=tool_key,
                name=f"Dynamic {template.name}",
                description=f"Generated tool for: {request}",
                code=code,
                parameters=params,
                template_name=tool_type,
            )

            # Cache the tool
            self._compiled_cache[tool_key] = tool

            logger.info(f"Generated dynamic tool: {tool.name}")
            return tool

        except Exception as e:
            logger.error(f"Tool generation failed: {e}")
            return None

    def _get_defaults(self, tool_type: str) -> dict[str, Any]:
        """Get default parameters for each tool type."""
        defaults = {
            "quantum_circuit": {
                "qubits": 3,
                "depth": 5,
                "gates": "h,cnot",
                "backend": "aer_simulator",
            },
            "math_analysis": {
                "function": "x**2",
                "range_start": -10,
                "range_end": 10,
                "points": 100,
                "method": "analyze",
            },
            "data_processor": {
                "operation": "summary",
                "format": "csv",
                "columns": "all",
                "filter": "None",
            },
        }
        return defaults.get(tool_type, {})


class DynamicAtomTool(AtomTool):
    """
    A dynamically generated AtomTool that compiles and executes generated code.
    """

    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        code: str,
        parameters: dict[str, Any],
        template_name: str,
    ):
        # Create signature from parameters
        signature = ToolSignature()
        for param_name, param_value in parameters.items():
            param_type = "integer" if isinstance(param_value, int) else "string"
            signature.add_param(param_name, param_type, f"Parameter {param_name}")

        super().__init__(
            key=key,
            name=name,
            description=description,
            signature=signature,
            category="dynamic",
        )

        self.code = code
        self.parameters = parameters
        self.template_name = template_name
        self._compiled_function = None
        self._compile_code()

    def _compile_code(self):
        """Compile the generated code into an executable function."""
        try:
            # Create execution scope
            scope = {
                "__builtins__": __builtins__,
                "print": print,  # Allow print for debugging
            }

            # Execute the code to define the function
            exec(self.code, scope)

            # Find the main function (assumes first function defined)
            for name, obj in scope.items():
                if callable(obj) and not name.startswith("__"):
                    self._compiled_function = obj
                    break

            if not self._compiled_function:
                raise RuntimeError("No callable function found in generated code")

            logger.info(f"Successfully compiled dynamic tool: {self.key}")

        except Exception as e:
            logger.error(f"Code compilation failed: {e}")
            self._compiled_function = None

    async def call(self, **kwargs) -> ToolResult:
        """Execute the dynamically generated tool."""
        try:
            if not self._compiled_function:
                return ToolResult(success=False, error="Tool not properly compiled")

            logger.info(f"Executing dynamic tool: {self.name}")

            # Merge provided parameters with defaults
            execution_params = {**self.parameters, **kwargs}

            # Execute the compiled function
            result = self._compiled_function(**execution_params)

            return ToolResult(
                success=True,
                result=result,
                metadata={
                    "tool_type": "dynamic",
                    "template": self.template_name,
                    "parameters_used": execution_params,
                },
            )

        except Exception as e:
            logger.error(f"Dynamic tool execution failed: {e}")
            return ToolResult(success=False, error=f"Execution error: {e!s}")


# Global instance
_generator = DynamicToolGenerator()


def generate_dynamic_tool(request: str) -> DynamicAtomTool | None:
    """Factory function to generate dynamic tools."""
    return _generator.generate_tool(request)


if __name__ == "__main__":
    print("Dynamic Tool Generator Test")
    print("=" * 50)

    # Test quantum circuit generation
    request = "build quantum circuit with qubits=4 depth=3 gates=h,cnot"
    tool = generate_dynamic_tool(request)

    if tool:
        print(f"Generated tool: {tool.name}")
        print(f"Parameters: {tool.parameters}")
        print("Code preview:")
        print(tool.code[:200] + "...")
    else:
        print("Tool generation failed")
