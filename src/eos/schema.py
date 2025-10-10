"""
EOS Schema Validation and Parsing

Provides validation and parsing capabilities for E-UPUSF Orchestration Schema
specifications using JSON Schema validation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from jsonschema import Draft202012Validator


@dataclass
class ValidationResult:
    """Result of EOS schema validation"""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str | None = None


@dataclass
class EOSMeta:
    """EOS metadata configuration"""
    run_id: str
    owner: str
    provenance: list[str] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)


@dataclass
class EOSProblem:
    """Problem definition for orchestration"""
    title: str
    statement: str
    objectives: list[str]
    constraints: list[str] = field(default_factory=list)
    stakeholders: list[str] = field(default_factory=list)
    risk_tolerance: dict[str, str] = field(default_factory=dict)


@dataclass
class CynefinPrior:
    """Cynefin framework probability distribution"""
    simple: float
    complicated: float
    complex: float
    chaotic: float
    
    def __post_init__(self):
        """Validate probability distribution sums to ~1.0"""
        total = self.simple + self.complicated + self.complex + self.chaotic
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Cynefin probabilities must sum to 1.0, got {total}"
            )


@dataclass
class EOSContext:
    """Context analysis configuration"""
    cynefin_prior: CynefinPrior
    domain_hints: list[str] = field(default_factory=list)
    uncertainty_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class EOSExpert:
    """Expert definition for MoE routing"""
    id: str
    kind: str  # tool, agent, human
    inputs: list[str]
    outputs: list[str]
    cost: dict[str, float]
    risk: dict[str, float]
    quality_prior: float
    fit_hints: list[str] = field(default_factory=list)


@dataclass
class EOSMethod:
    """Method definition in registry"""
    id: str
    states: list[str]
    entry_criteria: dict[str, Any]
    operators: list[str]
    artifacts_out: list[str]


@dataclass
class EOSSpec:
    """Complete EOS specification"""
    eos_version: str
    meta: EOSMeta
    problem: EOSProblem
    context: EOSContext
    resources: dict[str, Any]
    methods_registry: list[EOSMethod]
    ladder: dict[str, Any]
    experts: list[EOSExpert]
    routing: dict[str, Any]
    pipeline: dict[str, Any]
    evaluation: dict[str, Any]
    governance: dict[str, Any]
    telemetry: dict[str, Any]


class EOSValidator:
    """EOS specification validator using JSON Schema"""
    
    def __init__(self, schema_path: Path | None = None):
        """Initialize validator with schema"""
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.json"
        
        with open(schema_path) as f:
            self.schema = json.load(f)
        
        self.validator = Draft202012Validator(self.schema)
    
    def validate_yaml(self, yaml_path: Path) -> ValidationResult:
        """Validate EOS YAML specification"""
        try:
            with open(yaml_path) as f:
                spec_data = yaml.safe_load(f)
            
            return self.validate_dict(spec_data)
            
        except yaml.YAMLError as e:
            return ValidationResult(
                valid=False,
                errors=[f"YAML parsing error: {e}"]
            )
        except FileNotFoundError:
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {yaml_path}"]
            )
    
    def validate_dict(self, spec_data: dict[str, Any]) -> ValidationResult:
        """Validate EOS specification as dictionary"""
        result = ValidationResult(valid=True)
        
        # Extract schema version
        result.schema_version = spec_data.get("eos_version")
        
        # Validate against JSON schema
        try:
            self.validator.validate(spec_data)
        except jsonschema.ValidationError as e:
            result.valid = False
            result.errors.append(f"Schema validation error: {e.message}")
            
            # Add path context for nested errors
            if e.absolute_path:
                path_str = " -> ".join(str(p) for p in e.absolute_path)
                result.errors.append(f"Error path: {path_str}")
        
        except jsonschema.SchemaError as e:
            result.valid = False
            result.errors.append(f"Schema definition error: {e.message}")
        
        # Additional semantic validations
        if result.valid:
            self._validate_semantics(spec_data, result)
        
        return result
    
    def _validate_semantics(self, spec_data: dict[str, Any],
                            result: ValidationResult) -> None:
        """Perform additional semantic validations"""
        
        # Validate Cynefin probabilities sum to 1.0
        cynefin = spec_data.get("context", {}).get("cynefin_prior", {})
        if cynefin:
            total = sum(cynefin.values())
            if not (0.99 <= total <= 1.01):
                result.warnings.append(
                    f"Cynefin probabilities sum to {total:.3f}, "
                    "should be ~1.0"
                )
        
        # Validate method references in pipeline
        method_ids = {m["id"] for m in spec_data.get("methods_registry", [])}
        pipeline_states = spec_data.get("pipeline", {}).get("states", [])
        
        for state in pipeline_states:
            for method_id in state.get("methods_candidates", []):
                if method_id not in method_ids:
                    result.warnings.append(
                        f"Pipeline state '{state['name']}' references "
                        f"unknown method '{method_id}'"
                    )
        
        # Validate expert fit hints
        expert_hints = set()
        for expert in spec_data.get("experts", []):
            expert_hints.update(expert.get("fit_hints", []))
        
        valid_hints = method_ids | {"Observe", "Analyze", "Synthesize",
                                    "Implement", "Evaluate", "Lift",
                                    "Decompose", "Descend"}
        
        invalid_hints = expert_hints - valid_hints
        if invalid_hints:
            result.warnings.append(
                f"Expert fit hints reference unknown methods/states: "
                f"{', '.join(invalid_hints)}"
            )


class EOSParser:
    """Parser for converting validated EOS specs to typed objects"""
    
    @staticmethod
    def parse_spec(spec_data: dict[str, Any]) -> EOSSpec:
        """Parse validated spec data into typed EOSSpec object"""
        
        # Parse meta
        meta_data = spec_data["meta"]
        meta = EOSMeta(
            run_id=meta_data["run_id"],
            owner=meta_data["owner"],
            provenance=meta_data.get("provenance", []),
            reproducibility=meta_data.get("reproducibility", {})
        )
        
        # Parse problem
        prob_data = spec_data["problem"]
        problem = EOSProblem(
            title=prob_data["title"],
            statement=prob_data["statement"],
            objectives=prob_data["objectives"],
            constraints=prob_data.get("constraints", []),
            stakeholders=prob_data.get("stakeholders", []),
            risk_tolerance=prob_data.get("risk_tolerance", {})
        )
        
        # Parse context
        ctx_data = spec_data["context"]
        cynefin_data = ctx_data["cynefin_prior"]
        cynefin_prior = CynefinPrior(
            simple=cynefin_data["simple"],
            complicated=cynefin_data["complicated"],
            complex=cynefin_data["complex"],
            chaotic=cynefin_data["chaotic"]
        )
        
        context = EOSContext(
            cynefin_prior=cynefin_prior,
            domain_hints=ctx_data.get("domain_hints", []),
            uncertainty_thresholds=ctx_data.get("uncertainty_thresholds", {})
        )
        
        # Parse methods registry
        methods = []
        for method_data in spec_data["methods_registry"]:
            methods.append(EOSMethod(
                id=method_data["id"],
                states=method_data["states"],
                entry_criteria=method_data["entry_criteria"],
                operators=method_data["operators"],
                artifacts_out=method_data["artifacts_out"]
            ))
        
        # Parse experts
        experts = []
        for expert_data in spec_data["experts"]:
            experts.append(EOSExpert(
                id=expert_data["id"],
                kind=expert_data["kind"],
                inputs=expert_data["inputs"],
                outputs=expert_data["outputs"],
                cost=expert_data["cost"],
                risk=expert_data["risk"],
                quality_prior=expert_data["quality_prior"],
                fit_hints=expert_data.get("fit_hints", [])
            ))
        
        return EOSSpec(
            eos_version=spec_data["eos_version"],
            meta=meta,
            problem=problem,
            context=context,
            resources=spec_data["resources"],
            methods_registry=methods,
            ladder=spec_data["ladder"],
            experts=experts,
            routing=spec_data["routing"],
            pipeline=spec_data["pipeline"],
            evaluation=spec_data["evaluation"],
            governance=spec_data["governance"],
            telemetry=spec_data["telemetry"]
        )


class EOSSchema:
    """Main entry point for EOS schema operations"""
    
    def __init__(self, schema_path: Path | None = None):
        self.validator = EOSValidator(schema_path)
        self.parser = EOSParser()
    
    def load_and_validate(self, yaml_path: Path) -> tuple[ValidationResult,
                                                          EOSSpec | None]:
        """Load, validate, and parse EOS specification"""
        # Validate first
        validation_result = self.validator.validate_yaml(yaml_path)
        
        if not validation_result.valid:
            return validation_result, None
        
        # Parse if validation successful
        try:
            with open(yaml_path) as f:
                spec_data = yaml.safe_load(f)
            
            spec = self.parser.parse_spec(spec_data)
            return validation_result, spec
            
        except Exception as e:
            validation_result.valid = False
            validation_result.errors.append(f"Parsing error: {e}")
            return validation_result, None
    
    def validate_only(self, yaml_path: Path) -> ValidationResult:
        """Validate EOS specification without parsing"""
        return self.validator.validate_yaml(yaml_path)


# CLI utilities for EOS validation
def main():
    """CLI entry point for EOS validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate E-UPUSF Orchestration Schema specifications"
    )
    parser.add_argument("spec_file", help="EOS YAML specification file")
    parser.add_argument("--schema", help="Custom JSON schema file")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON format")
    
    args = parser.parse_args()
    
    # Initialize validator
    schema_path = Path(args.schema) if args.schema else None
    eos_schema = EOSSchema(schema_path)
    
    # Validate specification
    spec_path = Path(args.spec_file)
    result = eos_schema.validate_only(spec_path)
    
    # Output results
    if args.json:
        output = {
            "valid": result.valid,
            "schema_version": result.schema_version,
            "errors": result.errors,
            "warnings": result.warnings
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"EOS Validation Results for {spec_path}")
        print(f"Schema Version: {result.schema_version}")
        print(f"Valid: {'✅ PASS' if result.valid else '❌ FAIL'}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
    
    # Exit with appropriate code
    exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()