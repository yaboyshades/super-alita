#!/usr/bin/env python3
"""
Constitutional gRPC Service Generator

Automates creation of constitutional gRPC services following all 14
articles of the Super-Alita Architectural Constitution.

Usage:
  python generate_service.py --service-name MyService --proto-file service.proto --output-dir ./out
  python generate_service.py --config development_kits/constitutional_grpc_v1/examples/service_config.yaml --proto-file development_kits/constitutional_grpc_v1/examples/example_service.proto --output-dir ./out

Constitutional Compliance:
- Article I: Library-First (reuses established templates and patterns)
- Article II: Test-First (generates comprehensive test suite)
- Article III: Simplicity Gate (keeps functions small, focused)
- Article V: Clarity (explicit configuration and docs)
- Article VIII: Automation of Expertise (this script embodies automation)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

# Add project root to path for imports (for compliance checker reuse)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclasses.dataclass
class ServiceMethod:
    name: str
    request_type: str
    response_type: str
    description: str
    implementation_type: str


class ConstitutionalServiceGenerator:
    """
    Generator for constitutional gRPC services.

    Implements Article VIII (Automation of Expertise) by automating the
    creation of constitutionally-compliant gRPC services.
    """

    def __init__(self, kit_path: Path):
        self.kit_path = kit_path
        self.templates_path = kit_path / "templates"
        self.examples_path = kit_path / "examples"

        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Constitutional validation thresholds (defaults)
        self.constitutional_thresholds = {
            "max_function_lines": 50,
            "min_test_coverage": 80,
            "max_complexity": 10,
        }

        logger.info("Constitutional gRPC Service Generator initialized")

    def generate_service(
        self,
        service_name: str,
        proto_file: Path,
        output_dir: Path,
        config: dict[str, Any] | None = None,
        unified_integration: bool | None = None,
        constitutional_threshold: float | None = None,
    ) -> bool:
        """Generate a complete constitutional gRPC service."""
        try:
            logger.info(f"🚀 Generating constitutional gRPC service: {service_name}")

            if not self._validate_inputs(service_name, proto_file, output_dir):
                return False

            # Parse protobuf definition
            proto_info = self._parse_proto_file(proto_file)
            if proto_info is None:
                return False

            # Create merged service configuration
            service_config = self._create_service_config(
                service_name,
                proto_info,
                config_override=config,
                unified_integration=unified_integration,
                constitutional_threshold=constitutional_threshold,
                proto_file=proto_file,
            )

            # Render service files
            self._ensure_dir(output_dir)
            if not self._generate_service_files(service_config, output_dir):
                return False

            # Render tests
            if not self._generate_test_files(service_config, output_dir):
                return False

            # Render documentation
            if not self._generate_documentation(service_config, output_dir):
                return False

            # Validate constitutional compliance (best-effort)
            try:
                if not self._validate_constitutional_compliance(output_dir):
                    logger.warning("Generated service has constitutional violations")
            except Exception as e:  # Best-effort; do not fail generation
                logger.debug(f"Compliance check skipped due to: {e}")

            logger.info("🎉 Constitutional gRPC service generated successfully")
            logger.info(f"📂 Output directory: {output_dir}")
            self._print_next_steps(service_name, output_dir)
            return True

        except Exception as e:
            logger.error(f"Service generation failed: {e}")
            return False

    def _validate_inputs(
        self, service_name: str, proto_file: Path, output_dir: Path
    ) -> bool:
        # Service name sanity
        if not service_name or not service_name.replace("_", "").isidentifier():
            logger.error(f"❌ Invalid service name: {service_name}")
            return False

        # Proto
        if not proto_file.exists():
            logger.error(f"❌ Proto file not found: {proto_file}")
            return False
        if proto_file.suffix != ".proto":
            logger.error(f"❌ Proto file must have .proto extension: {proto_file}")
            return False

        # Output dir should be empty or non-existent
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.error(f"❌ Output directory not empty: {output_dir}")
            return False
        return True

    def _parse_proto_file(self, proto_file: Path) -> dict[str, Any] | None:
        """Parse .proto to extract package, services and methods (simple regex)."""
        try:
            content = proto_file.read_text(encoding="utf-8")

            proto_info: dict[str, Any] = {
                "package": "",
                "services": [],
            }

            pkg_match = re.search(r"^\s*package\s+([\w\.]+)\s*;", content, re.MULTILINE)
            if pkg_match:
                proto_info["package"] = pkg_match.group(1)

            service_iter = re.finditer(r"service\s+(\w+)\s*\{([\s\S]*?)\}", content)
            for s in service_iter:
                svc_name = s.group(1)
                body = s.group(2)
                methods: list[ServiceMethod] = []
                for m in re.finditer(
                    r"rpc\s+(\w+)\s*\(\s*([\w\.]+)\s*\)\s*returns\s*\(\s*([\w\.]+)\s*\)",
                    body,
                ):
                    method_name, req_t, resp_t = m.group(1), m.group(2), m.group(3)
                    methods.append(
                        ServiceMethod(
                            name=method_name,
                            request_type=req_t.split(".")[-1],
                            response_type=resp_t.split(".")[-1],
                            description=f"Handle {method_name} requests",
                            implementation_type=self._infer_implementation_type(
                                method_name
                            ),
                        )
                    )
                proto_info["services"].append({"name": svc_name, "methods": methods})

            logger.info(
                f"✅ Parsed proto file: {len(proto_info['services'])} services found"
            )
            return proto_info
        except Exception as e:
            logger.error(f"❌ Failed to parse proto file: {e}")
            return None

    def _infer_implementation_type(self, method_name: str) -> str:
        method_lower = method_name.lower()
        if "health" in method_lower:
            return "health_check"
        if "status" in method_lower:
            return "status_check"
        if "validate" in method_lower:
            return "validation"
        if "process" in method_lower or "execute" in method_lower:
            return "unified_processing"
        return "custom"

    def _create_service_config(
        self,
        service_name: str,
        proto_info: dict[str, Any],
        config_override: dict[str, Any] | None,
        *,
        unified_integration: bool | None,
        constitutional_threshold: float | None,
        proto_file: Path,
    ) -> dict[str, Any]:
        # Select first service as primary
        main_service = proto_info["services"][0] if proto_info.get("services") else None

        # Derive sensible defaults
        service_class = (
            main_service["name"] if isinstance(main_service, dict) else service_name
        )
        service_snake = snake_case(service_class)
        servicer_module = f"{service_snake}_servicer"
        proto_module_default = proto_file.stem

        # Extract overrides from provided config
        cfg = config_override or {}
        cfg_service = cfg.get("service", {}) if isinstance(cfg, dict) else {}
        cfg_server = cfg.get("server", {}) if isinstance(cfg, dict) else {}
        cfg_const = cfg.get("constitutional", {}) if isinstance(cfg, dict) else {}
        cfg_methods = cfg.get("methods", []) if isinstance(cfg, dict) else []
        cfg_integration = cfg.get("integration", {}) if isinstance(cfg, dict) else {}

        # Merge method definitions: prefer proto, fall back to config
        methods = (
            main_service["methods"]
            if main_service and main_service.get("methods")
            else []
        )
        if not methods and cfg_methods:
            methods = [
                ServiceMethod(
                    name=m.get("name", "Method"),
                    request_type=m.get("request_type", "Request"),
                    response_type=m.get("response_type", "Response"),
                    description=m.get("description", ""),
                    implementation_type=m.get("implementation_type", "custom"),
                )
                for m in cfg_methods
            ]

        # Final configuration payload consumed by templates
        unified_flag = (
            unified_integration
            if unified_integration is not None
            else bool(cfg_integration.get("unified_integration", False))
        )

        threshold_val = (
            constitutional_threshold
            if constitutional_threshold is not None
            else float(cfg_const.get("compliance_threshold", 0.75))
        )

        service_config: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "service_name": service_name,
            "service_class": service_class,
            "service_snake": service_snake,
            "servicer_module": servicer_module,
            "proto_module": cfg_service.get("proto_module", proto_module_default),
            "package": proto_info.get("package", ""),
            "proto_package": cfg_service.get("proto_package", proto_info.get("package", "")),
            "default_port": int(cfg_server.get("default_port", 50051)),
            "max_workers": int(cfg_server.get("max_workers", 10)),
            "unified_integration": unified_flag,
            "constitutional_threshold": threshold_val,
            "service_methods": [dataclasses.asdict(m) for m in methods],
        }

        logger.debug(
            f"Merged service configuration: {json.dumps(service_config, indent=2)}"
        )
        return service_config

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _render_to(
        self, template_name: str, context: dict[str, Any], dest: Path
    ) -> None:
        template = self.jinja_env.get_template(template_name)
        output = template.render(**context)
        dest.write_text(output, encoding="utf-8")
        logger.info(f"✅ Generated {dest}")

    def _generate_service_files(self, cfg: dict[str, Any], out_dir: Path) -> bool:
        try:
            # Package init
            (out_dir / "__init__.py").write_text("", encoding="utf-8")

            # Servicer
            servicer_file = out_dir / f"{cfg['servicer_module']}.py"
            self._render_to("servicer_template.py.j2", cfg, servicer_file)

            # Server
            server_file = out_dir / "server.py"
            self._render_to("server_template.py.j2", cfg, server_file)

            # Middleware (service-local copy for portability)
            middleware_file = out_dir / "constitutional_middleware.py"
            self._render_to("middleware_template.py.j2", cfg, middleware_file)
            return True
        except Exception as e:
            logger.error(f"Failed to generate service files: {e}")
            return False

    def _generate_test_files(self, cfg: dict[str, Any], out_dir: Path) -> bool:
        try:
            test_filename = f"test_{cfg['servicer_module']}.py"
            test_file = out_dir / test_filename
            self._render_to("test_template.py.j2", cfg, test_file)
            return True
        except Exception as e:
            logger.error(f"Failed to generate test files: {e}")
            return False

    def _generate_documentation(self, cfg: dict[str, Any], out_dir: Path) -> bool:
        try:
            readme_file = out_dir / "README.md"
            self._render_to("README_template.md", cfg, readme_file)
            return True
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            return False

    def _validate_constitutional_compliance(self, out_dir: Path) -> bool:
        """Run the included compliance checker (best-effort)."""
        try:
            # Lazy import to avoid hard dependency when executed standalone
            from development_kits.constitutional_grpc_v1.validate_compliance import (
                ConstitutionalComplianceChecker,
            )

            checker = ConstitutionalComplianceChecker(out_dir)
            report = checker.analyze_service()
            logger.info(
                f"🏛️ Constitutional compliance — overall: {report.overall_score:.2f} "
                f"status: {'COMPLIANT' if report.is_compliant else 'NON-COMPLIANT'}"
            )
            return True
        except Exception as e:
            logger.debug(f"Compliance validation unavailable: {e}")
            return False

    def _print_next_steps(self, service_name: str, out_dir: Path) -> None:
        logger.info(
            "\nNext steps:\n"
            f"  1) Compile protobufs (grpc_tools.protoc)\n"
            f"  2) Review {out_dir / 'README.md'} for usage\n"
            f"  3) Run tests: pytest -q {out_dir}\n"
            f"  4) Start server: python {out_dir / 'server.py'}\n"
        )


def _load_yaml_config(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Constitutional gRPC service generator")
    p.add_argument("--service-name", required=True, help="Service display name")
    p.add_argument("--proto-file", required=True, type=Path, help="Path to .proto file")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    p.add_argument(
        "--config", type=Path, help="Optional YAML config to override defaults"
    )
    p.add_argument(
        "--unified-integration",
        action="store_true",
        help="Enable unified integration hooks",
    )
    p.add_argument(
        "--constitutional-threshold",
        type=float,
        default=None,
        help="Override constitutional compliance threshold (0.0-1.0)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    kit_path = Path(__file__).resolve().parent
    gen = ConstitutionalServiceGenerator(kit_path)

    cfg = _load_yaml_config(args.config)

    ok = gen.generate_service(
        service_name=args.service_name,
        proto_file=args.proto_file,
        output_dir=args.output_dir,
        config=cfg,
        unified_integration=bool(args.unified_integration),
        constitutional_threshold=args.constitutional_threshold,
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
