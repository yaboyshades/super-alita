"""Telemetry helpers for unified intelligence responses."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _coerce_to_dict(advice: Any) -> dict[str, Any]:
    """Best-effort conversion of advice objects into dictionaries."""
    if advice is None:
        return {}
    if isinstance(advice, dict):
        return advice
    if hasattr(advice, "dict") and callable(advice.dict):
        return advice.dict()  # type: ignore[no-any-return]
    if hasattr(advice, "to_dict") and callable(advice.to_dict):
        return advice.to_dict()  # type: ignore[no-any-return]
    if hasattr(advice, "__dict__"):
        return dict(advice.__dict__)
    return {}


def _format_float(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:.3f}"


class TelemetryHeaders:
    """Generate and validate telemetry headers for unified intelligence."""

    DECISION_HEADER = "X-UI-Decision"
    FUSED_SCORE_HEADER = "X-UI-Fused-Score"
    WORKFLOW_HEADER = "X-UI-Workflow"
    CONSTITUTION_HEADER = "X-UI-Constitution"
    MANGLE_CONF_HEADER = "X-UI-Mangle-Conf"
    CODE_ANALYSIS_CONF_HEADER = "X-UI-Code-Analysis-Conf"
    CODE_ANALYSIS_FINDINGS_HEADER = "X-UI-Code-Analysis-Findings"

    @classmethod
    def from_advice(cls, advice: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {}
        try:
            decision = advice.get("decision")
            if isinstance(decision, str):
                headers[cls.DECISION_HEADER] = decision

            fused = advice.get("scores", {}).get("fused")
            formatted = _format_float(fused)
            if formatted is not None:
                headers[cls.FUSED_SCORE_HEADER] = formatted

            workflow_label = advice.get("workflow_result", {}).get("label")
            if not workflow_label:
                workflow_label = advice.get("telemetry", {}).get("workflow_label")
            if workflow_label:
                headers[cls.WORKFLOW_HEADER] = str(workflow_label)

            constitution = advice.get("constitutional_compliance", {}).get("overall_score")
            formatted = _format_float(constitution)
            if formatted is not None:
                headers[cls.CONSTITUTION_HEADER] = formatted

            mangle_conf = advice.get("mangle_insights", {}).get("confidence")
            formatted = _format_float(mangle_conf)
            if formatted is not None:
                headers[cls.MANGLE_CONF_HEADER] = formatted

            contributors = advice.get("scores", {}).get("contributors", {})
            formatted = _format_float(contributors.get("code_analysis"))
            if formatted is not None:
                headers[cls.CODE_ANALYSIS_CONF_HEADER] = formatted

            findings = advice.get("telemetry", {}).get("code_analysis_findings")
            if findings is not None:
                headers[cls.CODE_ANALYSIS_FINDINGS_HEADER] = str(findings)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to generate telemetry headers: %s", exc)
            headers.setdefault(cls.DECISION_HEADER, "error")
        return headers

    @classmethod
    def validate_headers(cls, headers: dict[str, str]) -> dict[str, Any]:
        result = {"valid": True, "issues": [], "warnings": [], "suggestions": []}
        required = [cls.DECISION_HEADER, cls.FUSED_SCORE_HEADER]
        for name in required:
            if name not in headers:
                result["issues"].append(f"Missing required header: {name}")
                result["valid"] = False

        decision = headers.get(cls.DECISION_HEADER)
        if decision and decision not in {"proceed", "revise", "block", "error"}:
            result["issues"].append(
                f"Invalid decision value: {decision}. Expected one of proceed/revise/block/error"
            )
            result["valid"] = False

        for score_header in (cls.FUSED_SCORE_HEADER, cls.CONSTITUTION_HEADER, cls.MANGLE_CONF_HEADER, cls.CODE_ANALYSIS_CONF_HEADER):
            if score_header in headers:
                try:
                    value = float(headers[score_header])
                except ValueError:
                    result["issues"].append(f"Invalid numeric header: {score_header}")
                    result["valid"] = False
                    continue
                if not 0.0 <= value <= 1.0:
                    result["warnings"].append(
                        f"Header {score_header} value {value} is outside the [0, 1] range"
                    )
        return result


@dataclass
class TelemetryEnvelope:
    request_id: str
    response: Any
    telemetry: dict[str, Any]

    @classmethod
    def wrap_response(cls, body: Any, request_id: str) -> dict[str, Any]:
        payload: dict[str, Any]
        telemetry: dict[str, Any]

        if isinstance(body, dict):
            payload = dict(body)
            telemetry = dict(payload.get("telemetry", {}))
        elif hasattr(body, "dict") and callable(body.dict):  # type: ignore[attr-defined]
            payload = body.dict()  # type: ignore[assignment]
            telemetry = dict(payload.get("telemetry", {}))
        else:
            payload = {"data": body}
            telemetry = {}

        telemetry.setdefault("request_id", request_id)
        telemetry.setdefault("processing_complete", True)

        envelope = cls(request_id=request_id, response=payload, telemetry=telemetry)
        return asdict(envelope)


class TelemetryMiddleware:
    """Adds telemetry headers and envelopes to HTTP responses."""

    def __init__(self, unified_engine: Any | None = None):
        self.unified_engine = unified_engine
        self.request_count = 0
        self.header_errors = 0

    def generate_headers(self, advice: Any) -> dict[str, str]:
        advice_dict = _coerce_to_dict(advice)
        return TelemetryHeaders.from_advice(advice_dict)

    async def process_response(self, response: Any, advice: Any | None = None) -> Any:
        if advice is None:
            return response

        advice_dict = _coerce_to_dict(advice)

        try:
            self.request_count += 1

            headers = TelemetryHeaders.from_advice(advice_dict)
            for name, value in headers.items():
                try:
                    response.headers[name] = value
                except Exception:  # pragma: no cover - response may lack headers
                    logger.debug("Response object has no headers attribute")

            validation = TelemetryHeaders.validate_headers(headers)
            if not validation["valid"]:
                self.header_errors += 1
                logger.warning("Telemetry header validation failed: %s", validation["issues"])

            if getattr(response, "media_type", None) == "application/json" and hasattr(response, "body"):
                body = response.body
                if isinstance(body, (bytes, bytearray)):
                    try:
                        payload = json.loads(body.decode())
                    except Exception:  # pragma: no cover - defensive parsing
                        payload = body.decode(errors="ignore")
                else:
                    payload = body

                request_id = advice_dict.get("telemetry", {}).get("request_id", "unknown")
                envelope = TelemetryEnvelope.wrap_response(payload, request_id)
                encoded = json.dumps(envelope).encode()
                response.body = encoded
                if hasattr(response, "headers"):
                    response.headers["Content-Length"] = str(len(encoded))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Telemetry processing failed: %s", exc)

        return response

    def get_stats(self) -> dict[str, Any]:
        error_rate = 0.0
        if self.request_count:
            error_rate = self.header_errors / self.request_count
        return {
            "requests_processed": self.request_count,
            "header_errors": self.header_errors,
            "error_rate": error_rate,
        }


def add_telemetry_headers(response: Any, advice: Any) -> None:
    headers = TelemetryHeaders.from_advice(_coerce_to_dict(advice))
    for name, value in headers.items():
        try:
            response.headers[name] = value
        except Exception:  # pragma: no cover
            logger.debug("Unable to set header %s", name)


def create_telemetry_envelope(advice: Any, request_id: str) -> dict[str, Any]:
    return TelemetryEnvelope.wrap_response(_coerce_to_dict(advice), request_id)
