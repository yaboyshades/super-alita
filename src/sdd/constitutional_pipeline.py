"""Constitutional SDD Pipeline.

Implements the Specification-Driven Development workflow with integrated
constitutional validation at each stage: specify, plan, and tasks.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..constitutional import ConstitutionalScorer
from ..core.yaml_utils import safe_dump, safe_load
from .models import (
    ConstitutionalAlignment,
    ConstitutionalValidation,
    NextStepGuidance,
    NextStepItem,
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TaskBreakdown,
    TasksRequest,
    TasksResponse,
)


class ConstitutionalSDDPipeline:
    """SDD pipeline with constitutional validation at each gate."""

    def __init__(self, workspace_root: Path | None = None):
        """Initialize the constitutional SDD pipeline."""
        self.workspace_root = workspace_root or Path.cwd()
        self.constitutional_scorer = ConstitutionalScorer()
        self.specs_dir = self.workspace_root / "specs"
        self.specs_dir.mkdir(exist_ok=True)

        # Constitutional compliance threshold
        self.compliance_threshold = 0.75

    async def specify(self, request: SpecifyRequest) -> SpecifyResponse:
        """Execute the /specify phase with constitutional validation."""
        branch_name = request.branch_name
        feature_dir: Path
        spec_file: Path
        feature_id: str
        feature_name: str

        if request.spec_file:
            spec_file = Path(request.spec_file)
            feature_dir = spec_file.parent
            feature_dir.mkdir(parents=True, exist_ok=True)
        elif request.feature_dir:
            feature_dir = Path(request.feature_dir)
            feature_dir.mkdir(parents=True, exist_ok=True)
            spec_file = feature_dir / "spec.md"
        else:
            feature_id = self._generate_feature_id(request.user_input)
            feature_name = self._slugify(request.user_input)
            feature_dir = self.specs_dir / f"{feature_id}-{feature_name}"
            feature_dir.mkdir(parents=True, exist_ok=True)
            spec_file = feature_dir / "spec.md"
            branch_name = branch_name or f"{feature_id}-{feature_name}"

        if (
            not request.spec_file
            and request.feature_dir
            and not spec_file.exists()
        ):
            spec_file.touch()

        if request.spec_file is None and request.feature_dir is None:
            feature_id_val = (
                feature_dir.name[:3]
                if feature_dir.name[:3].isdigit()
                else None
            )
            feature_id = feature_id_val or self._generate_feature_id(
                request.user_input
            )
            feature_name = (
                feature_dir.name.split("-", 1)[1]
                if "-" in feature_dir.name
                else self._slugify(request.user_input)[:50]
            )
        else:
            if branch_name and branch_name[:3].isdigit():
                feature_id = branch_name[:3]
            else:
                feature_id = self._generate_feature_id(request.user_input)
            if branch_name and "-" in branch_name:
                feature_name = branch_name.split("-", 1)[1]
            else:
                feature_name = (
                    feature_dir.name.split("-", 1)[1]
                    if "-" in feature_dir.name
                    else self._slugify(request.user_input)[:50]
                )
                branch_name = branch_name or f"{feature_id}-{feature_name}"

        guidance = self._collect_next_step_guidance(
            feature_id=feature_id,
            feature_dir=feature_dir,
            spec_file=spec_file,
            context=request.context,
            user_input=request.user_input,
        )
        specification = self._generate_specification(
            user_input=request.user_input,
            context=request.context,
            feature_id=feature_id,
            guidance=guidance,
        )

        spec_file.write_text(specification, encoding="utf-8")
        metadata_path = feature_dir / "next_steps.yaml"
        metadata_path.write_text(
            safe_dump(guidance.model_dump()), encoding="utf-8"
        )

        constitutional_compliance: dict[str, ConstitutionalValidation] = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_specification(
                specification
            )
            overall_score = self._calculate_overall_score(
                constitutional_compliance
            )
            threshold_met = overall_score >= self.compliance_threshold

        return SpecifyResponse(
            success=True,
            specification=specification,
            feature_id=feature_id,
            feature_path=str(spec_file),
            branch_name=branch_name,
            feature_name=feature_name,
            spec_file_path=str(spec_file),
            feature_dir=str(feature_dir),
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            next_steps=self._summarize_next_step_guidance(
                guidance, threshold_met
            ),
            next_step_guidance=guidance,
            next_step_metadata_path=self._relative_to_workspace(metadata_path),
            timestamp=datetime.now(),
        )

    async def plan(self, request: PlanRequest) -> PlanResponse:
        """Execute the /plan phase with constitutional validation."""
        # Read specification
        if request.specification_path is None:
            # Should be materialized by the enhanced framework; guard just in case
            raise ValueError("specification_path is required for planning")
        spec_path = Path(request.specification_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification not found: {spec_path}")

        specification = spec_path.read_text(encoding="utf-8")
        feature_dir = spec_path.parent
        feature_id = (
            request.feature_id
            if getattr(request, "feature_id", None)
            else self._derive_feature_id_from_path(str(spec_path))
        )
        raw_guidance = self._load_next_step_guidance(feature_dir, feature_id)
        metadata_rel = None
        if raw_guidance:
            raw_guidance = self._advance_guidance_for_plan(raw_guidance)
            metadata_rel = self._persist_guidance(feature_dir, raw_guidance)
        else:
            metadata_path = feature_dir / "next_steps.yaml"
            if metadata_path.exists():
                metadata_rel = self._relative_to_workspace(metadata_path)

        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(
            specification, request.technology_stack, request.constraints
        )

        # Write plan file
        plan_file = feature_dir / "implementation-plan.md"
        plan_file.write_text(implementation_plan, encoding="utf-8")

        # Generate supporting documents
        supporting_docs = self._generate_supporting_documents(
            feature_dir, specification, request.technology_stack
        )

        # Constitutional validation if requested
        constitutional_compliance: dict[str, ConstitutionalValidation] = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_implementation_plan(
                implementation_plan
            )
            overall_score = self._calculate_overall_score(
                constitutional_compliance
            )
            threshold_met = overall_score >= self.compliance_threshold

        guidance_summary: list[str] = []
        if raw_guidance:
            guidance_summary = self._summarize_next_step_guidance(
                raw_guidance, threshold_met
            )
        plan_next_steps = self._merge_next_step_summaries(
            guidance_summary, self._get_plan_next_steps(threshold_met)
        )

        return PlanResponse(
            success=True,
            feature_id=feature_id,
            implementation_plan=implementation_plan,
            plan=implementation_plan,
            plan_path=str(plan_file),
            supporting_documents=supporting_docs,
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            technology_recommendations=self._get_tech_recommendations(
                request.technology_stack
            ),
            architecture_decisions=self._extract_architecture_decisions(
                implementation_plan
            ),
            next_steps=plan_next_steps,
            next_step_guidance=raw_guidance,
            next_step_metadata_path=metadata_rel,
            timestamp=datetime.now(),
        )

    async def tasks(self, request: TasksRequest) -> TasksResponse:
        """Execute the /tasks phase with constitutional validation."""
        # Read or materialize implementation plan
        plan_path = None
        if getattr(request, "plan_path", None):
            plan_path = Path(request.plan_path)  # type: ignore[arg-type]
        elif getattr(request, "plan", None):
            # Materialize raw plan text
            feature_id = getattr(request, "feature_id", None) or "inline"
            feature_dir = self.specs_dir / f"{feature_id}-inline-plan"
            feature_dir.mkdir(parents=True, exist_ok=True)
            plan_file_tmp = feature_dir / "implementation-plan.md"
            plan_file_tmp.write_text(request.plan or "", encoding="utf-8")
            plan_path = plan_file_tmp
        else:
            raise FileNotFoundError(
                "No plan provided: supply plan_path or raw plan content in 'plan'"
            )

        assert plan_path is not None
        if not plan_path.exists():
            raise FileNotFoundError(
                f"Implementation plan not found: {plan_path}"
            )

        implementation_plan = plan_path.read_text(encoding="utf-8")
        feature_dir = plan_path.parent
        feature_id = (
            request.feature_id
            if getattr(request, "feature_id", None)
            else self._derive_feature_id_from_path(str(plan_path))
        )
        raw_guidance = self._load_next_step_guidance(feature_dir, feature_id)
        metadata_rel = None
        if raw_guidance:
            raw_guidance = self._advance_guidance_for_tasks(raw_guidance)
            metadata_rel = self._persist_guidance(feature_dir, raw_guidance)
        else:
            metadata_path = feature_dir / "next_steps.yaml"
            if metadata_path.exists():
                metadata_rel = self._relative_to_workspace(metadata_path)

        # Generate task breakdown
        tasks_breakdown = self._generate_task_breakdown(
            implementation_plan, request.priority_focus, request.team_size
        )

        # Parse structured tasks
        structured_tasks = self._parse_structured_tasks(tasks_breakdown)

        guidance_tasks: list[TaskBreakdown] = []
        if raw_guidance:
            guidance_tasks = self._convert_guidance_to_tasks(
                raw_guidance, [task.id for task in structured_tasks]
            )
            if guidance_tasks:
                structured_tasks.extend(guidance_tasks)
                tasks_breakdown = self._append_guidance_markdown(
                    tasks_breakdown, guidance_tasks
                )

        # Write tasks file (after appending guidance follow-ups if any)
        tasks_file = feature_dir / "tasks.md"
        tasks_file.write_text(tasks_breakdown, encoding="utf-8")

        # Constitutional validation if requested
        constitutional_compliance: dict[str, ConstitutionalValidation] = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_task_breakdown(
                tasks_breakdown
            )
            overall_score = self._calculate_overall_score(
                constitutional_compliance
            )
            threshold_met = overall_score >= self.compliance_threshold

        # Calculate estimates and critical path
        total_hours = sum(task.estimated_hours for task in structured_tasks)
        critical_path = self._calculate_critical_path(structured_tasks)

        guidance_summary: list[str] = []
        if raw_guidance:
            guidance_summary = self._summarize_next_step_guidance(
                raw_guidance, threshold_met
            )
        task_next_steps = self._merge_next_step_summaries(
            guidance_summary, self._get_tasks_next_steps(threshold_met)
        )

        return TasksResponse(
            success=True,
            feature_id=feature_id,
            tasks_breakdown=tasks_breakdown,
            tasks_path=str(tasks_file),
            tasks=structured_tasks,
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            estimated_total_hours=total_hours,
            critical_path=critical_path,
            next_steps=task_next_steps,
            next_step_guidance=raw_guidance,
            next_step_metadata_path=metadata_rel,
            timestamp=datetime.now(),
        )

    def _derive_feature_id_from_path(self, path_str: str) -> str:
        """Derive a feature identifier from a spec/plan/tasks path."""
        try:
            resolved = Path(path_str).resolve()
            parent_name = resolved.parent.name
            if len(parent_name) >= 3 and parent_name[:3].isdigit():
                return parent_name[:3]
            return (
                parent_name.split("-", 1)[0]
                if "-" in parent_name
                else parent_name
            )
        except Exception:  # noqa: BLE001
            return "unknown"

    def _generate_feature_id(self, _user_input: str) -> str:
        """Generate a unique feature ID."""
        # Get next sequential number
        existing_features = [
            d
            for d in self.specs_dir.iterdir()
            if d.is_dir() and d.name[:3].isdigit()
        ]
        next_num = len(existing_features) + 1
        return f"{next_num:03d}"

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""

        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")[:50]

    TASK_PRIORITY_BY_GATE = {
        "library_first": "high",
        "test_first": "critical",
        "simplicity": "high",
        "integration_first": "high",
        "clarity": "medium",
        "counterfactual": "medium",
    }

    GATE_TO_ARTICLE = {
        "library_first": "Article I - Library-First Development",
        "test_first": "Article II - Test-First Development",
        "simplicity": "Article III - Simplicity Gate",
        "integration_first": "Article IV - Integration-First Testing",
        "clarity": "Article V - Clarity and Unambiguity",
        "counterfactual": "Article VI - Counterfactual Justification",
    }

    def _collect_next_step_guidance(
        self,
        feature_id: str,
        feature_dir: Path,
        spec_file: Path,
        context: dict[str, Any],
        user_input: str,
    ) -> NextStepGuidance:
        """Derive structured next-step guidance for the specification."""
        default_owner = str(
            context.get("owner")
            or context.get("feature_owner")
            or context.get("primary_owner")
            or "unassigned"
        )

        clarifications = self._gather_clarifications(
            spec_file=spec_file,
            context=context,
            user_input=user_input,
            default_owner=default_owner,
        )
        artefacts = self._gather_research_artefacts(
            feature_dir=feature_dir,
            default_owner=default_owner,
        )
        commands = self._build_command_items(
            feature_id=feature_id,
            default_owner=default_owner,
        )
        metadata_reference = self._relative_to_workspace(
            feature_dir / "next_steps.yaml"
        )
        alignment = self._build_alignment(
            clarifications, artefacts, commands, metadata_reference
        )

        return NextStepGuidance(
            feature_id=feature_id,
            clarifications=clarifications,
            artefacts=artefacts,
            commands=commands,
            constitutional_alignment=alignment,
        )

    def _advance_guidance_for_plan(
        self, guidance: NextStepGuidance
    ) -> NextStepGuidance:
        """Mark clarifications complete and refresh command status after planning."""
        updated = guidance.model_copy(deep=True)
        for item in updated.clarifications:
            if item.status != "complete":
                item.status = "complete"
                if not item.rationale:
                    item.rationale = (
                        "Clarification resolved during planning phase."
                    )
        for item in updated.commands:
            action_lower = item.action.lower()
            if "/plan" in action_lower and item.status != "complete":
                item.status = "complete"
                if not item.rationale:
                    item.rationale = "Planning command executed."
            if "/tasks" in action_lower and item.status == "complete":
                item.status = "pending"
        updated.generated_at = datetime.now()
        return updated

    def _advance_guidance_for_tasks(
        self, guidance: NextStepGuidance
    ) -> NextStepGuidance:
        """Update guidance statuses when generating tasks."""
        updated = guidance.model_copy(deep=True)
        for item in updated.commands:
            if "/tasks" in item.action.lower() and item.status != "complete":
                item.status = "in_progress"
                if not item.rationale:
                    item.rationale = "Tasks are being generated."
        updated.generated_at = datetime.now()
        return updated

    def _persist_guidance(
        self, feature_dir: Path, guidance: NextStepGuidance
    ) -> str:
        """Write guidance back to disk and return workspace-relative path."""
        metadata_path = feature_dir / "next_steps.yaml"
        metadata_path.write_text(
            safe_dump(guidance.model_dump()), encoding="utf-8"
        )
        return self._relative_to_workspace(metadata_path)

    def _append_guidance_markdown(
        self,
        base_content: str,
        tasks: list[TaskBreakdown],
    ) -> str:
        """Append guidance-derived tasks to markdown representation."""
        lines = [base_content.rstrip(), "", "## Guidance Follow-ups"]
        for task in tasks:
            lines.append(f"### {task.title}")
            lines.append(f"- Priority: {task.priority}")
            if task.constitutional_requirements:
                lines.append(
                    f"- Constitutional Focus: {task.constitutional_requirements[0]}"
                )
            if task.description:
                lines.append(f"- Notes: {task.description}")
        return "\n".join(lines) + "\n\n"

    def _convert_guidance_to_tasks(
        self,
        guidance: NextStepGuidance,
        existing_ids: list[str],
    ) -> list[TaskBreakdown]:
        """Transform outstanding guidance items into supplemental task entries."""
        if not guidance:
            return []

        tasks: list[TaskBreakdown] = []
        counter = 0
        seen_ids = set(existing_ids)

        def next_identifier() -> str:
            nonlocal counter
            counter += 1
            candidate = f"NS-{counter:02d}"
            while candidate in seen_ids:
                counter += 1
                candidate = f"NS-{counter:02d}"
            seen_ids.add(candidate)
            return candidate

        def determine_priority(gate: str) -> str:
            return self.TASK_PRIORITY_BY_GATE.get(gate, "medium")

        actionable = guidance.artefacts + guidance.commands
        for item in actionable:
            if item.status == "complete":
                continue
            article = self.GATE_TO_ARTICLE.get(
                item.gate, item.gate.replace("_", " ").title()
            )
            description_parts: list[str] = []
            if item.rationale:
                description_parts.append(item.rationale)
            if item.linked_artifact:
                description_parts.append(
                    f"Linked artefact: {item.linked_artifact}"
                )
            description = (
                "\n".join(description_parts)
                or "Follow up on outstanding guidance item."
            )
            acceptance = (
                f"Evidence captured at {item.linked_artifact}"
                if item.linked_artifact
                else "Evidence documented"
            )
            tasks.append(
                TaskBreakdown(
                    id=next_identifier(),
                    title=item.action,
                    description=description,
                    priority=determine_priority(item.gate),
                    estimated_hours=2,
                    dependencies=[],
                    acceptance_criteria=[acceptance],
                    constitutional_requirements=[article],
                )
            )
        return tasks

    def _load_next_step_guidance(
        self,
        feature_dir: Path,
        feature_id: str,
    ) -> NextStepGuidance | None:
        """Load persisted next-step guidance if available."""
        metadata_path = feature_dir / "next_steps.yaml"
        if not metadata_path.exists():
            return None

        try:
            raw_data = safe_load(metadata_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

        if not raw_data:
            return None

        if "feature_id" not in raw_data:
            raw_data["feature_id"] = feature_id

        try:
            return NextStepGuidance(**raw_data)
        except Exception:  # noqa: BLE001
            return None

    def _gather_clarifications(
        self,
        spec_file: Path,
        context: dict[str, Any],
        user_input: str,
        default_owner: str,
    ) -> list[NextStepItem]:
        """Collect clarification items from context hints."""
        clarifications_input = context.get("clarifications") or []
        if isinstance(clarifications_input, str):
            clarifications_input = [clarifications_input]
        rel_spec = self._relative_to_workspace(spec_file)
        items: list[NextStepItem] = []
        for entry in clarifications_input:
            text = str(entry).strip()
            if not text:
                continue
            items.append(
                NextStepItem(
                    action=f"Resolve clarification: {text}",
                    owner=default_owner,
                    linked_artifact=f"{rel_spec}#clarifications",
                    gate="clarity",
                    status="pending",
                    rationale="Provided via /specify context clarifications",
                    source="clarification",
                )
            )
        return items

    def _gather_research_artefacts(
        self,
        feature_dir: Path,
        default_owner: str,
    ) -> list[NextStepItem]:
        """Derive artefact follow-ups from research notes."""
        research_path = feature_dir / "research.md"
        if not research_path.exists():
            return []

        tasks = self._parse_research_next_steps(research_path)
        items: list[NextStepItem] = []
        for task in tasks:
            gate = self._infer_gate_for_text(task)
            linked = self._infer_linked_artifact(task, feature_dir)
            items.append(
                NextStepItem(
                    action=task,
                    owner=default_owner,
                    linked_artifact=linked,
                    gate=gate,
                    status="pending",
                    rationale="Research next-step recommendation",
                    source="artefact",
                )
            )
        return items

    def _build_command_items(
        self,
        feature_id: str,
        default_owner: str,
    ) -> list[NextStepItem]:
        """Recommend workflow commands that keep SDD aligned."""
        items = [
            NextStepItem(
                action=f"Run /plan {feature_id} once clarifications are resolved",
                owner=default_owner or "feature-owner",
                linked_artifact=f"specs/{feature_id}/spec.md",
                gate="test_first",
                status="pending",
                rationale="Planning must wait until specification is unambiguous",
                source="command",
            ),
            NextStepItem(
                action=f"Invoke /tasks {feature_id} after the plan passes constitutional checks",
                owner=default_owner or "feature-owner",
                linked_artifact=f"specs/{feature_id}/spec.md",
                gate="integration_first",
                status="pending",
                rationale="Tasks phase depends on validated plan",
                source="command",
            ),
        ]
        return items

    def _parse_research_next_steps(self, research_path: Path) -> list[str]:
        text = research_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tasks: list[str] = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("## next steps"):
                in_section = True
                continue
            if in_section and stripped.startswith("## "):
                break
            if in_section and stripped.startswith("- "):
                tasks.append(stripped[2:].strip())
        return tasks

    def _infer_gate_for_text(self, text: str) -> str:
        lowered = text.lower()
        if any(
            keyword in lowered for keyword in ("contract", "test", "coverage")
        ):
            return "test_first"
        if any(
            keyword in lowered for keyword in ("data model", "model", "schema")
        ):
            return "simplicity"
        if any(
            keyword in lowered
            for keyword in ("documentation", "doc", "clarify")
        ):
            return "clarity"
        if any(
            keyword in lowered for keyword in ("integration", "environment")
        ):
            return "integration_first"
        if any(
            keyword in lowered for keyword in ("research", "reuse", "library")
        ):
            return "library_first"
        return "counterfactual"

    def _infer_linked_artifact(self, text: str, feature_dir: Path) -> str:
        rel_dir = self._relative_to_workspace(feature_dir)
        if "data-model" in text:
            return f"{rel_dir}/data-model.md"
        if "quickstart" in text:
            return f"{rel_dir}/quickstart.md"
        if "contract" in text:
            return f"{rel_dir}/contracts"
        return f"{rel_dir}/spec.md"

    def _build_alignment(
        self,
        clarifications: list[NextStepItem],
        artefacts: list[NextStepItem],
        commands: list[NextStepItem],
        metadata_reference: str,
    ) -> list[ConstitutionalAlignment]:
        items_by_gate: dict[str, list[NextStepItem]] = {}
        for item in clarifications + artefacts + commands:
            items_by_gate.setdefault(item.gate, []).append(item)

        alignments: list[ConstitutionalAlignment] = []
        for gate, gate_items in items_by_gate.items():
            article = self.GATE_TO_ARTICLE.get(
                gate, gate.replace("_", " ").title()
            )
            summary = (
                f"{len(gate_items)} item(s) pending to satisfy {article}."
            )
            evidence = f"{metadata_reference}#{gate}"
            alignments.append(
                ConstitutionalAlignment(
                    gate=gate, summary=summary, evidence=evidence
                )
            )
        return alignments

    def _relative_to_workspace(self, path: Path) -> str:
        try:
            return str(
                path.resolve().relative_to(self.workspace_root.resolve())
            )
        except ValueError:
            return str(path)

    def _safe_format_value(self, value: str) -> str:
        return str(value).replace("{", "{{").replace("}", "}}").strip()

    def _format_checkbox_list(
        self, items: list[NextStepItem], default_message: str
    ) -> str:
        if not items:
            return f"- [x] {self._safe_format_value(default_message)}"
        lines: list[str] = []
        for item in items:
            article = self.GATE_TO_ARTICLE.get(
                item.gate, item.gate.replace("_", " ").title()
            )
            owner = item.owner or "unassigned"
            lines.append(
                f"- [ ] Owner: {self._safe_format_value(owner)} — {self._safe_format_value(item.action)} (supports {article})"
            )
        return "\n".join(lines)

    def _format_alignment_list(
        self, items: list[ConstitutionalAlignment], default_message: str
    ) -> str:
        if not items:
            return f"- {self._safe_format_value(default_message)}"
        lines: list[str] = []
        for item in items:
            article = self.GATE_TO_ARTICLE.get(
                item.gate, item.gate.replace("_", " ").title()
            )
            evidence = item.evidence or "next_steps.yaml"
            lines.append(
                f"- {article}: {self._safe_format_value(item.summary)} (evidence: {self._safe_format_value(evidence)})"
            )
        return "\n".join(lines)

    def _format_command_list(
        self, items: list[NextStepItem], default_message: str
    ) -> str:
        if not items:
            return f"1. {self._safe_format_value(default_message)}"
        lines: list[str] = []
        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {self._safe_format_value(item.action)}")
        return "\n".join(lines)

    def _generate_specification(
        self,
        user_input: str,
        context: dict[str, Any],
        feature_id: str,
        guidance: NextStepGuidance,
    ) -> str:
        created_date = datetime.now().strftime("%Y-%m-%d")
        author = self._safe_format_value(
            context.get("author") or "SDD Framework"
        )
        problem_description = self._safe_format_value(
            context.get("problem_description") or user_input
        )
        additional_notes = self._safe_format_value(
            context.get("notes") or "None provided"
        )
        clarifications_text = self._format_checkbox_list(
            guidance.clarifications,
            "No outstanding clarifications. Ready for /plan.",
        )
        artefacts_text = self._format_checkbox_list(
            guidance.artefacts,
            "No supporting artefacts requested.",
        )
        alignment_text = self._format_alignment_list(
            guidance.constitutional_alignment,
            "Gate coverage will be generated once clarifications exist.",
        )
        commands_text = self._format_command_list(
            guidance.commands,
            "Specification is ready; run /plan when stakeholders approve.",
        )
        template = """# Feature Specification Template

**Feature ID:** {feature_id}
**Created:** {created_date}
**Status:** Draft
**Constitutional Compliance Score:** _To be calculated_

---

## Problem Statement

> **Focus on the WHAT and WHY, not the HOW**
> Be explicit about what you are trying to build and why.

{problem_description}

---

## User Stories

### Primary User Stories

> Each user story should follow the format: "As a [user type], I want [goal] so that [benefit]"

1. **User Story 1**
   - **As a** [user type]
   - **I want** [goal]
   - **So that** [benefit]

   **Acceptance Criteria:**
   - [ ] Criterion 1
   - [ ] Criterion 2
   - [ ] Criterion 3

2. **User Story 2**
   - **As a** [user type]
   - **I want** [goal]
   - **So that** [benefit]

   **Acceptance Criteria:**
   - [ ] Criterion 1
   - [ ] Criterion 2
   - [ ] Criterion 3

3. **User Story 3**
   - **As a** [user type]
   - **I want** [goal]
   - **So that** [benefit]

   **Acceptance Criteria:**
   - [ ] Criterion 1
   - [ ] Criterion 2
   - [ ] Criterion 3

### Secondary User Stories

> Additional user stories for edge cases or future considerations

---

## Functional Requirements

### Core Functionality

1. **Requirement 1:** [Description]
   - **Given** [context]
   - **When** [action]
   - **Then** [expected result]

2. **Requirement 2:** [Description]
   - **Given** [context]
   - **When** [action]
   - **Then** [expected result]

### Business Rules

1. [Business rule 1]
2. [Business rule 2]
3. [Business rule 3]

---

## Non-Functional Requirements

### Performance Requirements
- [Performance requirement 1]
- [Performance requirement 2]

### Security Requirements
- [Security requirement 1]
- [Security requirement 2]

### Usability Requirements
- [Usability requirement 1]
- [Usability requirement 2]

---

## Constraints & Assumptions

### Constraints
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

### Assumptions
- [Assumption 1]
- [Assumption 2]
- [Assumption 3]

---

## Success Criteria

### Definition of Done
- [ ] All user stories implemented and tested
- [ ] All acceptance criteria verified
- [ ] Performance requirements met
- [ ] Security requirements validated
- [ ] Constitutional compliance score >= 0.75

### Success Metrics
- [Metric 1]: [Target value]
- [Metric 2]: [Target value]
- [Metric 3]: [Target value]

---

## Review & Acceptance Checklist

> **Constitutional Framework Validation**

### Article I: Library-First Development
- [ ] Existing solutions researched and documented
- [ ] Decision to build vs. adopt existing libraries justified
- [ ] Library dependencies identified and evaluated

### Article II: Test-First Development
- [ ] Testable acceptance criteria defined
- [ ] Test strategy outlined
- [ ] Quality gates specified

### Article III: Simplicity Gate
- [ ] Scope is clearly bounded and minimal
- [ ] Complex requirements broken into simpler components
- [ ] Feature avoids unnecessary complexity

### Article IV: Integration-First Testing
- [ ] Integration points identified
- [ ] End-to-end testing scenarios defined
- [ ] Integration requirements specified

### Article V: Clarity and Unambiguity
- [ ] Requirements are clear and unambiguous
- [ ] Terms and concepts are well-defined
- [ ] No contradictory requirements

### Article VI: Counterfactual Justification
- [ ] Alternative approaches considered
- [ ] Decision rationale documented
- [ ] Trade-offs explicitly stated

### General Quality Checks
- [ ] User stories follow standard format
- [ ] Acceptance criteria are testable
- [ ] Requirements are prioritized
- [ ] Dependencies on other features identified
- [ ] Assumptions and constraints documented
- [ ] Success criteria defined and measurable

---

## Next Steps Guidance

> **Constitutional directive:** resolve outstanding clarifications and collect evidence before invoking /plan.
> Ensure every item ties back to the constitutional gates and keeps the specification scoped to the WHAT and WHY.

### Outstanding Clarifications
{clarifications_text}

### Required Artefacts & Evidence
{artefacts_text}

### Constitutional Gate Alignment
{alignment_text}

### Command Checklist
{commands_text}

---

## Additional Notes

{additional_notes}

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | {created_date} | {author} | Initial specification |

---
"""
        return template.format(
            feature_id=self._safe_format_value(feature_id),
            created_date=created_date,
            problem_description=problem_description,
            additional_notes=additional_notes,
            clarifications_text=clarifications_text,
            artefacts_text=artefacts_text,
            alignment_text=alignment_text,
            commands_text=commands_text,
            author=author,
        )

    def _merge_next_step_summaries(self, *collections: list[str]) -> list[str]:
        """Merge step summaries while preserving order and removing duplicates."""
        seen: set[str] = set()
        merged: list[str] = []
        for collection in collections:
            for item in collection:
                if item and item not in seen:
                    seen.add(item)
                    merged.append(item)
        return merged

    def _summarize_next_step_guidance(
        self, guidance: NextStepGuidance, threshold_met: bool
    ) -> list[str]:
        summary: list[str] = []
        if guidance.clarifications:
            summary.append(
                f"Resolve {len(guidance.clarifications)} outstanding clarification item(s)."
            )
        if guidance.artefacts:
            summary.append(
                f"Produce {len(guidance.artefacts)} supporting artefact(s) before planning."
            )
        summary.extend(item.action for item in guidance.commands[:2])
        if not summary:
            summary = self._get_specify_next_steps(threshold_met)
        return summary

    def _generate_implementation_plan(
        self,
        _specification: str,
        tech_stack: str,
        _constraints: dict[str, Any],
    ) -> str:
        """Generate implementation plan."""
        return f"""# Implementation Plan

## Architecture Overview
Based on the specification, this implementation follows constitutional principles.

## Technology Stack
{tech_stack or 'To be determined based on requirements'}

## Project Structure
Following the Simplicity Gate (<=3 projects):
1. Core Library
2. API/CLI Interface
3. Tests and Documentation

## Implementation Phases

### Phase 1: Foundation (Test-First)
- [ ] Set up test infrastructure
- [ ] Implement core data models
- [ ] Create basic API structure

### Phase 2: Core Features
- [ ] Implement main functionality
- [ ] Add CLI interface (Constitutional requirement)
- [ ] Integration testing with real environments

### Phase 3: Polish & Documentation
- [ ] Performance optimization
- [ ] Complete documentation
- [ ] Final constitutional review

## Constitutional Compliance Strategy
- **Library-First**: Research existing solutions before implementation
- **Test-First**: 80% minimum test coverage, TDD workflow
- **Simplicity**: Keep cyclomatic complexity < 10
- **Integration-First**: Test with real data and environments
- **Clarity**: Comprehensive documentation and clear naming
- **Counterfactual**: Document architectural decisions

## Risk Mitigation
- Technical risks identified and mitigated
- Dependencies evaluated for constitutional compliance
- Complexity kept minimal per Simplicity Gate

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _generate_task_breakdown(
        self, _implementation_plan: str, priority_focus: str, team_size: int
    ) -> str:
        """Generate task breakdown."""
        return f"""# Task Breakdown

## Priority Focus: {priority_focus.title()}

## Epic 1: Foundation & Infrastructure
### Task 1.1: Test Infrastructure Setup
- **Priority**: Critical
- **Estimated Hours**: 8
- **Dependencies**: None
- **Description**: Set up pytest, coverage, CI/CD pipeline
- **Acceptance Criteria**:
  - [ ] Test framework configured
  - [ ] Coverage reporting enabled
  - [ ] CI/CD pipeline functional

### Task 1.2: Core Data Models
- **Priority**: Critical
- **Estimated Hours**: 12
- **Dependencies**: 1.1
- **Description**: Implement core data structures with tests first
- **Acceptance Criteria**:
  - [ ] All models tested (TDD)
  - [ ] Validation logic complete
  - [ ] Documentation generated

## Epic 2: Core Implementation
### Task 2.1: Business Logic
- **Priority**: High
- **Estimated Hours**: 20
- **Dependencies**: 1.2
- **Description**: Implement main feature functionality
- **Acceptance Criteria**:
  - [ ] All requirements implemented
  - [ ] Test coverage >= 80%
  - [ ] Performance benchmarks met

### Task 2.2: CLI Interface
- **Priority**: High
- **Estimated Hours**: 10
- **Dependencies**: 2.1
- **Description**: Constitutional requirement for CLI interface
- **Acceptance Criteria**:
  - [ ] Text-in, text-out interface
  - [ ] Help documentation complete
  - [ ] Error handling robust

## Epic 3: Integration & Quality
### Task 3.1: Integration Testing
- **Priority**: Medium
- **Estimated Hours**: 16
- **Dependencies**: 2.2
- **Description**: Real environment testing (not mocks)
- **Acceptance Criteria**:
  - [ ] End-to-end workflows tested
  - [ ] Real data integration verified
  - [ ] Performance under load validated

### Task 3.2: Documentation & Review
- **Priority**: Medium
- **Estimated Hours**: 12
- **Dependencies**: 3.1
- **Description**: Complete documentation and constitutional review
- **Acceptance Criteria**:
  - [ ] All documentation complete
  - [ ] Constitutional compliance verified
  - [ ] Ready for deployment

## Team Size Optimization
Team Size: {team_size} developer(s)
Estimated Timeline: {78 // team_size} days (assuming 6 hours/day)

## Constitutional Requirements per Task
- All tasks must maintain constitutional compliance
- Test-first development required
- Library-first research before custom implementation
- Integration testing with real environments
- Clear documentation and decision rationale

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _validate_specification(
        self, specification: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate specification against constitutional articles."""
        # Use the constitutional scorer
        result = self.constitutional_scorer.score_specification(specification)

        validations = {}
        for violation in result.violations:
            article = violation.article
            if article not in validations:
                validations[article] = ConstitutionalValidation(
                    article=article,
                    compliant=False,
                    score=0.5,
                    violations=[],
                    suggestions=[],
                )
            validations[article].violations.append(violation.message)
            if violation.suggestion:
                validations[article].suggestions.append(violation.suggestion)

        # Add passing articles
        all_articles = [
            "Article I",
            "Article II",
            "Article III",
            "Article IV",
            "Article V",
            "Article VI",
        ]
        for article in all_articles:
            if article not in validations:
                validations[article] = ConstitutionalValidation(
                    article=article,
                    compliant=True,
                    score=result.overall_score,
                    violations=[],
                    suggestions=[],
                )

        return validations

    def _validate_implementation_plan(
        self, implementation_plan: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate implementation plan against constitutional articles."""
        return self._validate_specification(implementation_plan)

    def _validate_task_breakdown(
        self, tasks_breakdown: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate task breakdown against constitutional articles."""
        return self._validate_specification(tasks_breakdown)

    def _calculate_overall_score(
        self, validations: dict[str, ConstitutionalValidation]
    ) -> float:
        """Calculate overall constitutional compliance score."""
        if not validations:
            return 1.0

        total_score = sum(v.score for v in validations.values())
        return total_score / len(validations)

    def _get_specify_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for specify phase."""
        if threshold_met:
            return [
                "Review and refine the specification",
                "Validate requirements with stakeholders",
                "Run /plan to create implementation plan",
            ]
        else:
            return [
                "Address constitutional violations",
                "Improve specification clarity",
                "Re-run /specify after corrections",
            ]

    def _get_plan_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for plan phase."""
        if threshold_met:
            return [
                "Review implementation plan",
                "Validate technology choices",
                "Run /tasks to generate task breakdown",
            ]
        else:
            return [
                "Address constitutional violations in plan",
                "Simplify architecture if needed",
                "Re-run /plan after corrections",
            ]

    def _get_tasks_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for tasks phase."""
        if threshold_met:
            return [
                "Begin implementation with test-first approach",
                "Start with highest priority tasks",
                "Maintain constitutional compliance throughout",
            ]
        else:
            return [
                "Address constitutional violations in tasks",
                "Ensure test-first approach in all tasks",
                "Re-run /tasks after corrections",
            ]

    def _generate_supporting_documents(
        self, feature_dir: Path, _specification: str, _tech_stack: str
    ) -> list[str]:
        """Generate supporting documents."""
        docs = []

        # API contract
        api_file = feature_dir / "api-contract.md"
        api_content = f"""# API Contract

## Endpoints
TBD based on requirements

## Data Models
TBD based on specification

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        api_file.write_text(api_content, encoding="utf-8")
        docs.append(str(api_file))

        return docs

    def _get_tech_recommendations(self, _current_stack: str) -> list[str]:
        """Get technology recommendations."""
        return [
            "Use established libraries (Library-First principle)",
            "Prefer simple, well-documented tools",
            "Ensure CLI interface capability",
            "Choose tools with good testing support",
        ]

    def _extract_architecture_decisions(self, _plan: str) -> list[str]:
        """Extract architecture decisions from plan."""
        return [
            "Modular architecture for simplicity",
            "Test-first development approach",
            "CLI interface for constitutional compliance",
            "Integration testing with real environments",
        ]

    def _parse_structured_tasks(
        self, _tasks_content: str
    ) -> list[TaskBreakdown]:
        """Parse tasks content into structured format."""
        # Simple parsing - in production this would be more sophisticated
        tasks = [
            TaskBreakdown(
                id="1.1",
                title="Test Infrastructure Setup",
                description="Set up pytest, coverage, CI/CD pipeline",
                priority="critical",
                estimated_hours=8,
                dependencies=[],
                acceptance_criteria=[
                    "Test framework configured",
                    "Coverage reporting enabled",
                    "CI/CD pipeline functional",
                ],
                constitutional_requirements=["Test-First development"],
            ),
            TaskBreakdown(
                id="1.2",
                title="Core Data Models",
                description="Implement core data structures with tests first",
                priority="critical",
                estimated_hours=12,
                dependencies=["1.1"],
                acceptance_criteria=[
                    "All models tested (TDD)",
                    "Validation logic complete",
                    "Documentation generated",
                ],
                constitutional_requirements=["Test-First", "Clarity"],
            ),
        ]
        return tasks

    def _calculate_critical_path(
        self, tasks: list[TaskBreakdown]
    ) -> list[str]:
        """Calculate critical path through tasks."""
        # Simple implementation - in production would use proper scheduling
        return [task.id for task in tasks if task.priority == "critical"]
