"""
Rule engine for code analysis.

Adapted from mangle_code_scaffold_v2/scripts/run_engine.py
"""

import logging
import sqlite3
from contextlib import contextmanager

from .models import Finding

logger = logging.getLogger(__name__)


class RuleEngine:
    """Engine for running SQL-based rules against code facts database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.rules: dict[str, str] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        """Register the default set of analysis rules."""

        # Untested functions with high complexity; prefer the descriptive rule name but keep
        # the historical alias to avoid breaking existing integrations.
        untested_sql = """
        SELECT s.symbol, s.file, c.score AS complexity
        FROM symbol s
        JOIN complexity c ON c.symbol = s.symbol
        WHERE s.kind = 'function'
          AND c.score >= 0.3
          AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
        """
        self.rules["untested_complex_functions"] = untested_sql
        self.rules["untested_function"] = untested_sql

        # Complex functions with no inbound or outbound calls.
        self.rules["orphan_complex"] = """
        WITH indeg AS (
          SELECT callee AS sym, COUNT(*) AS d FROM calls GROUP BY callee
        ), outdeg AS (
          SELECT caller AS sym, COUNT(*) AS d FROM calls GROUP BY caller
        )
        SELECT s.symbol, s.file, c.score
        FROM symbol s
        JOIN complexity c ON c.symbol = s.symbol
        LEFT JOIN indeg i ON i.sym = s.symbol
        LEFT JOIN outdeg o ON o.sym = s.symbol
        WHERE s.kind='function'
          AND c.score >= 0.6
          AND IFNULL(i.d,0)=0 AND IFNULL(o.d,0)=0
        """

        # Circular dependencies between files (bidirectional edge).
        cycle_sql = """
        SELECT a.fileA, a.fileB
        FROM dep a
        JOIN dep b ON a.fileA = b.fileB AND a.fileB = b.fileA
        WHERE a.fileA <> a.fileB
        """
        self.rules["circular_dependencies"] = cycle_sql
        self.rules["cycle"] = cycle_sql

        # High-complexity functions that are invoked but not covered by tests.
        hot_sql = """
        WITH indeg AS (
          SELECT callee AS sym, COUNT(*) AS d FROM calls GROUP BY callee
        )
        SELECT s.symbol, s.file, c.score AS complexity, IFNULL(i.d,0) AS indegree
        FROM symbol s
        JOIN complexity c ON c.symbol = s.symbol
        LEFT JOIN indeg i ON i.sym = s.symbol
        WHERE s.kind='function'
          AND c.score >= 0.5
          AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
          AND IFNULL(i.d,0) >= 1
        """
        self.rules["hot_paths"] = hot_sql
        self.rules["hot_path"] = hot_sql

        # Configuration cascade: functions inside central settings/env modules without direct tests.
        config_sql = """
        SELECT s.symbol, s.file
        FROM symbol s
        WHERE REPLACE(s.file, '\\', '/') IN ('src/core/settings.py', 'src/core/env.py')
          AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
        """
        self.rules["config_cascade_breaks"] = config_sql

        # Functions that appear to emit JSON but never import the json module.
        self.rules["reinvention_json"] = """
        SELECT s.symbol, s.file
        FROM symbol s
        WHERE s.kind='function'
          AND (LOWER(s.symbol) LIKE '%::to_json%' OR LOWER(s.symbol) LIKE '%::json_%')
          AND NOT EXISTS (
            SELECT 1 FROM imports i WHERE i.file = s.file AND i.module = 'json'
          )
        """

    def add_rule(self, name: str, sql: str):
        """Add a custom rule."""
        self.rules[name] = sql
        logger.info(f"Added custom rule: {name}")

    def remove_rule(self, name: str):
        """Remove a rule."""
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed rule: {name}")

    @staticmethod
    def _canonical_rule_name(name: str) -> str:
        """Map legacy rule aliases to canonical names."""
        aliases = {
            "untested_function": "untested_complex_functions",
            "cycle": "circular_dependencies",
            "hot_path": "hot_paths",
        }
        return aliases.get(name, name)

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def run_rule(self, rule_name: str) -> list[Finding]:
        """
        Run a single rule and return findings.

        Args:
            rule_name: Name of the rule to run

        Returns:
            List of findings from this rule

        Raises:
            ValueError: If rule doesn't exist
        """
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found")

        sql = self.rules[rule_name]
        findings = []

        try:
            with self.get_db_connection() as conn:
                rows = conn.execute(sql).fetchall()

                canonical_name = self._canonical_rule_name(rule_name)

                for row in rows:
                    finding = Finding(rule_name=canonical_name)

                    if canonical_name == "untested_complex_functions":
                        finding.symbol = row["symbol"]
                        finding.file = row["file"]
                        finding.complexity = row["complexity"]
                    elif canonical_name == "orphan_complex":
                        finding.symbol = row["symbol"]
                        finding.file = row["file"]
                        finding.complexity = row["score"]
                    elif canonical_name == "circular_dependencies":
                        finding.file_a = row["fileA"]
                        finding.file_b = row["fileB"]
                    elif canonical_name == "hot_paths":
                        finding.symbol = row["symbol"]
                        finding.file = row["file"]
                        finding.complexity = row["complexity"]
                        finding.indegree = row["indegree"]
                    elif canonical_name == "config_cascade_breaks" or canonical_name == "reinvention_json":
                        finding.symbol = row["symbol"]
                        finding.file = row["file"]

                    metadata = dict(row)
                    metadata.setdefault("source_rule", rule_name)
                    if canonical_name == "config_cascade_breaks":
                        metadata.setdefault("note", "central config logic lacks direct tests")
                    finding.metadata = metadata

                    findings.append(finding)
        except Exception as e:
            logger.error(f"Error running rule '{rule_name}': {e}")
            # Return empty list on error rather than failing completely

        return findings

    def run_all_rules(
        self, rule_names: list[str] | None = None
    ) -> dict[str, list[Finding]]:
        """
        Run multiple rules and return all findings.

        Args:
            rule_names: List of rule names to run, or None for all rules

        Returns:
            Dictionary mapping rule names to their findings
        """
        if rule_names is None:
            seen = set()
            ordered: list[str] = []
            for name in self.rules.keys():
                canonical = self._canonical_rule_name(name)
                if canonical in seen:
                    continue
                seen.add(canonical)
                ordered.append(name)
            rule_names = ordered

        results: dict[str, list[Finding]] = {}
        for rule_name in rule_names:
            if rule_name in self.rules:
                canonical = self._canonical_rule_name(rule_name)
                if canonical in results:
                    continue
                findings = self.run_rule(rule_name)
                results[canonical] = findings
                logger.info(f"Rule '{canonical}': {len(findings)} findings")
            else:
                logger.warning(f"Rule '{rule_name}' not found, skipping")

        return results

    def get_summary(self, findings: dict[str, list[Finding]]) -> dict[str, int]:
        """Generate summary counts from findings."""
        return {rule: len(findings_list) for rule, findings_list in findings.items()}
