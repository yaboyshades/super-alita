"""
Mangle rules for SDD reasoning.

This module contains deductive rules written in Mangle syntax for reasoning
about code quality, test coverage, constitutional compliance, and feature completeness.

These rules enable automated analysis of:
- Untested functions
- Incomplete features
- Constitutional violations
- Code-to-specification traceability
- Dependency analysis
"""

MANGLE_RULES = """
// =============================================================================
// CORE REASONING RULES FOR SDD FRAMEWORK
// =============================================================================

// -----------------------------------------------------------------------------
// Test Coverage Rules
// -----------------------------------------------------------------------------

// A function is untested if it exists but no test is declared for it
untested_function(Func) :-
  function(Func, File, Module),
  !test_for(_, Func).

// A function is well_tested if it has multiple test cases
well_tested_function(Func) :-
  function(Func, File, Module),
  test_for(Test1, Func),
  test_for(Test2, Func),
  Test1 != Test2.

// A module has poor test coverage if more than half its functions are untested
poor_test_coverage(Module) :-
  module(Module, File),
  function_count(Module, Total),
  untested_count(Module, Untested),
  Untested > Total / 2.

// Helper rule to count functions in a module
function_count(Module, Count) :-
  module(Module, _),
  function(_, _, Module) |> do fn:group_by(Module), let Count = fn:Count().

// Helper rule to count untested functions in a module
untested_count(Module, Count) :-
  module(Module, _),
  untested_function(Func),
  function(Func, _, Module) |> do fn:group_by(Module), let Count = fn:Count().

// -----------------------------------------------------------------------------
// Feature Completeness Rules
// -----------------------------------------------------------------------------

// A feature is incomplete if it has an acceptance criterion that is not implemented by any test
incomplete_feature(FeatureID) :-
  acceptance_criterion(CritID, FeatureID, _),
  !implements_criterion(_, CritID).

// A feature is fully_implemented if all its acceptance criteria are covered by tests
fully_implemented_feature(FeatureID) :-
  spec(FeatureID, _),
  !incomplete_feature(FeatureID).

// A specification is orphaned if no code references it
orphaned_spec(FeatureID) :-
  spec(FeatureID, _),
  !code_for_feature(FeatureID, _).

// -----------------------------------------------------------------------------
// Constitutional Compliance Rules
// -----------------------------------------------------------------------------

// Article I: Library-First Development
// A module violates Article I if it reimplements functionality available in external dependencies
violates_library_first(Module) :-
  module(Module, _),
  function(Func, _, Module),
  external_dependency(Lib),
  reimplements_library_function(Func, Lib).

// Article II: Test-First Development
// A function violates Article II if it was implemented without corresponding tests
violates_test_first(Func) :-
  function(Func, File, Module),
  !test_for(_, Func),
  !function_doc(Func, _).  // Allow documented functions to be exempted

// Article III: Simplicity Gate
// A function violates Article III if it's too complex (heuristic: >10 lines or >3 parameters)
violates_simplicity(Func) :-
  function(Func, File, Module),
  function_complexity(Func, Complexity),
  Complexity > 10.

// Article IV: Integration-First Testing
// A feature violates Article IV if it has unit tests but no integration tests
violates_integration_first(FeatureID) :-
  spec(FeatureID, _),
  has_unit_tests(FeatureID),
  !has_integration_tests(FeatureID).

// Article V: Clarity and Unambiguity
// A function violates Article V if it lacks documentation
violates_clarity(Func) :-
  function(Func, File, Module),
  !function_doc(Func, _),
  !test_for(_, Func).  // Functions with tests might be self-documenting

// Article VI: Counterfactual Justification
// A design decision violates Article VI if no alternative was considered
violates_counterfactual(Decision) :-
  design_decision(Decision, Context),
  !alternative_considered(Decision, _).

// -----------------------------------------------------------------------------
// Code-Specification Traceability Rules
// -----------------------------------------------------------------------------

// Find all code related to a feature by tracing its tests back to the functions they cover
code_for_feature(FeatureID, Func) :-
  acceptance_criterion(CritID, FeatureID, _),
  implements_criterion(Test, CritID),
  test_for(Test, Func).

// Find all specifications that relate to a specific function
spec_for_function(Func, FeatureID) :-
  code_for_feature(FeatureID, Func).

// Find functions that implement multiple features (potential coupling issue)
multi_feature_function(Func) :-
  spec_for_function(Func, Feature1),
  spec_for_function(Func, Feature2),
  Feature1 != Feature2.

// -----------------------------------------------------------------------------
// Dependency Analysis Rules
// -----------------------------------------------------------------------------

// A module has circular dependencies if it imports something that eventually imports it back
circular_dependency(Module1, Module2) :-
  imports(Module1, Module2),
  dependency_path(Module2, Module1).

// Helper rule for transitive dependency paths
dependency_path(From, To) :-
  imports(From, To).

dependency_path(From, To) :-
  imports(From, Intermediate),
  dependency_path(Intermediate, To).

// A module is a dependency hotspot if many other modules depend on it
dependency_hotspot(Module) :-
  module(Module, _),
  dependent_count(Module, Count),
  Count > 5.

// Helper rule to count dependents
dependent_count(Module, Count) :-
  module(Module, _),
  imports(_, Module) |> do fn:group_by(Module), let Count = fn:Count().

// -----------------------------------------------------------------------------
// Code Quality Rules
// -----------------------------------------------------------------------------

// A function is complex if it has many parameters or long implementation
complex_function(Func) :-
  function(Func, File, Module),
  function_complexity(Func, Complexity),
  Complexity > 15.

// A class is overly large if it has too many methods
large_class(ClassName) :-
  class(ClassName, File, Module),
  method_count(ClassName, Count),
  Count > 20.

// Helper rule to count methods in a class
method_count(ClassName, Count) :-
  class(ClassName, _, _),
  method(_, ClassName, _) |> do fn:group_by(ClassName), let Count = fn:Count().

// A file is a god file if it contains too many classes or functions
god_file(File) :-
  module(_, File),
  entity_count(File, Count),
  Count > 50.

// Helper rule to count entities (classes + functions) in a file
entity_count(File, Count) :-
  module(_, File),
  (function(_, File, _); class(_, File, _)) |> do fn:group_by(File), let Count = fn:Count().

// -----------------------------------------------------------------------------
// Helper Predicates
// -----------------------------------------------------------------------------

// Check if a file is part of a library/module
part_of(File, Lib) :-
  module(Module, File),
  string_prefix(Lib, Module).

// Check if a function has a specific tag or attribute
has_tag(Func, Tag) :-
  function(Func, File, _),
  cli_command(Func, File),
  Tag = "cli".

has_tag(Func, Tag) :-
  function_doc(Func, Doc),
  string_contains(Doc, Tag).

// Check if a module has unit tests
has_unit_tests(FeatureID) :-
  code_for_feature(FeatureID, Func),
  test_for(Test, Func),
  test_function(Test, TestFile),
  string_contains(TestFile, "test_").

// Check if a module has integration tests
has_integration_tests(FeatureID) :-
  code_for_feature(FeatureID, Func),
  test_for(Test, Func),
  test_function(Test, TestFile),
  string_contains(TestFile, "integration").

// Heuristic for function complexity (simplified)
function_complexity(Func, 5) :-
  function(Func, _, _),
  !function_doc(Func, _).

function_complexity(Func, 3) :-
  function(Func, _, _),
  function_doc(Func, _).

// Heuristic for detecting reimplementation of library functions
reimplements_library_function(Func, Lib) :-
  function(Func, _, _),
  external_dependency(Lib),
  string_contains(Func, "parse"),
  Lib = "argparse".

reimplements_library_function(Func, Lib) :-
  function(Func, _, _),
  external_dependency(Lib),
  string_contains(Func, "request"),
  Lib = "requests".

// =============================================================================
// QUERY EXAMPLES AND COMMON PATTERNS
// =============================================================================

// Find all constitutional violations for a comprehensive audit
constitutional_violation(Article, Violator) :-
  violates_library_first(Violator),
  Article = "I".

constitutional_violation(Article, Violator) :-
  violates_test_first(Violator),
  Article = "II".

constitutional_violation(Article, Violator) :-
  violates_simplicity(Violator),
  Article = "III".

constitutional_violation(Article, Violator) :-
  violates_integration_first(Violator),
  Article = "IV".

constitutional_violation(Article, Violator) :-
  violates_clarity(Violator),
  Article = "V".

constitutional_violation(Article, Violator) :-
  violates_counterfactual(Violator),
  Article = "VI".

// Find all quality issues in one query
quality_issue(Type, Entity) :-
  untested_function(Entity),
  Type = "untested_function".

quality_issue(Type, Entity) :-
  complex_function(Entity),
  Type = "complex_function".

quality_issue(Type, Entity) :-
  large_class(Entity),
  Type = "large_class".

quality_issue(Type, Entity) :-
  god_file(Entity),
  Type = "god_file".

quality_issue(Type, Entity) :-
  circular_dependency(Entity, _),
  Type = "circular_dependency".

// Find all incomplete work
incomplete_work(Type, Entity) :-
  incomplete_feature(Entity),
  Type = "incomplete_feature".

incomplete_work(Type, Entity) :-
  orphaned_spec(Entity),
  Type = "orphaned_spec".

incomplete_work(Type, Entity) :-
  poor_test_coverage(Entity),
  Type = "poor_test_coverage".
"""

# Common query patterns for the CLI
COMMON_QUERIES = {
    "untested functions": "untested_function(Func)",
    "functions are untested": "untested_function(Func)",
    "what functions are untested": "untested_function(Func)",
    "incomplete features": "incomplete_feature(FeatureID)",
    "features are incomplete": "incomplete_feature(FeatureID)",
    "constitutional violations": "constitutional_violation(Article, Violator)",
    "violates constitution": "constitutional_violation(Article, Violator)",
    "quality issues": "quality_issue(Type, Entity)",
    "incomplete work": "incomplete_work(Type, Entity)",
    "complex functions": "complex_function(Func)",
    "circular dependencies": "circular_dependency(Module1, Module2)",
    "orphaned specs": "orphaned_spec(FeatureID)",
    "dependency hotspots": "dependency_hotspot(Module)",
    "well tested functions": "well_tested_function(Func)",
    "multi feature functions": "multi_feature_function(Func)",
    "library first violations": "violates_library_first(Module)",
    "test first violations": "violates_test_first(Func)",
    "simplicity violations": "violates_simplicity(Func)",
    "clarity violations": "violates_clarity(Func)",
}


def get_query_for_question(question: str) -> str | None:
    """
    Map natural language questions to Mangle queries.

    Args:
        question: Natural language question about the codebase

    Returns:
        Mangle query string if found, None otherwise
    """
    question_lower = question.lower()

    for pattern, query in COMMON_QUERIES.items():
        if pattern in question_lower:
            return query

    return None


def get_available_queries() -> list[str]:
    """
    Get list of available query patterns.

    Returns:
        List of natural language query patterns
    """
    return list(COMMON_QUERIES.keys())
