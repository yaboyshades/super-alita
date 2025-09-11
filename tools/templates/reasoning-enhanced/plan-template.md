# {{ PROJECT_NAME }} - Reasoning-Validated Technical Plan

## Planning Context
- Specification Source: {{ SPEC_PATH }}
- Planning Phase: {{ PHASE_NUMBER }}
- Cross-Phase Validation: ENABLED
- Reasoning Engine: MANGLE v{{ VERSION }}

## Architecture Facts (MANGLE)
```mangle
uses_technology("{{ PROJECT_NAME }}", "{{ TECH_STACK }}").
has_constraint("{{ PROJECT_NAME }}", "{{ CONSTRAINT }}").
supports_pattern("{{ ARCHITECTURE }}", "{{ PATTERN }}").
requires_expertise("{{ TECH_STACK }}", "{{ SKILL_LEVEL }}").
estimated_timeline("{{ PROJECT_NAME }}", {{ WEEKS }}).
team_capacity("{{ PROJECT_NAME }}", {{ CAPACITY }}).
```

## Planning Rules
```mangle
feasible_implementation(Project, Technology) :-
  uses_technology(Project, Technology),
  satisfies_constraints(Technology, Project),
  has_required_expertise(Team, Technology),
  within_timeline(Project, Technology).

valid_architecture(Architecture) :-
  supports_scalability(Architecture),
  meets_security_requirements(Architecture),
  compatible_with_existing_systems(Architecture),
  maintainable_complexity(Architecture).

realistic_timeline(Project, Timeline) :-
  estimated_effort(Project, Effort),
  team_capacity(Project, Capacity),
  Timeline >= (Effort / Capacity) * 1.2.

high_risk_choice(Technology) :-
  new_technology(Technology),
  tight_timeline(Project),
  limited_expertise(Team, Technology).
```

## Cross-Phase Verification Queries
```mangle
?- consistent_plan("{{ PROJECT_NAME }}", "{{ SPEC_NAME }}").
?- feasible_implementation("{{ PROJECT_NAME }}", ?).
?- high_risk_choice(?).
?- realistic_timeline("{{ PROJECT_NAME }}", {{ TIMELINE }}).
```

