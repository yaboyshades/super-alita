# {{ PROJECT_NAME }} - Reasoning-Optimized Task Breakdown

## Task Optimization Context
- Source Plan: {{ PLAN_PATH }}
- Optimization Level: {{ OPTIMIZATION_LEVEL }}
- Dependency Analysis: COMPLETE
- Resource Optimization: ENABLED

## Task Facts (MANGLE)
```mangle
task("{{ TASK_ID }}", "{{ TASK_NAME }}", {{ EFFORT_HOURS }}).
depends_on("{{ TASK_1 }}", "{{ TASK_2 }}").
requires_skill("{{ TASK_ID }}", "{{ SKILL_TYPE }}").
estimated_effort("{{ TASK_ID }}", {{ HOURS }}).
blocks("{{ TASK_1 }}", "{{ TASK_2 }}").
can_parallelize("{{ TASK_1 }}", "{{ TASK_2 }}").

team_member("{{ MEMBER_ID }}", "{{ NAME }}", [{{ SKILLS }}]).
available_capacity("{{ MEMBER_ID }}", {{ HOURS_PER_WEEK }}).
skill_match("{{ MEMBER_ID }}", "{{ SKILL }}", {{ PROFICIENCY_LEVEL }}).
```

## Sequencing Rules
```mangle
ready_to_start(Task) :-
  task(Task, _, _),
  all_dependencies_completed(Task),
  required_skills_available(Task),
  no_blocking_issues(Task).

optimal_sequence(Tasks) :-
  minimizes_idle_time(Tasks),
  respects_dependencies(Tasks),
  balances_workload(Tasks),
  maximizes_parallelization(Tasks).

optimal_assignment(Task, Member) :-
  task(Task, _, _),
  team_member(Member, _, Skills),
  requires_skill(Task, Skill),
  member(Skill, Skills),
  available_capacity(Member, Capacity),
  estimated_effort(Task, Effort),
  Capacity >= Effort.

critical_path_task(Task) :-
  task(Task, _, _),
  longest_dependency_chain(Task),
  no_slack_time(Task).
```

