
# Decision Policies

<cite>
**Referenced Files in This Document**   
- [src/core/decision_policy.py](file://src/core/decision_policy.py)
- [src/core/decision_policy_v1.py](file://src/core/decision_policy_v1.py)
- [src/ladder/policies/bandit.py](file://src/ladder/policies/bandit.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Decision Policy Engine](#core-decision-policy-engine)
3. [Policy Evaluation Process](#policy-evaluation-process)
4. [Configuration Parameters](#configuration-parameters)
5. [Strategy Selection Logic](#strategy-selection-logic)
6. [Relationship with Decision Engine](#relationship-with-decision-engine)
7. [Policy Conflicts and Resolution](#policy-conflicts-and-resolution)
8. [Customization and Extension](#customization-and-extension)
9. [Conclusion](#conclusion)

## Introduction
The Decision Policies component in the Super Alita framework serves as the central mechanism for determining optimal actions based on user input, context, and available capabilities. This system combines intent classification, capability matching, utility calculation, and strategy selection to generate executable plans. The framework supports multiple policy implementations, including multi-armed bandit algorithms and contextual decision-making, enabling adaptive behavior over time. This document provides a comprehensive overview of the implementation details, configuration options, and operational flow of the decision policy system.

## Core Decision Policy Engine

The core decision policy engine is implemented through the `DecisionPolicyEngine` class, which serves as the primary interface for decision-making within the Super Alita framework. This engine processes user messages, extracts structured information, and generates executable plans through a multi-stage evaluation process. The implementation exists in two nearly identical files—`decision_policy.py` and `decision_policy_v1.py`—with the latter serving as the canonical internal implementation while maintaining the former as the public import path for backward compatibility.

The engine follows a five-step decision process: intent classification, goal synthesis, capability resolution, utility calculation, and plan construction. It integrates with various system components including the intent classifier, goal synthesizer, utility calculator, and plan builder to transform natural language input into structured execution plans. The engine maintains state through bandit statistics that track the performance of different capabilities over time, enabling learning from past decisions.

```mermaid
classDiagram
    class