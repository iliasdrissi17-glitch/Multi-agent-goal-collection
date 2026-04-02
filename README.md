```markdown
# Warehouse Multi-Agent Goal Collection

**Global coordination once. Local intelligence thereafter.**

How do you make a fleet of robots work efficiently together when they cannot communicate during execution, only observe a small occluded portion of the world, and get permanently disabled after a single collision?

This project implements a decentralized multi-robot autonomy stack for cooperative pickup-and-delivery in warehouse environments. A one-shot global planner assigns work before the episode starts; after that, each robot operates independently using only local perception, reactive conflict handling, and differential-drive control.

---

## At a glance

- Multi-robot pickup and delivery under partial observability
- One-time global task allocation, no runtime communication
- Decentralized execution with local adaptation
- Reactive handling of dynamic obstacles and disappearing goals
- Corridor management, yielding, and detour generation
- Real-time control for differential-drive robots

---

## Why this problem is hard

This is not just a path-planning exercise.

Each robot must:
- collect shared goals
- transport them to collection points
- avoid static obstacles
- avoid other robots as dynamic obstacles
- act under line-of-sight sensing only
- recover when another robot takes its intended goal first
- do all of this without communicating after initialization

The challenge is not only motion generation. It is **fleet-level coordination under uncertainty**, where poor local decisions can destroy global performance.

---

## The core idea

The system is built around a simple but powerful split:

### 1. Global coordination before execution
A centralized planner uses full initial knowledge of the environment to compute:
- task allocation
- pickup and delivery ordering
- waypoint-level routing
- collection-point assignment
- coordination metadata serialized into a shared plan

### 2. Local intelligence during execution
Once the simulation starts, each robot runs independently and:
- follows its assigned task sequence
- tracks a continuous path
- reacts to visible robots and goal changes
- yields, detours, or backs off in local conflicts
- updates behavior based only on local observations

**Shared plan. Local decisions. No runtime communication.**

---

## Demo

### Emergent multi-agent behavior

<p align="center">
  <img src="assets/videos/config1.gif" width="45%"/>
  <img src="assets/videos/config2.gif" width="45%"/>
</p>

<p align="center">
  <img src="assets/videos/scenario3.gif" width="45%"/>
  <img src="assets/videos/scenario4.gif" width="45%"/>
</p>

### What these episodes show

- **Config 1**: baseline cooperative pickup and delivery
- **Config 2**: denser obstacle-aware routing
- **Config 3**: stronger robot-robot interaction and local conflict handling
- **Config 4**: higher congestion with decentralized adaptation

The system exhibits coordinated fleet behavior without runtime communication, relying only on shared initialization and local perception.

---

## System architecture

### Global planner

The global planner computes the high-level structure of execution:
- which robot should collect which goals
- in what order
- through which collection points
- along which waypoint paths

This phase is where the system exploits full world knowledge.

### Runtime controller

Each robot then runs the same decentralized policy:
- read local observations
- track the current task and phase
- follow a path continuously
- detect local conflicts
- apply yield / detour / backoff logic
- output wheel commands

This gives the fleet a consistent global strategy while preserving runtime autonomy.

---

## What is actually implemented

The runtime policy is not a simple “go to target” controller.

It includes:
- task-state management
- path-progress tracking
- lookahead-based continuous target generation
- local detour insertion around dynamic blockers
- priority logic around collection points
- narrow-corridor backoff and release behavior
- reassignment when goals disappear from local observations

In practice, this means the robot can:
- continue executing even when the world changes locally
- avoid deadlocks in tight spaces
- preserve throughput without relying on communication

---

## Motion model

The robots are modeled as differential-drive systems.

### State

```text
x = [x, y, psi]
```

where:
- `x, y` are planar coordinates
- `psi` is heading

### Control

```text
u = [omega_l, omega_r]
```

where:
- `omega_l` is the left wheel angular velocity
- `omega_r` is the right wheel angular velocity

The simulator clips commands to wheel-speed limits, but overly aggressive commands are still penalized through the actuation-effort metric.

---

## Path tracking

The controller uses continuous path tracking rather than naive waypoint hopping.

It maintains:
- the current path polyline
- progress along that path
- a forward lookahead target

This produces smoother behavior and reduces:
- oscillation
- stop-and-go motion
- wasted travel distance

---

## Dynamic obstacle handling

Other robots are treated as first-class dynamic obstacles.

When another robot blocks the near-horizon path, the controller can:
- detect blockage along the active path
- compute a local geometric detour
- splice that detour into the current route
- rejoin the original path once clearance is restored

This is one of the key reasons the system remains usable in dense interactions.

---

## Conflict resolution

### Around collection points
The agent applies explicit priority rules based on:
- carrying state
- relative distance
- convergence behavior
- tie-breaking logic

### In narrow corridors
The agent can:
- detect corridor entry conditions
- back off to a hold point
- wait for the other robot to clear
- release the corridor safely
- avoid oscillatory re-entry

This is where the implementation stops feeling like a toy controller and starts feeling like an engineered multi-robot system.

---

## Observability model

Robots do not get full runtime state.

Each agent only sees:
- nearby robots within sensing range and line of sight
- nearby goals that are still available
- static obstacles known from initialization

That makes the runtime problem fundamentally **partially observable**, which is why the architecture uses:
- strong global initialization
- lightweight local adaptation

---

## Problem constraints

The system is designed around the official exercise rules:

- each robot can carry **at most one goal**
- goals are collected automatically on contact
- goals are delivered automatically inside collection zones
- no communication is allowed after global planning
- collisions permanently disable robots
- execution must remain real-time compatible

These constraints make safety and local conflict handling central to the design.

---

## Performance priorities

The score function strongly rewards:
- delivered goals
- zero collisions
- early completion
- short travel distance
- low actuation effort
- low computation time

So the real design objective is:

**maximize throughput without paying for collisions, delay, or wasted motion.**

---

## Score structure

```text
score = goals * 100
       - collisions * 500
       + time_bonus
       - distance_penalty
       - effort_penalty
       - compute_penalty
```

That means:
- throughput matters
- safety matters even more
- efficiency is not cosmetic, it affects the score directly

---

## Repository structure

```text
multi-agent-goal-collection/
├── README.md
├── src/
│   └── pdm4ar/
│       └── exercises/
│           └── ex14/
│               ├── agent.py
│               └── (optional helper modules)
└── exercises_def/
    └── ex14/
        ├── ex14.py
        ├── perf_metrics.py
        ├── random_config.py
        └── configs
```

---

## Key components

| Component | Role |
|----------|------|
| `Pdm4arGlobalPlanner` | One-shot global coordination and task allocation |
| `GlobalPlanMessage` | Serialized plan broadcast to all robots |
| `Pdm4arAgent` | Decentralized runtime policy |
| `GoalTask` | Pickup and delivery task structure |
| `AgentPlan` | Ordered task list assigned to a robot |
| `get_commands()` | Real-time control loop |

---

## Execution flow

### Before the episode
- ingest environment, robots, goals, and collection points
- compute the global assignment and routing structure
- serialize the shared global plan
- broadcast it once to all agents

### During the episode
- each robot deserializes its own task list
- follows the assigned pickup and delivery sequence
- reacts to local obstacles and robot interactions
- adapts when its intended goal is no longer available

---

## Requirements

This project requires:

- Python 3.10+
- `numpy`
- `scipy`
- `shapely`
- `pydantic`

Install the public dependencies with:

```bash
pip install numpy scipy shapely pydantic
```

> **Important:** This project also depends on the **PDM4AR course simulation framework**, including `dg-commons`, which is **not installable directly via pip**.

---

## Current status

> **This repository is not standalone-runnable.**

It contains:
- the full global planning logic
- the decentralized runtime control policy
- the coordination and conflict-handling system

Full execution still requires the original course environment, including:
- the simulator
- the observation pipeline
- robot models
- scenario definitions
- exercise entry points

---

## Running the exercise

Within the original PDM4AR framework:

```bash
python path/to/src/pdm4ar/main.py -e 14
```

or

```bash
python path/to/src/pdm4ar/main.py --exercise 14
```

---

## What this project demonstrates

- Multi-agent systems engineering
- Decentralized autonomy under communication constraints
- Real-time control under partial observability
- Dynamic obstacle handling in robot fleets
- Conflict resolution in shared workspaces
- Practical robotics system design shaped by measurable performance criteria

---

## Design strengths

- Clear separation between coordination and control
- Runtime policy designed for partial observability
- Explicit handling of robot-robot interaction
- Better-than-trivial local conflict logic
- Strong alignment between system design and evaluation metric

---

## Limitations

- no explicit prediction model for other robots
- no learned coordination policy
- local conflict resolution remains heuristic
- performance may degrade in extreme congestion
- runtime is reactive rather than trajectory-optimization-based

---

## Future improvements

High-value next steps include:
- predictive multi-agent collision avoidance
- auction-based or graph-based task allocation
- reservation-based corridor management
- trajectory-level planning instead of waypoint-level pursuit
- learning-based coordination priors
- explicit deadlock recovery policies
- deployment on real differential-drive robots

---

## Bottom line

This project is a real multi-robot autonomy system, not just a simulator submission.

It combines:
- global task allocation
- decentralized execution
- obstacle-aware routing
- continuous path tracking
- local negotiation logic
- real-time differential-drive control

**When robots cannot talk, the system design has to.**

---

## Author

Ilias Drissi
```