# Architecture

## Layers

### 1. High-level agent
Plans tasks using semantic concepts and chooses the embodiment best suited for each subtask.

Examples:

- inspect an area;
- move to a named location;
- pick or place an object;
- take off or land;
- stand or walk.

### 2. Skill layer
Provides stable semantic operations. This is the API the agent sees.

The skill layer must remain independent of:

- simulator choice;
- motor protocol;
- control frequency;
- whether hardware is real or simulated.

### 3. Embodiment adapter
Maps semantic skills to a robot-specific controller.

An adapter is responsible for:

- connection lifecycle;
- observations;
- validation;
- dispatching skills;
- translating results and errors.

### 4. Policy/control layer
Runs at robot-appropriate frequencies.

Examples:

- LeRobot VLA/manipulation policy for XLeRobot;
- trajectory or flight controller for Crazyflie;
- locomotion policy / whole-body controller for humanoid.

### 5. Backend
Simulator or physical robot.

## What LeRobot owns

LeRobot is primarily used for:

- robot interfaces;
- datasets;
- learned policies;
- training and evaluation workflows;
- sim-to-real-compatible abstractions.

It does **not** need to be the single physics engine.

## Safety invariant

The high-level language-model agent never directly controls raw motors, thrust, servo PWM, or joint torque.

High-frequency closed-loop control stays inside deterministic or learned robot-specific controllers.
