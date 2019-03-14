# Wood-Berry Distillation Tower Simulation Package

## Folders

1. Benchmark_Plots
  - Step test plots
  - Optimal steady state input dynamics

2. ControlLoops
  - Decoupled system drawings

3. FIS: Contextual Bandit Fault Identification System

4. FTC_Case1: MeOH priority.

5. FTC_Case2: Water priority.

6. FTC_Case3: Both considered.
 - Velocity_Case4_initial: Initial results
 - Velocity_Case4_Ucost: Reward contains deltaU cost
 - Case1_Case2_using_MISO: Case 1 and case 2 using the MISO RL Code
 - SMDP: SMDP Case studies of case 3

7. MATLAB_Files:
  - Decoupler: Decoupled Wood-Berry distillation tower
  - wood_berry_distillation: Raw Wood-Berry distillation tower m file.  Has transfer function and conversion from tf2ss.

8. Paper_Plots
  - Bad system dynamic plots (inv response, overshoot, oscillations)
  - Case study plots: <br>
	Case 1: Top product priority <br>
	Case 2: Bottom product priority <br>
	Case 3: Top & bottom priority <br>
	Case 4: Adaptability Study <br>

## Modules

1. Decoupled_Distillation: **Decoupled** Wood-Berry distillation tower simulation with PI regulatory controllers.

2. Woodberry_Distillation: Initial Wood-Berry distillation tower simulation with PI regulatory controllers.

3. Woodberry_Distillation_RL: Initial Wood-Berry distillation tower simulation with RL in supervisory layer.

4. RL_Module_Position: Position reinforcement learning module. Takes absolute states and actions.

5. RL_Module_Velocity: Velocity reinforcement learning module. Takes relative actions given error.

6. RL_Module_Velocity_MIMO_SMDP: MIMO SMDP reinforcement learning module. Considers semi-Markov processes.

7. RL_Module_Velocity_SMDP: SMDP reinforcement learning module.
