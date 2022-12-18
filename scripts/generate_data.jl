## Load packages
using Environments
using PureMCTS

## Generate data for single pendulum
env = InvertedPendulumEnv
opts = PendulumOpts()
file_name = "single_pendulum" 
full_budget = round.(Int, exp10.(range(1, 5, 81)))
full_budget = full_budget[(full_budget .>= 30)]

b = planner_batch(
    budget=full_budget,
    horizon=4:30,
    γ=0.5:0.05:1.0,
    exploration_param=[0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
)
execute_batch(b, file_name, opts, envfun=env, show_progress=true)

## Generate data for double pendulum
env = InvertedDoublePendulumEnv
opts = PendulumOpts()
file_name = "double_pendulum" 
full_budget = round.(Int, exp10.(range(3, 6, 31)))

b = planner_batch(
    budget=full_budget,
    horizon=10:2:40,
    γ=0.7:0.05:1.0,
    exploration_param=[0, 2, 4, 8, 16, 32],
)
execute_batch(b, file_name, opts, envfun=env, show_progress=true)

