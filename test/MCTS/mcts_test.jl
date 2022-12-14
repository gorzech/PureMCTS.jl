struct DummyEnv <: AbstractEnvironment end
Environments.reset!(env::DummyEnv, seed) = 0.0
Environments.step!(env::DummyEnv, action) = 0.0, 1.0, false, ()
Environments.state(env::DummyEnv) = 1.0
Environments.setstate!(env::DummyEnv, state) = nothing
Environments.action_space(env::DummyEnv) = [1, 2]
Environments.isdone(env::DummyEnv, state) = false

mutable struct TestEnv <: AbstractEnvironment
    steps :: Int
end
Environments.reset!(env::TestEnv, seed) = (env.steps = 0)
Environments.step!(env::TestEnv, action) = begin
    env.steps += 1
    env.steps, 10.0 + env.steps, false, ()
end
Environments.state(env::TestEnv) = env.steps
Environments.setstate!(env::TestEnv, state) = (env.steps = state)
Environments.action_space(env::TestEnv) = [1]
Environments.isdone(env::TestEnv, state::Int) = false

@testset "select on new tree return one of the nodes" begin
    tree = Planner(DummyEnv())
    node, total_reward = @test_nowarn select!(tree)
    @test isroot(node)
    @test total_reward == 0.0
end

@testset "Check expand on new tree" begin
    tree = Planner(DummyEnv())
    root_node = tree.tree
    new_node = expand!(root_node, tree)
    @test isroot(root_node)
    @test isroot(AbstractTrees.parent(new_node))
    @test length(children(root_node)) == 2
    first_child = children(root_node)[1]
    @test !isroot(first_child)

    @test new_node.value.state == 1.0
    @test new_node.value.reward == 1.0
    @test new_node.value.value == 0.0
    @test new_node.value.visits == 0
end

@testset "Check run! based on simplistic environment" begin
    env = TestEnv(0)
    mcts = Planner(env, γ = 0.5, budget = 6, horizon = 3)
    @test_nowarn run!(mcts)
    @test length(children( mcts.tree)) == 1
    @test env.steps == 3
    run!(mcts)
    @test treeheight(mcts.tree) == 2
    run!(mcts)
    @test treeheight(mcts.tree) == 3
    run!(mcts)
    @test treeheight(mcts.tree) == 3
end

@testset "Check plan! based on simplistic environment" begin  
    env = TestEnv(0)
    mcts = Planner(env, γ = 0.5, budget = 6, horizon = 3)
    node = @test_nowarn plan!(mcts)
    @test AbstractTrees.parent(node) === mcts.tree
    @test node.value.state == 1
    @test node.value.reward == 11
    @test node.value.value == 10.125
    @test node.value.visits == 2

    @test length(children(node)) == 1
    child_node = children(node)[1]
    @test PureMCTS.isleaf(child_node)
    @test child_node.value.state == 2
    @test child_node.value.reward == 12
    @test child_node.value.value == 10.125
    @test child_node.value.visits == 1
end

@testset "Check plan! based on dummy env" begin
    env = DummyEnv()
    mcts = Planner(env, γ = 0.6, budget = 9, horizon = 3)
    node = @test_nowarn plan!(mcts)
    @test AbstractTrees.parent(node) === mcts.tree
    @test node.value.state == 1
    @test node.value.reward == 1
    @test node.value.value == 0.6 + 0.6 * 0.6 + 0.6 ^ 3
    @test node.value.visits == 1

    # @test treebreadth(node) == 1 
    @test treebreadth(mcts.tree) == 4
end

@testset "Check run_planner! based on dummy env" begin
    env = DummyEnv()
    mcts = Planner(env, γ = 0.6, budget = 12, horizon = 3)
    
    @test_nowarn run_planner!(mcts, max_steps = 2)
    @test treebreadth(mcts.tree) == 5
    node = children(mcts.tree)[1]
    @test node.value.state == 1
    @test node.value.reward == 1
    @test node.value.value == 0.6 + 0.6 * 0.6 + 0.6 ^ 3
    @test node.value.visits == 2
end