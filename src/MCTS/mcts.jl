mutable struct NodeValue{T}
    state::Union{Nothing,T}
    reward::Float64
    value::Float64
    visits::Int
    NodeValue{T}() where {T} = new{T}(nothing, 0, 0, 0)
    NodeValue(state::T) where {T} = new{T}(state, 0, 0, 0)
end

nodevalue(nv::NodeValue) = (nv.value, nv.reward, nv.visits)

struct Planner
    env::AbstractEnvironment
    temperature::Float64
    γ::Float64
    horizon::Int
    budget::Int
    tree::TreeNode{NodeValue}
end

function Planner(
    env::AbstractEnvironment;
    seed::Union{Nothing,Int} = nothing,
    temperature = 100.0,
    γ = 0.9,
    horizon = 10,
    budget = 100,
)
    reset!(env, seed)
    initial_state = state(env)
    nv = NodeValue(initial_state)
    tn = TreeNode{NodeValue}(nv)
    Planner(env, temperature, γ, horizon, budget, tn)
end

selection_value(node::TreeNode, mcts::Planner) =
    node.value.value + mcts.temperature / (node.value.visits + 1)

function select_action_id(node, mcts::Planner)::Union{Nothing,Int}
    if isleaf(node)
        return nothing
    end
    selection_values = [selection_value(n, mcts) for n in children(node)]
    return rand_max(selection_values)
end

select_action_id(mcts::Planner) = select_action_id(mcts.tree, mcts)

function select(mcts::Planner)
    node = mcts.tree
    reward_multiplier = mcts.γ
    total_reward = 0.0 # root's reward can be only 0
    while !isleaf(node) && !isdone(node.value.state)
        action_id = select_action_id(node, mcts)
        node = children(node)[action_id]
        reward_multiplier *= mcts.γ
        total_reward += reward_multiplier * node.value.reward
    end
    node, total_reward
end

function expand!(node::TreeNode, mcts::Planner)
    done = false
    if isnothing(node.value.state)
        pn = AbstractTrees.parent(node)
        setstate!(mcts.env, pn.value.state)
        action_id = findfirst(map(n -> n === node, children(pn)))
        _, reward, done, _ = step!(mcts.env, action_space(mcts.env)[action_id])
        node.value.state = state(mcts.env)
        node.value.reward = reward
    end

    if isleaf(node) && (!done || isroot(node)) && depth(node) < mcts.horizon
        leafs = length(action_space(mcts.env))
        node.children = [
            TreeNode{NodeValue}(NodeValue{typeof(state(mcts.env))}(), node) for _ = 1:leafs
        ]
    end
    nothing
end

function simulate!(node::TreeNode, total_reward::Float64, mcts::Planner)
    env = mcts.env
    actions = action_space(env)
    d = depth(node)
    reward_multiplier = mcts.γ^d
    done = isdone(node.value.state)
    setstate!(env, node.value.state)
    total_reward += reward_multiplier * node.value.reward
    while !done && d <= mcts.horizon
        action = rand(actions)
        _, reward, done, _ = step!(env, action)
        reward_multiplier *= mcts.γ
        total_reward += reward_multiplier * reward
        d += 1
    end
    return total_reward
end

function backpropagate!(node::TreeNode, total_reward::Float64, mcts::Planner)
    while !isroot(node)
        node.value.visits += 1
        node.value.value += (total_reward - node.value.value) / node.value.visits
        node = AbstractTrees.parent(node)
    end
    nothing
end

function run!(mcts::Planner)
    node, total_reward = select(mcts)
    expand!(node, mcts)
    total_reward = simulate!(node, total_reward, mcts)
    backpropagate!(node, total_reward, mcts)
end

function plan!(mcts::Planner)
    episodes = mcts.budget ÷ mcts.horizon
    for i = 1:episodes+1
        # if i % 10 == 0
        #     @debug ('{} / {}'.format(i+1, self.config['episodes']))
        run!(mcts)
    end
    # action_space(mcts.env)[select_action_id(mcts)]
    children(mcts.tree)[select_action_id(mcts)]
end

function run_planner!(mcts)
    episodes_before_done = 0
    while !isdone(mcts.tree.value.state) #&& episodes_before_done < 20
        new_root = plan!(mcts)
        # new_root.children = ()
        # mcts = Planner(mcts.env, mcts.temperature, mcts.γ, mcts.horizon, mcts.budget, new_root)
        mcts.tree.value = new_root.value
        mcts.tree.children = () # reset strategy
        episodes_before_done += 1
        # println(episodes_before_done)
        # println(mcts.tree.value)
    end
    println(episodes_before_done)
end