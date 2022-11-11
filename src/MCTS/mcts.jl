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
    seed::Union{Nothing,Int}=nothing,
    temperature=100.0,
    γ=0.9,
    horizon=10,
    budget=100
)
    Environments.reset!(env, seed)
    initial_state = state(env)
    nv = NodeValue(initial_state)
    tn = TreeNode{NodeValue}(nv)
    Planner(env, temperature, γ, horizon, budget, tn)
end

# return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
# selection_value(node::TreeNode, mcts::Planner) =
#     value(node).value + mcts.temperature / (value(node).visits + 1)

function selection_value(node::TreeNode, mcts::Planner)
    p = AbstractTrees.parent(node)
    p_i = 1 / length(children(p))
    value(node).value + mcts.temperature * p_i * sqrt(value(p).visits) / (value(node).visits + 1)
end

function select_action_id(node, mcts::Planner)::Union{Nothing,Int}
    if isleaf(node)
        return nothing
    end
    selection_values = [selection_value(n, mcts) for n in children(node)]
    return rand_max(selection_values)
end

select_action_id(mcts::Planner) = select_action_id(mcts.tree, mcts)

function step_empty_state_node!(node, env)
    if isnothing(value(node).state)
        pn = AbstractTrees.parent(node)
        setstate!(env, pn.value.state)
        action_id = findfirst(map(n -> n === node, children(pn)))
        _, reward, _, _ = step!(env, action_space(env)[action_id])
        value(node).state = state(env)
        value(node).reward = reward
    end
end

function select!(mcts::Planner)
    node = mcts.tree
    reward_multiplier = 1.0
    total_reward = 0.0
    while !isleaf(node) && !isdone(value(node).state)
        total_reward += reward_multiplier * value(node).reward
        reward_multiplier *= mcts.γ
        action_id = select_action_id(node, mcts)
        node = children(node)[action_id]
    end
    step_empty_state_node!(node, mcts.env)
    total_reward += reward_multiplier * value(node).reward
    node, total_reward
end


function expand!(node::TreeNode, mcts::Planner)
    done = isdone(value(node).state)
    if isleaf(node) && (!done || isroot(node)) && depth(node) <= mcts.horizon
        leafs = length(action_space(mcts.env))
        addchildren!(node, (NodeValue{typeof(state(mcts.env))}() for _ = 1:leafs))
        child_node = rand(children(node))
        step_empty_state_node!(child_node, mcts.env)
        return child_node
    end
    return node # it is node at terminal state
end

function simulate!(node::TreeNode, total_reward::Float64, mcts::Planner)
    env = mcts.env
    actions = action_space(env)
    d = depth(node)
    reward_multiplier = mcts.γ^(d - 1)
    done = isdone(value(node).state)
    setstate!(env, value(node).state)
    total_reward += reward_multiplier * value(node).reward
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
    while true
        value(node).visits += 1
        value(node).value += (total_reward - value(node).value) / value(node).visits
        if isroot(node)
            break
        end
        node = AbstractTrees.parent(node)
    end
    nothing
end

function run!(mcts::Planner)
    node, total_reward = select!(mcts)
    new_node = expand!(node, mcts)
    total_reward = simulate!(new_node, total_reward, mcts)
    backpropagate!(new_node, total_reward, mcts)
end

function plan!(mcts::Planner)
    episodes = mcts.budget ÷ mcts.horizon
    for i = 1:episodes
        run!(mcts)
    end
    new_node = children(mcts.tree)[select_action_id(mcts)]
    step_empty_state_node!(new_node, mcts.env) # in rare cases this happens
    return new_node
end

function run_planner!(mcts; render_env=false, max_steps=Inf)
    completed_episodes = 0
    if render_env
        render!(mcts.env)
    end
    steps = 0
    while !isdone(value(mcts.tree).state)
        new_root = plan!(mcts)
        if render_env
            setstate!(mcts.env, value(new_root).state)
            render!(mcts.env)
        end
        completed_episodes += 1
        steps += 1
        steps < max_steps || break
        # reset strategy
        reset!(mcts.tree, NodeValue(value(new_root).state))
    end
    # @info "Planner completed $completed_episodes episodes"
    return completed_episodes
end
