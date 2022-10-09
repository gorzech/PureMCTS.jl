mutable struct NodeValue
    state::Union{Nothing,Vector{Float64}}
    reward::Float64
    value::Float64
    visits::Int
    NodeValue() = new(nothing, 0, 0, 0)
    NodeValue(state) = new(state, 0, 0, 0)
end

nodevalue(nv::NodeValue) = nv.value

struct Planner
    env::AbstractEnvironment
    temperature::Float64
    γ::Float64
    tree::TreeNode{NodeValue}
end

function Planner(env::AbstractEnvironment, seed::Union{Nothing, Int} = nothing, temperature = 100.0, γ = 0.9)
    reset!(env, seed)
    initial_state = state(env)
    nv = NodeValue(initial_state)
    tn = TreeNode{NodeValue}(nv)
    Planner(env, temperature, γ, tn)
end

selection_value(node::TreeNode, mcts::Planner) =
    node.data.value + mcts.temperature / (node.data.visits + 1)

function select(mcts::Planner)
    node = mcts.tree
    while !isempty(children(node))
        selection_values = selection_value.(children(node), mcts)
        node = children(node)[rand_max(selection_values)]
    end
    node
end

function expand!(node::TreeNode, mcts::Planner)
    leafs = length(action_space(mcts.env))
    node.children = [TreeNode{NodeValue}(NodeValue(), node) for _ = 1:leafs]
    nothing
end

