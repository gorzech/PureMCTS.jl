mutable struct NodeValue
    state::Union{Nothing,Vector{Float64}}
    value::Float64
    visits::Int
    NodeValue() = new(nothing, 0, 0)
    NodeValue(state) = new(state, 0, 0)
end

struct Planner
    env<:AbstractEnvironment
    temperature::Float64
    γ::Float64
    tree::TreeNode{NodeValue}

    function Planner(env::AbstractEnvironment, temperature = 100.0, γ = 0.9)
        
        new(TreeNode{NodeValue}(NodeValue(initial_state)), temperature, γ)
    end
end

selection_value(node::TreeNode, mcts::Planner) =
    node.data.value + mcts.temperature / (node.data.visits + 1)

function select(mcts::Planner)
    node = mcts.tree
    while !isnothing(children(node))
        selection_values = selection_value.(children(node), mcts)
        node = children(node)[rand_max(selection_values)]
    end
    node
end

function expand!(node::TreeNode, leafs::Int)
    node.children = [TreeNode{NodeValue()} for _ = 1:leafs]
end
