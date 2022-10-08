struct Node
    value :: Float64
    reward :: Float64
    state :: Union{Nothing, Vector{Float64}}
end

Node() = Node(0.0, 0.0, nothing)

struct TreeNode
    node :: Node
    visits :: Int
    parent :: Union{Nothing, Int}
    children :: Union{Nothing, Vector{Int}}
end

TreeNode() = TreeNode(Node(), 0, nothing, nothing)
TreeNode(parent :: Int) = TreeNode(Node(), 0, parent, nothing)

struct Tree
    nodes :: Vector{TreeNode}
end

Tree() = Tree([TreeNode()])

get_node(tree::Tree, id::Int) = tree.nodes[id]