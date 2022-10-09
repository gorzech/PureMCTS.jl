mutable struct TreeNode{T}
    value :: T
    parent :: Union{Nothing, TreeNode{T}}
    children :: Union{Nothing, Vector{TreeNode{T}}}
    function TreeNode{T}(data, parent=nothing, children=nothing) where T
        new{T}(data, parent, children)
    end
end 

AbstractTrees.ParentLinks(::Type{TreeNode}) = StoredParents()

AbstractTrees.children(node::TreeNode) = node.children

AbstractTrees.parent(node::TreeNode) = node.parent

AbstractTrees.nodevalue(node::TreeNode) = node.data
