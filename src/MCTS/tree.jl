mutable struct TreeNode{T}
    value :: T
    parent :: Union{Nothing, TreeNode{T}}
    children :: Union{Tuple{}, Vector{TreeNode{T}}}
    function TreeNode{T}(data, parent=nothing, children=()) where T
        new{T}(data, parent, children)
    end
end 

AbstractTrees.ParentLinks(::Type{TreeNode}) = StoredParents()

AbstractTrees.children(node::TreeNode) = node.children

AbstractTrees.parent(node::TreeNode) = node.parent

nodevalue(value) = value
AbstractTrees.nodevalue(node::TreeNode) = nodevalue(node.value)

function depth(node::TreeNode) 
    if isroot(node)
        1
    else 
        depth(AbstractTrees.parent(node)) + 1
    end
end

isleaf(node::TreeNode) = isempty(node.children)