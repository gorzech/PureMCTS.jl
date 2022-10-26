mutable struct TreeNode{T}
    value::T
    parent::Union{Nothing,TreeNode{T}}
    children::Union{Tuple{},Vector{TreeNode{T}}}
    function TreeNode{T}(data::T, parent = nothing, children = ()) where {T}
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

function addchildren!(node::TreeNode{T}, childrens) where {T}
    node.children = [TreeNode{T}(c, node) for c in childrens]
    nothing
end

value(node::TreeNode) = node.value

function reset!(node::TreeNode{T}, value::T) where {T}
    node.children = ()
    node.value = value
    nothing
end
