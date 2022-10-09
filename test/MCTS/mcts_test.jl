@testset "select on new tree just return root" begin
    tree = Planner(InvertedPendulumEnv())
    node = @test_nowarn select(tree)
    @test isroot(node)
end

@testset "check expand on new tree" begin
    tree = Planner(InvertedPendulumEnv())
    root_node = tree.tree
    expand!(root_node, tree)
    @test isroot(root_node)
    @test length(children(root_node)) > 0
    first_child = children(root_node)[1]
    @test !isroot(first_child)
    should_be_root = AbstractTrees.parent(first_child)
    @test isroot(should_be_root)
end