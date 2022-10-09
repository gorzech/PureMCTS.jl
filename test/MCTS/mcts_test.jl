@testset "select on new tree just return root" begin
    tree = Planner([1.0])
    node = @test_nowarn select(tree)
    @test isroot(node)
end

