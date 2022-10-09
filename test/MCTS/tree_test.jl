@testset "Initialize tree with single node" begin
    @test_nowarn TreeNode{Int}(0)
end
