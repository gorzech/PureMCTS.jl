@testset "Initialize empty node" begin
    @test_nowarn Node()
end

@testset "Initialize empty tree node" begin
    @test_nowarn TreeNode()
    @test_nowarn TreeNode(1)
end

@testset "Initialize new tree with only root node" begin
    @test_nowarn Tree()
end