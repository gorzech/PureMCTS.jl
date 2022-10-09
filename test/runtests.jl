# using Revise
using AbstractTrees
using Environments
using PureMCTS
using Test

@testset "PureMCTS.jl" begin
    # Write your tests here.
    include("helper_test.jl")
    include("MCTS/tree_test.jl")
    include("MCTS/mcts_test.jl")
end
