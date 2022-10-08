@testset "rand_max with only one maximum" begin
    using PureMCTS: rand_max
    @test rand_max([1]) == 1
    @test rand_max([0, 0, 0, 1, 7, 4, 3, 1]) == 5
    @test rand_max([1.0, 1.1, -100.0, 2e-1]) == 2
end

@testset "rand_max with more maxima" begin
    using PureMCTS: rand_max
    @test_nowarn rand_max([1, 1, 1])
    @test 0 < rand_max([1, 1, 1]) < 4
    @test any([1, 3] .== rand_max([1.0, -2.0, 1.0])) 
end