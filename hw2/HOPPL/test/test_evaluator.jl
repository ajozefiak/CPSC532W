
# All testing is performed on the given and new test files,
# already converted to JSON format by Daphne

using Test, HypothesisTests

include("../evaluator.jl")
include("../../FOPPL/read_json.jl")

# Deterministic Tests
@testset "Deterministic Tests" begin

    test_path = "../CS532-HW5/programs/tests/deterministic/test_"
    truth_path = "../CS532-HW5/programs/tests/deterministic/test_"

    @testset "Deterministic" begin
        for i in 1:12
            exp = read_json(test_path*string(i)*".json")
            truth = read_json(truth_path*string(i)*".truth")
            res = evaluate(exp)
            @test isapprox(res,truth,rtol=1e-6)
        end
    end

    @testset "hash-map" begin
        exp = read_json(test_path*string(13)*".json")
        truth = read_json(truth_path*string(13)*".truth")
        res = evaluate(exp)
        @test res == Dict(6 => 2, 1 => 3.2)
    end

    @testset "Deterministic control flow" begin
        for i in 14:15
            exp = read_json(test_path*string(i)*".json")
            truth = read_json(truth_path*string(i)*".truth")
            res = evaluate(exp)
            @test res == truth
        end
    end

end

@testset "HOPPL Deterministic Tests" begin

    test_path = "../CS532-HW5/programs/tests/hoppl-deterministic/test_"
    truth_path = "../CS532-HW5/programs/tests/hoppl-deterministic/test_"

    @testset "HOPPLDeterministic" begin
        for i in 1:12
            exp = read_json(test_path*string(i)*".json")
            truth = read_json(truth_path*string(i)*".truth")
            res = evaluate(exp)
            @test isapprox(res,truth,rtol=1e-6)
        end
    end
end

@testset "Probabilistic Tests" begin

    test_path = "../CS532-HW5/programs/tests/probabilistic/test_"
    truth_path = "../CS532-HW5/programs/tests/probabilistic/test_"

    num_samples = 1e4
    max_p_value = 1e-4

    @testset "Standard Distributions" begin
        for i in [1 2 3 4 6]
            exp = read_json(test_path*string(i)*".json")
            truth = read_json(truth_path*string(i)*".truth")

            dist = primitives[truth[1]](truth[2:end]...)

            samples = evaluate_n(exp,n=num_samples)
            # Need to cast as Float64 for KS Test
            samples = Float64[samples...]
            p_val = pvalue(ExactOneSampleKSTest(samples,dist))

            @test p_val > max_p_value
        end
    end
    @testset "Mixture Model" begin

        # Hard-coded because there is only one mixture model test
        exp = read_json(test_path*"5.json")
        dist = MixtureModel(Normal[Normal(-1,0.3),Normal(1,0.3)],[0.1, 0.9])

        samples = evaluate_n(exp,n=num_samples)
        # Need to cast as Float64 for KS Test
        samples = Float64[samples...]
        p_val = pvalue(ExactOneSampleKSTest(samples,dist))

        @test p_val > max_p_value

    end
end
