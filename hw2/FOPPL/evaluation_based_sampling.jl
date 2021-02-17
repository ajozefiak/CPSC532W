# from daphne import daphne
# from tests import is_tol, run_prob_test,load_truth

# Includes a map to primitives
include("primitives.jl")

# Sampling Discrete Distribution needs to account for 0 vs 1 based indexing
function sample_dist(dist)
    if typeof(dist) <: DiscreteNonParametric
        return rand(dist) - 1
    else
        return rand(dist)
    end
end

function evaluate_program_init(ast)
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # ρ stores all functions
    ρ = Dict()

    # Handle the case where we have at least one function
    if length(ast) > 1
        for i in 1:length(ast)-1
            push!(ρ, ast[i][2] => ast[i][3:4])
        end
    end

    return evaluate_program(ast[end],l=Dict(),ρ=ρ)
end


function evaluate_program(ast;l=Dict(),ρ=Dict())
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # Handle the case were e is a vector
    if typeof(ast) <: Array
        if size(ast) == (1,)
            return evaluate_program(ast[1],l=l,ρ=ρ)


        # Handle "sample" statement
        elseif ast[1] == "sample"
            return sample_dist(evaluate_program(ast[2],l=l,ρ=ρ))


        # Handle "observe" statement
        elseif ast[1] == "observe"
            return evaluate_program(ast[3],l=l,ρ=ρ)


        # Handle "let" statement
        elseif ast[1] == "let"
            c = evaluate_program(ast[2][2],l=l,ρ=ρ)
            push!(l, ast[2][1] => c)
            return evaluate_program(ast[3], l=l,ρ=ρ)


        # (Lazily) Handle "if" statement
        elseif ast[1] == "if"
            if evaluate_program(ast[2],l=l)
                return evaluate_program(ast[3],l=l,ρ=ρ)
            else
                return evaluate_program(ast[4],l=l,ρ=ρ)
            end

        # Handle procedure calls
        elseif haskey(ρ,ast[1])

            f = ρ[ast[1]]

            l_local = Dict()
            for  i in 2:length(ast)
                push!(l_local, f[1][i-1] => evaluate_program(ast[i],l=l,ρ=ρ))
            end

            return evaluate_program(f[2],l=l_local,ρ=ρ)

        # Handle constant statement
        else
            return primitives[ast[1]](evaluate_program.(ast[2:end],l=l,ρ=ρ)...)
        end

    # Handle the case where e is a singleton
    else

        if haskey(l,ast)
            return l[ast]
        else
            return ast
        end
    end

end

# TODO: In the future make this more efficient,
# and use an equivalent of a stream in Julia
function get_samples_eval(ast,n)
    samples = Float64[]
    for i in 1:n
        push!(samples,evaluate_program_init(ast))
    end
    return samples
end
