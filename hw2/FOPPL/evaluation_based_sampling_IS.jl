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

# likelihood of Discrete Distribution needs to account for 0 vs 1 based indexing
function log_prob(d1,c2)
    if typeof(d1) <: DiscreteNonParametric
        return log(pdf(d1,c2+1))
    else
        return log(pdf(d1,c2))
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

    # initialize logσ
    # Note, it is easier to pass around an array rather than making
    # evaluate_program output a tuple. TODO: Will this break AD?
    logσ = [0.0]

    return (evaluate_program(ast[end],l=Dict(),ρ=ρ,logσ=logσ), logσ[1])
end


function evaluate_program(ast;l=Dict(),ρ=Dict(),logσ=[0.0])
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # Handle the case were e is a vector
    if typeof(ast) <: Array
        if size(ast) == (1,)
            return evaluate_program(ast[1],l=l,ρ=ρ,logσ=logσ)


        # Handle "sample" statement
        elseif ast[1] == "sample"
            return sample_dist(evaluate_program(ast[2],l=l,ρ=ρ,logσ=logσ))


        # Handle "observe" statement
        elseif ast[1] == "observe"
            d1 = evaluate_program(ast[2],l=l,ρ=ρ,logσ=logσ)
            c2 = evaluate_program(ast[3],l=l,ρ=ρ,logσ=logσ)
            logσ[1] += log_prob(d1,c2)
            return evaluate_program(ast[3],l=l,ρ=ρ,logσ=logσ)


        # Handle "let" statement
        elseif ast[1] == "let"
            c = evaluate_program(ast[2][2],l=l,ρ=ρ,logσ=logσ)
            push!(l, ast[2][1] => c)
            return evaluate_program(ast[3], l=l,ρ=ρ,logσ=logσ)


        # (Lazily) Handle "if" statement
        elseif ast[1] == "if"
            if evaluate_program(ast[2],l=l,ρ=ρ,logσ=logσ)
                return evaluate_program(ast[3],l=l,ρ=ρ,logσ=logσ)
            else
                return evaluate_program(ast[4],l=l,ρ=ρ,logσ=logσ)
            end

        # Handle procedure calls
        elseif haskey(ρ,ast[1])

            f = ρ[ast[1]]

            l_local = Dict()
            for  i in 2:length(ast)
                push!(l_local, f[1][i-1] => evaluate_program(ast[i],l=l,ρ=ρ,logσ=logσ))
            end

            return evaluate_program(f[2],l=l_local,ρ=ρ,logσ=logσ)

        # Handle constant statement
        else
            return primitives[ast[1]](evaluate_program.(ast[2:end],l=l,ρ=ρ,logσ=logσ)...)
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
# Output: samples[i] ≡ (ith sample, logσ of ith sample)
function get_samples_eval(ast,n)
    samples = []
    for i in 1:n
        push!(samples,evaluate_program_init(ast))
    end
    return samples
end

function get_mean(samples,i)
    n = length(samples)
    sum = 0
    sumw = 0
    for i in 1:n
        W = samples[i][2]
        w = float(ℯ)^W
        sum += w*samples[i][1]
        sumw += w
    end
    return sum / sumw
end
