using Random, Distributions, FunctionalCollections


# Helper functions that help map Julia functions to FOPPL syntax

function vector(x...)
    vec = @Persistent []
    for i in x
        vec = push(vec,i)
    end
    return vec
end

# TODO: test getting from multidimensional vector or vector of vectors
function getidx(A,i)
    if typeof(A) <: PersistentHashMap{Any,Any}
        return A[i]
    else
        return A[i+1]
    end
end

# TODO: In the future check if this mutation will impact autodifferentiation
function putidx(A,i,x)
    if typeof(A) <: PersistentHashMap{Any,Any}
        A = assoc(A,i,x)
    else
        A = assoc(A,i+1,x)
    end
    return A
end

function hashmap(x...)
    i = 1
    dict = @Persistent Dict()
    while i < length(x)
        dict = assoc(dict, x[i], x[i+1])
        i += 2
    end
    return dict
end

function conj_helper(A,x)
    B = @Persistent [x]
    return append(B,A)
end

#=
function hashmap(x...)
    i = 1
    dict = Dict()
    while i < length(x)
        push!(dict, x[i]=>x[i+1])
        i += 2
    end
    return dict
end
=#

function second(A)
    return A[2]
end

function rest(A)
    return A[2:end]
end

function Discrete(p)
    p = p ./ sum(p)
    return Categorical(p...)
end

# TODO: In the future, make these mat functions more general

function mat_transpose(A)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    # perform transpose
    mat_A = transpose(mat_A)
    # convert back to vector of vectors
    out_A = []
    for i in 1:size(mat_A)[1]
        push!(out_A,mat_A[i,:])
    end
    return out_A
end

function mat_mul(A,B)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    mat_B = [Bi[j] for Bi in B, j in 1:length(B[1])]
    # perform multiplication
    mat_AB = mat_A*mat_B
    # convert back to vector of vectors
    out_AB = []
    for i in 1:size(mat_AB)[1]
        push!(out_AB,mat_AB[i,:])
    end
    return out_AB
end

function mat_add(A,B)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    mat_B = [Bi[j] for Bi in B, j in 1:length(B[1])]
    # perform addition
    mat_AplusB = mat_A+mat_B
    # convert back to vector of vectors
    out_AplusB = []
    for i in 1:size(mat_AplusB)[1]
        push!(out_AplusB,mat_AplusB[i,:])
    end
    return out_AplusB
end


function mat_sub(A,B)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    mat_B = [Bi[j] for Bi in B, j in 1:length(B[1])]
    # perform addition
    mat_AsubB = mat_A-mat_B
    # convert back to vector of vectors
    out_AsubB = []
    for i in 1:size(mat_AsubB)[1]
        push!(out_AsubB,mat_AsubB[i,:])
    end
    return out_AsubB
end

function mat_tanh(A)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    # perform tanh
    mat_A = tanh.(mat_A)
    # convert back to vector of vectors
    out_A = []
    for i in 1:size(mat_A)[1]
        push!(out_A,mat_A[i,:])
    end
    return out_A
end

function mat_repmat(A,d...)
    # convert to matrix
    mat_A = [Ai[j] for Ai in A, j in 1:length(A[1])]
    # perform repmat
    mat_A = repeat(mat_A, outer=d)
    # convert back to vector of vectors
    out_A = []
    for i in 1:size(mat_A)[1]
        push!(out_A,mat_A[i,:])
    end
    return out_A
    return
end

# Helper function for casting input type to Dirichlet distribution
dirichlet(α) = Dirichlet(float.(α))

and(x,y) = x && y

or(x,y) = x || y

if_opp(cond,e1,e2) = cond ? e1 : e2

function sample_dist(dist)
    if typeof(dist) <: DiscreteNonParametric
        return rand(dist) - 1
    else
        return rand(dist)
    end
end

function observe_dist(dist,c)
    return c
end

# List of primitives that map FOPPL syntax to Julia functions

primitives = Dict(
    "abs"           => abs,
    "+"             => +,
    "-"             => -,
    "sqrt"          => sqrt,
    "*"             => *,
    "/"             => /,
    ">"             => >,
    "<"             => <,
    ">="            => >=,
    "<="            => <=,
    "="             => ==,
    "log"           => log,
    "and"           => and,
    "or"            => or,
    "vector"        => vector,
    "get"           => getidx,
    "put"           => putidx,
    "first"         => first,
    "second"        => second,
    "rest"          => rest,
    "last"          => last,
    "peek"          => last,
    "empty?"        => isempty,
    "conj"          => conj_helper,
    "append"        => append,
    "hash-map"      => hashmap,
    "sample"        => sample_dist,
    "observe"       => observe_dist,
    "normal"        => Normal,
    "beta"          => Beta,
    "exponential"   => Exponential,
    "uniform"       => Uniform,
    "discrete"      => Discrete,
    "gamma"         => Gamma,
    "dirichlet"     => dirichlet,
    "uniform-continuous" => Uniform,
    "flip"          => Bernoulli,
    "mat-transpose" => mat_transpose,
    "mat-mul"       => mat_mul,
    "mat-add"       => mat_add,
    "mat-sub"       => mat_sub,
    "mat-tanh"      => mat_tanh,
    "mat-repmat"    => mat_repmat,
    "if"            => if_opp,
    "alpha"         => ""
)

function new_env(env)
    outer = copy(env)
    return Dict{String,Any}("outer" => outer)
end

function env_haskey(env,key)
    if haskey(env,key)
        return true
    elseif haskey(env,"outer")
        return env_haskey(env["outer"],key)
    else
        return false
    end
end

function env_find(env,key)
    if haskey(env,key)
        return env[key]
    else
        return env_find(env["outer"],key)
    end
end

function env_update(env,key,value)
    if haskey(env,key)
        env[key] *= value
        return
    else
        return push_address(env["outer"],key,value)
    end
end

function push_address(env,addr)
    return env_update(env,"alpha",addr)
end
