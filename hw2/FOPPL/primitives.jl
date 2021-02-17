using Random, Distributions

# Helper functions that help map Julia functions to FOPPL syntax

function vector(x...)
    vec = []
    for i in x
        push!(vec,i)
    end
    return vec
end

# TODO: test getting from multidimensional vector or vector of vectors
function getidx(A,i)
    if typeof(A) <: Dict
        return A[i]
    else
        return A[i+1]
    end
end

# TODO: In the future check if this mutation will impact autodifferentiation
function putidx(A,i,x)
    if typeof(A) <: Dict
        push!(A, i => x)
    else
        A[i+1] = x
    end
    return A
end

function hashmap(x...)
    i = 1
    dict = Dict()
    while i < length(x)
        push!(dict, x[i]=>x[i+1])
        i += 2
    end
    return dict
end

function second(A)
    return A[2]
end

function rest(A)
    return A[2:end]
end

function Discrete(p)
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

# List of primitives that map FOPPL syntax to Julia functions

primitives = Dict(
    "+"           => +,
    "-"           => -,
    "sqrt"        => sqrt,
    "*"           => *,
    "/"           => /,
    ">"           => >,
    "<"           => <,
    ">="          => >=,
    "<="          => <=,
    "vector"      => vector,
    "get"         => getidx,
    "put"         => putidx,
    "first"       => first,
    "second"      => second,
    "rest"        => rest,
    "last"        => last,
    "append"      => push!,
    "hash-map"    => hashmap,
    "normal"      => Normal,
    "beta"        => Beta,
    "exponential" => Exponential,
    "uniform"     => Uniform,
    "discrete"    => Discrete,
    "mat-transpose" => mat_transpose,
    "mat-mul"       => mat_mul,
    "mat-add"       => mat_add,
    "mat-sub"       => mat_sub,
    "mat-tanh"      => mat_tanh,
    "mat-repmat"    => mat_repmat
)
