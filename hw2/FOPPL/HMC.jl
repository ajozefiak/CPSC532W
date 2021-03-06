include("primitives.jl")
# gibbs.jl gives functionality to sample from posterior once
include("gibbs.jl")

using Zygote

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



####################################
# Real stuff
####################################
# TODO: Fix abuse of notation with the use of AST
# TODO: Should use of a dictionary for variables be changed to a fixed length array?
# The use of a dictionary is an artifact from the evaluation based samplers.

# Given AST for a probability distribution of a node,
# return a differentiable map from all nodes, in particular parent nodes,
# to a distribution object
function create_map_to_dist(ast,nodes)
    if typeof(ast) <: Array
        if size(ast) == (1,)
            temp_map = create_map_to_dist(ast[1],nodes)
            return V -> temp_map(V)

        else
            temp_maps = [create_map_to_dist(sub_ast,nodes) for sub_ast in ast[2:end]]
            opp = primitives[ast[1]]
            return V -> opp([f(V) for f in temp_maps]...)
        end
    else
        if string(ast) ∈ nodes
            return V -> V[ast]
        else
            return V -> ast
        end
    end
end

# Given AST of the graphical model, return a disctionary
# that maps each node to a map that generates its distribution object
function get_dists(ast)
    dists = Dict()
    nodes = ast[2]["V"]
    for v in nodes
        temp_map = create_map_to_dist(ast[2]["P"][v][2],nodes)
        push!(dists, v => temp_map)
    end
    return dists
end

# Compute kinetic energy given momentum R and mass matrix M
function compute_K(R,M)
    # TODO: could this be computed better if M is diagonal?
    if size(M) == (1,)
        return (0.5*R'*R)/M[1]
    elseif typeof(M) <: Vector
        return 0.5*R'*(R ./ M)
    else
        return 0.5 * R'*(M \ R)
    end
end

# Compute potential energy given variable assignment, l, and map
# from variables to distributions
function compute_U(l,dists)
    E_U = 0
    for v in keys(l)
        E_U -= log_prob(dists[v](l),l[v])
    end
    return E_U
end

function compute_∇U(U,l,n)
    ∇U = zeros(n)
    ∇U_dict = U'(l)

    k = 1
    for v in l
        if v[1][1:6] == "sample" && haskey(∇U_dict,v[1])
            ∇U[k] = ∇U_dict[v[1]]
            k += 1
        end
    end
    return ∇U
end

# TODO: deal with mass
function update_l!(l,R,ϵ)
    k = 1
    for v in l
        if v[1][1:6] == "sample"
            l[v[1]] = v[2] + ϵ*R[k]
            k +=1
        end
    end
    return l
end

function leapfrog(l,R,T,ϵ,U)
    for t in 1:T
        ∇U = compute_∇U(U,l,length(R))
        R .= R - 0.5*ϵ*∇U

        update_l!(l,R,ϵ)

        ∇U = compute_∇U(U,l,length(R))
        R .= R - 0.5*ϵ*∇U
    end
    return l,R
end

# ast is the graphical model
# n is the number of samples
# T, ϵ, and M are the hyperparameters for HMC
# Note, M must have the proper size
function HMC(ast,n,T,ϵ,M)

    samples = []

    # Remove later
    neg_log_density = []

    # Initialize a dictionary for variables
    l = Dict()

    # TODO: get rid of ρ here
    ρ = Dict()
    # TODO refactor sample_from_joint
    # Get initial samples into l
    sample_from_joint(ast,l,ρ)

    dists = get_dists(ast)

    num_sampled_vars = size(M)[1]

    # Define the Hamiltonian and Potential and Kinetic energy functions
    # K(R) could be defined better
    K(R) = compute_K(R,M)
    U(l) = compute_U(l,dists)
    H(l,R) = U(l) + K(R)

    # Ensure that initial sample is does not have infinite NLL,
    # otherwise it cannot be differentiated
    while U(l) ==  Inf
        sample_from_joint(ast,l,ρ)
    end

    # Get n samples
    for i in 1:n
        R = rand(MvNormal(zeros(num_sampled_vars), M))
        l_temp = copy(l)
        R_temp = copy(R)

        # Modification here occurs inplace
        leapfrog(l_temp,R_temp,T,ϵ,U)

        u = rand(Uniform(0,1))

        if u < exp(-H(l_temp,R_temp) + H(l,R))
            # update l, TODO: make this faster/better
            for v in l
                l[v[1]] = l_temp[v[1]]
            end
            push!(samples,evaluate_program_graph(ast[3],l=l,ρ=ρ))
        else
            if !isempty(samples)
                push!(samples,samples[end])
            else
                push!(samples,evaluate_program_graph(ast[3],l=l,ρ=ρ))
            end
        end
        push!(neg_log_density,U(l))
    end

    return samples, neg_log_density
end
