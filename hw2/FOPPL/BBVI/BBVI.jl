include("../primitives.jl")
# Get sample_from_joint function
include("../gibbs.jl")
using Zygote

function sample_dist(dist)
    if typeof(dist) <: DiscreteNonParametric
        return rand(dist) - 1
    else
        return rand(dist)
    end
end

# likelihood of Discrete Distribution needs to account for 0 vs 1 based indexing
function logprob(d,c)
    if typeof(d) <: DiscreteNonParametric
        return logpdf(d,c+1)
    else
        return logpdf(d,c)
    end
end

function get_traversal(ast)
    # Create a graph g
    nodes = ast[2]["V"]
    num_nodes = length(nodes)
    arcs = ast[2]["A"]

    nodes_map = Dict()
    for i in 1:num_nodes
        push!(nodes_map, nodes[i] => i)
    end

    g = SimpleDiGraph(num_nodes)

    for pair in arcs
        if haskey(nodes_map, pair[1])
            out_node = nodes_map[pair[1]]
            for in_node in pair[2]
                if haskey(nodes_map,in_node)
                    add_edge!(g,out_node,nodes_map[in_node])
                end
            end
        end
    end

    # Compute ancestral sampling traversal order
    traversal = topological_sort_by_dfs(g)
    return traversal
end


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

function eval_BBVI_step(ast,traversal,l,dists,logW,q,λ,G)
    nodes = ast[2]["V"]
    for i in traversal
        v = nodes[i]
        # Two cases, either the variable is unobserved or observed
        if v[1:6] == "sample"
            # get prior distribution
            p = dists[v](l)
            # sample from proposal distribution
            q_dist = q[v](λ[v]...)
            c = sample_dist(q_dist)
            l[v] = c
            # get gradient
            logprob_λ = prms -> logprob(q[v](prms...),c)
            grad = [gradient(logprob_λ,λ[v])[1]...]
            push!(G, v => grad)
            # update logW
            logW[1] += (logprob(p,c) - logprob(q_dist,c))
        else
            p = dists[v](l)
            logW[1] += logprob(p,l[v])
        end
    end
    return
end

# Returns elbo_gradient ĝ, sharing the same datastructure as λ
function elbo_gradients(weighted_grads,λ)
    # Initialize ĝ
    ĝ = Dict()
    for v in λ
        push!(ĝ,v[1] => zeros(size(λ[v[1]])))
    end

    G = [y[1] for y in weighted_grads]
    logW = [y[2] for y in weighted_grads]
    L = length(G)
    for v in ĝ
        for i in 1:L
            if haskey(G[i],v[1])
                ĝ[v[1]] += logW[i] .* G[i][v[1]]
            end
        end
        ĝ[v[1]] = ĝ[v[1]] ./ L
    end

    #TODO handle b̂
    return ĝ
end

# λ is updated by projected SGD using gradient ĝ
function optimizer_step!(λ,q,ĝ,t)
    for v in λ
        λ[v[1]] += (1/t) .* ĝ[v[1]]

        # Projection step
        if q[v[1]] <: Normal
            if λ[v[1]][2] < 0.0
                λ[v[1]][2] = 1e-3
            end
        end
    end
end

# ast is the compiled program by Daphne in graphical form,
# n is the number of samples, T is the number of optimization steps,
# L is the number of samples for per iteration of the optimization
function BBVI(ast,n,T,L)

    # Optimization Step

    # TODO: refactor to do this better, including handling of ρ
    # Initialize variable assignment by sampling from joint
    l = Dict()
    ρ = Dict()
    sample_from_joint(ast,l,ρ)

    # Get traversal order for ancestral sampling
    traversal = get_traversal(ast)

    # distribution maps for each random variable
    dists = get_dists(ast)

    # Initialize proposal distributions
    # TODO: decide if these dictionaries should be combined
    q = Dict()
    λ = Dict()
    nodes = ast[2]["V"]
    for v in nodes
        if v[1:6] == "sample"
            # Create a proposal distribution
            d = dists[v](l)

            d_type = typeof(d)
            push!(q,v => d_type)

            parameters = params(d)
            push!(λ,v => [parameters...])
        end
    end

    # Do the actual optimization
    for t in 1:T
        weighted_grads = []
        for s in 1:L
            # Initialize logW and G for new sample
            logW = [0.0]
            G = Dict()
            eval_BBVI_step(ast,traversal,l,dists,logW,q,λ,G)
            push!(weighted_grads, (G,logW[1]))
        end
        ĝ = elbo_gradients(weighted_grads,λ)
        optimizer_step!(λ,q,ĝ,t)
    end

    return q,λ
    # Sampling Step
    samples = []
    # TODO
    return samples

end
