using LightGraphs

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

##############################################################################
##############################################################################

# Takes as input a directed graph and returns the sampling order according to
# ancestral sampling
function compute_traversal(g)
    g_temp = copy(g)
    num_nodes = nv(g_temp)
    traversal = []
    while length(traversal) < num_nodes
        next_samples = findall(v->v==0,indegree(g_temp))
        # Remove nodes already in traversal
        next_samples = setdiff(next_samples,traversal)
        traversal = vcat(traversal, next_samples)

        # remove edges, note this should be implemented better
        for i in 1:num_nodes
            for v in next_samples
                rem_edge!(g_temp,v,i)
            end
        end
    end
    return traversal
end

function evaluate_program_graph(ast;l=Dict(),ρ=Dict())
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # Handle the case were e is a vector
    if typeof(ast) <: Array
        if size(ast) == (1,)
            return evaluate_program_graph(ast[1],l=l,ρ=ρ)


        # Handle "sample" statement
    elseif ast[1] == "sample*"
            return sample_dist(evaluate_program_graph(ast[2],l=l,ρ=ρ))


        # Handle "observe" statement
    elseif ast[1] == "observe*"
            return evaluate_program_graph(ast[3],l=l,ρ=ρ)


        # Handle "let" statement
        elseif ast[1] == "let"
            c = evaluate_program_graph(ast[2][2],l=l,ρ=ρ)
            push!(l, ast[2][1] => c)
            return evaluate_program_graph(ast[3], l=l,ρ=ρ)


        # (Lazily) Handle "if" statement
        elseif ast[1] == "if"
            if evaluate_program_graph(ast[2],l=l)
                return evaluate_program_graph(ast[3],l=l,ρ=ρ)
            else
                return evaluate_program_graph(ast[4],l=l,ρ=ρ)
            end

        # Handle procedure calls
        elseif haskey(ρ,ast[1])

            f = ρ[ast[1]]

            l_local = Dict()
            for  i in 2:length(ast)
                push!(l_local, f[1][i-1] => evaluate_program_graph(ast[i],l=l,ρ=ρ))
            end

            return evaluate_program_graph(f[2],l=l_local,ρ=ρ)

        # Handle constant statement
        # First handle hash-map case, potentially fix the code here
        elseif ast[1] == "hash-map"
            return hashmap(vcat(ast[2:end]...)...)

        else
            return primitives[ast[1]](evaluate_program_graph.(ast[2:end],l=l,ρ=ρ)...)
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

function sample_from_joint(ast)
    # Handle functions
    ρ = Dict()

    # Create a graph g
    nodes = ast[2]["V"]
    num_nodes = length(nodes)
    arcs = ast[2]["A"]
    num_arcs = length(arcs)

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
    traversal = compute_traversal(g)

    # Setup a dictionary for sampled values
    l = Dict()
    for i in traversal
        compute_SampObs = evaluate_program_graph(ast[2]["P"][nodes[i]],l=l,ρ=ρ)
        push!(l,nodes[i] => compute_SampObs)
    end

    return evaluate_program_graph(ast[3],l=l,ρ=ρ)

end


# TODO: In the future make this more efficient,
# and use an equivalent of a stream in Julia
function get_samples_graph(ast,n)
    samples = []
    for i in 1:n
        push!(samples,sample_from_joint(ast))
    end
    return samples
end


##############################################
# Gibbs sampling with MH
##############################################

# NOTE: assumes l and ρ are already initialized
function sample_from_joint(ast,l,ρ)
    # Handle functions

    # Create a graph g
    nodes = ast[2]["V"]
    num_nodes = length(nodes)
    arcs = ast[2]["A"]
    num_arcs = length(arcs)

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
    traversal = compute_traversal(g)

    # Setup a dictionary for sampled values

    for i in traversal
        compute_SampObs = evaluate_program_graph(ast[2]["P"][nodes[i]],l=l,ρ=ρ)
        push!(l,nodes[i] => compute_SampObs)
    end

    return evaluate_program_graph(ast[3],l=l,ρ=ρ)

end

function accept(ast,l,ρ,x,x_new)
    logα  = 0
    l_new = copy(l)
    l_new[x[1]] = x_new

    children = ast[2]["A"][x[1]]
    for  child in children
        logα += log_prob(evaluate_program_graph(ast[2]["P"][child][2],l=l_new,ρ=ρ),l_new[child])
        logα -= log_prob(evaluate_program_graph(ast[2]["P"][child][2],l=l,ρ=ρ),l[child])
    end

    return float(ℯ)^logα
end

function gibbs_step(ast,l,ρ)
    # Iterate over unobserved variables

    for x in l
        # Only iterate over unobserved RVs
        if x[1][1:6] == "sample"
            # x[1] is the variable key in the dictionary, x[2] is the value

            # TODO: refactor to make this nicer to look at
            x_new = sample_dist(evaluate_program_graph(ast[2]["P"][x[1]][2],l=l,ρ=ρ))
            α = accept(ast,l,ρ,x,x_new)
            u = rand(Uniform(0,1))
            if u < α
                l[x[1]] = x_new
            end
        end
    end
    # Return the samples after having performed a single gibbs step
    return evaluate_program_graph(ast[3],l=l,ρ=ρ)
end

###############################################################################
# Helpers for log density
###############################################################################

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

# Compute potential energy given variable assignment, l, and map
# from variables to distributions
function compute_U(l,dists)
    E_U = 0
    for v in keys(l)
        E_U -= log_prob(dists[v](l),l[v])
    end
    return E_U
end

###############################################################################
###############################################################################

function gibbs(ast,n)
    samples = []
    # Sample initial sample from prior distribution
    # Need to return dict of variable assignment
    l = Dict()
    ρ = Dict()
    # TODO refactor sample_from_joint
    init_sample = sample_from_joint(ast,l,ρ)

    neg_log_density = []
    dists = get_dists(ast)
    U(l) = compute_U(l,dists)

    # get n samples using gibbs sampling
    for i in 1:n
        # TODO: write gibbs_step, by refactoring earlier code
        push!(samples,gibbs_step(ast,l,ρ))
        push!(neg_log_density,U(l))
    end
    return samples, neg_log_density
end
