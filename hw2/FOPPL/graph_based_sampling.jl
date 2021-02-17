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
