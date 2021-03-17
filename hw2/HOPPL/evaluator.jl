include("primitives.jl")

function evaluate(exp;env=nothing)
    if env == nothing
        env = copy(primitives)
    end

    if typeof(exp) <: Array

        if length(exp) > 1

            # TODO: handle alpha
            if exp[1] == "fn"
                # Check if all variables are bound
                if all([env_haskey(env,exp[2][i]) for i in 2:length(exp[2])])
                    # TODO handle observe statements, right now end == 3 iff
                    # no observe statement
                    return evaluate(exp[end],env=env)
                else
                    return exp
                end

            elseif exp[1] == "if"
                if evaluate(exp[2],env=env)
                    return evaluate(exp[3],env=env)
                else
                    return evaluate(exp[4],env=env)
                end

            elseif typeof(exp[1]) <: String
                args = [evaluate(exp[i],env=env) for i in 3:length(exp)]
                f = env_find(env,exp[1])

                # Handle case where f is defined in the evaluate_program
                if typeof(f) <: Array
                    local_env = new_env(env)
                    for i in 3:length(exp)
                        push!(local_env, f[2][i-1] => args[i-2])
                    end
                    return evaluate(f,env=local_env)

                # Otherwise f is a primitive function
                else
                    return env_find(env,exp[1])(args...)
                end

            elseif typeof(exp[1]) <: Array
                args = []
                for i in 3:length(exp)
                    # Handle case where argument is a function
                    if exp[i][1] == "fn"
                        push!(args,exp[i])
                    # Otherwise evaluate the expression
                    else
                        push!(args,evaluate(exp[i],env=env))
                    end
                end

                # The case where exp[1] is an fn
                if exp[1][1] == "fn"
                    local_env = new_env(env)
                    for i in 3:length(exp)
                        push!(local_env, exp[1][2][i-1] => args[i-2])
                    end
                    return evaluate(exp[1],env=local_env)

                # The case where exp[1] is non-fn statement and so we don't know
                # how to bind variables
                else
                    f = evaluate(exp[1],env=env)

                    if typeof(f) <: Array
                        local_env = new_env(env)
                        for i in 3:length(exp)
                            push!(local_env, f[2][i-1] => args[i-2])
                        end
                        return evaluate(f,env=local_env)

                    # Otherwise f is a primitive function
                    else
                        return env_find(env,exp[1])(args...)
                    end
                end

            end

        end

    else
        if env_haskey(env,exp)
            return env_find(env,exp)
        else
            return exp
        end
    end
end


function evaluate_n(exp;n=1)
    samples = []
    for i in 1:n
        push!(samples,evaluate(exp))
    end
    return samples
end
