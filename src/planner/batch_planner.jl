using ProgressMeter
using DataFrames: DataFrame, empty!, push!, nrow, rename!, antijoin
using CSV
using Dates
using Random: shuffle!

function planner_batch(;
    budget=round.(Int, exp10.(range(3, 6, 31))),
    γ=0.96:0.005:0.995,
    horizon=20:5:60,
    exploration_param=[8],
    seed_shift=1:10,
    file_name::Union{Nothing,String,Vector{String}}=nothing,
    starting_seed=4839633
)
    seeds = starting_seed .+ seed_shift

    it = collect(Base.product(horizon, budget, exploration_param, γ, seeds))
    shuffle!(it)
    if isnothing(file_name)
        return it
    end

    dff = if file_name isa String
        CSV.read(file_name * ".csv", DataFrame)
    else
        CSV.read(file_name, DataFrame)
    end
    
    df_it = DataFrame(it)
    rename!(df_it, ["horizon", "budget", "exploration_param", "gamma", "seed"])

    df_result = antijoin(df_it, dff, on=[:horizon, :budget, :exploration_param, :gamma, :seed])
    return Tuple.(eachrow(df_result))
end

function execute_batch(
    planner_it,
    file_name_prefix,
    opts=PendulumOpts();
    envfun=InvertedDoublePendulumEnv,
    show_progress=false,
    start_new_file=true
)
    file_name = file_name_prefix * ".csv"
    if start_new_file && isfile(file_name)
        @error "File already exists! Aborting."
        return nothing
    elseif !start_new_file && !isfile(file_name)
        @error "File did not exists, while it is expected! Aborting."
        return nothing
    end
    nit = length(planner_it)
    date_format = "yyyy-mm-dd HH:MM:SS"
    @info string(
        Dates.format(now(), date_format),
        " Expected to get ",
        length(planner_it),
        " results!",
    )

    count = 0
    lk = Threads.ReentrantLock()
    df = DataFrame(
        seed=12554,
        horizon=40,
        budget=1_000_000,
        exploration_param=500,
        gamma=0.975,
        steps=100,
    )
    empty!(df)
    p = !show_progress || Progress(length(planner_it))
    Threads.@threads for j in eachindex(planner_it)
        env = envfun(opts)
        mcts = Planner(
            env,
            seed=planner_it[j][5],
            horizon=planner_it[j][1],
            budget=planner_it[j][2],
            exploration_param=planner_it[j][3],
            γ=planner_it[j][4],
        )
        res = run_planner!(mcts)
        !show_progress || next!(p)
        lock(lk) do
            push!(df, [planner_it[j][5], planner_it[j][1], planner_it[j][2], planner_it[j][3], planner_it[j][4], res])
            if nrow(df) >= 10
                count += nrow(df)
                CSV.write(file_name, df, append=!start_new_file)
                start_new_file = false
                empty!(df)
                show_progress ||
                    @info "$(Dates.format(now(), date_format)) Iteration $count/$nit"
            end
        end
    end

    if nrow(df) > 0
        count += nrow(df)
        CSV.write(file_name, df, append=!start_new_file)
        empty!(df)
        show_progress || @info "$(Dates.format(now(), date_format)) Iteration $count/$nit"
    end
    nothing
end