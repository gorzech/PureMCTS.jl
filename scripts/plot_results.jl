## Import packages
using DataFrames
using CSV
using Plots
using Plots: heatmap
using Statistics

function export_heatmap_plots(df; fig_dir = ".")
    gdf = groupby(df, [:horizon, :budget, :exploration_param, :gamma])
    # Compute statistics
    stat_df = combine(gdf, :steps => mean, :steps => std)

    gr()
    mkpath(fig_dir)
    fntsm = Plots.font("Times", pointsize = 11)
    fntlg = Plots.font("Times", pointsize = 11)
    default(titlefont = fntlg, guidefont = fntlg, tickfont = fntsm, legendfont = fntsm)
    default(size = (480, 320)) #Plot canvas size
    all_gamma = sort!(unique(stat_df[:, :gamma]))
    all_cp = sort!(unique(stat_df[:, :exploration_param]))
    png_file(g, cp, stat) = "$fig_dir/$(stat)_g_$(g)_cp_$cp.png"
    file_mean(g, cp) = png_file(g, cp, "mean")
    file_std(g, cp) = png_file(g, cp, "std")
    for g in all_gamma
        for t in all_cp
            df_plot = subset(
                stat_df,
                [:gamma, :exploration_param] => (_g, _t) -> _g .== g .&& _t .== t,
            )
            sort!(df_plot, [:budget, :horizon])
            b = unique(df_plot[:, :budget])
            h = unique(df_plot[:, :horizon])

            v = reshape(df_plot[:, :steps_mean], (length(h), length(b)))
            plt1 = heatmap(
                b,
                h,
                v,
                clims = (0, 200),
                xaxis = :log,
                c = cgrad(:tokyo, rev = false),
                xlabel = "Budget [-]",
                ylabel = "Horizon [-]",
            )
            Plots.savefig(plt1, file_mean(g, t))
            v = reshape(df_plot[:, :steps_std], (length(h), length(b)))
            plt2 = heatmap(
                b,
                h,
                v,
                clims = (0, 80),
                xaxis = :log,
                c = cgrad(:tokyo, rev = true),
                xlabel = "Budget [-]",
                ylabel = "Horizon [-]",
            )
            Plots.savefig(plt2, file_std(g, t))
        end
    end
    println("Generation of the plots completed.")
end

## Single pendulun plots
file_name = "single_pendulum.csv"
fig_dir = "single_pendulum"
df = CSV.read(file_name, DataFrame)
export_heatmap_plots(df, fig_dir = fig_dir)

## Double pendulum plots
file_name = "double_pendulum.csv"
fig_dir = "double_pendulum"
df = CSV.read(file_name, DataFrame)
export_heatmap_plots(df, fig_dir = fig_dir)
