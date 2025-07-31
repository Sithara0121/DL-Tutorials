using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select
auto = dataset("ISLR", "Auto")
println(first(auto, 3))
@show describe(auto)
y, X = unpack(auto, ==(:MPG))
train, test = partition(eachindex(y), 0.5, shuffle=true, rng=444)
LR = @load LinearRegressor pkg = MLJLinearModels
using Plots

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    xlabel!("Horsepower")
    ylabel!("MPG")
end

lm = LR()
mlm = machine(lm, select(X, :Horsepower), y)
fit!(mlm, rows=train)
mse = rms(MLJ.predict(mlm, rows=test), y[test])^2
xx = (Horsepower=range(50, 225, length=100) |> collect,)
yy = MLJ.predict(mlm, xx)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label="Data", legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy, label="Fit", legend=:topright, linewidth=3, color=:orange)
    xlabel!("Horsepower")
    ylabel!("MPG")
end
hp = X.Horsepower
Xhp = DataFrame(hp1=hp, hp2=hp .^ 2, hp3=hp .^ 3)

LinMod = Pipeline(
    FeatureSelector(features=[:hp1]),
    LR()
)

LinMod.feature_selector.features = [:hp1] # poly of degree 1
lr1 = machine(LinMod, Xhp, y) # poly of degree 1 (line)
fit!(lr1, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2] # poly of degree 2
lr2 = machine(LinMod, Xhp, y)
fit!(lr2, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2, :hp3] # poly of degree 3
lr3 = machine(LinMod, Xhp, y)
fit!(lr3, rows=train)

get_mse(lr) = rms(MLJ.predict(lr, rows=test), y[test])^2

@show get_mse(lr1)
@show get_mse(lr2)
@show get_mse(lr3)

hpn = xx.Horsepower
Xnew = DataFrame(hp1=hpn, hp2=hpn .^ 2, hp3=hpn .^ 3)

yy1 = MLJ.predict(lr1, Xnew)
yy2 = MLJ.predict(lr2, Xnew)
yy3 = MLJ.predict(lr3, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label=false, size=(800, 600))
    plot!(xx.Horsepower, yy1, label="Order 1", linewidth=3, color=:orange,)
    plot!(xx.Horsepower, yy2, label="Order 2", linewidth=3, color=:green,)
    plot!(xx.Horsepower, yy3, label="Order 3", linewidth=3, color=:red,)
    xlabel!("Horsepower")
    ylabel!("MPG")
end
Xhp = DataFrame([hp .^ i for i in 1:10], :auto)

cases = [[Symbol("x$j") for j in 1:i] for i in 1:10]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)
#train + test => give you 10 splits of train+test
# Now we're left with fitting the tuned model

mtm = machine(tm, Xhp, y)
fit!(mtm)


rep = report(mtm)
res = rep.plotting
rep.best_model

Xnew = DataFrame([hpn .^ i for i in 1:10], :auto)
yy5 = MLJ.predict(mtm, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy5, color=:orange, linewidth=4, legend=false)
    xlabel!("Horsepower")
    ylabel!("MPG")
end

using DataFrames
LinMod = Pipeline(
    FeatureSelector(features=[:Nmae]),
    LR()
)
names_cols = names(select(X, Not(:Name)))

cases = [[Symbol(names_cols[i]) for i in 1:j] for j in 1:lastindex(names_cols)]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)
mtm = machine(tm, X, y)
fit!(mtm)
rep = report(mtm)

res = rep.plotting
rep.best_model

best_models_mse_mean = mean(rep.best_history_entry.per_fold[1])^2
best_models_mse_std = std(rep.best_history_entry.per_fold[1])^2

