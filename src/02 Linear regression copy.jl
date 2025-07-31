

using MLJ
models()
filter(model) = model.is_pure_julia && model.is_supervised && model.prediction_type == :probabilistic
models(filter)
models("XGB")
measures("F1")
LR = @load LinearRegressor pkg = MLJLinearModels

import RDatasets: dataset
import DataFrames: describe, select, Not, rename!
data = dataset("MASS", "Boston")
println(first(data, 3))
@show describe(data)
data = coerce(data, autotype(data, :discrete_to_continuous))
y = data.MedV
X = select(data, Not(:MedV))
mdls = models(matching(X, y))
model = LR()
X_uni = select(X, :LStat) # only a single feature
println("Xuni")
println(first(X_uni, 3))
mach_uni = machine(model, X_uni, y)
fit!(mach_uni)
ŷ = MLJ.predict(mach_uni, X_uni)
round(rsquared(ŷ, y), sigdigits=4)
println("rsquared")

println(first(round(rsquared(ŷ, y), sigdigits=4), 3))
fp = fitted_params(mach_uni)
@show fp.coefs
@show fp.intercept
using Plots
plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600), xlabel="LStat")
Xnew = (LStat=collect(range(extrema(X.LStat)..., length=100)),)
plot!(Xnew.LStat, MLJ.predict(mach_uni, Xnew), linewidth=3, color=:orange)
mach = machine(model, X, y)
fit!(mach)
fp = fitted_params(mach)
coefs = fp.coefs
intercept = fp.intercept
for (name, val) in coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(intercept, sigdigits=3))")
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)
res = ŷ .- y
begin
    plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800, 600)))
    hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end
mean(y)
histogram(res, normalize=true, size=(800, 600), label="residual")
X2 = hcat(X, X.LStat .* X.Age)
rename!(X2, :x1 => :interaction)
mach = machine(model, X2, y)
fit!(mach)
ŷ = MLJ.predict(mach, X2)
round(rsquared(ŷ, y), sigdigits=4)
using DataFrames
X3 = DataFrame(hcat(X.LStat, X.LStat .^ 2), [:LStat, :LStat2])
mach = machine(model, X3, y)
fit!(mach)
ŷ = MLJ.predict(mach, X3)
round(rsquared(ŷ, y), sigdigits=4)
Xnew = (LStat=Xnew.LStat, LStat2=Xnew.LStat .^ 2)
plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))
plot!(Xnew.LStat, MLJ.predict(mach, Xnew), linewidth=3, color=:orange)
