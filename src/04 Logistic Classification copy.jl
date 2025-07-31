# ## Stock market data

using MLJ
import RDatasets: dataset #import does not bring the package and its fucntions in the namespace
import DataFrames: DataFrame, describe, select, Not
import StatsBase: countmap, cor, var

smarket = dataset("ISLR", "Smarket")
@show size(smarket)
@show names(smarket)
r3(x) = round(x, sigdigits=3)
r3(pi)
ϵ = 0.1
println(describe(smarket, :mean, :std, :eltype))
X = select(smarket, Not(:Direction))
using Plots

function plotter(cr::Matrix{Float64}, cols::Vector{Symbol})::Nothing
    (n, m) = size(cr)

    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="Pearson Correlation Coefficients")

    annotate!([(j, i, text(round(cr[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])

    savefig("./pearson_correlations.png")
    return nothing
end

cm = X |> Matrix |> cor
plotter(round.(cm, sigdigits=1), Symbol.(names(X)))

# Let's see what the `:Volume` feature looks like:

using Plots

begin
    plot(X.Volume, size=(800, 600), linewidth=2, legend=false)
    xlabel!("Tick number")
    ylabel!("Volume")
end


# ### Logistic Regression

# We will now try to train models; the target `:Direction` has two classes: `Up` and `Down`; it needs to be interpreted as a categorical object, and we will mark it as a _ordered factor_ to specify that 'Up' is positive and 'Down' negative (for the confusion matrix later):

y = coerce(y, OrderedFactor)
levels(y)

# Note that in this case the default order comes from the lexicographic order which happens  to map  to  our intuition since `D`  comes before `U`.

cm = countmap(y)
categories, vals = collect(keys(cm)), collect(values(cm))
Plots.bar(categories, vals, title="Bar Chart Example", legend=false)
ylabel!("Number of occurrences")
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels
X2 = select(X, Not([:Year, :Today]))
classif = machine(LogisticClassifier(), X2, y)
fit!(classif)
ŷ = MLJ.predict(classif, X2)
ŷ[1:3]

cross_entropy(ŷ, y) |> r3

ŷ = predict_mode(classif, X2)
misclassification_rate(ŷ, y) |> r3

@show cm = confusion_matrix(ŷ, y)

@show false_positive(cm)
@show accuracy(ŷ, y) |> r3
@show accuracy(cm) |> r3  # same thing
@show ppv(cm) |> r3   # a.k.a. precision
@show recall(cm) |> r3
@show f1score(ŷ, y) |> r3
@show fpr(ŷ, y) |> r3

train = 1:findlast(X.Year .< 2005)
test = last(train)+1:length(y)

fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

X3 = select(X2, [:Lag1, :Lag2])
classif = machine(LogisticClassifier(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

Xnew = (Lag1=[1.2, 1.5], Lag2=[1.1, -0.8])
ŷ = MLJ.predict(classif, Xnew)
ŷ |> println

mode.(ŷ)


