using Flux, Statistics, CSV, DataFrames, Dates, Printf
using OneHotArrays: onehotbatch, onecold
using MLUtils: DataLoader

now_str() = Dates.format(now(), "HH:MM:SS")
macro log(msg)
    :(println("[", now_str(), "] ", $(esc(msg))))
end

# ---------- 1. Data ----------
@log "Downloading Wine Quality (red) ..."
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df  = CSV.read(download(url), DataFrame)
df.class = ifelse.(df.quality .≤ 4, "Low",
            ifelse.(df.quality .≤ 6, "Medium", "High"))
classes = ["Low", "Medium", "High"]

df = df[shuffle(1:end), :]
X = Matrix{Float32}(select(df, Not([:quality, :class])))'
y = onehotbatch(df.class, classes)

split = 1300
Xtr, Xte = X[:, 1:split], X[:, split+1:end]
ytr, yte = y[:, 1:split], y[:, split+1:end]

# ---------- 2. Helper: train once ----------
function train_eval(lr, hidden; epochs=30)
    model = Chain(
        Dense(11, hidden, relu),
        Dense(hidden, 3),
        softmax
    )
    opt = Flux.setup(Flux.Adam(lr), model)
    loader = DataLoader((Xtr, ytr), batchsize=32, shuffle=true)

    for _ in 1:epochs
        Flux.train!((m,x,y)->Flux.crossentropy(m(x),y), model, loader, opt)
    end
    acc = mean(onecold(model(Xte)) .== onecold(yte))
    return acc
end

# ---------- 3. Grid search ----------
lrs   = [0.1, 0.03, 0.01]
hids  = [8, 16, 32]
results = []

@log "Starting grid search ..."
for lr in lrs, h in hids
    acc = train_eval(lr, h)
    push!(results, (lr=lr, hidden=h, acc=acc))
    @printf "[Grid] lr=%.2f  hidden=%2d  acc=%.2f%%\n" lr h acc*100
end

# ---------- 4. Report ----------
best = argmax(r -> r.acc, results)
@log "Best: lr=$(best.lr), hidden=$(best.hidden), accuracy=$(round(best.acc*100, digits=2))%"