using Flux, Statistics, CSV, DataFrames, Dates, Printf
using OneHotArrays: onehotbatch, onecold
using MLUtils: DataLoader
using BSON: @save

now_str() = Dates.format(now(), "HH:MM:SS")
macro log(msg)
    :(println("[", now_str(), "] ", $(esc(msg))))
end

# ---------- 1. Data ----------
@log "Downloading Wine Quality (red) ..."
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df  = CSV.read(download(url), DataFrame)

# 3-class target
df.class = ifelse.(df.quality .≤ 4, "Low",
            ifelse.(df.quality .≤ 6, "Medium", "High"))
classes = ["Low", "Medium", "High"]

# shuffle & split
df = df[shuffle(1:end), :]
X = Matrix{Float32}(select(df, Not([:quality, :class])))'   # 11×N
y = onehotbatch(df.class, classes)

split = 1300
Xtr, Xte = X[:, 1:split], X[:, split+1:end]
ytr, yte = y[:, 1:split], y[:, split+1:end]

@log "Train: $(size(Xtr,2))  Test: $(size(Xte,2))"

# ---------- 2. Build the BEST model ----------
# model = Chain(
#     Dense(11, 32, tanh),
#     Dense(32, 3),
#     softmax
# )
# Early-stopping
# Stop training when validation accuracy hasn’t improved for 10 epochs.
# Batch-normalization
# Add BatchNorm layers after each Dense to stabilize training.
model = Chain(
    Dense(11, 32), BatchNorm(32), relu,
    Dense(32, 16), BatchNorm(16), relu,
    Dense(16, 3), softmax
)
opt_state = Flux.setup(Flux.Adam(0.003), model)   # slightly lower LR
loader = DataLoader((Xtr, ytr), batchsize=32, shuffle=true)

@log "Fine-tuning best model (lr=0.01, hidden=32) for 100 epochs ..."

loss_fn(m, x, y) = Flux.crossentropy(m(x), y)

for epoch in 1:100
    epoch_loss = 0.0f0
    batches = 0
    for (x, y) in loader
        l, grads = Flux.withgradient(model) do m
            loss_fn(m, x, y)
        end
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += l
        batches += 1
    end
    epoch % 10 == 0 &&
        @log "Epoch $epoch/100  avg-loss = $(round(epoch_loss/batches, digits=4))"
end

# ---------- 3. Evaluate ----------
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
acc = accuracy(Xte, yte)
@log "Final test accuracy: $(round(acc*100, digits=2))%"

# ---------- 4. Save ----------
@save "best_wine_model.bson" model
@log "Model saved to best_wine_model.bson"