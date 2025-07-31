using Flux, Statistics, CSV, DataFrames, Dates
using OneHotArrays: onehotbatch, onecold
using MLUtils: DataLoader

now_str() = Dates.format(now(), "HH:MM:SS")
macro log(msg)
    :(println("[", now_str(), "] ", $(esc(msg))))
end

# ---------- 1. Download Wine Quality ----------
@log "Downloading Wine Quality (red) ..."
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df  = CSV.read(download(url), DataFrame)
@log "Loaded: $(size(df,1)) rows, $(size(df,2)) columns"

# ---------- 2. Create 3-class target ----------
df.class = ifelse.(df.quality .≤ 4, "Low",
            ifelse.(df.quality .≤ 6, "Medium", "High"))
classes = ["Low", "Medium", "High"]
@log "Class counts: $(combine(groupby(df, :class), nrow))"

# ---------- 3. Shuffle & split ----------
df = df[shuffle(1:end), :]
X = Matrix{Float32}(select(df, Not([:quality, :class])))'   # 11×N
y = onehotbatch(df.class, classes)

split = 1300                       # 1300 train, 299 test
X_train, X_test = X[:, 1:split], X[:, split+1:end]
y_train, y_test = y[:, 1:split], y[:, split+1:end]

@log "Train: $(size(X_train,2))  Test: $(size(X_test,2))"

# ---------- 4. Model ----------
model = Chain(
    Dense(11, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 3),
    softmax
)
opt_state = Flux.setup(ADAM(0.001), model)
@log "Parameters: $(sum(length, Flux.params(model)))"

# ---------- 5. DataLoader ----------
train_loader = DataLoader((X_train, y_train), batchsize=32, shuffle=true)

# ---------- 6. Train ----------
@log "Training 50 epochs ..."
for epoch in 1:50
    epoch_loss = 0.0f0
    batches = 0
    for (x, y) in train_loader
        l, grads = Flux.withgradient(model) do m
            Flux.crossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += l; batches += 1
    end
    epoch % 10 == 0 &&
        @log "Epoch $epoch/50  avg-loss = $(round(epoch_loss/batches, digits=4))"
end

# ---------- 7. Evaluate ----------
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
acc = accuracy(X_test, y_test)
@log "Final test accuracy: $(round(acc*100, digits=2))%"
