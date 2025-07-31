using Flux, Statistics, CSV, DataFrames, Dates
using OneHotArrays: onehotbatch, onecold
using MLUtils: DataLoader, splitobs
using BSON: @save

now_str() = Dates.format(now(), "HH:MM:SS")
macro log(msg)
    :(println("[", now_str(), "] ", $(esc(msg))))
end

# ---------- 1. Data ----------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df  = CSV.read(download(url), DataFrame)
df.class = ifelse.(df.quality .≤ 4, "Low",
            ifelse.(df.quality .≤ 6, "Medium", "High"))
classes = ["Low", "Medium", "High"]

X = Matrix{Float32}(select(df, Not([:quality, :class])))'
y = onehotbatch(df.class, classes)

# 80-10-10 split (train/val/test)
N = size(X, 2)
(Xtr, ytr), (Xval, yval), (Xte, yte) = splitobs((X, y); at=(0.8, 0.1, 0.1))
# ---------- 2. Model ----------
model = Chain(
    Dense(11, 32),
    Dropout(0.3),
    relu,
    Dense(32, 16),
    Dropout(0.2),
    relu,
    Dense(16, 3),
    softmax
)
opt_state = Flux.setup(Flux.Adam(0.003), model)

# ---------- 3. Early-stopping ----------
best_val_acc = 0.0
patience = 10
patience_left = patience
train_loader = DataLoader((Xtr, ytr), batchsize=32, shuffle=true)

loss_reg(m, x, y) = Flux.crossentropy(m(x), y) + 1e-4 * sum(norm, Flux.params(m))

@log "Training with early-stopping (max 200 epochs) ..."
for epoch in 1:200
    Flux.train!(loss_reg, model, train_loader, opt_state)

    val_acc = mean(onecold(model(Xval)) .== onecold(yval))
    epoch % 10 == 0 && @log "Epoch $epoch  val-acc = $(round(val_acc*100, digits=2))%"

    if val_acc > best_val_acc + 0.001
        best_val_acc = val_acc
        patience_left = patience
    else
        patience_left -= 1
        if patience_left == 0
            @log "Early-stopping at epoch $epoch  best-val-acc = $(round(best_val_acc*100, digits=2))%"
            break
        end
    end
end

# ---------- 4. Final test ----------
test_acc = mean(onecold(model(Xte)) .== onecold(yte))
@log "Final test accuracy: $(round(test_acc*100, digits=2))%"

# ---------- 5. Save ----------
@save "regularised_wine_model.bson" model
@log "Model saved to regularised_wine_model.bson"