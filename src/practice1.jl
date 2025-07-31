using Flux, Statistics, CSV, DataFrames, Dates
using OneHotArrays: onehotbatch, onecold
using MLUtils: DataLoader

# ---------- tiny helper ----------
now_str() = Dates.format(now(), "HH:MM:SS")

macro log(msg)
    :(println("[", now_str(), "] ", $(esc(msg))))
end

# ---------- 1. Dataset ----------
@log "Downloading Iris dataset ..."
const URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
const COLS = [:sepal_length, :sepal_width, :petal_length, :petal_width, :class]

df = CSV.File(download(URL), header=false) |> DataFrame
rename!(df, COLS)
@log "Dataset loaded: $(size(df,1)) rows, $(size(df,2)) columns"

# ---------- 2. Shuffle ----------
df = df[shuffle(1:end), :]
@log "Dataset shuffled"

# ---------- 3. Features / labels ----------
X = Matrix{Float32}(select(df, Not(:class)))'
classes = unique(df.class)
y = onehotbatch(df.class, classes)
@log "Features matrix: $(size(X)),  Classes: $(join(classes, ", "))"

# ---------- 4. Split ----------
X_train, X_test = X[:, 1:120], X[:, 121:150]
y_train, y_test = y[:, 1:120], y[:, 121:150]
@log "Train set: $(size(X_train,2)) samples,  Test set: $(size(X_test,2)) samples"

# ---------- 5. Model ----------
model = Chain(Dense(4, 16, relu), Dense(16, 3), softmax)
opt_state = Flux.setup(ADAM(0.01), model)
@log "Model built: $(sum(length, Flux.params(model))) parameters"

# ---------- 6. DataLoader ----------
train_loader = DataLoader((X_train, y_train), batchsize=16, shuffle=true)
@log "Training for 100 epochs ..."

# ---------- 7. Training loop ----------
for epoch in 1:100
    epoch_loss = 0.0f0
    batches = 0
    for (x, y) in train_loader
        l, grads = Flux.withgradient(model) do m
            Flux.crossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += l
        batches += 1
    end
    epoch_loss /= batches
    epoch % 10 == 0 && @log "Epoch $epoch/100  avg-loss = $(round(epoch_loss, digits=4))"
end

# ---------- 8. Evaluate ----------
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
acc = accuracy(X_test, y_test)
@log "Final test accuracy: $(round(acc*100, digits=2))%"