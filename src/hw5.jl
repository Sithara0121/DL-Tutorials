#Step 1: Setting up the environment
using Pkg
Pkg.add("Flux")
Pkg.add("CUDA") # If you're using GPU acceleration
Pkg.add("DataFrames")
Pkg.add("MLDatasets")
Pkg.add("Plots")
#Step 2: LeNet5 Architecture in Flux.jl
using Flux
using MLDatasets
using Plots

# Load CIFAR-10 dataset
train_images, train_labels = CIFAR10.traindata()
test_images, test_labels = CIFAR10.testdata()

# Preprocess images (normalize and reshape)
train_images = float.(train_images) ./ 255.0
test_images = float.(test_images) ./ 255.0

train_images = permutedims(train_images, (3, 2, 1))  # Convert (32, 32, 3, N) to (N, 32, 32, 3)
test_images = permutedims(test_images, (3, 2, 1))

# LeNet5 architecture
function lenet5()
    Chain(
        Conv((5, 5), 3=>6, relu),  # First convolutional layer (5x5, 3 input channels, 6 output channels)
        MaxPool((2, 2)),            # First max pooling
        Conv((5, 5), 6=>16, relu), # Second convolutional layer (5x5, 6 input channels, 16 output channels)
        MaxPool((2, 2)),            # Second max pooling
        Flatten(),
        Dense(16 * 5 * 5, 120, relu),  # Fully connected layer
        Dense(120, 84, relu),           # Fully connected layer
        Dense(84, 10)                   # Output layer (10 classes)
    )
end

# Instantiate the model
model = lenet5()

# Loss function (cross-entropy)
loss(x, y) = Flux.crossentropy(model(x), y)

# Optimizer (Adam)
opt = ADAM()

# Train the model
function train!(model, data, labels, opt, epochs=6)
    for epoch in 1:epochs
        for (x, y) in zip(data, labels)
            gs = gradient(() -> loss(x, y), params(model))
            Flux.Optimise.update!(opt, params(model), gs)
        end
        println("Epoch $epoch complete")
    end
end

# Convert images and labels to the appropriate format
train_data = [(train_images[:, :, :, i], train_labels[i]) for i in 1:size(train_images, 4)]
test_data = [(test_images[:, :, :, i], test_labels[i]) for i in 1:size(test_images, 4)]

# Train for 6 epochs
train!(model, train_data, train_labels, opt, epochs=6)

# Test accuracy
function test_accuracy(model, data)
    correct = 0
    total = 0
    for (x, y) in data
        prediction = Flux.argmax(model(x))
        correct += (prediction == y)
        total += 1
    end
    return correct / total
end

accuracy = test_accuracy(model, test_data)
println("Test accuracy: $accuracy")
#Step 3: Effect of New Examples vs. Repeated Examples
# Subset sizes and epochs for training
subset_sizes = [10000, 20000, 30000]
epochs = [6, 3, 2]

accuracies = []

for (subset_size, epoch) in zip(subset_sizes, epochs)
    # Select a subset of the training data
    subset_data = train_data[1:subset_size]
    
    # Reinitialize the model for each experiment
    model = lenet5()
    
    # Train for the specific number of epochs
    train!(model, subset_data, train_labels, opt, epochs=epoch)
    
    # Test the accuracy
    accuracy = test_accuracy(model, test_data)
    push!(accuracies, accuracy)
end

# Plot the results
plot(subset_sizes, accuracies, marker=:circle, xlabel="Training Set Size", ylabel="Test Accuracy", label="Test Accuracy")
#Step 4: Effect of Filter Size on Performance (LeNet3 and LeNet7)
# LeNet3 with (3, 3) filters
function lenet3()
    Chain(
        Conv((3, 3), 3=>6, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 6=>16, relu),
        MaxPool((2, 2)),
        Flatten(),
        Dense(16 * 5 * 5, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10)
    )
end

# LeNet7 with (7, 7) filters
function lenet7()
    Chain(
        Conv((7, 7), 3=>6, relu),
        MaxPool((2, 2)),
        Conv((7, 7), 6=>16, relu),
        MaxPool((2, 2)),
        Flatten(),
        Dense(16 * 5 * 5, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10)
    )
end

# Compare LeNet5, LeNet3, and LeNet7
models = [lenet5(), lenet3(), lenet7()]
accuracies_filter = []

for model_fn in models
    model = model_fn()
    train!(model, train_data, train_labels, opt, epochs=6)
    accuracy = test_accuracy(model, test_data)
    push!(accuracies_filter, accuracy)
end

# Plotting
bar(["LeNet5", "LeNet3", "LeNet7"], accuracies_filter, ylabel="Test Accuracy", title="Effect of Filter Size")
#Step 5: Visualizing Learned Features (Feature Maps)
using Flux: @nn

# Extracting feature maps
function visualize_features(model, image)
    feature_maps = []
    
    # Function to capture the output of each convolution layer
    function capture_features(x)
        push!(feature_maps, x)
        return x
    end
    
    # Modify the model to capture feature maps
    layers = []
    for layer in model
        if typeof(layer) <: Conv
            push!(layers, x -> capture_features(layer(x)))
        else
            push!(layers, layer)
        end
    end
    
    new_model = Chain(layers...)
    
    # Pass the image through the modified model
    _ = new_model(image)
    
    return feature_maps
end

# Visualizing feature maps
image = test_images[:, :, :, 1]  # Use the first image in the test set
feature_maps = visualize_features(model, image)

# Plot feature maps
for (i, feature_map) in enumerate(feature_maps)
    num_filters = size(feature_map, 3)
    for j in 1:num_filters
        heatmap(feature_map[:, :, j], color=:viridis, title="Feature Map $i - Filter $j")
    end
end
