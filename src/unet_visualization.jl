"""
Makes image of the raw data overlaid with a translucent label.

# Arguments

- `img`: raw data image (2D)
- `label`: labels for the image
- `weight`: mask of which label values to include. Pixels with a weight of 0 will be plotted in the
    raw, but not labeled, data; pixels with a weight of 1 will be plotted with raw data overlaid with label.

## Optional keyword arguments
- `contrast::Real`: Contrast factor for raw image. Default 1 (no adjustment)
- `label_intensity::Real`: Intensity of label, from 0 to 1. Default 0.5.
"""
function view_label_overlay(img, label, weight; contrast::Real=1, label_intensity::Real=0.5)
    img_gray = map(x->min(x, 0.5), img * contrast * (1 - label_intensity) ./ maximum(img))
    green = img_gray + [weight[x] == 0 ? 0 : (1-label[x]) * label_intensity for x in CartesianIndices(size(img))]
    red = img_gray + [weight[x] == 0 ? 0 : label[x] * label_intensity for x in CartesianIndices(size(img))]
    blue = img_gray
    rgb_stack = RGB.(red, green, blue)
    return rgb_stack
end

"""
Generates an image which compares the predictions of the neural net with the label.
Green = match, red = mismatch.
Assumes the predictions and labels are binary 2D arrays.

# Arguments

- `predicted`: neural net predictions
- `actual`: actual labels
- `weight`: pixel weights; weight of 0 is ignored and not plotted.
"""
function visualize_prediction_accuracy_2D(predicted, actual, weight)
    inaccuracy = map(abs, predicted - actual)
    green = [weight[x] == 0 ? 0 : 1 - inaccuracy[x] for x in CartesianIndices(size(inaccuracy))]
    red = [weight[x] == 0 ? 0 : inaccuracy[x] for x in CartesianIndices(size(inaccuracy))]
    blue = zeros(size(inaccuracy))
    rgb_stack = RGB.(red, green, blue)
    return rgb_stack
end


"""
Generates an image which compares the predictions of the neural net with the label.
Green = match, red = mismatch.
Assumes the predictions and labels are binary 3D arrays.

# Arguments

- `predicted`: neural net predictions
- `actual`: actual labels
- `weight`: pixel weights; weight of 0 is ignored and not plotted.
"""
function visualize_prediction_accuracy_3D(predicted, actual, weight)
    @manipulate for z=1:size(raw)[3]
        visualize_prediction_accuracy_2D(predicted[:,:,z], actual[:,:,z], weight[:,:,z])
    end
end


"""
Makes grid out of many smaller plots. 

# Arguments

- `plots`: List of plots
- `cols::Integer`: Number of columns in array of plots to be created
- `size`: Size of resulting plot per row.
"""
function make_plot_grid(plots, cols::Integer, size)
    if length(plots) < cols
        cols = length(plots)
    end
    rows = length(plots)÷cols
    p = plot(layout=(rows, cols), size=(size[1], size[2]*rows))
    for (i,plot) in enumerate(plots)
        row = length(plots)÷i + 1
        col = (i-1)%cols + 1
        plot!(p[row,col], plot)
    end
    return p
end

"""
Compares multiple different neural network predictions of the raw dataset,
in comparison with the label and weight samples.

# Arguments

- `raw`: 2D raw dataset
- `label`: labels on raw dataset. Set to `nothing` to avoid displaying labels (for instance, on a testing datset).
- `weight`: weights on the labels. Set to `nothing` if you are not displaying labels.
- `predictions_array`: various predictions of the raw dataset.

# Optional keyword arguments

- `cols::Integer`: maximum number of columns in the plot. Default 7.
- `size`: size of plot per row. Default (1800, 750).
"""
function display_predictions_2D(raw, label, weight, predictions_array; cols::Integer=7, size=(1800,750))
    plots = []
    if label != nothing
        push!(plots, plot(view_label_overlay(raw, label, weight, contrast=2), aspect_ratio=1, showaxis=false, legend=false, flip=false))
        push!(plots, plot(view_label_overlay(weight, label, weight, contrast=2), aspect_ratio=1, showaxis=false, legend=false, flip=false))
    else
        push!(plots, heatmap(raw, aspect_ratio=1, showaxis=false, legend=false, fillcolor=:viridis))
    end

    for predictions in predictions_array
        push!(plots, plot(view_label_overlay(predictions, label, weight, img_max=1), aspect_ratio=1, showaxis=false, legend=false, flip=false))
        if label != nothing
            push!(plots, plot(visualize_prediction_accuracy(predictions_last, label, weight), aspect_ratio=1, showaxis=false, flip=false, legend=false))
        end
    end
    return make_plot_grid(plots, cols, size)
end

"""
Compares multiple different neural network predictions of the raw dataset,
using an interactive slider to toggle between z-planes of the 3D dataset.

# Arguments

- `raw`: 3D raw dataset
- `label`: labels on raw dataset. Set to `nothing` to avoid displaying labels (for instance, on a testing datset).
- `weight`: weights on the labels. Set to `nothing` if you are not displaying labels.
- `predictions_array`: various predictions of the raw dataset.

# Optional keyword arguments

- `cols::Integer`: maximum number of columns in the plot. Default 7.
- `size`: size of plot per row. Default (1800, 750).
"""
function display_predictions_3D(raw, label, weight, predictions_array; cols::Integer=7, size=(1800,750))
    @manipulate for z=1:size(raw)[3]
        display_predictions_2D(raw[:,:,z], label[:,:,z], weight[:,:,z], [predictions[:,:,z] for predictions in predictions_array]; cols=cols, size=size)
    end
end

