"""
Converts `centroids` into an image mask of size `imsize`
"""
function centroids_to_img(imsize, centroids)
    return map(x->Tuple(x) in centroids, CartesianIndices(imsize))
end

"""
Plots instance segmentation image `img_roi`, where each object is given a different color.
Can also plot raw data and semantic segmentation data for comparison.

# Arguments

- `raw`: 3D raw image. If set to `nothing`, it will not be plotted.
- `predicted`: 3D semantic segmentation image. If set to `nothing`, it will not be plotted.
- `img_roi`: 3D instance segmentation image

# Optional keyword arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
- `plot_size`: size of the plot
- `axis`: axis to project, default 3
"""
function view_roi_3D(raw, predicted, img_roi; color_brightness=0.3, plot_size=(600,600), axis=3)
    plot_imgs = []
    if raw != nothing
        max_img = maximum(raw)
        push!(plot_imgs, map(x->RGB.(x/max_img, x/max_img, x/max_img), raw))
    end
    if predicted != nothing
        push!(plot_imgs, map(x->RGB.(x,x,x), predicted))
    end
    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    push!(plot_imgs, map(x->colors[x+1], img_roi))
    @manipulate for z=1:size(plot_imgs[1])[axis]
        i = [(dim == axis) ? z : Colon() for dim=1:3]
        make_plot_grid([img[i[1],i[2],i[3]] for img in plot_imgs], length(plot_imgs), plot_size)
    end
end


"""
Plots instance segmentation image `img_roi`, where each object is given a different color.
Can also plot raw data and semantic segmentation data for comparison.

# Arguments

- `raw`: 2D raw image. If set to `nothing`, it will not be plotted.
- `predicted`: 2D semantic segmentation image. If set to `nothing`, it will not be plotted.
- `img_roi`: 2D instance segmentation image

# Optional keyword arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
"""
function view_roi_2D(raw, predicted, img_roi; color_brightness=0.3)
    plot_imgs = []
    if raw != nothing
        max_img = maximum(raw)
        push!(plot_imgs, map(x->RGB.(x/max_img, x/max_img, x/max_img), raw))
    end
    if predicted != nothing
        push!(plot_imgs, map(x->RGB.(x,x,x), predicted))
    end
    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    push!(plot_imgs, map(x->colors[x+1], img_roi))
    make_plot_grid(plot_imgs, length(plot_imgs), plot_size)
end

"""
Displays the `centriods` of an image, superimposed on the image `img`. Assumes 3D image and centroids.
"""
function view_centroids_3D(img, centroids)
    @manipulate for z=1:size(img)[3]
        view_centroids_2D(img[:,:,z], centroids)
    end
end

"""
Displays the `centriods` of an image, superimposed on the image `img`. Assumes 2D image.
The centroids can be 3D; only the first two dimensions will be used.
"""
function view_centroids_2D(img, centroids)
    Plots.heatmap(img, fillcolor=:viridis, aspect_ratio=1, showaxis=false, flip=false)
    Plots.scatter!(map(x->x[2], valid_centroids), map(x->x[1], valid_centroids), flip=false, seriesalpha=0.3)
end
