"""
Converts `centroids` into an image mask of size `imsize`
"""
function centroids_to_img(imsize, centroids)
    return map(x->Tuple(x) in centroids, CartesianIndices(imsize))
end

"""
Plots instance segmentation image `img_roi`, where each object is given a different color.

# Arguments

- `img_roi`: 3D instance segmentation image

# Optional keyword Arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
"""
function view_roi_3D(img_roi; color_brightness=0.3)
    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    img_plot = map(x->colors[x+1], img_roi)
    @manipulate for z=1:size(img_plot)[3]
        Plots.plot(img_plot[:,:,z])
    end
end


"""
Plots instance segmentation image `img_roi`, where each object is given a different color.

# Arguments

- `img_roi`: 2D instance segmentation image

# Optional keyword Arguments

- `color_brightness`: minimum RGB value (out of 1) that an object will be plotted with
"""
function view_roi_2D(img_roi; color_brightness=0.3)
    num = maximum(img_roi)+1
    colors = [RGB.(color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand(), color_brightness+(1-color_brightness)*rand()) for i=1:num]
    colors[1] = RGB.(0,0,0)
    img_plot = map(x->colors[x+1], img_roi)
    Plots.plot(img_plot)
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
