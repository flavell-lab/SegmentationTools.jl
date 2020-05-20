"""
Runs instance segmentation on all given frames and outputs to various files.
Returns dictionary of results and a list of error frames (most likely because the worm was not in the field of view).

# Arguments

- `rootpath::String`, `frame::Integer`, `img_prefix::String`, `mhd_path::String`, and `channel::Integer`: 
    Reads MHD file from `rootpath/mhd_path/img_prefix_tchannel.mhd` and outputs resulting image.
- `prediction_path::String`: Reads UNet predictions from `rootpath/prediction_path/frame_predictions.h5`
- `centroids_output_path::String`: Path to output centroids (relative to `rootpath`)
- `activity_output_path::String`: Path to output activity (relative to `rootpath`)

# Optional keyword arguments

- `min_vol::Real`: smallest neuron volume. Default the volume of a spheroid with radius 1 but elongated by a factor of 3 in the z-direction.
- `kernel_σ`: Gaussian kernel size to filter distance image for local peak detection. Default (0.5, 0.5, 1.5).
- `min_distance::Real`: minimum distance between two local peaks. Default 2.
- `threshold::Real`: UNet output threshold before the pixel is considered foreground. Default 0.75.
"""
function instance_segmentation_output(rootpath::String, frame::Integer, img_prefix::String, mhd_path::String, channel::Integer, prediction_path::String,
            centroids_output_path::String, activity_output_path::String;
            min_vol=volume(1, (1,1,3)), kernel_σ=(0.5,0.5,1.5), min_distance=2, threshold=0.75)
    n = length(frames)
    results = Dict()
    error_frames = []
    @showprogress for i in 1:n
        frame = frames[i]
        try
            img_roi, centroids, activity = instance_segmentation(rootpath, frame, img_prefix, mhd_path, channel, prediction_path,
                min_vol=min_vol, kernel_σ=kernel_σ, min_distance=min_distance, threshold=threshold)

            results[frame] = (img_roi, centroids, activity)
            
            centroid_path = joinpath(rootpath, centroids_output_path, "$(frame).txt")
            write_centroids(centroids, centroid_path)

            activity_path = joinpath(rootpath, activity_output_path, "$(frame).txt")
            write_activity(activity, activity_path)
        catch e
            println("ERROR: "*string(e));
            append!(error_frames, frame)
        end
    end
    return (results, error_frames)
end

"""
Computes volume from a radius and a sampling ratio
"""
volume(radius, sampling_ratio) = (4 / 3) * π * sum(radius .* sampling_ratio)


"""
Runs instance segmentation on a frame. Removes detected objects that are too small to be neurons.

# Arguments

- `rootpath::String`, `frame::Integer`, `img_prefix::String`, `mhd_path::String`, and `channel::Integer`: 
    Reads MHD file from `rootpath/mhd_path/img_prefix_tchannel.mhd` and outputs resulting image.
- `prediction_path::String`: Reads UNet predictions from `rootpath/prediction_path/frame_predictions.h5`

# Optional keyword arguments

- `min_vol::Real`: smallest neuron volume. Default the volume of a spheroid with radius 1 but elongated by a factor of 3 in the z-direction.
- `kernel_σ`: Gaussian kernel size to filter distance image for local peak detection. Default (0.5, 0.5, 1.5).
- `min_distance::Real`: minimum distance between two local peaks. Default 2.
- `threshold::Real`: UNet output threshold before the pixel is considered foreground. Default 0.75.
"""
function instance_segmentation(rootpath::String, frame::Integer, img_prefix::String, mhd_path::String, channel::Integer, prediction_path::String;
        min_vol::Real=volume(1, (1,1,3)), kernel_σ=(0.5,0.5,1.5), min_distance::Real=2, threshold=0.75)
    # read image
    img = read_mhd(rootpath, img_prefix, mhd_path, frame, channel)
    predicted_th = load_predictions(joinpath(rootpath, prediction_path, "$(frame)_predictions.h5"), threshold=threshold)
    img_b = remove_small_objects(predicted_th, min_vol);
    img_roi = segment_instance(img_b, kernel_σ=kernel_σ, min_distance=min_distance);

    # get centroids of ROIs
    centroids = get_centroids(img_roi);

    # get activity of ROIs
    activity = get_activity(img_roi, img);

    return (img_roi, centroids, activity)
end


"""
Gets centroids from image ROI `img_roi`.
"""
function get_centroids(img_roi)
    centroids = []
    indices = CartesianIndices(img_roi)
    for i=1:maximum(img_roi)
        total = sum(img_roi .== i)
        if total == 0
            continue
        end
        push!(centroids, map(x->round(x), Tuple(sum((img_roi .== i) .* indices))./total))
    end
    return centroids
end

"""
Returns the average activity of all ROIs in `img_roi` in the given image `img`.
"""
function get_activity(img_roi, img)
    activity = []
    for i=1:maximum(img_roi)
        total = sum(img_roi .== i)
        if total == 0
            continue
        end
        push!(activity, map(x->round(x), sum(img[img_roi .== i]) / total))
    end
    return activity
end
