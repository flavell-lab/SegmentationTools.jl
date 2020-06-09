"""
Runs instance segmentation on all given frames and outputs to various files (centroids, activity measurements, and image ROIs).
Skips a given output method if the corresponding output directory was empty
Returns dictionary of results and a list of error frames (most likely because the worm was not in the field of view).

# Arguments

- `rootpath::String`, `frames`, `img_prefix::String`, `mhd_path::String`, and `channel::Integer`: 
    Reads MHD files from, eg, `rootpath/mhd_path/img_prefix_t0123_ch2.mhd` for frame=123 and channel=2, and outputs resulting image.
- `prediction_path::String`: Reads UNet predictions from `rootpath/prediction_path/frame_predictions.h5`
- `centroids_output_path::String`: Path to output centroids (relative to `rootpath`)
- `activity_output_path::String`: Path to output activity (relative to `rootpath`)
- `roi_output_path::String`: Path to output image ROIs (relative to `rootpath`)

# Optional keyword arguments

- `min_vol::Real`: smallest neuron volume. Default the volume of a spheroid with radius 1 but elongated by a factor of 3 in the z-direction.
- `kernel_σ`: Gaussian kernel size to filter distance image for local peak detection. Default (0.5, 0.5, 1.5).
- `min_distance::Real`: minimum distance between two local peaks. Default 2.
- `threshold::Real`: UNet output threshold before the pixel is considered foreground. Default 0.75.
"""
function instance_segmentation_output(rootpath::String, frames, img_prefix::String, 
            mhd_path::String, channel::Integer, prediction_path::String,
            centroids_output_path::String, activity_output_path::String, roi_output_path::String;
            min_vol=volume(1, (1,1,3)), threshold=0.75)
    n = length(frames)
    results = Dict()
    error_frames = Dict()
    @showprogress for i in 1:n
        frame = frames[i]
        try
            mhd_str = joinpath(rootpath, mhd_path, img_prefix*"_t"*string(frame, pad=4)*"_ch$(channel).mhd")
            img_roi, centroids, activity = instance_segmentation(rootpath, frame, mhd_str, prediction_path,
                min_vol=min_vol, threshold=threshold)

            results[frame] = (img_roi, centroids, activity)
            
            if centroids_output_path != ""
                centroid_path = joinpath(rootpath, centroids_output_path, "$(frame).txt")
                create_dir(joinpath(rootpath, centroids_output_path))
                write_centroids(centroids, centroid_path)
            end

            if activity_output_path != ""
                activity_path = joinpath(rootpath, activity_output_path, "$(frame).txt")
                create_dir(joinpath(rootpath, activity_output_path))
                write_activity(activity, activity_path)
            end

            if roi_output_path != ""
                roi_path = joinpath(rootpath, roi_output_path, "$(frame)")
                create_dir(joinpath(rootpath, roi_output_path))
                mhd = MHD(mhd_str)
                spacing = split(mhd.mhd_spec_dict["ElementSpacing"], " ")
                write_raw("$(roi_path).raw", map(x->UInt16(x), img_roi))
                write_MHD_spec("$(roi_path).mhd", spacing[1], spacing[end], size(img_roi)[1],
                    size(img_roi)[2], size(img_roi)[3], "$(frame).raw")
            end
        catch e
            error_frames[i] = string(e)
        end
    end
    if length(keys(error_frames)) > 0
        println("WARNING: Errors in some frames.")
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

- `rootpath::String`: Working directory.
- `frame::Integer`: Frame number of the image in question.
- `mhd::String`: Path to mhd file.
- `prediction_path::String`: Reads UNet predictions from `rootpath/prediction_path/frame_predictions.h5`

# Optional keyword arguments

- `min_vol::Real`: smallest neuron volume. Default the volume of a spheroid with radius 1 but elongated by a factor of 3 in the z-direction.
- `threshold::Real`: UNet output threshold before the pixel is considered foreground. Default 0.75.
"""
function instance_segmentation(rootpath::String, frame::Integer, mhd::String, prediction_path::String;
        min_vol::Real=volume(1, (1,1,3)), threshold=0.75)
    # read image
    img = read_img(MHD(mhd))
    predicted_th = load_predictions(joinpath(rootpath, prediction_path, "$(frame)_predictions.h5"), threshold=threshold)
    img_b = remove_small_objects(predicted_th, min_vol);
    img_roi = consolidate_labeled_img(labels_map(fast_scanning(img_b, 0.5)));

    # get centroids of ROIs
    centroids = get_centroids(img_roi);

    # get activity of ROIs
    activity = get_activity(img_roi, img);

    return (img_roi, centroids, activity)
end


""" Converts an instance-segmentation image to a ROI image. """
function consolidate_labeled_img(labeled_img)
    labels = []
    bkg_label = 0
    max_sum = 0
    for i=minimum(labeled_img):maximum(labeled_img)
        s = sum(labeled_img .== i)
        if s > 0
            append!(labels, i)
        end
        if s > max_sum
            bkg_label=i
            max_sum = s
        end
    end
    label_dict = Dict()
    label_dict[bkg_label] = UInt16(0)
    count = 1
    for i=1:length(labels)
        if labels[i] != bkg_label
            label_dict[labels[i]] = UInt16(count)
            count = count + 1
        end
    end
    return map(x->label_dict[x], labeled_img)
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
            push!(activity, 0)
        end
        push!(activity, map(x->round(x), sum(img[img_roi .== i]) / total))
    end
    return activity
end
