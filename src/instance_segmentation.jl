"""
Runs instance segmentation on all given frames and outputs to various files (centroids, activity measurements, and image ROIs).
Skips a given output method if the corresponding output directory was empty
Returns dictionary of results and a list of error frames (most likely because the worm was not in the field of view).

# Arguments

- `rootpath::String`, `frames`, `img_prefix::String`, `mhd_path::String`, and `channel::Integer`: 
    Reads MHD files from, eg, `rootpath/mhd_path/img_prefix_t0123_ch2.mhd` for frame=123 and channel=2, and outputs resulting image.
- `prediction_path::String`: Reads UNet predictions from `rootpath/prediction_path/frame_predictions.h5`
- `roi_output_path::String`: Path to output image ROIs (relative to `rootpath`)

# Optional keyword arguments

- `centroids_output_path::String`: Path to output centroids (relative to `rootpath`). Set to "" (default) to not output centroids.
- `activity_output_path::String`: Path to output activity (relative to `rootpath`). Set to "" (default) to not output activity.
- `min_neuron_size::Integer`: smallest neuron size, in voxels. Default 10.
- `threshold::Real`: UNet output threshold before the pixel is considered foreground. Default 0.75.
"""
function instance_segmentation_output(rootpath::String, frames, img_prefix::String, 
            mhd_path::String, channel::Integer, prediction_path::String, roi_output_path::String;
            centroids_output_path::String="", activity_output_path::String="",
            min_neuron_size::Integer=10, threshold=0.75)
    n = length(frames)
    results = Dict()
    error_frames = Dict()
    @showprogress for i in 1:n
        frame = frames[i]
        try
            predictions = load_predictions(joinpath(rootpath, prediction_path, "$(frame)_predictions.h5")) .> threshold
            img_roi = instance_segmentation(predictions, min_neuron_size=min_neuron_size)
            
            results[frame] = img_roi
            mhd_str = joinpath(rootpath, mhd_path, img_prefix*"_t"*string(frame, pad=4)*"_ch$(channel).mhd")
            
            if centroids_output_path != ""
                centroids = get_centroids(img_roi)
                centroid_path = joinpath(rootpath, centroids_output_path, "$(frame).txt")
                create_dir(joinpath(rootpath, centroids_output_path))
                write_centroids(centroids, centroid_path)
            end
            
            if activity_output_path != ""
                img = read_img(MHD(mhd_str))
                activity = get_activity(img_roi, img)
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

- `predictions: UNet predictions array

# Optional keyword arguments

- `min_neuron_size::Integer`: smallest neuron size, in voxels. Default 7.
"""
function instance_segmentation(predictions; min_neuron_size::Integer=7)
    return consolidate_labeled_img(labels_map(fast_scanning(predictions, 0.5)), min_neuron_size);
end


""" Converts an instance-segmentation image to a ROI image. Ignores ROIs smaller than the minimum size. """
function consolidate_labeled_img(labeled_img, min_neuron_size)
    labels = []
    bkg_label = 0
    max_sum = 0
    for i=minimum(labeled_img):maximum(labeled_img)
        s = sum(labeled_img .== i)
        if s > min_neuron_size
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
    return map(x->(x in keys(label_dict) ? label_dict[x] : UInt16(0)), labeled_img)
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

"""
Finds the convex hull of a set of `points`, counting how many lines intersect each point.
"""
function find_convex_hull(points)
    hull = Dict()
    
    for p1 in points
        for p2 in points
            if p1 == p2
                continue
            end
            d = floor(distance(p1,p2,zscale=1))
            for t=0:d
                pt = Tuple(map(x->Int32(round(x)),p2 .+ t.*(p1.-p2)./d))
                if pt in keys(hull)
                    hull[pt][2] = hull[pt][2] + 1
                    continue
                end
                if pt in points
                    hull[pt] = [1,1]
                else
                    hull[pt] = [0,1]
                end
            end
        end
    end
    return hull
end

"""
Returns all points from `img_roi` corresponding to region `roi`.
"""
function get_points(img_roi, roi)
    return [Tuple(x) for x in CartesianIndices(size(img_roi)) if img_roi[x] == roi]
end

"""
Computes the distance between two points `p1` and `p2`, with the z-axis scaled by `zscale`.
"""
function distance(p1, p2; zscale=1)
    dim_dist = collect(Float64, (p1 .- p2) .* (p1 .- p2))
    dim_dist[3] *= zscale^2
    return sqrt(sum(dim_dist))
end

"""
Does watershed segmentation on a set of `points` and their convex `hull`, with the intent of splitting concave neurons.
Scales z-axis by `zscale` (default 1) and expands first segmented neuron by `init_scale` to determine second neuron location.
"""
function hull_watershed(points, hull; zscale=1, init_scale=0.7)
    weight = sum([(1 - hull[pt][1]) * hull[pt][2] for pt in keys(hull)])
    # get the farthest point from the concave points and declare it the center of the first neuron
    dist_1 = [sum([distance(p1, p2, zscale=zscale) * (1 - hull[p2][1]) * hull[p2][2] for p2 in keys(hull)])/weight for p1 in points]
    farthest_1 = argmax(dist_1)
    neuron_1_init_hull = Dict()
    for p in keys(hull)
        if distance(p, points[farthest_1], zscale=zscale) < dist_1[farthest_1] * init_scale
            neuron_1_init_hull[p] = 0
        else
            neuron_1_init_hull[p] = 1
        end
    end
    
    # get the farthest point from the first neuron and concave points and declare it the center of the second neuron
    dist_2 = [sum([distance(p1, p2, zscale=zscale) * (1 - neuron_1_init_hull[p2]) for p2 in keys(hull)])/sum([1 - neuron_1_init_hull[pt] for pt in keys(hull)]) for p1 in points]
    farthest_2 = argmax(dist_2)
    
    # sort points based on which center they're closer to
    min_dim = [minimum([points[i][l] for i=1:length(points)]) - 1 for l = 1:length(points[1])]
    max_dim = [maximum([points[i][l] for i=1:length(points)]) for l = 1:length(points[1])]
    watershed_img = zeros(Tuple(max_dim[l] .- min_dim[l] for l = 1:length(min_dim)))
    for idx in CartesianIndices(watershed_img)
        point = collect(Tuple(idx)) .+ min_dim
        watershed_img[idx] = -sum([distance(point, p2, zscale=zscale) * (1 - hull[p2][1]) * hull[p2][2] for p2 in keys(hull)])/weight
    end
    
    watershed_markers = zeros(Int64, size(watershed_img))
    watershed_markers[CartesianIndex(Tuple(collect(points[farthest_1]) .- min_dim))] = 1
    watershed_markers[CartesianIndex(Tuple(collect(points[farthest_2]) .- min_dim))] = 2
    
    results = labels_map(watershed(watershed_img, watershed_markers))
    
    neuron_1 = [p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == 1]
    neuron_2 = [p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == 2]
    
    return (neuron_1, neuron_2)
end


function watershed_threshold(points, centroid_matches, predictions)
    min_dim = [minimum([points[i][l] for i=1:length(points)]) - 1 for l = 1:length(points[1])]
    max_dim = [maximum([points[i][l] for i=1:length(points)]) for l = 1:length(points[1])]
    watershed_img = zeros(Tuple(max_dim[l] .- min_dim[l] for l = 1:length(min_dim)))
    for idx in CartesianIndices(watershed_img)
        point = collect(Tuple(idx)) .+ min_dim
        watershed_img[idx] = -predictions[CartesianIndex(Tuple(point))]
    end
    watershed_markers = zeros(Int64, size(watershed_img))
    for (i, centroid) in enumerate(centroid_matches)
        watershed_markers[CartesianIndex(Tuple(collect(map(x->Int32(x), centroid)) .- min_dim))] = i
    end
    
    results = labels_map(watershed(watershed_img, watershed_markers))
    
    neurons = [[p for p in points if results[CartesianIndex(Tuple(collect(p) .- min_dim))] == i] for i=1:length(centroid_matches)]
    
    return neurons
end

function detect_incorrect_merges(img_roi, predictions, thresholds, neuron_sizes)
    bad_rois = Dict()
    for t in 1:length(thresholds)
        img_roi_t = instance_segmentation(predictions .> thresholds[t], min_neuron_size=neuron_sizes[t])
        centroids = get_centroids(img_roi_t)
        for i=1:maximum(img_roi)
            points = get_points(img_roi, i)
            centroid_matches = []
            for centroid in centroids
                if centroid in points
                    push!(centroid_matches, centroid)
                end
            end
            if length(centroid_matches) >= length(get(bad_rois, i, [0,0]))
                bad_rois[i] = centroid_matches
            end
        end
    end
    return bad_rois
end

function instance_segmentation_threshold(img_roi, predictions; thresholds=[0.8, 0.9], neuron_sizes=[4,4])
    img_roi_new = copy(img_roi)
    bad_rois = detect_incorrect_merges(img_roi, predictions, thresholds, neuron_sizes)
    idx = maximum(img_roi) + 1
    for roi in keys(bad_rois)
        neurons = watershed_threshold(get_points(img_roi, roi), bad_rois[roi], predictions)
        for i in 1:length(neurons)
            if i == 1
                continue
            end
            neuron = neurons[i]
            for pt in neuron
                img_roi_new[CartesianIndex(pt)] = idx
            end
            idx = idx + 1
        end
    end
    return img_roi_new
end



"""
Instance segments an image `img_roi` given set of `points` with a given convex `hull` via watershedding.
Discards neurons with size less than `min_neuron_size` (default 10), scales z-axis by `zscale` (default 1),
and expands first segmented neuron by `init_scale` to determine second neuron location.
"""
function instance_segment_hull(img_roi, points, hull; min_neuron_size=10, zscale=1, init_scale=0.7)
    neuron_1, neuron_2 = hull_watershed(points, hull, zscale=zscale, init_scale=init_scale)
    failed_flag = (length(neuron_1) < min_neuron_size || length(neuron_2) < min_neuron_size)
    if failed_flag
        return (img_roi, true)
    end
    m = maximum(img_roi)
    img_roi_new = copy(img_roi)
    for pt in neuron_2
        img_roi_new[CartesianIndex(pt)] = m+1
    end
    return (img_roi_new, false)
end

"""
Generates a score for how concave a set of `points` with a given convex `hull` is.
"""
function concave_score(points, hull)
    return sum([(1-x[1])*x[2] for x in values(hull)]) * length(hull) / sum([x[2] for x in values(hull)]) * log(length(values(hull))) / length(values(hull))
end

"""
Detects if a given set of `points` is concave given its `hull`, by comparing its concavity score with `threshold_scale`.
"""
function is_concave(points, hull; threshold_scale=0.3)
    return concave_score(points, hull) > threshold_scale
end

"""
Finds the `num_neurons` (default 10) most concave neurons in an image `img_roi`.
Concave neurons must also meet the `threshold_scale`.
"""
function find_concave_neurons(img_roi; num_neurons=10, threshold_scale=0.3)
    hull_points = Dict()
    top = [[0.0,0.0] for i=1:num_neurons]
    for roi in 1:maximum(img_roi)
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        score = concave_score(points, hull)
        if score > max(top[1][2], threshold_scale)
            top[1][1] = roi
            top[1][2] = score
            top = sort(top, by=x->x[2])
        end
        hull_points[roi] = (points, hull)
    end
    concave = Dict()
    best_score = 0
    for t in top
        if t[1] != 0
            if best_score == 0
                best_score = t[2]
            end
            concave[t[1]] = hull_points[t[1]]
        end
    end
    return (concave, best_score)
end

"""
Recursively segments all concave neurons in an image. 

# Arguments
- `img_roi`: Image to segment

# Optional keyword arguments
- `threshold_scale::Real`: Neurons less concave than this won't be segmented. Default 0.3
- `num_neurons::Real`: Maximum number of concave neurons per frame. Defaul 10.
- `zscale::Real`: Scale of z-axis relative to xy plane. Default 2.78.
- `min_neuron_size::Integer`: Minimum size of a neuron (in pixels). Default 10.
- `scale_recurse_multiply::Real`: Factor to increase the concavity threshold for recursive segmentation. Default 1.5.
- `init_scale::Real`: Amount to expand first neuron before computing location of second neuron. Default 0.7. 
"""
function instance_segment_concave(img_roi; threshold_scale::Real=0.3, init_scale::Real=0.7, zscale::Real=2.78, min_neuron_size::Integer=10, scale_recurse_multiply::Real=1.5, num_neurons::Integer=10)
    concave, score = find_concave_neurons(img_roi, threshold_scale=threshold_scale, num_neurons=num_neurons)
    threshold_scale_recurse = score * scale_recurse_multiply
    concave_queue = Queue{Int64}()
    errors = []
    for roi in keys(concave)
        enqueue!(concave_queue, roi)
    end
    while !isempty(concave_queue)
        roi = dequeue!(concave_queue)
        img_roi, error = instance_segment_hull(img_roi, concave[roi][1], concave[roi][2], zscale=zscale, min_neuron_size=min_neuron_size, init_scale=init_scale)
        if error
            append!(errors, roi)
            continue
        end
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        if is_concave(points, hull, threshold_scale=threshold_scale_recurse)
            concave[roi] = (points, hull)
            enqueue!(concave_queue, roi)
        end
        # newest neuron will be the highest number
        roi = maximum(img_roi)
        points = get_points(img_roi, roi)
        hull = find_convex_hull(points)
        if is_concave(points, hull, threshold_scale=threshold_scale_recurse)
            concave[roi] = (points, hull)
            enqueue!(concave_queue, roi)
        end
    end
    return (img_roi, errors)
end

