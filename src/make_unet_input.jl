"""
Gets all neighbors of a voxel (including diagonally).

# Arguments

- `idx`: coordinate to find neighbors of
- `s`: shape of image (to avoid finding neighbors outside the image boundary).
"""
function get_neighbors_diagonal(idx, s)
    nbs = [[]]
    for elt in zip(idx, s)
        new_nbs = []
        for nb in nbs
            if elt[1] != 1
                new_nb = copy(nb)
                push!(new_nbs, append!(new_nb, elt[1] - 1))
            end
            if elt[1] != elt[2]
                new_nb = copy(nb)
                push!(new_nbs, append!(new_nb, elt[1] + 1))
            end
            new_nb = copy(nb)
            push!(new_nbs, append!(new_nb, elt[1]))
        end
        nbs = new_nbs
    end
    return filter(x->x!=idx, map(x->Tuple(x),nbs))
end

"""
Gets all neighbors of a voxel (not including diagonally) within an N-dimensional image.

# Arguments

- `idx`: coordinate to find neighbors of
- `s`: shape of image (to avoid finding neighbors outside the image boundary).
"""
function get_neighbors_cartesian(idx, s)
    idx_arr = collect(Tuple(idx))
    nbs = []
    for i in 1:length(idx)
        if idx[i] != 1
            nb = copy(idx_arr)
            nb[i] = idx[i] - 1
            push!(nbs, nb)
        end
        if idx[i] != s[i]
            nb = copy(idx_arr)
            nb[i] = idx[i] + 1
            push!(nbs, nb)
        end
    end
    return map(x->CartesianIndex(Tuple(x)),nbs)
end


"""
Creates a weight map from a given labeled dataset.
Unlabeled data (label 0) has weight 0 and background data (label 2) far from foreground data
has weight 1. Foreground data (label 1) has a higher weight.
Background data near the foreground (also label 2) has weight exponentially decaying
down from the foreground weight. "Background-gap" data that serves to mark boundaries between foreground objects
(label 3) has the highest weight.

# Arguments

- `label`: labeled dataset to turn into weights

# Optional keyword arguments

- `scale_xy::Real`: Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 0.36.
- `scale_z::Real`: Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.
- `metric::String`: Metric to compute distance. Default taxicab (currently, no other metrics are implemented, but they will likely be added in a future release.)
- `weight_foreground::Real`: weight of foreground (1) label
- `weight_bkg_gap::Real`: weight of background-gap (3) label
- `delete_boundary::Bool`: whether to set the weight of foreground (2) pixels adjacent to background (1 and 3) pixels to 0. Default false.
- `scale_bkg_gap::Bool`: whether to upweight background-gap pixels for each neuron pixel they border.
"""
function create_weights(label; scale_xy::Real=0.36, scale_z::Real=1, metric::String="taxicab", weight_foreground::Real=4,
        weight_bkg_gap::Real=24, delete_boundary::Bool=false, scale_bkg_gap::Bool=false)
    weights = zeros(size(label))
    # add flat weight to background, so regions far from neurons get counted
    # this will be overwritten with higher weights near neurons
    bkg = [label[x] == 2 for x in CartesianIndices(size(weights))]
    weights = weights .+ bkg
    # over-weight neurons and regions near neurons by neuron-to-background ratio
    if weight_foreground == "proportional"
        multiplier = sum(label .== 2) / sum(label .== 1)
    else
        multiplier = weight_foreground
    end
    to_update = Queue{Tuple}()
    visited = Dict()
    decay_z = 2.0^(-scale_z)
    decay_xy = 2.0^(-scale_xy)
    for coord in CartesianIndices(size(weights))
        if label[coord] == 1
            enqueue!(to_update, (coord, multiplier, coord))
            weights[coord] = multiplier
            visited[coord] = true
        else
            visited[coord] = false
        end
    end
    count = 0
    while !isempty(to_update)
        item = dequeue!(to_update)
        nbs = get_neighbors_cartesian(item[1], size(weights))
        for nb in nbs
            if metric == "euclidean"
                throw("Euclidean metric not fully functional.")
                #weight = multiplier * get_weight(nb, item[3], decay, scale_xy, scale_z)
            elseif metric == "taxicab"
                # nb is a neighbor of our item - check whether to use xy or z scaling
                weight = item[2] * ((item[1][3] != nb[3]) ? decay_z : decay_xy)
            else
                throw("Metric "*metric*" not implemented.")
            end
            # weight is still above background levels, so we will overwrite background
            if weight > 1 && weight > weights[nb]
                enqueue!(to_update, (nb, weight, item[3]))
                weights[nb] = weight
            end
        end
        count += 1
    end
    # don't weight unlabeled points at all (we have to calculate them above as otherwise nearby background would be weighted incorrectly)
    labeled = [label[x] != 0 for x in CartesianIndices(size(weights))]
    weights = weights .* labeled
    
    # don't weight boundary points and make neurons smaller (neural net needs more flexibility here, and their labeling is hard to get consistent)
    if delete_boundary
        for idx in CartesianIndices(size(weights))
            nbs = get_neighbors_cartesian(idx, size(weights))
            for nb in nbs
                # only reweight pixels in xy plane
                if nb[3] != idx[3]
                    continue
                end
                # only overwrite neuron pixels bordering background
                if (label[idx] == 1) && (label[nb] != label[idx]) && (weights[nb] != 0)
                    weights[idx] = 0
                end
            end
        end
    end

    for idx in CartesianIndices(size(weights))
        if !scale_bkg_gap
            weights[idx] = weight_bkg_gap
            continue
        end
        nbs = get_neighbors_cartesian(idx, size(weights))
        neuron_nbs = 0
        if label[idx] == 3 
            for nb in nbs
                if label[nb] == 1
                    neuron_nbs = neuron_nbs + 1 
                end
            end
            weights[idx] = weight_bkg_gap * (neuron_nbs + 1)
        end
    end
     
    return collect(map(x->convert(Float64, x), weights))
end


"""
Generates an HDF5 file, to be input to the UNet, out of a raw image file and a label file. Assumes 3D data.

# Arguments

- `rootpath::String`: Working directory path; all other paths are relative to this.
- `hdf5_path::String`: Path to HDF5 output file to be generated.
- `nrrd_path::String`: Path to NRRD file containing labels. If set to the empty string, only the raw data will be added to the HDF5 file.
- `mhd_path::String`: Path to MHD file containing raw data.

# Optional keyword arguments

- `crop`: `[crop_x, crop_y, crop_z], where every point with coordinates not in the given ranges is cropped out.
- `transpose::Bool`: whether to transpose the x-y coordinates of the image.
- `weight_strategy::String`: method to generate weights. Default and recommended is neighbors, which weights background pixels nearby foreground higher.
    Alternative is proportional, which will weight foreground and background constantly at a value inversely proportional to the number of pixels in those weights.
    The proportional weight function will ignore labels that are not 1 or 2 (including the background-gap label 3).
    This parameter has no effect if `nrrd_path` is the empty string.
- `metric::String`: metric used to infer distance. Default (and only metric currently implemented) is taxicab. 
    This parameter has no effect if `nrrd_path` is the empty string.
- `scale_xy::Real`: Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 0.36.
- `scale_z::Real`: Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.
- `weight_foreground::Real`: weight of foreground (1) label
- `weight_bkg_gap::Real`: weight of background-gap (3) label
- `delete_boundary::Bool`: whether to set the weight of foreground (2) pixels adjacent to background (1 and 3) pixels to 0. Default false.
"""
function make_hdf5(rootpath::String, hdf5_path::String, nrrd_path::String, mhd_path::String; crop=nothing, transpose::Bool=false, weight_strategy::String="neighbors", 
        metric::String="taxicab", scale_xy::Real=0.36, scale_z::Real=1, weight_foreground::Real=4, weight_bkg_gap::Real=24, delete_boundary::Bool=false)
    make_label = (nrrd_path != "")
    if make_label
        label = collect(load(joinpath(rootpath, nrrd_path)))
    end
    raw = read_img(MHD(joinpath(rootpath, mhd_path)))
    if crop != nothing
        if make_label
            label = label[crop[1], crop[2], crop[3]]
        end
        raw = raw[crop[1], crop[2], crop[3]]
    end
    if transpose
        if make_label
            label = permutedims(label, [2,1,3])
        end
        raw = permutedims(raw, [2,1,3])
    end
    if make_label
        # Don't weight unlabeled, weight all neurons and bkg equally (corrected for number of pixels)
        if weight_strategy == "proportional"
            weight_arr = [0, round(sum(label .== 2)/sum(label .== 1)), 1]
            weight_fn = x->weight_arr[Int16(x)+1]
            weight = map(weight_fn, label)
        # Don't weight unlabeled, weight background pixels nearby neuron pixels higher.
        elseif weight_strategy == "neighbors"
            weight = create_weights(label, scale_xy=scale_xy, scale_z=scale_z, weight_foreground=weight_foreground, weight_bkg_gap=weight_bkg_gap, metric=metric, delete_boundary=delete_boundary)
        else
            throw("Weight strategy "*weight_strategy*" not implemented.")
        end
    end
    # Neurons are labeled as 1, assume everything else is background, but weight only real background
    if make_label
        label_new = map(x->Int16(x), label .== 1)
    end
    # weighting
    # weight neuron pixels more heavily since there are fewer of them
    f = h5open(joinpath(rootpath, hdf5_path), "w")
    f["raw"] = collect(raw)
    if make_label
        f["label"] = collect(label_new)
        f["weight"] = collect(weight)
    end
    close(f)
end