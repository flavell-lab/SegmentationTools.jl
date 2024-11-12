#using FileIO
#using Printf
#using UNet2D
#using Statistics
#using Images
#using ImageMorphology
#using ImageFiltering
#using LinearAlgebra
#using NRRDIO
#using Interpolations
#using FlavellBase: maxprj, rescale_to_range, create_dir

"""
    img_to_noise_sampler(
        img, cutoff_percent=0.96, dtype=UInt16
    )

Returns a vector containing the dimmest `cutoff_percent` pixels of `img`.

# Arguments
- `img`: image to sample from (2D or 3D)
- `cutoff_percent`: proportion of dimmest pixels to sample
- `dtype`: desired datatype for sampler
"""
function img_to_noise_sampler(img, cutoff_percent=0.96, dtype=UInt16)
    if dtype <: Integer
        img = round.(img)
    end
    sampler = vec(dtype.(img))
    cutoff_level = quantile(sampler, cutoff_percent)
    sampler = sampler[sampler .<= cutoff_level]
    return sampler
end

"""
    noisify(
        img, mask::BitMatrix, sampler
    )

Replaces pixels of `img` that are `false` in `mask` with noise from `sampler`.
Note: will error if any dim of `img` is larger than corresponding dim of `mask` (they should be the same).

# Arguments
- `img`: image to apply noise to (2D)
- `mask`: BitMatrix indicating where to keep original pixel values (2D)
- `sampler`: vector of values for noise sampling
"""
function noisify(img, mask::BitMatrix, sampler)
    noisified_img = [mask[i,j] ? img[i,j] : rand(sampler) for i in 1:size(img, 1), j in 1:size(img, 2)]
    return noisified_img
end

"""
    create_disk(
        radius::Int
    )

Creates a circular structuring element with a specified radius.

# Arguments
- `radius`: radius of the disk
"""
function create_disk(radius::Int)
    center = radius + 1
    strel = falses(2*radius + 1, 2*radius + 1)
    for x in 1:size(strel, 1)
        for y in 1:size(strel, 2)
            if sqrt((center - x)^2 + (center - y)^2) <= radius
                strel[x, y] = true
            end
        end
    end
    return strel
end

"""
    smooth_blob(
        bitmatrix::BitMatrix, kernel_radius::Int = 8
    )

Smooths binary blobs in a given bitmatrix using morphological closing.

# Arguments
- `bitmatrix`: binary image to smooth
- `kernel_radius`: radius of the smoothing kernel
"""
function smooth_blob(bitmatrix::BitMatrix, kernel_radius::Int = 8)
    # Create a circular structuring element
    selem = create_disk(kernel_radius)

    # Apply morphological closing
    closed = closing(bitmatrix, selem)

    return closed
end

"""
    get_bright_pixels(
        img, percentage
    )

Identifies the brightest pixels in an image based on a percentage threshold.

# Arguments
- `img`: image to analyze
- `percentage`: proportion of brightest pixels to consider
"""
function get_bright_pixels(img, percentage)
    threshold = quantile(img[:], 1 - percentage)
    return img .> threshold
end

"""
    get_boundary(
        mask
    )

Extracts the boundary of a binary mask using an edge detection filter.

# Arguments
- `mask`: binary mask from which to extract the boundary
"""
function get_boundary(mask)
    # Use the Laplacian filter to detect edges
    filter = [-1 -1 -1; -1 8 -1; -1 -1 -1]
    boundary_img = imfilter(Float64.(mask), filter)
    return boundary_img .> 0
end

"""
    expand_into_bright_pixels(
        img, mask, bright_pixels
    )

Expands the mask to include neighboring bright pixels.

# Arguments
- `img`: image to analyze
- `mask`: binary mask to expand
- `bright_pixels`: binary mask indicating bright pixel locations
"""
function expand_into_bright_pixels(img, mask, bright_pixels)
    boundary = get_boundary(mask)
    changes = false

    for i in 2:size(img, 1)-1
        for j in 2:size(img, 2)-1
            if boundary[i, j]
                for di in -1:1
                    for dj in -1:1
                        if bright_pixels[i+di, j+dj] && !mask[i+di, j+dj]
                            mask[i+di, j+dj] = true
                            changes = true
                        end
                    end
                end
            end
        end
    end
    return changes
end

"""
    remove_bisections(
        img, mask_old, brightest_cutoff_pct
    )

Expands the mask such that no bright pixels are on its boundary (ie: no potential neuron bisections).

# Arguments
- `img`: image to analyze
- `mask_old`: initial mask to modify
- `brightest_cutoff_pct`: proportion of brightest pixels to retain
"""
function remove_bisections(img, mask_old, brightest_cutoff_pct)
    mask = copy(mask_old)
    bright_pixels = get_bright_pixels(img, brightest_cutoff_pct)
    while expand_into_bright_pixels(img, mask, bright_pixels)
        continue
    end
    return mask
end

"""
    dilate_mask(
        mask, radius
    )

Dilates a binary mask using a specified radius.

# Arguments
- `mask`: binary mask to dilate
- `radius`: dilation radius
"""
function dilate_mask(mask, radius)
    # Create a structuring element for dilation
    selem = trues(2*radius + 1, 2*radius + 1)
    return dilate(mask, selem)
end

"""
    remove_small_blobs(
        mask::BitMatrix
    )

Removes small disconnected blobs from a binary mask, retaining only the largest blob.

# Arguments
- `mask`: binary mask to process
"""
function remove_small_blobs(mask::BitMatrix)
    # Label connected components
    labels = label_components(mask)

    # Count pixels in each blob
    blob_counts = Dict{Int, Int}()
    for val in labels
        blob_counts[val] = get(blob_counts, val, 0) + 1
    end

    # If no blobs (only bg) just return
    if length(blob_counts) == 1
        return mask
    else
        # Remove 0 label (background) from the counts
        delete!(blob_counts, 0)

        # Find the largest blob label
        largest_blob_label = argmax(blob_counts)[1]

        # Create a new bitmatrix where only the largest blob is retained
        cleaned_mask = labels .== largest_blob_label

        return cleaned_mask
    end
end

"""
    mask_to_points(
        mask
    )

Converts a binary mask into a list of coordinates of true values.

# Arguments
- `mask`: binary mask to convert
"""
function mask_to_points(mask)
    points = []
    for i = 1:size(mask, 1) # row (y)
        for j = 1:size(mask, 2) # column (x)
            if mask[i, j]
                push!(points, [j, -i]) # image oriented as displayed, in fourth quadrant
            end
        end
    end
    return points
end

"""
    rotate_point(
        p, angle_rad
    )

Rotates a point counter-clockwise by a given angle in radians.

# Arguments
- `p`: point as a 2D vector
- `angle_rad`: rotation angle in radians
"""
function rotate_point(p, angle_rad)
    # rotates counter-clockwise about top-left corner (origin)
    rotation_matrix = [cos(angle_rad) -sin(angle_rad); sin(angle_rad) cos(angle_rad)]
    return rotation_matrix * p
end

"""
    bounding_box(
        points
    )

Finds the bounding box coordinates for a set of points (no rotation).

# Arguments
- `points`: list of points as 2D vectors to bound
"""
function bounding_box(points)
    min_x, min_y, max_x, max_y = Inf, Inf, -Inf, -Inf
    for p in points
        min_x = min(min_x, p[1])
        min_y = min(min_y, p[2])
        max_x = max(max_x, p[1])
        max_y = max(max_y, p[2])
    end
    return (min_x, min_y, max_x, max_y)
end

"""
    minimum_rotated_bbox(
        mask; ref_angle=0.0
    )

Finds the smallest bounding box for a binary mask, allowing rotation. Picks rotation closest to `ref_angle`

# Arguments
- `mask`: binary mask to analyze

# Optional keyword arguments
- `ref_angle`: reference angle (in radians) for orientation. Default 0.0
"""
function minimum_rotated_bbox(mask; ref_angle=0.0)
    height, width = size(mask)
    points = mask_to_points(mask)
    min_bbox = (Inf, Inf, -Inf, -Inf)
    min_area = Inf
    optimal_angle = 0.0

    for angle_rad in (ref_angle -(pi/4)):(pi/360):(ref_angle +(pi/4))  # increment by 0.5 degrees
        #angle_rad = deg2rad(angle_deg)
        rotated_points = [rotate_point(p, angle_rad) for p in points]
        bbox = bounding_box(rotated_points)

        area = (bbox[3] - bbox[1]) * (bbox[4] - bbox[2])
        if area < min_area
            min_area = area
            min_bbox = bbox
            optimal_angle = angle_rad
        end
    end

    # bbox coord system is y-shifted (og corner stays at origin) and y-flipped from image
    new_height = ceil(Int, sqrt(height^2 + width^2) * max(abs(sin(optimal_angle + atan(height/width))),
                                                          abs(sin(optimal_angle - atan(height/width)))))
    new_width = ceil(Int, sqrt(height^2 + width^2) * max(abs(sin(optimal_angle + atan(width/height))),
                                                        abs(sin(optimal_angle - atan(width/height)))))
    # shift entire image from quadrant 1 towards quadrant 4
    y_change = max(0,
                   width * sin(optimal_angle),
                   height * sin(optimal_angle - (pi/2)),
                   sqrt(height^2 + width^2) * sin(optimal_angle - atan(height/width)))
    # shift entire image from quadrant 3 towards quadrant 4
    x_change = max(0,
                   height * -sin(optimal_angle),
                   width * -sin(optimal_angle + (pi/2)),
                   sqrt(height^2 + width^2) * -sin(optimal_angle + atan(width/height)))
    min_x = max(1, floor(Int, min_bbox[1] + x_change))
    max_x = min(new_width, ceil(Int, min_bbox[3] + x_change))
    min_y = max(1, floor(Int, -(min_bbox[4] - y_change))) # flip to quadrant 1
    max_y = min(new_height, ceil(Int, -(min_bbox[2] - y_change)))  # flip to quadrant 1

    return [min_y, max_y, min_x, max_x], optimal_angle, new_height, new_width
end

"""
    rotate_mask(
        mask, angle_rad, fill_val=0.0
    )

Rotates a binary mask by a given angle.

# Arguments
- `mask`: binary mask to rotate
- `angle_rad`: rotation angle in radians
- `fill_val`: fill value for areas outside the original mask
"""
function rotate_mask(mask, angle_rad, fill_val=0.0)
    # fill_val specifies what value to assign new pixels
    # imrotate rotates clockwise for some reason
    rotated_mask = parent(imrotate(mask, -angle_rad, fill_val) .> 0.5)
    return rotated_mask
end

"""
    rotate_image(
        image, angle_rad, fill_val=0.0, interp_method=Linear()
    )

Rotates an image by a given angle.

# Arguments
- `image`: image to rotate
- `angle_rad`: rotation angle in radians
- `fill_val`: fill value for new pixels
- `interp_method`: interpolation method
"""
function rotate_image(image, angle_rad, fill_val=0.0, interp_method=Linear())
    # fill_val specifies what value to assign new pixels (default Nan)
    # imrotate rotates clockwise for some reason
    rotated_image = parent(imrotate(image, -angle_rad, fill_val, interp_method))
    return rotated_image
end

"""
    actually_crop(
        img, crop_params; sampler=[0]
    )

Crops an image based on specified parameters, filling areas as needed.

# Arguments
- `img`: image to crop
- `crop_params`: coordinates for cropping; takes the form ([x_min, x_max], [y_min, y_max], [z_min, z_max])

# Optional keyword arguments
- `sampler`: values to fill for padding. Default [0]
"""
function actually_crop(img, crop_params; sampler=[0])
    (crop_x, crop_y, _) = crop_params
    image = copy(img)
    dtype = typeof(image[1,1])
    sampler = dtype.(sampler)
    # Expand if dims are too low
    height, width = size(image)
    if crop_y[1] < 1
        diff = 1 - crop_y[1]
        image = vcat(rand(sampler, diff, width), image)
        height += diff
        crop_y[1] += diff
        crop_y[2] += diff
    end
    if crop_y[2] > height
        diff = crop_y[2] - height
        image = vcat(image, rand(sampler, diff, width))
        height += diff
    end
    if crop_x[1] < 1
        diff = 1 - crop_x[1]
        image = hcat(rand(sampler, height, diff), image)
        width += diff
        crop_x[1] += diff
        crop_x[2] += diff
    end
    if crop_x[2] > width
        diff = crop_x[2] - width
        image = hcat(image, rand(sampler, height, diff))
        width += diff
    end

    return image[crop_y[1]:crop_y[2], crop_x[1]:crop_x[2]]
end

"""
    pca_rotate(
        mask; ref_angle=0.0
    )

Finds the principal component rotation angle for a binary mask. Picks rotation closest to `ref_angle`.
Note: Puts "long" side on y-axis.

# Arguments
- `mask`: binary mask to analyze

# Optional keyword arguments
- `ref_angle`: reference angle for rotation. Default 0.0
"""
function pca_rotate(mask; ref_angle=0.0)
    # Step 1: Extract the coordinates of the `true` values
    coords = findall(mask)
    y, x = getindex.(coords, 1), getindex.(coords, 2)

    # Step 2: Center the data
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    data = hcat(x_centered, y_centered)

    # Step 3: Perform SVD
    U, S, Vt = svd(data)

    # Step 4: Extract the principal components
    pc1 = Vt[:, 1] # The first principal component
    pc2 = Vt[:, 2] # The second principal component

    # The angle between the first principal component and the x-axis can be found using the arctangent function
    # Since pc1 is a unit vector, its x-component is cos(angle) and y-component is sin(angle)

    # Calculate the angle for the first principal component
    angle_pc1_rad = atan(pc1[2], pc1[1])  # This returns the angle in radians

    # Put that component on the y-axis
    # TODO: determine desired long axis from crop size
    angle_pc1_rad += pi/2

    # Pick 180 rotation closest to reference angle
    angle_diff = mod((ref_angle - angle_pc1_rad) +pi, 2pi) -pi
    if angle_diff > pi/2
        angle_pc1_rad += pi
    elseif angle_diff < -pi/2
        angle_pc1_rad -= pi
    end

    # Keep btw -pi and pi
    angle_pc1_rad = mod(angle_pc1_rad +pi, 2pi) -pi

    return angle_pc1_rad
end

"""
    crop_to_size_helper(
        mask::BitMatrix, desired_width::Int, desired_height::Int
    )

Crops a mask to specified dimensions without rotation, maximizing the amount of `true` values retained and padding opposite sides equally as needed.

# Arguments
- `mask`: binary mask to crop
- `desired_width`: target width for cropping
- `desired_height`: target height for cropping
"""
function crop_to_size_helper(mask::BitMatrix, desired_width::Int, desired_height::Int)
    test_mask = copy(mask)
    test_h, test_w = size(test_mask)

    # Shrink mask to smallest size containing all true vals
    top_crop = findfirst(row -> any(row), eachrow(test_mask)) -1
    bot_crop = test_h - findlast(row -> any(row), eachrow(test_mask))
    l_crop = findfirst(col -> any(col), eachcol(test_mask)) -1
    r_crop = test_w - findlast(col -> any(col), eachcol(test_mask))

    test_mask = test_mask[(top_crop +1):(test_h -bot_crop), (l_crop +1):(test_w -r_crop)]
    test_h, test_w = size(test_mask)

    # Expand if dims are too low
    if test_h < desired_height
        h_diff = desired_height - test_h
        top_crop -= floor(Int, h_diff/2)
        bot_crop -= ceil(Int, h_diff/2)
        test_mask = vcat(falses(floor(Int, h_diff/2), test_w), test_mask, falses(ceil(Int, h_diff/2), test_w))
        test_h = size(test_mask)[1]
    end
    if test_w < desired_width
        w_diff = desired_width - test_w
        l_crop -= floor(Int, w_diff/2)
        r_crop -= ceil(Int, w_diff/2)
        test_mask = hcat(falses(test_h, floor(Int, w_diff/2)), test_mask, falses(test_h, ceil(Int, w_diff/2)))
        test_w = size(test_mask)[2]
    end

    # Crop least amount of foreground if dims are too high
    loss = 0
    while size(test_mask) != (desired_height, desired_width)
        top_loss = count(test_mask[1,:])
        bot_loss = count(test_mask[test_h,:])
        l_loss = count(test_mask[:,1])
        r_loss = count(test_mask[:,test_w])
        if test_h > desired_height
            if bot_loss > top_loss
                test_mask = test_mask[2:test_h,:]
                loss += top_loss
                top_crop += 1
            else
                test_mask = test_mask[1:test_h-1,:]
                loss += bot_loss
                bot_crop += 1
            end
            test_h -= 1
        end
        if test_w > desired_width
            if r_loss > l_loss
                test_mask = test_mask[:,2:test_w]
                loss += l_loss
                l_crop += 1
            else
                test_mask = test_mask[:,1:test_w-1]
                loss += r_loss
                r_crop += 1
            end
            test_w -= 1
        end
    end

    return (loss, (top_crop, bot_crop, l_crop, r_crop))
end

"""
    crop_to_size(
        mask::BitMatrix, crop_size
    )

Crops a mask using crop_to_size_helper, but allowing for 90-degree rotation to maximize `true` retention.

# Arguments
- `mask`: binary mask to crop
- `crop_size`: target dimensions
"""
function crop_to_size(mask::BitMatrix, crop_size)
    desired_height, desired_width, _ = crop_size
    no_rot_loss, no_rot_crops = crop_to_size_helper(mask, desired_width, desired_height)
    q_rot_loss, q_rot_crops = crop_to_size_helper(mask, desired_height, desired_width)
    actual_height, actual_width = size(mask)

    if no_rot_loss < q_rot_loss
        # this will put desired_height side on y-axis
        return [1+no_rot_crops[1], actual_height-no_rot_crops[2], 1+no_rot_crops[3], actual_width-no_rot_crops[4]]
    else
        # TODO: rotate and edit theta
        # right now this will put desired_height side on x-axis
        return [1+q_rot_crops[1], actual_height-q_rot_crops[2], 1+q_rot_crops[3], actual_width-q_rot_crops[4]]
    end
end

"""
    get_crop_rotate_param(
        mask, vol, crop_size; prev_theta=0.0
    )

Calculates cropping and rotation parameters for a 3D volume.
First, attempts to find a rotation that keeps all `true` values of mask inside cropping area (z-dim flattened).
If that fails, puts PCA dim-1 on the long axis and maximizes `true` value retention when cropping.
Crops in z-dim by maximizing total pixel brightness.

# Arguments
- `mask`: binary mask
- `vol`: volume to process (3D)
- `crop_size`: target crop size

# Optional keyword arguments
- `prev_theta`: rotation angle of previous timepoint (for initialization). Default 0.0
"""
function get_crop_rotate_param(mask, vol, crop_size; prev_theta=0.0)
    # Try minimum bounding box first
    crop_params, theta, rotated_h, rotated_w = minimum_rotated_bbox(mask, ref_angle=prev_theta)
    bbox_crop_h = crop_params[2]-crop_params[1] +1
    bbox_crop_w = crop_params[4]-crop_params[3] +1
    pca_used = false
    if (bbox_crop_h <= crop_size[1] && bbox_crop_w <= crop_size[2])
        diff_h = crop_size[1] - bbox_crop_h
        diff_w = crop_size[2] - bbox_crop_w
        crop_params[1] -= floor(Int, diff_h/2)
        crop_params[2] += ceil(Int, diff_h/2)
        crop_params[3] -= floor(Int, diff_w/2)
        crop_params[4] += ceil(Int, diff_w/2)
    elseif (bbox_crop_w <= crop_size[1] && bbox_crop_h <= crop_size[2])
        diff_h = crop_size[2] - bbox_crop_h
        diff_w = crop_size[1] - bbox_crop_w
        crop_params[1] -= floor(Int, diff_h/2)
        crop_params[2] += ceil(Int, diff_h/2)
        crop_params[3] -= floor(Int, diff_w/2)
        crop_params[4] += ceil(Int, diff_w/2)
    else # If too big, go with PCA
        pca_used = true
        theta = pca_rotate(mask, ref_angle=prev_theta)
        rotated_mask = rotate_mask(mask, theta)
        rotated_h, rotated_w = size(rotated_mask)
        # Sets crop params to right size automatically
        crop_params = crop_to_size(rotated_mask, crop_size)
    end

    # Turn 90 towards prev_theta if oriented incorrectly
    if (crop_params[2] - crop_params[1] +1 != crop_size[1])
        angle_diff = mod((prev_theta - theta) +pi, 2pi) -pi
        if angle_diff > 0
            theta += pi/2
            new_cp3 = crop_params[1]
            new_cp4 = crop_params[2]
            new_cp1 = rotated_w - crop_params[4] +1
            new_cp2 = rotated_w - crop_params[3] +1
        else
            theta -= pi/2
            new_cp1 = crop_params[3]
            new_cp2 = crop_params[4]
            new_cp3 = rotated_h - crop_params[2] +1
            new_cp4 = rotated_h - crop_params[1] +1
        end
        crop_params = [new_cp1, new_cp2, new_cp3, new_cp4]
    end

    crop_x = [crop_params[3], crop_params[4]]
    crop_y = [crop_params[1], crop_params[2]]

    # Crop Z
    crop_z = [1, size(vol, 3)]
    loss_z = sum(mask .* vol[:,:,crop_z], dims=(1,2))
    while (crop_z[2] - crop_z[1] +1) > crop_size[3]
        if (loss_z[1] > loss_z[2])
            crop_z[2] -= 1
            loss_z[2] = sum(mask .* vol[:,:,crop_z[2]])
        else
            crop_z[1] += 1
            loss_z[1] = sum(mask .* vol[:,:,crop_z[1]])
        end
    end

    crop_params_transformed = [crop_x, crop_y, crop_z]
    
    return(crop_params_transformed, theta, pca_used)
end

"""
    crop_image(
        mask, image, crop_params, theta, noise_bool, noise_cutoff_pct; dtype=UInt16
    )

Crops and rotates an image according to given parameters, with optional noise application.
If `noise_bool==false`, pixels outside of mask will be set to 0.

# Arguments
- `mask`: binary mask of pixels to keep
- `image`: image to crop and rotate
- `crop_params`: coordinates for cropping; takes the form ([x_min, x_max], [y_min, y_max], [z_min, z_max])
- `theta`: rotation angle (to be done before cropping)
- `noise_bool`: toggle adding noise to image outside mask
- `noise_cutoff_pct`: proportion of dimmest pixels to sample during noise generation

# Optional keyword arguments
- `dtype`: desired data type for pixels in the cropped image. Default `UInt16`
"""
function crop_image(mask, image, crop_params, theta, noise_bool, noise_cutoff_pct; dtype=UInt16)
    if noise_bool
        noise_sampler = img_to_noise_sampler(image, noise_cutoff_pct, dtype)
        rotated_image = rotate_image(image, theta)
        if dtype <: Integer
            rotated_image = round.(rotated_image)
        end
        rotated_image = dtype.(rotated_image)
        rotated_mask = round.(Bool, rotate_image(mask, theta, false))
        rotated_image = noisify(rotated_image, rotated_mask, noise_sampler) # noisify after rotate allows extrapolated pixels to be noise
        cropped_image = actually_crop(rotated_image, crop_params; sampler=noise_sampler)
    else
        rotated_image = rotate_image(mask .* image, theta)
        if dtype <: Integer
            rotated_image = round.(rotated_image)
        end
        rotated_image = dtype.(rotated_image)
        cropped_image = actually_crop(rotated_image, crop_params)
    end

    return cropped_image
end

"""
    crop_vol(
        mask, vol, crop_params, theta, noise_bool, noise_cutoff_pct; dtype=UInt16
    )

Crops and rotates a volume according to given parameters, with optional noise application.
If `noise_bool==false`, pixels outside of mask will be set to 0.
Crops in z-dim by maximizing total pixel brightness.
Note: calls `crop_image` on each z-slice. TODO: do this more efficiently

# Arguments
- `mask`: binary mask of pixels to keep
- `vol`: volume to crop
- `crop_params`: coordinates for cropping; takes the form ([x_min, x_max], [y_min, y_max], [z_min, z_max])
- `theta`: rotation angle (to be done before cropping)
- `noise_bool`: toggle adding noise to image outside mask (applied to each z-slice)
- `noise_cutoff_pct`: proportion of dimmest pixels of each slice to sample during noise generation

# Optional keyword argument
- `dtype`: desired data type for pixels in the cropped volume. Default `UInt16`
"""
function crop_vol(mask, vol, crop_params, theta, noise_bool, noise_cutoff_pct; dtype=UInt16)
    cropped_vol = zeros(typeof(vol[1,1,1]), ((crop_params[2][2]-crop_params[2][1]+1), (crop_params[1][2]-crop_params[1][1]+1), (crop_params[3][2]-crop_params[3][1]+1)))

    for (z_slice_new, z_slice_old) in enumerate(crop_params[3][1]:crop_params[3][2])
        cropped_slice = crop_image(mask, vol[:,:,z_slice_old], crop_params, theta, noise_bool, noise_cutoff_pct, dtype=dtype)#round.(typeof(vol[1,1,1]), crop_image(mask, vol[:,:,z_slice], crop_params, theta, noise_bool, noise_cutoff_pct))
        cropped_vol[:,:,z_slice_new] = cropped_slice
    end

    cropped_mip = maxprj(cropped_vol, dims=3)
    cropped_mip = rescale_to_range.(cropped_mip, minimum(cropped_mip), maximum(cropped_mip), 0.0, 1.0)

    return cropped_vol, cropped_mip
end

"""
    crop_rotate_dset!(
        path_dir_nrrd::String, path_dir_nrrd_crop::String, path_dir_MIP_crop::String, path_model::String, t_range, ch_list, dict_crop_rot_param::Dict,
        spacing_axi::AbstractFloat, spacing_lat::AbstractFloat, f_basename::Function, save_MIP::Bool;
        confidence_cutoff=0.5, mask_smoothness=8, brightest_cutoff_pct=0.08, dilation_amount=2, noise_bool=true, noise_cutoff_pct=0.94, crop_size=(284, 120, 64)
    )

Crops each timepoint in `t_range` of a dataset using CropNet.
CropNet identifies the head of the worm (z-dim flattened), does some postprocessing, and replaces everything else with noise.
If `noise_bool==false`, non-head pixels will be set to 0.
Crops in z-dim by maximizing total pixel brightness.
Resulting `crop_size` is the same for all timepoints.

# Arguments
- `path_dir_nrrd`: path to source NRRD files (output of shear-correction)
- `path_dir_nrrd_crop`: path for saving cropped NRRD files
- `path_dir_MIP_crop`: path for saving cropped maximum intensity projections
- `path_model`: path to the CropNet model file
- `t_range`: time range for processing
- `ch_list`: list of channels
- `dict_crop_rot_param`: dictionary for cropping/rotation parameters (will be filled by function)
- `spacing_axi`: axial spacing for volume (only involved in writing NRRD)
- `spacing_lat`: lateral spacing for volume (only involved in writing NRRD)
- `f_basename`: function that returns the base name of the input image file
- `save_MIP`: whether to save MIP images

# Optional keyword arguments
- `confidence_cutoff`: threshold for binarizing CropNet output. Default 0.5
- `mask_smoothness`: radius of smoothing kernel in mask generation. Default 8
- `brightest_cutoff_pct`: proportion of brightest pixels to avoid bisecting in mask generation. Default 0.08
- `dilation_amount`: radius of expansion in mask generation. Default 2
- `noise_bool`: toggle adding noise to image outside mask (applied to each z-slice). Default true
- `noise_cutoff_pct`: proportion of dimmest pixels of each slice to sample during noise generation. Default 0.94
- `crop_size`: target crop size. Default (284, 120, 64)

# Output
- Cropped and rotated volumes are saved in `path_dir_nrrd_crop`
- Maximum intensity projections of volumes are saved in `path_dir_MIP_crop`
- In-place modification of the `dict_crop_rot_param` dictionary with the cropping and rotation parameters, mask, and cropping strategy for each time point
- Returns a tuple containing errors and time points where the worm might have been out of focus
"""
function crop_rotate_dset!(path_dir_nrrd::String, path_dir_nrrd_crop::String, path_dir_MIP_crop::String, path_model::String, t_range, ch_list, dict_crop_rot_param::Dict,
    spacing_axi::AbstractFloat, spacing_lat::AbstractFloat, f_basename::Function, save_MIP::Bool;
    confidence_cutoff=0.5, mask_smoothness=8, brightest_cutoff_pct=0.08, dilation_amount=2, noise_bool=true, noise_cutoff_pct=0.94, crop_size=(284, 120, 64))
    
    # Load model
    model = create_model(1, 1, 64, path_model)

    create_dir.([path_dir_nrrd_crop, path_dir_MIP_crop])
    dict_error = Dict{Int, Any}()
    focus_issues = []
    theta = 0.0 # to initialize theta search

    for t in t_range
#        try
            vol = read_img(NRRD(joinpath(path_dir_nrrd, f_basename(t, ch_list[1]) * ".nrrd")))
            vol = UInt16.(max.(0, vol))

            mip = maxprj(vol, dims=3)
            mip = rescale_to_range.(mip, minimum(mip), maximum(mip), 0.0, 1.0)
            mip = convert(Matrix{Float32}, mip)

            if haskey(dict_crop_rot_param, t)
                crop_params = dict_crop_rot_param[t]["crop"]
                theta = dict_crop_rot_param[t]["θ"]
                mask = dict_crop_rot_param[t]["mask"]
            else
                dict_crop_rot_param[t] = Dict()

                # Get crop params
                model_input = UNet2D.standardize(mip)
                model_result = eval_model(model_input, model)
                model_result_mask = model_result .> confidence_cutoff
                if count(model_result_mask) < 5 #!any(model_result_mask)
                    #TODO: report in dict_error
                    #TODO: actually set threshold mask size (currently 5px)
                    println("No worm detected in timepoint $(t)")
                    bg_cutoff = quantile(vec(model_input), 0.96) # we generally use 96% as noise-neuron cutoff
                    mask = model_input .> bg_cutoff
                    (crop_params, theta, pca_used) = get_crop_rotate_param(mask, vol, crop_size, prev_theta=theta)
                    mask = trues(size(mask)) # so nothing is replaced with noise
                else
                    smoothed_mask = smooth_blob(model_result_mask, mask_smoothness) # improves accuracy
                    no_bisections_mask = remove_bisections(mip, smoothed_mask, brightest_cutoff_pct) # don't want bisected neurons
                    dilated_mask = dilate_mask(no_bisections_mask, dilation_amount) # play it safe on the bisected neurons
                    cleaned_mask = remove_small_blobs(dilated_mask) # we only want the one blob
                    mask = cleaned_mask
                    (crop_params, theta, pca_used) = get_crop_rotate_param(mask, vol, crop_size, prev_theta=theta)
                end

                dict_crop_rot_param[t]["crop"] = crop_params
                dict_crop_rot_param[t]["θ"] = theta
                dict_crop_rot_param[t]["mask"] = mask
                dict_crop_rot_param[t]["pca_used"] = pca_used
            end

            for ch in ch_list
                bname = f_basename(t, ch)
                vol = read_img(NRRD(joinpath(path_dir_nrrd, bname * ".nrrd")))
                vol = UInt16.(max.(0, vol)) 

                # Crop
                (cropped_vol, cropped_mip) = crop_vol(mask, vol, crop_params, theta, noise_bool, noise_cutoff_pct)

                # Save
                path_base = joinpath(path_dir_nrrd_crop, bname)
                path_nrrd = path_base *".nrrd"
                path_png = joinpath(path_dir_MIP_crop, bname *".png")
                write_nrrd(path_nrrd, cropped_vol, (spacing_lat, spacing_lat, spacing_axi))
                if save_MIP
                    FileIO.save(path_png, clamp01nan.(cropped_mip))
                end
            end

            # still crop out-of-focus data but give error message
            if crop_params[3][1] <= 1 || crop_params[3][2] >= size(vol)[3]
                append!(focus_issues, t)
            end
#        catch e_
#            dict_error[t] = e_
#        end
    end

    return dict_error, focus_issues
end

"""
    crop_rotate_dset!(
        param_path::Dict, param::Dict, t_range, ch_list, dict_crop_rot_param::Dict; save_MIP::Bool=true,
        nrrd_dir_key::String="path_dir_nrrd_shearcorrect", nrrd_crop_dir_key::String="path_dir_nrrd_crop", mip_crop_dir_key::String="path_dir_MIP_crop",
        model_key::String="path_crop_model"
    )

Crops each timepoint in `t_range` of a dataset using CropNet, using configuration dictionaries for paths and parameters.
CropNet identifies the head of the worm (z-dim flattened), does some postprocessing, and replaces everything else with noise.
Crops in z-dim by maximizing total pixel brightness.

# Arguments

- `param_path`: dictionary containing paths for input/output files
- `param`: dictionary of additional parameters for processing
- `t_range`: time range for processing
- `ch_list`: list of channels to process
- `dict_crop_rot_param`: dictionary storing cropping and rotation parameters

# Optional keyword arguments
- `save_MIP`: whether to save MIP images. Default true
- `nrrd_dir_key`: key in `param_path` for the input NRRD directory. Default "path_dir_nrrd_shearcorrect"
- `nrrd_crop_dir_key`: key in `param_path` for the cropped (output) NRRD directory. Default "path_dir_nrrd_crop"
- `mip_crop_dir_key`: key in `param_path` for the cropped (output) MIP directory. Default "path_dir_MIP_crop"
- `model_key`: key in `param_path` for the path to the CropNet model file. Default "path_crop_model"
"""
function crop_rotate_dset!(param_path::Dict, param::Dict, t_range, ch_list, dict_crop_rot_param::Dict; save_MIP::Bool=true,
    nrrd_dir_key::String="path_dir_nrrd_shearcorrect", nrrd_crop_dir_key::String="path_dir_nrrd_crop", mip_crop_dir_key::String="path_dir_MIP_crop",
    model_key::String="path_crop_model")
    path_dir_nrrd = param_path[nrrd_dir_key]
    path_dir_nrrd_crop = param_path[nrrd_crop_dir_key]
    path_dir_MIP_crop = param_path[mip_crop_dir_key]
    path_model = param_path[model_key]
    #threshold_size = param["crop_threshold_size"] #TODO: remove this param
    #threshold_intensity = param["crop_threshold_intensity"] #TODO: remove this param
    #TODO: allow save_MIP=false
    spacing_axi = param["spacing_axi"]
    spacing_lat = param["spacing_lat"]
    f_basename = param_path["get_basename"]
    crop_size = param["crop_size"]

    crop_rotate_dset!(path_dir_nrrd, path_dir_nrrd_crop, path_dir_MIP_crop, path_model, t_range, ch_list, dict_crop_rot_param,
        spacing_axi, spacing_lat, f_basename, save_MIP; crop_size=crop_size)
end

"""
    crop_rotate_image(
        image, crop_params, theta; mask=trues(size(image)), noise_bool=false, noise_cutoff_pct=0.0, crop_size=(284, 120, 64)
    )

Crops and rotates a single 2D image based on specified parameters

# Arguments
- `image`: image to crop and rotate
- `crop_params`: coordinates for cropping; takes the form ([x_min, x_max], [y_min, y_max], [z_min, z_max])
- `theta`: rotation angle (to be done before cropping)

# Optional keyword arguments
- `mask`: binary mask of pixels to keep. Default `trues(size(image))`
- `noise_bool`: toggle adding noise to image outside mask. Default false
- `noise_cutoff_pct`: proportion of dimmest pixels to sample during noise generation. Default 0.0
- `crop_size`: target crop size. Default (284, 120, 64)
"""
function crop_rotate_image(image, crop_params, theta; mask=trues(size(image)), noise_bool=false, noise_cutoff_pct=0.0, crop_size=(284, 120, 64))
    cropped_img = crop_image(mask, image, crop_params, theta, noise_bool, noise_cutoff_pct, dtype=typeof(image[1,1]))

    return cropped_img
end

"""
    crop_rotate_volume(
        vol, crop_params, theta; mask=trues(size(vol)[1:2]), noise_bool=false, noise_cutoff_pct=0.0, crop_size=(284, 120, 64)
    )

Crops and rotates a 3D volume based on specified parameters

# Arguments
- `vol`: volume to crop and rotate.
- `crop_params`: coordinates for cropping; takes the form ([x_min, x_max], [y_min, y_max], [z_min, z_max])
- `theta`: rotation angle (to be done before cropping)

# Optional keyword arguments
- `mask`: binary mask of pixels to keep. Default `trues(size(image))`
- `noise_bool`: toggle adding noise to image outside mask. Default false
- `noise_cutoff_pct`: proportion of dimmest pixels to sample during noise generation. Default 0.0
- `crop_size`: target crop size. Default (284, 120, 64)
"""
function crop_rotate_volume(vol, crop_params, theta; mask=trues(size(vol)[1:2]), noise_bool=false, noise_cutoff_pct=0.0, crop_size=(284, 120, 64))
    (cropped_vol, cropped_mip) = crop_vol(mask, vol, crop_params, theta, noise_bool, noise_cutoff_pct, dtype=typeof(vol[1,1,1]))

    return cropped_vol, cropped_mip
end