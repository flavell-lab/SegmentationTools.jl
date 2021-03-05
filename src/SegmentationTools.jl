module SegmentationTools

using FlavellBase, ImageDataIO, HDF5, Interact, MHDIO, Distributions, StatsBase, LinearAlgebra, PyCall,
    ProgressMeter, FileIO, NRRD, DataStructures, Images, Plots, ImageSegmentation, WormFeatureDetector,
    ImageTransformations, CoordinateTransformations, StaticArrays, Interpolations, Rotations

include("init.jl")
include("unet_visualization.jl")
include("make_unet_input.jl")
include("semantic_segmentation.jl")
include("instance_segmentation.jl")
include("centroid_visualization.jl")
include("crop_worm.jl")

export
    instance_segmentation_output,
    volume,
    instance_segmentation,
    consolidate_labeled_img,
    get_centroids,
    get_activity,
    create_weights,
    make_unet_input_h5,
    view_label_overlay,
    visualize_prediction_accuracy_2D,
    visualize_prediction_accuracy_3D,
    make_plot_grid,
    display_predictions_2D,
    display_predictions_3D,
    centroids_to_img,
    view_roi_2D,
    view_roi_3D,
    view_centroids_2D,
    view_centroids_3D,
    instance_segment_concave,
    get_points,
    distance,
    resample_img,
    compute_mean_iou,
    detect_incorrect_merges,
    watershed_threshold,
    instance_segmentation_threshold,
    instance_segmentation_watershed,
    get_crop_rotate_param,
    crop_rotate,
    crop_rotate!,
    uncrop_img_roi,
    uncrop_img_rois,
    call_unet
end # module
