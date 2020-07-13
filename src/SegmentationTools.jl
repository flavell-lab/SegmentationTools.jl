module SegmentationTools

using FlavellBase, ImageDataIO, HDF5, Interact, MHDIO, CaSegmentation, Distributions,
    ProgressMeter, FileIO, NRRD, DataStructures, Images, Plots, ImageSegmentation

include("unet_visualization.jl")
include("make_unet_input.jl")
include("instance_segmentation.jl")
include("centroid_visualization.jl")

export
    instance_segmentation_output,
    volume,
    instance_segmentation,
    consolidate_labeled_img,
    get_centroids,
    get_activity,
    create_weights,
    make_hdf5,
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
    view_centroids_3D
end # module
