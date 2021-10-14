var documenterSearchIndex = {"docs":
[{"location":"segment/#Neuron-Segmentation-API","page":"Neuron Segmentation API","title":"Neuron Segmentation API","text":"","category":"section"},{"location":"segment/#Semantic-Segmentation","page":"Neuron Segmentation API","title":"Semantic Segmentation","text":"","category":"section"},{"location":"segment/","page":"Neuron Segmentation API","title":"Neuron Segmentation API","text":"make_unet_input_h5\ncall_unet","category":"page"},{"location":"segment/#SegmentationTools.make_unet_input_h5","page":"Neuron Segmentation API","title":"SegmentationTools.make_unet_input_h5","text":"Makes a UNet input file. This function supports making files either for training or prediction.\n\nArguments\n\nimg_raw::Array: Raw image\nimg_label::Union{Nothing, Array}: Image label. If nothing, label and weight will not be generated.\npath_h5::String: Path to HDF5 output directory.\ncrop (optional, default nothing): [crop_x, crop_y, crop_z], where every point with coordinates not in the given ranges is cropped out.\ntranspose::Bool (optional, default false): whether to transpose the x-y coordinates of the image.\nweight_strategy::String (optional): method to generate weights from labels. Default and recommended is neighbors, which weights background pixels nearby foreground higher.  Alternative is proportional, which will weight foreground and background constantly at a value inversely proportional to the number of pixels in those weights.  The proportional weight function will ignore labels that are not 1 or 2 (including the background-gap label 3).  This parameter has no effect if img_label is the empty string.\nmetric::String (optional): metric used to infer distance. Default (and only metric currently implemented) is taxicab.   This parameter has no effect if img_label is the empty string.\nscale_xy::Real (optional): Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 1.   This parameter has no effect if img_label is the empty string.\nscale_z::Real (optional): Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.   This parameter has no effect if img_label is the empty string.\nweight_foreground::Real (optional, default 6): weight of foreground (1) label   This parameter has no effect if img_label is the empty string.\nweight_bkg_gap::Real (optional, default 10): weight of background-gap (3) label   This parameter has no effect if img_label is the empty string.\nboundary_weight (optional): weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.   This parameter has no effect if img_label is the empty string.\nbin_scale (optional): scale to bin image in each dimension [X,Y,Z]. Default 1,1,1.\nSN_reduction_factor (optional): amount to reduce. Default 1 (no reduction)\nSN_percent (optional): percentile to estimate std of image from. Default 16.\nscale_bkg_gap::Bool (optional): whether to upweight background-gap pixels for each neuron pixel they border. Default false.   This parameter has no effect if img_label is the empty string.\n\n\n\n\n\nMakes UNet input files from all files in a directory. This function supports making files either for training or prediction.\n\nArguments\n\npath_mhd::String: Path to raw data MHD files\npath_nrrd::Union{Nothing, String}: Path to NRRD label files. If nothing, labels and weights will not be generated.\npath_h5::String: Path to HDF5 output files.\ncrop (optional, default nothing): [crop_x, crop_y, crop_z], where every point with coordinates not in the given ranges is cropped out.\ntranspose::Bool (optional, default false): whether to transpose the x-y coordinates of the image.\nweight_strategy::String (optional): method to generate weights from labels. Default and recommended is neighbors, which weights background pixels nearby foreground higher.  Alternative is proportional, which will weight foreground and background constantly at a value inversely proportional to the number of pixels in those weights.  The proportional weight function will ignore labels that are not 1 or 2 (including the background-gap label 3).  This parameter has no effect if nrrd_path is the empty string.\nmetric::String (optional): metric used to infer distance. Default (and only metric currently implemented) is taxicab.   This parameter has no effect if nrrd_path is the empty string.\nscale_xy::Real (optional): Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 1.   This parameter has no effect if nrrd_path is the empty string.\nscale_z::Real (optional): Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.   This parameter has no effect if nrrd_path is the empty string.\nweight_foreground::Real (optional, default 6): weight of foreground (1) label   This parameter has no effect if nrrd_path is the empty string.\nweight_bkg_gap::Real (optional, default 10): weight of background-gap (3) label   This parameter has no effect if nrrd_path is the empty string.\nboundary_weight (optional): weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.   This parameter has no effect if nrrd_path is the empty string.\nbin_scale (optional): scale to bin image in each dimension [X,Y,Z]. Default 1,1,1.\nSN_reduction_factor (optional): amount to reduce. Default 1 (no reduction)\nSN_percent (optional): percentile to estimate std of image from. Default 16.\nscale_bkg_gap::Bool (optional): whether to upweight background-gap pixels for each neuron pixel they border. Default false.   This parameter has no effect if nrrd_path is the empty string.\n\n\n\n\n\nGeneratesn HDF5 file, to be input to the UNet, out of a raw image file and a label file. Assumes 3D data.\n\nArguments\n\nparam_path::Dict: Dictionary containing paths to directories and a get_basename function that returns MHD file names, including:\npath_dir_unet_data: Path to UNet input and output data\npath_dir_mhd::String: Path to MHD files\nt_range: Time points to watershed\nch_marker::Int: Marker channel\nf_basename::Function: Function that returns the name of MHD files\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.call_unet","page":"Neuron Segmentation API","title":"SegmentationTools.call_unet","text":"Makes a local copy of a parameter file, modifies directories in that parameter file, then calls the UNet.\n\nArguments\n\nparam_path: path parameter dictionary including:\npath_root_process: Root data path\npath_dir_unet_data: Path to UNet input and output directory\npath_unet_pred: Path to the predict.py file in pytorch-3d-unet installation\npath_unet_param: Path to UNet prediction parameter file\npath_unet_py_env: Path to a script that initializes the relevant environment variables for the UNet to run\n\n\n\n\n\n","category":"function"},{"location":"segment/#Watershed-Threshold-Instance-Segmentation","page":"Neuron Segmentation API","title":"Watershed-Threshold Instance Segmentation","text":"","category":"section"},{"location":"segment/","page":"Neuron Segmentation API","title":"Neuron Segmentation API","text":"instance_segmentation_watershed\ninstance_segmentation_threshold\ndetect_incorrect_merges\nwatershed_threshold\ninstance_segmentation\nconsolidate_labeled_img","category":"page"},{"location":"segment/#SegmentationTools.instance_segmentation_watershed","page":"Neuron Segmentation API","title":"SegmentationTools.instance_segmentation_watershed","text":"Runs watershed instance segmentation on all given frames and can output to various files (centroids, activity measurements, and image ROIs). Skips a given output method if the corresponding output directory was empty. Returns dictionary of results and a list of error frames (most likely because the worm was not in the field of view).\n\nArguments\n\nparam::Dict: Dictionary containing parameters, including:\nseg_threshold_unet: Confidence threshold of the UNet output for a pixel to be counted as a neuron.\nseg_min_neuron_size: Minimum neuron size, in voxels\nseg_threshold_watershed: Confidence thresholds of the UNet output for a pixel to be counted as a neuron in each watershed step\nseg_watershed_min_neuron_sizes: Minimum neuron sizes, in voxels, in each watershed step\nparam_path::Dict: Dictionary containing paths to directories and a get_basename function that returns MHD file names, including:\npath_dir_unet_data: Path to UNet output data (input to the watershed algorithm)\npath_dir_roi: Path to non-watershed ROI ouptut data\npath_dir_roi_watershed: Path to watershed ROI output data\npath_dir_marker_signal: Path to marker channel signal output data\npath_dir_centroid: Path to centroid output data\npath_dir_mhd::String: Path to MHD files\nt_range: Time points to watershed\nf_basename::Function: Function that returns the name of MHD files\nsave_centroid::Bool (optional): Whether to save centroids. Default false\nsave_signal::Bool (optional): Whether to save marker signal. Default false\nsave_roi::Bool (optional): Whether to save ROIs before watershedding. Default false\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.instance_segmentation_threshold","page":"Neuron Segmentation API","title":"SegmentationTools.instance_segmentation_threshold","text":"Further instance segments a preliminary ROI image by thresholding UNet predictions and checking if ROIs split during thresholding.\n\nArguments:\n\nimg_roi: image that maps points to their current ROIs\npredictions: UNet raw predictions\n\nOptional keyword arguments\n\nthresholds: Array of threshold values - at each value, check if an ROI was split. Default [0.7, 0.8, 0.9]\nneuron_sizes: Array of neuron size values, one per threshold.   Neurons that were found in a threshold that are smaller than the corresponding value are discarded and not counted for ROI split evidence.   Default [5,4,4]\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.detect_incorrect_merges","page":"Neuron Segmentation API","title":"SegmentationTools.detect_incorrect_merges","text":"Detects incorrectly merged ROIs via thresholding. Thresholds the UNet raw output multiple times, checking if  an ROI gets split into multiple, smaller ROIs at higher threshold values.\n\nArguments:\n\nimg_roi: Current ROIs for the image - the method checks each ROI for incorrect merging\npredictions: UNet raw output (not thresholded)\nthresholds: Array of threshold values - at each value, check if an ROI was split\nneuron_sizes: Array of neuron size values, one per threshold.   Neurons that were found in a threshold that are smaller than the corresponding value are discarded and not counted for ROI split evidence.\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.watershed_threshold","page":"Neuron Segmentation API","title":"SegmentationTools.watershed_threshold","text":"Watersheds an ROI, taking as input its peaks (found previously via thresholding) and the UNet raw output.\n\nArguments:\n\npoints: Set of points in the ROI to watershed.\ncentroid_matches: Set of centroids in the points in question - each centroid will spawn a new ROI via watershed.\npredictions: UNet raw output (not thresholded)\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.instance_segmentation","page":"Neuron Segmentation API","title":"SegmentationTools.instance_segmentation","text":"Runs instance segmentation on a frame. Removes detected objects that are too small to be neurons.\n\nArguments\n\n`predictions: UNet predictions array\n\nOptional keyword arguments\n\nmin_neuron_size::Integer: smallest neuron size, in voxels. Default 7.\n\n\n\n\n\n","category":"function"},{"location":"segment/#SegmentationTools.consolidate_labeled_img","page":"Neuron Segmentation API","title":"SegmentationTools.consolidate_labeled_img","text":"Converts an instance-segmentation image labeled_img to a ROI image. Ignores ROIs smaller than the minimum size min_neuron_size. \n\n\n\n\n\n","category":"function"},{"location":"segment/#Watershed-Concave-Instance-Segmentation-(currently-not-used)","page":"Neuron Segmentation API","title":"Watershed-Concave Instance Segmentation (currently not used)","text":"","category":"section"},{"location":"segment/","page":"Neuron Segmentation API","title":"Neuron Segmentation API","text":"instance_segment_concave","category":"page"},{"location":"segment/#SegmentationTools.instance_segment_concave","page":"Neuron Segmentation API","title":"SegmentationTools.instance_segment_concave","text":"Recursively segments all concave neurons in an image. \n\nArguments\n\nimg_roi: Image to segment\n\nOptional keyword arguments\n\nthreshold_scale::Real: Neurons less concave than this won't be segmented. Default 0.3\nnum_neurons::Real: Maximum number of concave neurons per frame. Defaul 10.\nzscale::Real: Scale of z-axis relative to xy plane. Default 1.\nmin_neuron_size::Integer: Minimum size of a neuron (in pixels). Default 10.\nscale_recurse_multiply::Real: Factor to increase the concavity threshold for recursive segmentation. Default 1.5.\ninit_scale::Real: Amount to expand first neuron before computing location of second neuron. Default 0.7. \n\n\n\n\n\n","category":"function"},{"location":"visualize/#Data-Visualization-API","page":"Data Visualization API","title":"Data Visualization API","text":"","category":"section"},{"location":"visualize/#Visualize-ROIs","page":"Data Visualization API","title":"Visualize ROIs","text":"","category":"section"},{"location":"visualize/","page":"Data Visualization API","title":"Data Visualization API","text":"view_roi_3D\nview_roi_2D\nmake_plot_grid","category":"page"},{"location":"visualize/#SegmentationTools.view_roi_3D","page":"Data Visualization API","title":"SegmentationTools.view_roi_3D","text":"Plots instance segmentation image img_roi, where each object is given a different color. Can also plot raw data and semantic segmentation data for comparison.\n\nArguments\n\nraw: 3D raw image. If set to nothing, it will not be plotted.\npredicted: 3D semantic segmentation image. If set to nothing, it will not be plotted.\nimg_roi: 3D instance segmentation image\n\nOptional keyword arguments\n\ncolor_brightness: minimum RGB value (out of 1) that an object will be plotted with\nplot_size: size of the plot\naxis: axis to project, default 3\nraw_contrast: contrast of raw image, default 1\nlabeled_neurons: neurons that should have a specific color, as an array of arrays.\nlabel_colors: an array of colors, one for each array in labeled_neurons\nneuron_color: the color of non-labeled neurons. If not supplied, all of them will be random different colors.\noverlay_intensity: intensity of ROI overlay on raw image\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.view_roi_2D","page":"Data Visualization API","title":"SegmentationTools.view_roi_2D","text":"Plots instance segmentation image img_roi, where each object is given a different color. Can also plot raw data and semantic segmentation data for comparison.\n\nArguments\n\nraw: 2D raw image. If set to nothing, it will not be plotted.\npredicted: 2D semantic segmentation image. If set to nothing, it will not be plotted.\nimg_roi: 2D instance segmentation image\n\nOptional keyword arguments\n\ncolor_brightness: minimum RGB value (out of 1) that an object will be plotted with\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.make_plot_grid","page":"Data Visualization API","title":"SegmentationTools.make_plot_grid","text":"Makes grid out of many smaller plots. \n\nArguments\n\nplots: List of things to be plotted. Each item must be something that could be input to the plot function.\ncols::Integer: Number of columns in array of plots to be created\nplot_size: Size of resulting plot per row.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#Visualize-ROI-Centroids","page":"Data Visualization API","title":"Visualize ROI Centroids","text":"","category":"section"},{"location":"visualize/","page":"Data Visualization API","title":"Data Visualization API","text":"centroids_to_img\nview_centroids_3D\nview_centroids_2D","category":"page"},{"location":"visualize/#SegmentationTools.centroids_to_img","page":"Data Visualization API","title":"SegmentationTools.centroids_to_img","text":"Given an image of size imsize, converts centroids into an image mask of that size.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.view_centroids_3D","page":"Data Visualization API","title":"SegmentationTools.view_centroids_3D","text":"Displays the centroids of an image.\n\nArguments:\n\nimg: Image\ncentriods: Centroids of the image, to be superimposed on the image.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.view_centroids_2D","page":"Data Visualization API","title":"SegmentationTools.view_centroids_2D","text":"Displays the centroids of an image.\n\nArguments:\n\nimg: Image\ncentriods: Centroids of the image, to be superimposed on the image. They can be 3D; if they are, the first dimension will be ignored.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#Visualize-UNet","page":"Data Visualization API","title":"Visualize UNet","text":"","category":"section"},{"location":"visualize/","page":"Data Visualization API","title":"Data Visualization API","text":"view_label_overlay\nvisualize_prediction_accuracy_3D\nvisualize_prediction_accuracy_2D\ndisplay_predictions_3D\ndisplay_predictions_2D","category":"page"},{"location":"visualize/#SegmentationTools.view_label_overlay","page":"Data Visualization API","title":"SegmentationTools.view_label_overlay","text":"Makes image of the raw data overlaid with a translucent label.\n\nArguments\n\nimg: raw data image (2D)\nlabel: labels for the image\nweight: mask of which label values to include. Pixels with a weight of 0 will be plotted in the   raw, but not labeled, data; pixels with a weight of 1 will be plotted with raw data overlaid with label.\n\nOptional keyword arguments\n\ncontrast::Real: Contrast factor for raw image. Default 1 (no adjustment)\nlabel_intensity::Real: Intensity of label, from 0 to 1. Default 0.5.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.visualize_prediction_accuracy_3D","page":"Data Visualization API","title":"SegmentationTools.visualize_prediction_accuracy_3D","text":"Generates an image which compares the predictions of the neural net with the label. Green = match, red = mismatch. Assumes the predictions and labels are binary 3D arrays.\n\nArguments\n\npredicted: neural net predictions\nactual: actual labels\nweight: pixel weights; weight of 0 is ignored and not plotted.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.visualize_prediction_accuracy_2D","page":"Data Visualization API","title":"SegmentationTools.visualize_prediction_accuracy_2D","text":"Generates an image which compares the predictions of the neural net with the label. Green = match, red = mismatch. Assumes the predictions and labels are binary 2D arrays.\n\nArguments\n\npredicted: neural net predictions\nactual: actual labels\nweight: pixel weights; weight of 0 is ignored and not plotted.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.display_predictions_3D","page":"Data Visualization API","title":"SegmentationTools.display_predictions_3D","text":"Compares multiple different neural network predictions of the raw dataset, using an interactive slider to toggle between z-planes of the 3D dataset.\n\nArguments\n\nraw: 3D raw dataset\nlabel: labels on raw dataset. Set to nothing to avoid displaying labels (for instance, on a testing datset).\nweight: weights on the labels. Set to nothing if you are not displaying labels.\npredictions_array: various predictions of the raw dataset.\n\nOptional keyword arguments\n\ncols::Integer: maximum number of columns in the plot. Default 7.\nplot_size: size of plot per row. Default (1800, 750).\naxis: axis to project, default 3\ndisplay_accuracy::Bool: whether to display prediction accuracy (green for match, red for mismatch). Default true.\ncontrast::Real: contrast of raw image. Default 1.\n\n\n\n\n\n","category":"function"},{"location":"visualize/#SegmentationTools.display_predictions_2D","page":"Data Visualization API","title":"SegmentationTools.display_predictions_2D","text":"Compares multiple different neural network predictions of the raw dataset, in comparison with the label and weight samples. The order of the plots is a plot of the raw data, followed by a plot of the weights, followed by plots of raw predictions and prediction vs label differential (in the order the predictions were specified in the array).\n\nArguments\n\nraw: 2D raw dataset\nlabel: labels on raw dataset. Set to nothing to avoid displaying labels (for instance, on a testing datset).\nweight: weights on the labels. Set to nothing if you are not displaying labels.\npredictions_array: various predictions of the raw dataset.\n\nOptional keyword arguments\n\ncols::Integer: maximum number of columns in the plot. Default 7.\nplot_size: size of plot per row. Default (1800, 750).\ndisplay_accuracy::Bool: whether to display prediction accuracy (green for match, red for mismatch). Default true.\ncontrast::Real: contrast of raw image. Default 1.\n\n\n\n\n\n","category":"function"},{"location":"extract/#ROI-Data-Extraction-API","page":"ROI Data Extraction API","title":"ROI Data Extraction API","text":"","category":"section"},{"location":"extract/","page":"ROI Data Extraction API","title":"ROI Data Extraction API","text":"get_centroids\nget_activity","category":"page"},{"location":"extract/#SegmentationTools.get_centroids","page":"ROI Data Extraction API","title":"SegmentationTools.get_centroids","text":"Gets centroids from image ROI img_roi.\n\n\n\n\n\n","category":"function"},{"location":"extract/#SegmentationTools.get_activity","page":"ROI Data Extraction API","title":"SegmentationTools.get_activity","text":"Returns the average activity of all ROIs in an image.\n\nArguments\n\nimg_roi: ROI-labeled image\nimg: raw image\nactivity_func (optional, default mean): function to apply to pixel intensities of the ROI tocompute the activity for the entire ROI.\n\n\n\n\n\n","category":"function"},{"location":"crop/#Cropping-API","page":"Cropping API","title":"Cropping API","text":"","category":"section"},{"location":"crop/","page":"Cropping API","title":"Cropping API","text":"crop_rotate!\ncrop_rotate\nget_crop_rotate_param\nuncrop_img_rois\nuncrop_img_roi","category":"page"},{"location":"crop/#SegmentationTools.crop_rotate!","page":"Cropping API","title":"SegmentationTools.crop_rotate!","text":"Crops and rotates a set of images.\n\nArguments\n\nparam_path::Dict: Dictionary containing paths to directories and a get_basename function that returns MHD file names.\nparam::Dict: Dictionary containing parameters, including:\ncrop_threshold_intensity: minimum number of standard deviations above the mean that a pixel must be for it to be categorized as part of a feature\ncrop_threshold_size: minimum size of a feature for it to be categorized as part of the worm\nspacing_axi: Axial (z) spacing of the pixels, in um\nspacing_lat: Lateral (xy) spacing of the pixels, in um\nt_range: Time points to crop\nch_list: Channels to crop\ndict_crop_rot_param::Dict: For each time point, the cropping parameters to use for that time point.  If the cropping parameters at a time point are not found, they will be stored in the dictionary, modifying it.\nsave_MIP::Bool (optional): Whether to save png files. Default true\nmhd_dir_key::String (optional): Key in param_path to directory to input MHD files. Default path_dir_mhd_shearcorrect\nmhd_crop_dir_key::String (optional): Key in param_path to directory to output MHD files. Default path_dir_mhd_crop\nmip_crop_dir_key::String (optional): Key in param_path to directory to output MIP files. Default path_dir_MIP_crop\n\n\n\n\n\n","category":"function"},{"location":"crop/#SegmentationTools.crop_rotate","page":"Cropping API","title":"SegmentationTools.crop_rotate","text":"Rotates and then crops an image, optionally along with its head and centroid locations.\n\nArguments\n\nimg: image to transform (3D)\ncrop_x: crop amount in x-dimension\ncrop_y: crop amount in y-dimension\ncrop_z: crop amount in z-dimension\ntheta: rotation amount in xy-plane\nworm_centroid: centroid of the worm (to rotate around). NOT the centroids of ROIs in the worm.\n\nOptional keyword arguments\n\nfill: what value to put in pixels that were rotated in from outside the original image.\n\nIf kept at its default value \"median\", the median of the image will be used. Otherwise, it can be set to a numerical value.\n\ndegree: degree of the interpolation. Default Linear(); can set to Constant() for nearest-neighbors.\ndtype: type of data in resulting image. Default Int16.\ncrop_pad: amount to pad the cropping in each dimension. Default 5 pixels in each dimension.\nmin_crop_size: minimum size of cropped image. Default [210,96,51], the UNet input size.\n\nOutputs a new image that is the cropped and rotated version of img.\n\n\n\n\n\n","category":"function"},{"location":"crop/#SegmentationTools.get_crop_rotate_param","page":"Cropping API","title":"SegmentationTools.get_crop_rotate_param","text":"Generates cropping and rotation parameters from a frame by detecting the worm's location with thresholding and noise removal. The cropping parameters are designed to remove the maximum fraction of non-worm pixels.\n\nArguments\n\nimg: Image to crop\n\nOptional keyword arguments\n\nthreshold_intensity: Number of standard deviations above mean for a pixel to be considered part of the worm. Default 3.\nthreshold_size: Number of adjacent pixels that must meet the threshold to be counted. Default 10.\n\n\n\n\n\n","category":"function"},{"location":"crop/#SegmentationTools.uncrop_img_rois","page":"Cropping API","title":"SegmentationTools.uncrop_img_rois","text":"Uncrops all ROI images.\n\nArguments\n\nparam_path::Dict: Dictionary containing paths to files\nparam::Dict: Dictionary containing pipeline parameters\ncrop_params::Dict: Dictionary containing cropping parameters\nimg_size: Size of uncropped images to generate\nroi_cropped_key::String (optional): Key in param_path corresponding to locations of ROI images to uncrop. Default path_dir_roi_watershed.\nroi_uncropped_key::String (optional): Key in param_path corresponding to location to put uncropped ROI images. Default path_dir_roi_watershed_uncropped.\n\n\n\n\n\n","category":"function"},{"location":"crop/#SegmentationTools.uncrop_img_roi","page":"Cropping API","title":"SegmentationTools.uncrop_img_roi","text":"Uncrops an ROI image.\n\nArguments\n\nimg_roi: ROI image to uncrop\ncrop_params: Dictionary of cropping parameters\nimg_size: Size of uncropped image\ndegree (optional): Interpolation method used. Default Constant() which results in nearest-neighbor interpolation\ndtype (optional): Data type of uncropped image. Default Int16\n\n\n\n\n\n","category":"function"},{"location":"train/#UNet-Training-API","page":"UNet Training API","title":"UNet Training API","text":"","category":"section"},{"location":"train/","page":"UNet Training API","title":"UNet Training API","text":"create_weights\nresample_img\ncompute_mean_iou","category":"page"},{"location":"train/#SegmentationTools.create_weights","page":"UNet Training API","title":"SegmentationTools.create_weights","text":"Creates a weight map from a given labeled dataset. Unlabeled data (label 0) has weight 0 and background data (label 2) far from foreground data has weight 1. Foreground data (label 1) has a higher weight. Background data near the foreground (also label 2) has weight exponentially decaying down from the foreground weight. \"Background-gap\" data that serves to mark boundaries between foreground objects (label 3) has the highest weight.\n\nArguments\n\nlabel: labeled dataset to turn into weights\n\nOptional keyword arguments\n\nscale_xy::Real: Inverse of the distance in the xy-plane, in pixels, before the background data weight is halved. Default 0.36.\nscale_z::Real: Inverse of the distance in the z-plane, in pixels, before the background data weight is halved. Default 1.\nmetric::String: Metric to compute distance. Default taxicab (currently, no other metrics are implemented)\nweight_foreground::Real: weight of foreground (1) label\nweight_bkg_gap::Real: weight of background-gap (3) label\nboundary_weight: weight of foreground (2) pixels adjacent to background (1 and 3) pixels. Default nothing, which uses default foreground weight.\nscale_bkg_gap::Bool: whether to upweight background-gap pixels for each neuron pixel they border.\n\n\n\n\n\n","category":"function"},{"location":"train/#SegmentationTools.compute_mean_iou","page":"UNet Training API","title":"SegmentationTools.compute_mean_iou","text":"Computes the mean IOU between raw_file and a prediction_file HDF5 files.\n\nBy default, assumes a threshold of 0.5, but this can be changed with the threshold parameter.\n\n\n\n\n\n","category":"function"},{"location":"#SegmentationTools.jl-Documentation","page":"SegmentationTools.jl Documentation","title":"SegmentationTools.jl Documentation","text":"","category":"section"},{"location":"","page":"SegmentationTools.jl Documentation","title":"SegmentationTools.jl Documentation","text":"Pages = [\"crop.md\", \"segment.md\", \"visualize.md\", \"extract.md\", \"train.md\", \"util.md\"]","category":"page"},{"location":"util/#Utilities-API","page":"Utilities API","title":"Utilities API","text":"","category":"section"},{"location":"util/","page":"Utilities API","title":"Utilities API","text":"volume\nget_points\ndistance","category":"page"},{"location":"util/#SegmentationTools.volume","page":"Utilities API","title":"SegmentationTools.volume","text":"Computes volume from a radius and a sampling ratio\n\n\n\n\n\n","category":"function"},{"location":"util/#SegmentationTools.get_points","page":"Utilities API","title":"SegmentationTools.get_points","text":"Returns all points from img_roi corresponding to region roi.\n\n\n\n\n\n","category":"function"},{"location":"util/#SegmentationTools.distance","page":"Utilities API","title":"SegmentationTools.distance","text":"Computes the distance between two points p1 and p2, with the z-axis scaled by zscale (default 1).\n\n\n\n\n\n","category":"function"}]
}