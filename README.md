# SegmentationTools.jl

This package provides a set of visualization tools for segmentation. It is geared towards neuron segmentation,
but many of the tools can be used for other segmentation problems as well.

The API is available [here](https://flavell-lab.github.io/SegmentationTools.jl/dev/).

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, `WormFeatureDetector.jl`, and `CaSegmentation.jl` packages from the `flavell-lab` github page, and that you have [succesfully configured `WebIO`](https://juliagizmos.github.io/WebIO.jl/latest/providers/ijulia/) if you're using this package's visualization tools with Jupyter. It is designed to interface with the `pytorch-3dunet` package, also in the `flavell-lab` github page.

Additionally, the example code provided here requires the `ImageDataIO` package is loaded in the current Julia environment.

## Cropping the worm

Before performing any other operations on an image, it is useful to crop out non-worm regions; this will speed up the subsequent operations, and can also improve UNet output:

```julia
crop_parameters = Dict()
cropping_errors = crop_rotate!(param_path, param, t_range, [ch_marker], crop_parameters, save_MIP=true)
```

## Making weighted HDF5 files

The `make_unet_input_h5` method can generate appropriately-weighted HDF5 files from raw MHD data and labeled NRRD files:

```julia
make_unet_input_h5(param_path, path_mhd_crop, t_range, ch_marker, get_basename)
```

These HDF5 files can then be fed as input to the UNet.

## Visualizing the UNet's predictions

The `display_predictions_3D` method can display the UNet's predictions in comparison with the raw and labeled data.

```julia
# load the data (say it's dataset 50)
raw, label, weight = load_training_set("/path/to/data/50.h5")
# load the UNet's predictions
predictions = load_predictions("/path/to/data/50_predictions.h5")
# display the predictions
# the order of plots will be raw data + label, weights + label, predictions + label, predictions vs label match
display_predictions_3D(raw, label, weight, [predictions])
```

## Instance segmentation

After the UNet has been verified to be giving reasonable output, the next step is to turn the UNet's semantic segmentation into an instance segmentation. The `instance_segmentation_watershed` function does this instance segmentation and outputs the result to various files, which can be used during later steps in elastix registration:

```julia
results, errors = instance_segmentation_watershed(param, param_path, mhd_crop_dir, t_range, get_basename, save_centroid=true, save_signal=true, save_roi=true)
```

## Visualizing instance segmentation

Assuming that you have successfully instance segmented the data, you can view the resulting ROIs,
in comparison with the raw and predicted data, to ensure that instance segmentation was successful:

```julia
view_roi_3D(raw, predictions, img_roi)
```