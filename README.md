# SegmentationTools.jl

This package provides a set of visualization tools for segmentation. It is geared towards neuron segmentation,
but many of the tools can be used for other segmentation problems as well.

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, and `CaSegmentation.jl` packages from the `flavell-lab` github page, and that you have [succesfully configured `WebIO`](https://juliagizmos.github.io/WebIO.jl/latest/providers/ijulia/) if you're using this package's visualization tools with Jupyter. It is designed to interface with the `pytorch-3dunet` package, also in the `flavell-lab` github page.

Additionally, the example code provided here requires the `ImageDataIO` package is loaded in the current Julia environment.

## Making weighted HDF5 files

The `make_hdf5` method can generate appropriately-weighted HDF5 files from raw MHD data and labeled NRRD files:

```julia
make_hdf5("/path/to/data", "hdf5/01.h5", "label_cropped/01_label.nrrd", "img_cropped/01_img.mhd")
```

These HDF5 files can then be fed as input to the UNet.

## Visualizing the UNet's predictions

The `display_predictions_3D` method can display the UNet's predictions in comparison with the raw and labeled data.

```julia
# load the data
raw, label, weight = load_training_set("/path/to/data.h5")
# load the UNet's predictions
predictions = load_predictions("/path/to/predictions.h5")
# display the predictions
# the order of plots will be raw data + label, weights + label, predictions + label, predictions vs label match
display_predictions_3D(raw, label, weight, [predictions])
```

## Instance segmentation

After the UNet has been verified to be giving reasonable output, the next step is to turn the UNet's semantic segmentation into an instance segmentation. The `instance_segmentation_output` function does this instance segmentation and outputs the result to various files, which can be used during later steps in elastix registration:

```julia

```
