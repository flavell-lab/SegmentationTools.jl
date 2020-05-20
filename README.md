# SegmentationTools.jl

This package provides a set of visualization tools for segmentation. It is geared towards neuron segmentation,
but many of the tools can be used for other segmentation problems as well.

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, and `CaSegmentation.jl` packages from the `flavell-lab` github page, and that you have [succesfully configured `WebIO`](https://juliagizmos.github.io/WebIO.jl/latest/providers/ijulia/) if you're using this package's visualization tools with Jupyter. It is designed to interface with the `pytorch-3dunet` package, also in the `flavell-lab` github page.

## Semantic segmentation

