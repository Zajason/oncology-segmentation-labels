# Classical Methods

The classical methods are CPU-friendly and require no model checkpoints. They provide interpretable baselines for the box-to-mask problem.

## Thresholding

- **Otsu** chooses one global threshold.
- **Multi-Otsu** partitions an image into several intensity classes.
- **Adaptive thresholding** computes local thresholds and is more robust to uneven illumination.

## Region and Graph Methods

- **Watershed** treats the image as a topographic surface and grows regions from markers.
- **Connected Components** keeps connected foreground regions after binarization.
- **Random Walker** assigns pixels to seeds using graph diffusion.

## Active Contours

- **Chan-Vese** evolves a contour based on region homogeneity.
- **Morphological GAC** evolves contours using image gradients and morphological operators.

These methods are valuable because they are transparent and fast, but they generally underperform learned methods on noisy or heterogeneous medical images.
