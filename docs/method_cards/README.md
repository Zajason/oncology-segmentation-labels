# Method Cards

Short notes for the 12 mask-generation methods compared in the project.

| Method | Family | Strength | Main limitation |
|---|---|---|---|
| Otsu | Thresholding | Fast global baseline | Weak when foreground/background intensities overlap |
| Multi-Otsu | Thresholding | Handles multiple intensity bands | Still brittle on noisy ultrasound |
| Adaptive Thresholding | Local thresholding | Robust to local intensity variation | Can fragment objects |
| Watershed | Region segmentation | Separates localized objects | Sensitive to marker choice |
| Otsu + Watershed | Hybrid classical | Stronger reference baseline than either alone | Still handcrafted |
| Connected Components | Region analysis | Simple and interpretable | Depends on binarization quality |
| Random Walker | Graph segmentation | Spatially coherent masks | Slower and seed-sensitive |
| Chan-Vese | Active contour | Region-based contour evolution | Can be slow and initialization-sensitive |
| Morphological GAC | Active contour | Good boundary refinement | Parameter-sensitive and slower |
| SAM | Foundation model | Strong promptable segmentation | Requires checkpoint/GPU and may need domain tuning |
| U-Net | Learned model | Strong supervised medical baseline | Requires pixel-level masks for training |
| Guided U-Net | Learned box-guided model | Learns image + bbox segmentation behavior | Requires training and checkpoint management |
