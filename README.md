# Neural Postprocessing

This repository implements neural postprocessing method. It applies a set of random image degradation methods only on a certain instance (e.g. person). Given a fixed size of image patch including that instance, the network has to predict the parameters of the image processing methods that are applied on the instance.

# Usage

```python trainer.py```

# Covered image operations

1. White balance
2. Brightness
3. Contrast
4. Saturation
5. Gamma correction
6. Channel curves
7. Chromatic aberration
8. Blurring
9. Noise
10. JPEG artifact
