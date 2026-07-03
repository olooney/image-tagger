Are these the exact same image except for minor cropping, resizing, and filtering?

Left file: "{left_filename}" ({left_width}x{left_height})
Right file: "{right_filename}" ({right_width}x{right_height})

Return "left", "right", or "both" in the `keep` field.

Use "both" when the images are not clearly from the same original picture, or when you can't decide with image is
better (for example, if both are damaged in different ways but to a similar degree.)

If they are duplicates and you must choose one, keep the larger, higher-quality image. Prefer the version without cropping, added text, borders, letterboxing, watermarks, or artificial overlays. Prefer natural color grading, exposure, and contrast over heavy filters or degraded compression.

Treat changes in crop, scale, compression, color filtering, exposure, contrast, letterboxing, and small border bars as minor only when the images still appear to come from the same original picture. Return "both" for merely similar subjects, compositions, styles, or scenes.

If there is no meaningful difference in the images, choose the one that has the cleaner, more meaningful filename.

If all else fails and the images are either identical or some close as to make no difference, choose "left".
