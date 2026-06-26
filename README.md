Image Tagger
============

A command line utility to use vision model to organize images.

Features
--------

Extract image metadata using a vision model, such as category, genre, tags, and image
description.

Rename arbitrary image filenames to clean, human-readable filenames.

Normalize upload image formats, correcting mismatched extensions and converting
lossless or uncompressed formats to PNG and lossy formats to JPEG.

Prepare a static HTML gallery of images and metadata.

Move tagged images into sibling category directories.


Sample Gallery
--------------

View a sample [Art Gallery](https://olooney.github.io/image-tagger/gallery3/index.html) tagged with GPT-5.4.


CLI Usage
---------

You will need to put your OpenAI API key in the usual `OPENAI_API_KEY`
environment variable.

The upload workflow is available through `just` tasks:

```powershell
just convert [DIRECTORY]
just tag [DIRECTORY]
just rename [DIRECTORY]
just gallery [DIRECTORY]
just wall [DIRECTORY]
just shelve [DIRECTORY]
```

`just convert` prepares uploads for tagging. It corrects image extensions when
the file contents do not match the name, converts lossless or uncompressed
formats such as BMP and GIF to PNG, converts lossy formats such as WEBP, AVIF,
and HEIC to JPEG, and normalizes `.jpeg` filenames to `.jpg`.

If `DIRECTORY` is omitted, the tools use the configured uploads folder. By
default, metadata is written to `image_metadata.csv` inside that directory.

`just wall` creates an `index.html` image wall directly from every supported image
under `DIRECTORY`. It uses relative image paths, computes a median image aspect
ratio up front, and displays the images in equal-sized grid cells with a
click-to-open full-size overlay.

Vision Models
-------------

Supported vision model providers are:

| Provider | Model |
| --- | --- |
| `openai` | `gpt-5.4` |
| `gemma` | `gemma4:e4b` via Ollama |
| `qwen` | `qwen3.5:4b` via Ollama |

Python API
----------

You can also generate an `image_metadata.csv` file for a given directory of
images from Python like so:

```python
import image_tagger as it

filepaths = it.find_images(image_dir)
it.tag_images(filepaths, metadata_filename)
```

This file contains a description, tags, and other metadata that a vision model can
infer from looking at the image itself.

The metadata CSV contains a column called `clean_filename` which suggests
a new, clean filename for each file in the format `lower_snake_case.png`.
To automatically rename all the images listed in the CSV to their suggested
clean filenames, you can use:

```python
it.rename_images(metadata_filename, verbose=1, dry_run=False)
```

Finally, run:

```python
it.generate_gallery(metadata_filename, gallery_filename)
```

to generate a static `index.html` file which shows each image listed in
`image_metadata.csv` side-by-side with its inferred metadata. The gallery also has a
simple local search feature to demonstrate how the inferred metadata enables
better image searching.

To move renamed images into sibling directories matching the tagged category 
such as `../books/`, create those directories first and run:

```python
it.shelve_images(metadata_filename, verbose=1, dry_run=False)
```

Source
------

This [Jupyter notebook](https://github.com/olooney/image-tagger/blob/main/notebooks/Image%20Tagger%20Test.ipynb)
contains an example of use, including generating test images by scrambling
filenames and some summary visualizations.

The main
[`image_tagger.py`](https://github.com/olooney/image-tagger/blob/main/src/image_tagger.py)
contains the core tagging, renaming, shelving, and gallery code. The default
vision-model instructions live in
[`image_prompt.md`](https://github.com/olooney/image-tagger/blob/main/src/image_tagger_data/image_prompt.md)
and are loaded as `IMAGE_PROMPT_TEMPLATE`. Pass `--instructions-filename` on
the CLI, or `instructions_filename` from Python, to use a different prompt
template without editing the package data. The `csv_columns` variable contains
the names and order of the columns of the generated `image_metadata.csv` file.

Attribution
-----------

The images in the sample gallery mostly come from here:

1. [ICM Quality Mix Vol. 57 - Modern Martyrs](https://imgur.com/gallery/icm-quality-mix-vol-57-modern-martyrs-zcEiD6A)
2. [ICM Quality Mix Vol. 55 - Cooler Heads](https://imgur.com/gallery/icm-quality-mix-vol-55-cooler-heads-QQjYFFS)

ICM is a project of [MetaPathos](https://imgur.com/user/MetaPathos/posts) and
was chosen because it is an extremely diverse collection of images in
different styles and often oblique humor or references which should challenge
vision models.

In addition to the ICM images, there were also a few dozen other test images I
had previously used for gpt-4v. Most of these were chosen to exercise specific
features such as occluded object detection or susceptibility to malicious
prompts hidden within images.

