Image Tagger
============


Features
--------

Extract image metadata using a vision model, such as category, genre, tags, and image
description.

Rename arbitrary image filenames to clean, human-readable filenames.

Prepare a static HTML gallery of images and metadata.

Move tagged images into sibling category directories.


Usage
-----

You will need to put your OpenAI API key in a file called
`~/.openai/credentials.yaml` in this format:


    organization: "YOUR ORG KEY HERE" # Test Project
    api_key: "YOUR API KEY HERE"

The upload workflow is available through `just` tasks:

```powershell
just convert [DIRECTORY]
just tag [DIRECTORY]
just rename [DIRECTORY]
just gallery [DIRECTORY]
just shelve [DIRECTORY]
```

If `DIRECTORY` is omitted, the tools use the configured uploads folder. By
default, metadata is written to `image_metadata.csv` inside that directory.
Extra CLI arguments can be passed after `--`, for example:

```powershell
just tag -- -v --provider openai --extensions "jpg, png; gif"
```

Supported vision model providers are:

| Provider | Model |
| --- | --- |
| `openai` | `gpt-5.4` |
| `gemma` | `gemma4:e4b` via Ollama |
| `qwen` | `qwen3.5:4b` via Ollama |

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
`metadata.csv` side-by-side with its inferred metadata. The gallery also has a
simple local search feature to demonstrate how the inferred metadata enables
better image searching.

To move renamed images into sibling category directories such as `../books/`,
create those directories first and run:

```python
it.shelve_images(metadata_filename, verbose=1, dry_run=False)
```


Sample Gallery
--------------

View a sample [Image Tagger Gallery](https://olooney.github.io/image_tagger/gallery/index.html).

Or a newer [Art Gallery](https://olooney.github.io/image_tagger/gallery3/index.html) tagged with GPT-5.4.

For this sample gallery, I added a "notes" column with manual annotations to
document where it did notably well or poorly, or where its behavior is notably
different from the earlier gpt-4v.

The images in this gallery mostly come from here:

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


Source
------

This [Jupyter notebook](https://github.com/olooney/image_tagger/blob/main/notebooks/Image%20Tagger%20Test.ipynb)
contains an example of use, including generating test images by scrambling
filenames and some summary visualizations.

The main
[`image_tagger.py`](https://github.com/olooney/image_tagger/blob/main/src/image_tagger.py)
contains all of the relevant Python code. The variable
`NAME_IMAGE_PROMPT_TEMPLATE` holds the prompt used to instruct the vision model about
which metadata to generate and `csv_columns` contains the names and order of
the columns of the generated `metadata.csv` file. Editing those two variables
is the easiest way to customize the behavior of the entire project.
