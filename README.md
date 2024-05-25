Image Tagger
============


Features
--------

Extract image metadata using gpt-4o, such as category, genre, tags, and image
description.

Rename arbitrary image filenames to clean, human-readable filenames.

Prepare a static HTML gallery of the images.


Usage
-----

You will need to put your OpenAI API key in a file called
`.openai/credentials.yaml` in this format:


```python
organization: "YOUR ORG KEY HERE" # Test Project
api_key: "YOUR API KEY HERE"
```

Then you can generate a `metadata.csv` file for a given directory of images
like so:

```python
import image_tagger as it

filepaths = it.find_images(image_dir)
it.tag_images(filepaths, metadata_filename)
```

This file contains a discription, tags, and other metadata that gpt-4o can
infer from looking at the image itself.

The metadata CSV contains a column called `clean_filename` which suggests
a new, clean filename for each file in the format `lower_snake_case.png`.
To automatically rename all the images listed in the CSV to their suggested
clean filenames, you can use:

```python
it.autorename(metadata_filename, verbose=1, dry_run=False)
```

Finally, run:

```python
it.generate_gallery(metadata_filename, gallery_filename)
```

to generate an static `index.html` file which shows each image listed in
`metadata.csv` side-by-side with its inferred metadata. The gallery also as a
simple local search feature to demonstrate how the inferred metadata enables
better image searching.


Sample Gallery
--------------

View a sample [Image Tagger Gallery](https://olooney.github.io/image_tagger/gallery/index.html).

I've added a "notes" column with manual annotations that I added after
reviewing the results to document where it did notably well or poorly, or where
it's behavior is notably different from the earlier gpt-4v.

The images in this gallery come from here:

1. [ICM Quality Mix Vol. 57 - Modern Martyrs](https://imgur.com/gallery/icm-quality-mix-vol-57-modern-martyrs-zcEiD6A)
2. [ICM Quality Mix Vol. 55 - Cooler Heads](https://imgur.com/gallery/icm-quality-mix-vol-55-cooler-heads-QQjYFFS)

Plus a list of test images I had previously used for gpt-4v.


Source
------

This [Jupyter notebook](https://github.com/olooney/image_tagger/blob/main/src/Image%20Tagger%20Test.ipynb)
contains an example of use, including generating test images by scrambling
filenames and some summary visualizations.

The main
[`image_tagger.py`](https://github.com/olooney/image_tagger/blob/main/src/image_tagger.py)
contains all of the relevant Python code. The variable
`NAME_IMAGE_PROMPT_TEMPLATE` holds the prompt used to instruct gpt-4o about
which metadata to generate and `csv_columns` contains the names and order of
the columns of the generated `metadata.csv` file. Editing those two variables
is the easiest way to customize the behavior of the entire project.

