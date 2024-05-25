import os
import sys
import math
import re
import json
import base64
from urllib.parse import urlsplit
from io import BytesIO
import time
import csv
from datetime import datetime
import traceback
import random
import string

import numpy as np
import pandas as pd
from PIL import Image
import requests
import jinja2

from util import (
    connect_to_openai,
    retry_decorator,
    TemporarySeed
)

NAME_IMAGE_PROMPT_TEMPLATE = """Describe this image, come up with a good filename for it,
and determine category, genre, and tags for the image.

The description should be as detailed as possible. If the main subject is a person,
describe their appearance and pose.

The file extension MUST match the current filename extension.
If the current filename already adequately describes the image, use the current filename.
The filename must be less than 40 characters and should be less than 20 characters.
The filename should omit conjunctions, articles, prepositions, etc. 
The filename should use only the essential nouns and adjectives.
The filename must be all lowercase with no spaces or special characters. Use "_" to separate words.

Determine if the current filename (given below) already loosely matches the above
format (don't be too strict) and has a filename that makes sense; report that as the
boolean flag "filename_already_makes_sense".

The category should be one of "photo", "art", "comic", or "meme". 
Photographs of sculptures or paintings count as "art".
A "comic" is any cartoon regardless of humor.
A "meme" is an image which prominately features text (not merely text in the background.)
Only one category can be chosen.

The genre should be one of "sci-fi", "fantasy", "realism", etc.
Only one category can be chosen.

The tags should be a list of relevant topics or themes that may help users
to find this image while searching.

Format your answer as a JSON object in this example format:

    {{
        "description": "A cat sitting on a red blanket.",
        "category": "photo",
        "genre": "realism",
        "tags": ["cozy", "domestic", "hygge", "pet"],
        "filename_already_makes_sense": false,
        "filename": "cat_red_blanket.jpg"
    }}

Current filename: "{filename}"
"""

file_extension_blacklist = ['.mp3', '.mp4', '.pdf', '.docx', '.xlsx', '.csv', '.zip', '.gz', '.txt']

csv_columns = [
    'timestamp',
    'status',
    'total_tokens',
    'model',
    'original_filepath',
    'original_filename',
    'width',
    'height',
    'category',
    'genre',
    'filename',
    'clean_filename',
    'filename_already_makes_sense',
    'tags',
    'description'
]

client = connect_to_openai()


def clean_filename(filename):
    filename = filename.lower()
    filename = re.sub(r'^[^a-zA-Z_]+', '', filename) # strip leading whitespace
    filename = re.sub(r'[\s_-]+', '_', filename) # whitespace to underscore
    filename = re.sub(r'[^a-zA-Z0-9_.]', '', filename) # strip special characters
    filename = re.sub(r'[\s_-]*\.+', '.', filename) # whitespace before dot
    
    return filename


def fix_extension(current_filename, suggested_filename):
    current_ext = os.path.splitext(current_filename)[1].lower()
    suggested_base, suggested_ext = os.path.splitext(suggested_filename)
    if current_ext != suggested_ext:
        suggested_filename = suggested_base + current_ext
    return suggested_filename


def path_name_ext(path):
    dir_path = os.path.dirname(path)
    filename_with_ext = os.path.basename(path)
    filename, ext = os.path.splitext(filename_with_ext)
    if not dir_path.endswith(os.sep):
        dir_path += os.sep
    return (dir_path, filename, ext)


def scramble(filename):
    with TemporarySeed(seed=hash(filename)):
        return "".join(random.sample(string.ascii_letters, k=8))


def resize_image_to_fit(image, max_dimension=512):
    # read from disk if given as filename
    if isinstance(image, str):
        image = Image.open(image)
    original_width, original_height = image.size
    
    # Determine which dimension is larger and calculate scaling factor
    if max(original_width, original_height) > max_dimension:
        if original_width > original_height:
            scaling_factor = max_dimension / original_width
        else:
            scaling_factor = max_dimension / original_height
        
        # Calculate new dimensions based on scaling factor
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        
        # Resize the image to the new dimensions
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def base64_encode_image(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    img_byte_data = img_buffer.getvalue()
    
    # Encode to Base64
    base64_encoded_image = base64.b64encode(img_byte_data)
    base64_image_data = base64_encoded_image.decode('utf-8')

    return base64_image_data


@retry_decorator
def gpt_vision(url, prompt):
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {
              "role": "user",
              "content": [
                {"type": "text", "text": prompt},
                {
                  "type": "image_url",
                  "image_url": {
                      "url": url,
                  },
                },
              ],
            }
        ],
    )
    return chat_response


def tag_image(filepath: str) -> dict:
    image = None
    
    # handle local or remote images
    if filepath.startswith('http'):
        url = filepath
        filename = urlsplit(url).path.split('/')[-1]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = resize_image_to_fit(image)
    else:
        dir, filename = os.path.split(filepath)
        image = resize_image_to_fit(filepath)
        
    base64_image_data = base64_encode_image(image)
    url = f"data:image/jpeg;base64,{base64_image_data}"

    # prepare the prompt
    prompt = NAME_IMAGE_PROMPT_TEMPLATE.format(filename=filename)
    response = gpt_vision(url, prompt)
    json_string = response.choices[0].message.content
    data = json.loads(json_string)

    # clean up the suggested filename and fix the extension if necessary
    suggested_filename = clean_filename(data.get('filename', None))
    suggested_filename_fixed = fix_extension(filename, suggested_filename)

    # format the results
    data['clean_filename'] = suggested_filename_fixed
    data['original_filepath'] = filepath
    data['original_filename'] = filename
    data['total_tokens'] = response.usage.total_tokens
    data['tags'] = ";".join( tag.lower().strip() for tag in data['tags'] )
    data['model'] = response.model
    data['width'] = image.size[0]
    data['height'] = image.size[1]
    
    return data


def tag_images(filepaths, output_filename, retry_errors=False, verbose=1):
    file_already_exists = os.path.exists(output_filename)
    mode = 'a' if file_already_exists else 'w'
    
    processed_paths = set()
    if file_already_exists:
        with open(output_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if retry_errors:
                processed_paths = set(row['original_filepath'] for row in reader if row['status'] == 'ok')
            else:
                processed_paths = set(row['original_filepath'] for row in reader)
    
    with open(output_filename, mode, newline='', encoding='utf-8') as csvfile:
        columns = csv_columns
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not file_already_exists:
            writer.writeheader()
    
        for index, filepath in enumerate(filepaths):
            if any(filepath.lower().endswith(ext) for ext in file_extension_blacklist):
                continue
                
            if filepath in processed_paths:
                continue
            processed_paths.add(filepath)
            
            try:
                row = tag_image(filepath)
                row.update({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'ok'
                })
                writer.writerow(row)
                csvfile.flush()
                
                if verbose == 1:
                    print('.', end=('\n' if (index + 1) % 100 == 0 else ''))
                elif verbose > 1:
                    print(repr(row))
            except:
                error_message = traceback.format_exc()
                
                if verbose == 1:
                    print('e', end=('\n' if (index + 1) % 100 == 0 else ''))
                elif verbose > 1:
                    print(error_message)
                
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'original_filepath': filepath,
                    'status': 'error',
                    'description': error_message
                })


def find_images(dirs, max_days_old=None):
    if max_days_old is None:
        max_days_old = float('Inf')

    if isinstance(dirs, str):
        dirs = [ dirs ]
    
    current_time = time.time()
    
    filepaths = []
    for dir in dirs:
        # List all files in the directory and filter them
        filepaths += [
            os.path.join(dir, fn) 
            for fn in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, fn)) 
            and (current_time - os.path.getmtime(os.path.join(dir, fn))) < max_days_old*86400
        ]
        
    return filepaths


def scramble_image_directory(input_dir, output_dir, max_dimension=512):
    for filepath in find_images(input_dir):
        path, name, ext = path_name_ext(filepath)
        scrambled_name = scramble(name)
        new_filepath = os.path.join(output_dir, scrambled_name + ext)
        thumbnail = resize_image_to_fit(filepath, max_dimension)
        thumbnail.save(new_filepath)


def autorename(csv_filename, verbose=1, dry_run=False):
    metadata_df = pd.read_csv(csv_filename)
    for index, row in metadata_df.iterrows():
        source = row['original_filepath']
        
        if row['status'] != 'ok' or not row['clean_filename']:
            if verbose >= 2:
                print(f'skipping errored row {index} {source!r}')
            continue
    
        # old filename
        if not os.path.isfile(source):
            if verbose >= 1:
                print(f'source file {source!r} is missing!')
            continue
    
        # new filename
        directory, old_filename = os.path.split(source)
        new_filename = row['clean_filename']
        target = os.path.join(directory, new_filename)
    
        # check for no-op
        if target == source:
            if verbose >= 2:
                print(f'no rename necessary for {source!r}')
            continue
    
        # ensure extension matches
        source_ext = os.path.splitext(source)[1]
        target_base, target_ext = os.path.splitext(target)
        if source_ext.lower() != target_ext.lower():
            if verbose >= 1:
                print(f"Mismatched file extensions between {source!r} and {target!r}!")
            continue

        # check for name collisions
        if os.path.isfile(target):
            if verbose >= 1:
                print(f'target {target!r} already exists!')
            suffix = 2
            while True:
                target = target_base + "_" + str(suffix) + target_ext
                if not os.path.isfile(target):
                    break
                suffix += 1
            if verbose >= 1:
                print(f'proceeding with target {target!r}.')
            continue

        # actually perform the file rename
        if verbose >=2:
            print(f'renaming {source!r} to {target!r}...', end='')
        try:
            if not dry_run:
                os.rename(source, target)
            if verbose >= 2:
                print('success!')
        except Exception as e:
            if verbose >= 2:
                print('error!')
            else:
                print(f"error renaming {source!r} to {target!r}!")
            traceback.print_exc() 


def generate_gallery(csv_filename, output_filename, verbose=1):
    # Setup Jinja2 environment
    file_loader = jinja2.FileSystemLoader('.')
    env = jinja2.Environment(loader=file_loader)

    # read the metadata and prepare for merge
    metadata_df = pd.read_csv(csv_filename)
    items = metadata_df.to_dict('records')
    for item in items:
        item['formatted_timestamp'] = datetime.fromisoformat(item['timestamp']).strftime('%m/%d/%y %I:%M %p')
        item['tags'] = [ tag.strip() for tag in item['tags'].split(';') ]
        notes = item.get('notes', '')
        # filter out the NaNs that pandas uses for missing values.
        item['notes'] = notes if notes and isinstance(notes, str) else ''
    
    # Render the template with the data
    template = env.get_template('template.html')
    output = template.render(items=items)
    
    # Save the rendered HTML to a file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output)