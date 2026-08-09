Describe this image, come up with a good filename for it,
and determine category, genre, and tags for the image.

## Describe Image

Provide a detailed but concise description in one paragraph of 4-7 sentences.
Describe only information visible in the image. Avoid speculation.
If the main subject is a person, describe their appearance and pose.

## Propose Clean Filename

Follow these general filename rules unless a more specific rule below overrides them:

* The file extension MUST match the current filename extension.
* If the current filename already adequately describes the image, use the current filename.
* The filename should usually be less than 40 characters and should generally be more than 20 characters.
* The filename MUST always be LESS than 60 characters and MORE than 10 characters.
* The filename should omit conjunctions, articles, prepositions, etc.
* The filename should use only the essential nouns and adjectives.
* The filename must be all lowercase with no spaces or special characters. Use "_" to separate words.

If the image is a book cover, base the filename on the title while respecting the length,
case, extension, and separator rules above. Use ONLY the book title for the filename: Do NOT
include the author or extra words like "_book", "_cover", "cover_art", etc. Do NOT include the
subtitle of a book in the filename; a subtitle is identifiable by a colon separator or a different, smaller font.
Always omit the leading article ("a", "an", or "the") but retain all articles, conjunctions, and other words
inside the title. If the title is not in English, translate the title into English or use the
standard English title for the work if you know it. If the book cover is from a series, use the series
short, colloquial name as a prefix, e.g. "zap_z64_*" or "analog_*".

If the image is a piece of art and you either know the artist's name or it's in the original filename,
include their last name only (or mononym) as a suffix like so: "sunflowers_picasso.png". Do not
guess if you're not sure; it's fine to simply omit the artist's name as well: "sunflowers.png". This
rule applies only to artwork; never add the artist name to books, comics, or other categories.

Likewise, if the image is a photograph and the name of the subject is in the filename or a well-known
person, include it first in first-name-last-name order and qualify it with one or two words about
the pose, setting, or wardrobe: whatever makes the photograph distinctive. 

Examples:

* "secret_garden.png" (omit leading "the" from book title.)
* "other_side_of_the_sky.png" (omit leading "the", retain "of the" in the middle.)
* "child_playing_sunset.jpg" (short description of a photograph)
* "shark_existential_crisis.gif" (short description of a comic strip.)
* "marilyn_monroe_turtleneck.jpg" (model name + distinctive wardrobe.)
* "emily_rudd_cross_eyed.gif" (model name + distinctive pose.)

Counterexamples (DO NOT DO THESE!):

* "the_secret_garden.png" INCORRECT!
* "secret_garden_frances_burnett.png" INCORRECT!
* "secret_garden_book_cover.png" INCORRECT!

## Filename Already Makes Sense?

Determine if the current filename (given below) already loosely matches the above
format (don't be too strict) and has a filename that makes sense; report that as the
boolean flag "filename_already_makes_sense".

If the filename contains useful information such as the name or title but also
contains unacceptable formatting, dates, or random identifiers, then set the
"filename_already_makes_sense" flag to false and create a new filename that
retains the information in the original while correcting it and including
information from the image itself:

* "McKay-Jane-b6f2z9.jpg" -> "jane_mckay_dancing.jpg"
* "John Waterhouse: Circe Invidiosa 1892 - OilOnCanvas.jpg" -> "circe_invidiosa_waterhouse.png"

## Assign Category

Choose exactly one category from this authoritative list of library shelves:

    {categories}

The `default` shelf is intentionally excluded because it is an inbox, not a
category. Use the exact configured identifier and do not invent variants.

## Assign Genre

The genre should be one of "sci-fi", "fantasy", "comedy", "mystery", "horror",
"drama", "tragedy", "nonfiction", "nature", or "abstract". Only one genre can
be chosen. Other genres can be used if none of these fit, but strongly prefer
this list.

## List Tags

The tags should be a list of relevant topics or themes that may help users
to find this image while searching. Feel free to invent any tag that may
help the user. It is not necessary to use tags that are already adequately
covered by the category or genre. Never use hashtags!

Current filename: "{filename}"
