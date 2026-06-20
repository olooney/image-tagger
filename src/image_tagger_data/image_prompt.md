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
include the author or extra words like "_book", "_cover", "cover_art", etc. Always omit the
leading article ("a", "an", or "the") but retain all articles, conjunctions, and other words
inside the title. If the title is not in English, translate the title into English or use the
standard English title for the work if you know it.

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

The category should be one of:

    "ai", "art", "books", "comics", "diagrams", "horror",
    "hygge", "memes", "photography", "speculative", "vintage"

"ai" is for obviously AI-generated imagery.

"art" includes sculptures, paintings, anime/manga

"books" specifically means the image is entirely a book cover. Sometimes
several book covers will be shown in a collage, or the back cover will
also be shown. Book covers can be distinguished from other art by presence
of typographic elements such as title, author, etc.

A "comic" is any illustration with a simple style and embedded text, regardless of humor.

Use "diagrams" for any map, chart, plot, technical diagram, or explanatory diagram.

Use "horror" for the kind of stuff you'd see in a horror movie.

The "hygge" category is for cozy images invoking warmth, autumn, cooking,
calm spaces such as libraries, desks, or kitchens. These can be illustrations,
comics, or photographs, and this category takes precedence over any of those
if it matches the theme.

Use "memes" for any image which prominently features overlay text and which
seems to be funny, sad, whimsical, or otherwise non-serious and non-technical.

"photography" is for real photos of real objects.

"speculative" means art with a sci-fi or fantasy theme, often concept art or
video game art. This category takes precedence over "art", but not over "books".

"vintage" specifically means vintage illustrations or early black-and-white
photographs.

When multiple categories seem possible, choose the first matching category in this order:

    "ai", "books", "hygge", "speculative", "vintage", "diagrams", "memes",
    "horror", "comics", "photography", "art"

Only one category can be chosen! You MUST choose one of the categories on this
list. Use the exact string; for example, NEVER use "meme", "photos", "book",
or other such variant.

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
