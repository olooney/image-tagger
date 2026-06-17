Describe this image, come up with a good filename for it,
and determine category, genre, and tags for the image.

## Describe Image

Provide a detailed but concise description in one paragraph of 4-7 sentences.
Describe only information visible in the image. Avoid speculation.
If the main subject is a person, describe their appearance and pose.

## Propose Clean Filename

The file extension MUST match the current filename extension.
If the current filename already adequately describes the image, use the current filename.
The filename should usually be less than 40 characters and should generally be more than 20 characters.
The filename MUST always be LESS than 60 characters and MORE than 10 characters.
The filename should omit conjunctions, articles, prepositions, etc.
The filename should use only the essential nouns and adjectives.
The filename must be all lowercase with no spaces or special characters. Use "_" to separate words.

If the image is a book cover, base the filename on the title while respecting the above rules.
Use ONLY the book tile for the file name: Do NOT include the author or extra words like "_book",
"_cover", "cover_art", etc. Always omit the leading article ("a", "an", or "the".) Respect all of\
the above rules.

Examples:

* "secret_garden.png"
* "child_playing_sunset.jpg"
* "shark_existential_crisis.gif"

Counterexamples (DO NOT DO THESE!):

* "the_secret_garden.png" INCORRECT!
* "secret_garden_frances_burnett.png" INCORRECT!
* "secret_garden_book_cover.png" INCORRECT!

## Filename Already Makes Sense?

Determine if the current filename (given below) already loosely matches the above
format (don't be too strict) and has a filename that makes sense; report that as the
boolean flag "filename_already_makes_sense".

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

Use "memes" fon any image which prominently features overlay test and which
seems to be funny, sad, whimsical, or otherwise non-serious and non-technical.

"photography" is for real photos of real objects.

"speculative" means art with a sci-fi or fantasy theme, often concept art or
video game art. This category takes precedence over "art", but not over "books".

"vintage" specifically means vintage illustrations or early black-and-white
photographs.

Only one category can be chosen! You MUST choose one of the categories on this
list. Use the exact string; for example, NEVER use "meme", "photos", "book",
or other such variant.

## Assign Genre

The genre should be one of "sci-fi", "fantasy", "comedy", "mystery", "horror",
"drama", "tragedy", "nonfiction", "nature", or "abstract". Only one genre can
be chosen. Other genres can be used if non of these fit, but strongly prefer
this list.

## List Tags

The tags should be a list of relevant topics or themes that may help users
to find this image while searching. Feel free to invent any tag that may
help the user. It is not necessary to use tags that are already adequately
covered by the category or genre. Never use hashtags!

Current filename: "{filename}"
