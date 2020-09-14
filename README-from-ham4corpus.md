# ham4corpus
The #ham4corpus repo currently contains files with various types of information about the Original Broadway Cast recording of *Hamilton: An American Musical* (i.e. all words currently in the show, minus the one scene not on the OBC recording).

*Suggested uses:* Twitter bots, text visualization. 

All lyrics are by Lin-Manuel Miranda and copied from the LMM-annotated Genius.com hosting of the lyrics. Cast/character information is from Wikipedia. Links to all sources listed below.

# For example:

Using [Stéfan Sinclair](https://github.com/sgsinclair) and [Geoffrey Rockwell](https://github.com/GeoffreyRockwell)'s [Voyant Tools](http://voyant-tools.org) to explore the text of the musical, I discovered the following about the Hamilton lyrics en masse (in order as one text block, with no names of speakers/singers interspersed):

  * 21,351 total words and 2,939 unique word forms
  * Interesting frequent words: da (103 times, thanks George III), time (87), hamilton appears the same number of times wait does (79), room (71), burr (69), sir (56), satisfied (37), story (35), helpless (32).
  * Unsurprisingly, "sir" is the most frequent one-away collocate of "burr" (9 times).
  * Single word occurrence via microsearch (map of where a given word appears throughout the lyrics):

![Screenshot of Voyant microsearch for occurences of the word "wait" throughout the Hamilton lyrics](https://github.com/amandavisconti/ham4corpus/blob/master/wait_occurrences.png)
Red = the word "wait" vs the rest of the lyrics (the second-to-last red block is Burr's final "Wait!", last one is Eliza's "I can't wait to see you again")

![Screenshot of Voyant microsearch for occurences of the word "Hamilton" throughout the Hamilton lyrics](https://github.com/amandavisconti/ham4corpus/blob/master/hamilton_occurrences.png)
Red = the word "Hamilton" vs the rest of the lyrics (was surprised that the word "Hamilton" isn't used again after Burr's last "The world was wide enough for both Hamilton and me")

Explore the Voyant dashboards for both versions of the lyrics yourself at:
* [tinyurl.com/hamilton-lyrics-names](https://tinyurl.com/hamilton-lyrics-names) to explore the lyrics including the speaker names

* [tinyurl.com/just-hamilton-lyrics](https://tinyurl.com/just-hamilton-lyrics) to explore the lyrics without the speaker names:

# The files
## Lyrics, including character names before their parts
**Title:** All_Hamilton_Lyrics_Speakers

**What:** One file containing all lyrics sung in the Original Broadway Cast recording of *Hamilton*, with the name of the character singing each part appearing on the line above the beginning of their part. No empty lines between anything, just a solid block of *Hamilton*. Copied and pasted from the lyrics on Genius.com; the placement of simultaneous lyrics is broken up rather than side-by-side.

**Pulled from:** http://genius.com/albums/Lin-manuel-miranda/Hamilton-original-broadway-cast-recording

## Lyrics, not including character names before their parts
**Title:** All_Hamilton_Lyrics_No_Speakers

**What:** One file containing all lyrics sung in the Original Broadway Cast recording of *Hamilton*. No empty lines between anything, just a solid block of Hamilton.  Copied and pasted from the lyrics on Genius.com; the placement of simultaneous lyrics is broken up rather than side-by-side.

**Pulled from:** http://genius.com/albums/Lin-manuel-miranda/Hamilton-original-broadway-cast-recording

## Original Broadway Cast Actors & Character Names
**Title:** OBC_Cast_Actors_Character.json

**What:** Actors and the named characters played by them in the Original Broadway Cast recording of *Hamilton*. Actors who played multiple characters are listed multiple times.

**Pulled from:** https://en.wikipedia.org/wiki/Hamilton_(musical)#Principal_roles_and_major_casts

# How

I wish I could say my Python is non-rusty enough that I scraped Genius.com and Wikipedia to get this data, but it isn't and I was on hold on the phone, so I just cut and pasted everything into a text document and grepped:

^\s*?\r to remove blank lines

\[.* to remove the [character names]

If you'd like to learn more about doing cool digital things with text, check out [The Programming Historian](http://programminghistorian.org/lessons/) for novice-friendly, peer-reviewed lessons on data cleaning, distant reading, web scraping, and Python.
