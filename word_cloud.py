import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from text_to_frequency_dict import sorted_frequency_dict

def masked_image_from_freq_dict(text_dict, given_mask):

    wc = WordCloud(background_color="white", max_words=200, mask=given_mask, contour_width=3, contour_color='gold')
    # generate word cloud
    wc.generate_from_frequencies(text_dict)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# script taken from ham4corpus
hamilton_full_text = open('All_lyrics_Speakers', encoding="utf-8").read()
#print(sorted_frequency_dict(hamilton_full_text))

## .png taken from https://www.clipartkey.com/view/Tooimh_transparent-hamilton-png-hamilton-logo-transparent/
hamilton_mask = np.array(Image.open(path.join(d, "hamilton.png")))

masked_image_from_freq_dict(sorted_frequency_dict(hamilton_full_text), hamilton_mask)