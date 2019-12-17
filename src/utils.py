#organize text into corpus
import re
def get_text():
    filename = '../data/heart.txt'
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    #output = re.sub(r'\d+', '', raw_text)
    return raw_text
