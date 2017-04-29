import nltk
import string
from multiprocessing import Pool,freeze_support
from itertools import chain
# from gensim.parsing.preprocessing import STOPWORDS
import os
#import sys
import glob
import errno
# Preprocess documents .txt under a given folder.
import warnings

# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def read_file(fname):
    safe_fname = format_filename(fname)
    outfile = "p{0}".format(fname)
    return outfile
def format_filename(fname):
    """
    Convert a fanme into a safe string for file name.
    :return:STring
    """
    return ''.join(convert_valid(one_char) for one_char in fname)
def convert_valid(one_char):
    """
    Convert a character into _ if invalid.
    :return:String

    """
    valid_chars = "-_.%s%s" % (string.ascii_letters,string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


#======
    #Text preprocessing methods
#======
def prep_bbc(path_file,in_file):
    punkt = nltk.data.load('tokenizers/punkt/english.pickle')
    #nltk.download('punkt')
    out_file = read_file(in_file)
    with open(out_file,"w") as y_file:
     with open(path_file, "r") as x_file:
        for line in x_file:
            sentences = punkt.tokenize(line.lower())
            #------
            freeze_support()
            p = Pool()
            #Chop the iterable "sentence" into 4 chunks and give them to these pooled processes.
            # since the tokenizer works on a per sentence level, we can parallelize
            words2 = list(chain.from_iterable(p.map(nltk.tokenize.word_tokenize, sentences)))
            p.close()
            #------
            # Now remove words that consist of only punctuation characters and Stopwords
            # words2 = [word for word in words2 if word not in STOPWORDS]
            words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]
            #------
            # Remove contractions - words that begin with '
            words2 = [word for word in words2 if not (word.startswith("'") and len(word) <=2)]
            #-----
            # Adjust the format of output string
            words2 = ['{0} '.format(elem) for elem in words2]
            str=''.join(word for word in words2)
            y_file.write(str)
     x_file.close()
    y_file.close()

if __name__== '__main__':
    #nltk.download('punkt')
    #punkt = nltk.data.load('tokenizers/punkt/english.pickle')

    #stemmer = GermanStemmer()#Join your own directories together.
    news_agent = "bbc"

    mypaths = ['business','entertainment','politics','sport','tech']# swith between sub directories under "datasets/raw_data/bbc/", #
    #mypaths =['test']
    # do toy experiment on different categories of datasets

    base_folder = os.path.dirname(os.path.realpath(__file__))
    for path_end in mypaths:
        path = os.path.join(base_folder +r"\..\..\datasets\raw_data",news_agent,path_end)
        if os.path.exists(path):
          try:
            os.chdir(path)
            for file in glob.glob("*.txt"):
               pathfile = os.path.join(path,file)
               print(pathfile)
               #prep_bbc(pathfile,file)
              #Do somethinghere
          except errno.ENOENT:
              print("inValid file path")


