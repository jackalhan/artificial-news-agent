from gensim import corpora
from gensim import utils
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
import preprocessor as p
from collections import Counter
import codecs
import string
import os
import csv
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from six import iteritems
from gensim import corpora
dataFolder = 'data/'
inputFolder = 'input/'
outputFolder = 'output/'
articlesFolder = 'articles/'

def ReadAll():
    for filename in os.listdir(os.getcwd()):
        print (filename)

def VisitingFolder(FolderName):
    os.chdir(FolderName)

def CleanData(filenames, target):
    for item in filenames:
        with open(item, 'r', encoding='utf-8') as csvReadfile:
             reader = csv.DictReader(csvReadfile)
             compare = 'test'
             for row in reader:
                new = p.clean(row['tweet'])
                new2 = re.sub('[^A-Za-z0-9]+', ' ', new).replace('http', '')
                #remove all digits
                new3 =''.join([i for i in new2 if not i.isdigit()])
                #remove all multiple spaces and spaces at beginning, and convert to lower case
                new4 = new3.strip().lower()
                if len(new4) > 40 and compare != new4 :
                    compare = new4
                    target.writerow({'user': row['user'], 'text': new4, 'length': len(new4)})
def get_file_list(key):
    result = []
    filenames = os.listdir(os.getcwd())
    for item in filenames:
        if any(x in item for x in key):
            result.append(item)
    return result

def WriteAllTxt2Single(folderName, writer):
    VisitingFolder(folderName)
    filenames = os.listdir(os.getcwd())
    for item in filenames:
        with codecs.open(item, 'rb') as txtReadfile:
             reader = txtReadfile.readline()
             writer.write(reader + '\n'.encode('ascii'))
        txtReadfile.close()
    os.chdir('..')
def Sumlines(document):
    return sum(1 for line in open(document))

def Mergedocument(folderName, newfile):
    document = []
    VisitingFolder(folderName)
    filenames = os.listdir(os.getcwd())
    for item in filenames:
        with open(item, 'r') as txtReadfile:
             reader = txtReadfile.read()
             document.append(reader)
    txtReadfile.close()
    with open(newfile, 'w') as txtWritefile:
        for line in document:
            txtWritefile.writelines(line + '\n')
    txtWritefile.close()
    print(Sumlines(newfile))
    os.chdir('..')
def Totalwords(doc):
    with open(doc, 'r') as txtReadfile:
        words = [word for line in txtReadfile for word in line.split()]
        print (len(words))

def BuildDoc(docs):
    with open(docs, 'r') as txtReadfile:
        doc = [line.strip() for line in txtReadfile]
    txtReadfile.close()
    return doc

def Dictionry():
    # remove common words and tokenize
    documents = BuildDoc('single.txt')
    stoplist = BuildDoc('stopwords.txt')

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    from pprint import pprint  # pretty-printer
    pprint(texts)
    print (len(texts[0]))
    dictionary = corpora.Dictionary(texts)
    dictionary.save('dic.dict')  # store the dictionary, for future reference
    #print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    print(corpus)

def Transform():
    dictionary = corpora.Dictionary.load('dic.dict')
    corpus = corpora.MmCorpus('corpus.mm')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    corpora.BleiCorpus.serialize('corpus.lda-c', corpus)
    #for doc in corpus_tfidf:
    #   print(doc)

def Callmodel():
    corpus = corpora.MmCorpus('corpus.mm')
    dictionary = corpora.Dictionary.load('dic.dict')
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=150)
    lda.print_topics(150)
#################Followint are function calls:

# Mergedocument(articlesFolder, 'single.txt')
'''
with codecs.open('single.txt', 'wb') as txtWritefile:
     WriteAllTxt2Single(inputFolder, txtWritefile)
txtWritefile.close()
print (Sumlines('single.txt'))
'''
VisitingFolder(inputFolder)
#print (os.getcwd())
business = ['business', 'customer', 'profit', 'manager', 'staff', 'money', 'work', 'market', 'company', 'product']
entertainment = ['fun', 'music', 'movie', 'star','game','watch','style','life','song','celebration']
politics = ['government', 'trump', 'economy', 'energy', 'jobs', 'tax', 'war', 'democrats', 'iraq', 'reform']
sport = ['athlete', 'football', 'golf', 'hockey', 'rugby', 'swimming', 'tennis', 'baseball', 'nike', 'nba']
tech = ['4G', 'data mining', 'agile', 'big data', 'science', 'algorithm', 'engine', 'framework', 'html', 'cloud computing']

filenames = get_file_list(sport)

with open('CleanedTweets1.csv', 'w', encoding='utf-8', newline='') as csvWritefile:
     fieldnames = ['user', 'text', 'length']
     writer = csv.DictWriter(csvWritefile, fieldnames=fieldnames)
     writer.writeheader()
     CleanData(filenames, writer)

