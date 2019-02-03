import spacy
import en_core_web_sm
import nltk
from nltk.stem.porter import *

nlp = spacy.load('en_core_web_sm')

stemmer = PorterStemmer()

git_message = 'I am depressed'

doc = nlp(git_message)
issue = ''
for ent in doc.ents:
    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
    if ent.label_ == 'ISSU':
        issue = stemmer.stem(ent.text)
        print(issue)

