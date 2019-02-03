from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
#en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
   

# create sample documents
doc_a = "I'm really missing my family, I just want to go home. It's so hard living so far from my support systems. I feel so lonely all the time, I haven't made too many friends and I now live alone. I hate going back to my room at the end of the day to no one"
doc_b = "My anxiety is not improving, I've had such a hard time finding a counsellor that I feel comfortable talking to and who fits in my schedule. I'm not sure if I need an increase in my antidepressants or if maybe this feeling with go away as I get used to living alone. Every time I go home, I just want to cry. I don't know much longer I can do this for." 
doc_c = "I am excited and stressed about moving to university soon. I want to meet new people, but I know that the work is going to be so much harder than I can anticipate. I don't have many friends going to the same university as me, so I am a bit nervous but also happy that I have the opportunity to meet new people."
doc_d = "I really like playing the piano. My favourite type of songs to play are ragtime hits! This September, it will have been my 8th year playing and I hope to pass my grade 8 piano exam with distinction. Here's hoping!"
doc_e = "Today, a bat flew into my room! I swear, my window wasn't even open and yet it swooped right in and went into a box underneath my bed. I had absolutely no idea what to do, I was so anxious! I hesitantly took the box outside with my friend Miranda and set the bat free."
doc_f = "Frosh week was so fun! I met so many friends in my frosh group, doing things that I never could have imagined: playing in the mud, throwing oatmeal at people and climbing an old football pole. The anxiety towards school started to set in though, I am nervous I won't be smart enough to stay in the program. I feel like I may not deserve to be here and that everyone else will be better than me."
doc_g = "I am stressing over midterm season, I don't know how I am going to have time to do anything other than studying. There are multiple midterms every week, on top of regular classes, labs and assignments. I really want to do well, I can't do poorly, I want to do really well. I mean, I hope I can just do okay."
doc_h = "It's harder than I thought living so far away from home. I really miss my parents and my siblings and sometimes I just feel as though I can't cope. But I get through it every time I guess. I don't know why, it is just a source of anxiety for me. As much as I want to go home and see them, I worry that it will get harder to leave them."
doc_i = "I can't wait to go home for Thanksgiving! It's gonna be so nice to see my family, to eat good food and sleep in my bed. I worry though that I will have to spend a lot of time doing work and not be able to enjoy it. I will just have to do the best that I can."
doc_j = "I'm feeling quite overwhelmed. I can't wait to get a break from school so I can just watch TV and sleep. School has just been so involved and it feel nice once its all done."
doc_k = "My boyfriend and I are barely speaking. Ever since we went off to university, nothing has been the same. I hate that we're like this, and I really hope we don't break up. I do love him but I fear that we are drifting too far apart."
doc_l = "Anxiety is consuming my body, and my life. I just feel like I'm a slave to my school work and I hate how I'm letting it take over my whole life, even though I feel like there's nothing I can do about it. It has been days since I have washed my hair and I am so behind on laundry. But alas, the end is in sight."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g, doc_h, doc_i, doc_j, doc_k, doc_l]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stop_words]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=100)
print(ldamodel.print_topics(num_topics=4, num_words=4))