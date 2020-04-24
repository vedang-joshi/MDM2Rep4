#!/usr/bin/python

'''
Install tweepy:
$ git clone https://github.com/tweepy/tweepy.git
$ cd tweepy
$ pip install

Install pandas:
$ pip install pandas
'''

# Import the libraries
import tweepy
import string
import re
import random
import operator
import nltk
import scipy.sparse
from collections import defaultdict
from gensim import matutils, models
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from pprint import pprint
import pandas as pd
import gensim.corpora as corpora
import matplotlib.pyplot as plt


# Read in login file to connect up with Twitter API
loginFile = pd.read_csv("Login.csv")

# Get Twitter API credentials
consumerKey = loginFile["key"][0]
consumerSecret = loginFile["key"][1]
accessToken = loginFile["key"][2]
accessTokenSecret = loginFile["key"][3]

# Create the authentication object
authenticateObject = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticateObject.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in authentication information
apiObject = tweepy.API(authenticateObject, wait_on_rate_limit=True)


def coherence_funct(twitterHandle, apiObject):
    print("")
    print(twitterHandle)
    print("")
    # Extract 100 tweets from the twitter user
    tweetExtract = apiObject.user_timeline(screen_name=twitterHandle, count=1500, lang="en", tweet_mode="extended")

    # Create a dataframe with a column called Tweets
    tweetList = []
    for tweet in tweetExtract:
        tweetList.append(tweet.full_text)

    tweetsDataframe = pd.DataFrame(tweetList, columns=['Original_Tweets'])

    # Users may uncomment the lines below to get the dataframe showing the tweets obtained from the user.
    #print('''
    #Show the tweets obtained from twitter API on a PANDAS dataframe''')
    #print(tweetsDataframe)

    # Create a function to clean the tweets
    def text_cleaner(text):
        text = re.sub('@[A-Za-z0–9]+', '', text)                        # Removing @mentions
        text = re.sub('#', '', text)                                    # Removing hash tag
        text = re.sub('RT[\s]+', '', text)                              # Removing RT
        text = re.sub('https?:\/\/\S+', '', text)                       # Removing hyperlink
        text = text.lower()                                             # make text lowercase
        text = re.sub('\[.*?\]', '', text)                              # Remove text in square brackets
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
        text = re.sub('\w*\d\w*', '', text)                             # remove words containing numbers
        text = re.sub('[‘’“”…]', '', text)                              # remove additional punctuation
        text = re.sub('\n', '', text)                                   # remove nonsensical text missed earlier
        emoji = re.compile("["
            u"\U0001F600-\U0001F64F"                                    # emoticons
            u"\U0001F300-\U0001F5FF"                                    # symbols & pictographs
            u"\U0001F680-\U0001F6FF"                                    # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"                                    # flags (iOS)
            u"\U00002500-\U00002BEF"                                    # chinese characters
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  
            u"\u3030"
                          "]+", re.UNICODE)
        text = emoji.sub(r'', text)
        return text

    # Create a function to remove the stopwords
    stop = set(stopwords.words('english'))
    def remove_stopwords(x):
        stopList = []
        for word in x.split():
            if word not in (stop):
                stopList.append(word)
        return ' '.join(stopList)

    # Clean the tweets and remove stopwords
    tweetsDataframe['First_Clean_Tweets'] = tweetsDataframe['Original_Tweets'].apply(text_cleaner)
    tweetsDataframe['Second_Clean_Tweets'] = tweetsDataframe['First_Clean_Tweets'].apply(remove_stopwords)

    # Users may uncomment the lines below to get the dataframe after cleaning the tweets obtained from the user.
    #print('''
    #Show the tweets obtained from after cleaning the data i.e. after removing stopwords etc''')
    #print(tweetsDataframe)

    # Get data as string to take out noun phrases, verb phrases etc.
    changeDataframeToList = tweetsDataframe['First_Clean_Tweets'].tolist()
    dataframeToString = ' '.join(changeDataframeToList)
    newTokenizedList = []
    for line in changeDataframeToList:
        for word in line.split():
            newTokenizedList.append(word)
    dataframeToStringNospaces = " ".join(dataframeToString.split())

    # separate into noun phrases, adjectives, verb phrases etc.
    tokenizedSentence = nltk.sent_tokenize(dataframeToStringNospaces)

    nounList = []
    verbList = []
    adjectiveList = []
    adverbList = []
    for sentence in tokenizedSentence:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nounList.append(word)
            elif (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
                verbList.append(word)
            elif (pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
                adjectiveList.append(word)
            elif (pos == 'RB' or pos == 'RBR' or pos == 'RBS'):
                adverbList.append(word)

    print("")
    print("Noun Phrase List: ", nounList)
    print("")
    print("Verb Phrase List: ", verbList)
    print("")
    print("Adjectives List: ", adjectiveList)
    print("")
    print("Adverb List: ", adverbList)
    print("")

    # Topic Modelling: Assign topics to the subject talked about. Focusing on Noun/Noun Phrases as most of the meaning
    #                                                            of the sentence can be found in the noun phrases used.

    def nouns(text):
        # Given a string of text, tokenize the text and pull out only the nouns.
        def is_noun(position):
            return position[:2] == 'NN'
        tokenizedTxt = word_tokenize(text)
        allNouns = []
        for (word,position) in pos_tag(tokenizedTxt):
            if is_noun(position):
                allNouns.append(word)
        return ' '.join(allNouns)

    # Apply the nouns function to the transcripts to filter only on nouns
    dataframeNouns = pd.DataFrame(tweetsDataframe['Second_Clean_Tweets'].apply(nouns))

    # Users may uncomment the lines below to get the dataframe containing nouns.
    #print('''
    #Show the tweets obtained''')
    #print(dataframeNouns)

    # Re-add the additional stop words since we are recreating the document-term matrix

    with open('stopwords.txt', 'r') as file:
        addStopList = file.read().splitlines()

    appendedStopList = stop.union(addStopList)

    # Recreate a document-term matrix
    countVectorizer = CountVectorizer(stop_words=appendedStopList)
    dataCountVectorizer = countVectorizer.fit_transform(tweetsDataframe['Second_Clean_Tweets'])
    countVectorizerToArray = pd.DataFrame(dataCountVectorizer.toarray(), columns=countVectorizer.get_feature_names())
    countVectorizerToArray.index = dataframeNouns.index

    # Users may uncomment the lines below to get the document term matrix..
    #print('''
    #Show the document term matrix''')
    #print(countVectorizerToArray)

    # Create the gensim corpus
    termDocMatrix = countVectorizerToArray.transpose()
    sparseCounts = scipy.sparse.csr_matrix(termDocMatrix)
    gensimCorpus = matutils.Sparse2Corpus(sparseCounts)
    listOfList = []
    listOfList.append(newTokenizedList)
    # Create the vocabulary dictionary. Gensim requires mapping from word IDs to words
    id2word = corpora.Dictionary(listOfList)


    def coherence_values(dictionary, corpus, limit, start=1, step=1):
        # Returns:
        # coherenceValues : Coherence values corresponding to the LDA model with respective number of topics
        coherenceValues = []
        for num_topics in range(start, limit, step):
            modelTopic=models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                       passes=10, alpha='auto')
            # If required the line below, prints all the topics found.
            #pprint(modelTopic.print_topics())
            coherenceModel = CoherenceModel(model=modelTopic, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherenceValues.append(coherenceModel.get_coherence())

        return coherenceValues

    coherenceValues = coherence_values(dictionary=id2word, corpus=gensimCorpus, start=1, limit=30, step=1)
    limitOfTopics=30

    # Create an x-values list to plot against the coherence values in the coherenceValues list.
    xValuesList = []
    for i in range(1, limitOfTopics, 1):
        xValuesList.append(i)

    # Add the x-values list and the coherence values list to a dictionary, so both key and value can be accessed easily
    sortedDict = dict(zip(xValuesList, coherenceValues))
    maxTopics = max(sortedDict.items(), key=operator.itemgetter(1))[0]

    # The lines below may be uncommented if the coherence value graph is to be obtained.
    #plt.plot(xValuesList, coherenceValues, 'g-')
    #plt.title("original: only nouns")
    #plt.xlabel("Num Topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("coherence_values"), loc='best')
    #plt.show()

    # Here we try to generate text in the same style as the tweets from a specific user.
    # We use simple Markov chains to achieve this!

    def markov_chain(textInput):
        # The input is a string of text i.e. here it is the tweets from one particluar twitter handle
        # and the output is a dictionary of current words and potential future words that may occur in a tweet.
        # The training data used is the user generated tweets and the test data is the

        # Tokenize the text by word including punctuation
        tokenizedText = textInput.split(' ')

        # Initialize a default dictionary to hold all the current words and potential next words.
        # Extremely powerful as even if key value doesn't exist we can still input a key value pair in this default
        # dictionary; something which cannot be done in a normal dict.
        defaultDict = defaultdict(list)

        # Create a zipped list of all of the word pairs and put them in a list of next words
        # tokenizedText[0:-1] == all the words from start to all but one.
        # tokenizedText[1:] == all the words from the second word to the end.
        # Thus the first word will be paired to the second one, the second to the third, fourth and so on.
        for currentWord, nextWord in zip(tokenizedText[0:-1], tokenizedText[1:]):
            defaultDict[currentWord].append(nextWord)

        # Convert the default dict back into a dictionary
        convertedDict = dict(defaultDict)
        return convertedDict

    def generate_tweets(chain):
        # Input a dictionary generated from the markov_chain function in the form current word: next potential words list
        # to generate random sentences in the style of the original user.

        # Number of words in generated tweet
        counter = 15
        # Randomly pick a word and capitalize the first letter
        randomKey = random.choice(list(chain.keys()))
        generatedSentence = randomKey.capitalize()

        # Generate the second word from the value list. Set the new word as the first word. Repeat.
        for i in range(counter - 1):
            nextRandomWord = random.choice(chain[randomKey])
            randomKey = nextRandomWord
            generatedSentence += ' ' + nextRandomWord

        # End it with a fullstop
        generatedSentence += '.'
        return (generatedSentence)

    tweetDict = markov_chain(dataframeToStringNospaces)

    print("Text generation using simple Markov chains")
    # Range is number of tweets generated
    for _ in range(20):
        sentGen = generate_tweets(tweetDict)
        print("Generated tweet: ", sentGen)

    return maxTopics


###################### Main Execution #########################################################
# Initialise a number of topics list & a twitter handle list. Users may input
# custom twitter handles in the file twitter_handles.txt which will be read into the program.
topicNumList = []
handleList = []
with open('twitter_handles.txt', 'r') as file:
    twitterHandles = file.read().splitlines()

for i in twitterHandles:
    handleList.append(i)

# Get the topic number for the LDA model with the highest coherence value as calculated in the coherence_funct above.
for i in handleList:
    maxTopic = coherence_funct(i, apiObject)
    topicNumList.append(maxTopic)

# Plot the twitter handle list against the number of topics list
plt.plot(handleList, topicNumList, 'r-')
plt.title("Topic count for specific Twitter users")
plt.xlabel("Twitter handles")
plt.ylabel("Number of topics in tweets")
plt.xticks(rotation=90, ha='left')
plt.show()