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
from sklearn_crfsuite import CRF, metrics
from nltk import word_tokenize, pos_tag
from nltk.tag.util import untag
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
    tweetExtract = apiObject.user_timeline(screen_name=twitterHandle, count=70, lang="en", tweet_mode="extended")

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
    sentenceList = nltk.sent_tokenize(dataframeToStringNospaces)
    for sentence in sentenceList:
        tokenizedSentences = nltk.word_tokenize(str(sentence))

        # Get the tagged sentences from the pre tagged Penn Treebank corpus in NLTK's directory.

        taggedSentencesFromCorpus = nltk.corpus.treebank.tagged_sents()

        # Please refer to https://nlpforhackers.io/training-pos-tagger/ for the feature extraction function
        def feature_extraction_function(word, index):
            return {
                # Check for word itself
                'word': word[index],
                'is_first': index == 0,
                'is_last': index == len(word) - 1,
                'is_capitalized': word[index][0].upper() == word[index][0],
                'is_all_caps': word[index].upper() == word[index],
                'is_all_lower': word[index].lower() == word[index],
                # Indicates prefixes to words like using 'a' in 'ammoral' to show 'not moral'
                'prefix-1': word[index][0],
                # Indicates prefixes to words like using 'un' in 'unhappy' to show 'not happy'
                'prefix-2': word[index][:2],
                # Indicates prefixes to words like using 'dis' in 'disagree' to show 'not agree'
                'prefix-3': word[index][:3],
                # Indicates plurality to words like using 's' in 'bags' to show 'multiple bags'
                'suffix-1': word[index][-1],
                # Indicates past tense verbs eg. ending in 'ed'
                'suffix-2': word[index][-2:],
                # Indicates present participle verbs eg. ending in 'ing'
                'suffix-3': word[index][-3:],
                # Check for previous word and next word
                'previous_word': '' if index == 0 else word[index - 1],
                'next_word': '' if index == len(word) - 1 else word[index + 1],
                # Check for sub-string '-'
                'has_hyphen': '-' in word[index],
                'is_numeric': word[index].isdigit(),
                # Check if there are capital characters inside the string
                'capitals_inside': word[index][1:].lower() != word[index][1:]
            }

        # Split the dataset: 75% of the tagged sentences will be used for training and the remaining 25% will
        # be used for testing
        partitionText = int(0.75 * len(taggedSentencesFromCorpus))
        trainingSentences = taggedSentencesFromCorpus[:partitionText]  # From start to the partition
        testingSentences = taggedSentencesFromCorpus[
                           partitionText:]  # From partition to the end of the tagged sentences

        def pass_to_dataframe(taggedSentences):
            wordList = []   # X
            tagList = []    # Y

            for tagged in taggedSentences:
                wordList.append([feature_extraction_function(untag(tagged), index) for index in range(len(tagged))])
                tagList.append([tag for _, tag in tagged])

            return wordList, tagList

        # Each row in the dataframe is sequencial as CRF learns sequences - Unlike simple markov models which do not recall
        # sequences
        wordsTrain, tagsTrain = pass_to_dataframe(trainingSentences)
        wordsTest, tagsTest = pass_to_dataframe(testingSentences)

        # Use a CRFSuite model, and fit the training data to the model.
        model = CRF()
        # Fit the model to the trained words and corresponding tags
        model.fit(wordsTrain, tagsTrain)

        def position_tagger(sentence):
            featureWords = [feature_extraction_function(sentence, index) for index in range(len(sentence))]
            return list(zip(sentence, model.predict([featureWords])[0]))

        print(position_tagger(tokenizedSentences))

        tagsPrediction = model.predict(wordsTest)
        # Test for accuracy between the tested tags and the predicted tags
        accuracyModel = metrics.flat_accuracy_score(tagsTest, tagsPrediction)

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
            # If required, the line below, prints all the topics found.
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

    '''
    tweetDict = markov_chain(dataframeToStringNospaces)
    
    print("Text generation using simple Markov chains")
    # Range is number of tweets generated
    for _ in range(20):
        sentGen = generate_tweets(tweetDict)
        print("Generated tweet: ", sentGen)

    '''

    return maxTopics, accuracyModel


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
    maxTopic, modelAccuracy = coherence_funct(i, apiObject)
    topicNumList.append(maxTopic)

print("The accuracy of the CRF modelling is: ",modelAccuracy)


# Plot the twitter handle list against the number of topics list
plt.plot(handleList, topicNumList, 'ro')
plt.title("Topic count for Twitter users: Top 20 most followed British MPs")
plt.xlabel("Twitter handles")
plt.ylabel("Number of topics in tweets")
plt.xticks(rotation=90, ha='left')
plt.show()
