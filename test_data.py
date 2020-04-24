from nltk.corpus import brown
import nltk

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
# We train a UnigramTagger by specifying tagged sentence data as a parameter
# when we initialize the tagger.
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
print(unigram_tagger.evaluate(brown_tagged_sents))