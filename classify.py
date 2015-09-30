import nltk
import collections
from pprint import pprint
''' We define an "item" class. We instantiate an object of this class for every research article'''

class item():
	
	
	def __init__(self, name, category):
		self.name = name
		self.category = category
		self.feature = {}	# feature set of the article(being treated as data item)

	
	'''Reading data and creating a list of items(articles).'''

	def read_words(self):
	
		theString = open('Data/'+self.name,'r').read()
		allowedCharacters ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

		finalString = ''

		for eachCharacter in theString:
			if eachCharacter in allowedCharacters:
				finalString = finalString + eachCharacter
			else:
				finalString = finalString + ' '


		wordList = finalString.split()

		return wordList
		
	'''This creates the feature set of the article by giving back the frequency of every dictionary word in the article.'''

	def features(self, top_words):
		for w in top_words:
			self.feature["%s" % w] = self.read_words().count(w)

		return self.feature

	'''This returns the most frequent words from the article.'''	

	def return_frequent(self):
		properNameList = []
		wordList1 = self.read_words()
		for eachWord in wordList1:
			if eachWord.istitle() and len(eachWord) >= 4 and len(eachWord) <= 20:

				properNameList.append(eachWord)
			else:
				continue
 
			properNamesDictionary = {}

		for eachName in properNameList:
 
			if eachName in properNamesDictionary:
				continue
			else:
				cnt = properNameList.count(eachName)
				properNamesDictionary[eachName] = cnt

		frequent = sorted(properNamesDictionary, key=properNamesDictionary.get, reverse=True)[:10]
		return frequent


'''This is the classifier function. It return a trained classifier.'''

def classify_articles(items_classify, top_words_classify):
	training_set = []
	for item_iter in items_classify:
		item_iter.features(top_words_classify)
		tup = (item_iter.feature , item_iter.category)  # tup is a 2-element tuple
		training_set.append(tup)
	classifier1 = nltk.NaiveBayesClassifier.train(training_set)
	return classifier1




def main():
	items_train = []
	total_top_words = []
	k = open("metadata.txt" ,'r')

	'''Reading data and creating a list of items(articles).'''

	for line1 in k:
		l = line1.split(",")
		m = item(l[0],l[1])
		items_train.append(m)
		frequent_m = m.return_frequent()
		total_top_words.extend(frequent_m)


	'''Taking the most frequent words from the collection of words achieved by combining the words from every article.'''


	total_top_words = collections.Counter(total_top_words)
	total_top_words = sorted(total_top_words, key=total_top_words.get, reverse=True)[:50]
	print sorted(total_top_words)
	

	classifier = classify_articles(items_train , total_top_words)  #this is the classifer


	s = open("test_data.txt", 'r')
	items_test = []
	

	'''Reading test data and creating a list of items(test articles).'''

	for line in s:
		l = line.split(",")
		q = item(l[0],l[1])
		items_test.append(q)

	z = open("result.txt",'w+')		# this file stores resultant labels
	i = 0
	j = 0
	for item_to_be in items_test:
		features_test = item_to_be.features(total_top_words)
		category = classifier.classify(features_test)
		probabilities = classifier.prob_classify(features_test)

		z.write(item_to_be.name+","+category)
		j = j+1
		if category == item_to_be.category:
			i = i+1
			
	print ( i / (j*1.0))* 100
	print i


if __name__ == "__main__": main()


