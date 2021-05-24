from collections import defaultdict, Counter
import re
import string
import math


class TextProcessor:
    stopWords = set(['i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", "we'll",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't"])
    @staticmethod
    def getTokens(rawText):
        #convert to lowercase
        rawTextLowerCase = rawText.lower()
        # split into the words
        tokens = re.split(" |\n|:",rawTextLowerCase)
        
        #remove the null strings
        tokens = list(filter(None,tokens))

        #remove fullstops and commas
        for i in range(len(tokens)):
            tokens[i] = tokens[i].replace(",","").replace(".","")
        #remove stopwords 
        tokens = TextProcessor.removeStopWords(tokens)
        
        return tokens

    @staticmethod
    def removePunctuations(rawText):
        rawText = rawText.translate(str.maketrans('', '', string.punctuation))
        return rawText
    
    @staticmethod
    def removeStopWords(tokens):
        return [word for word in tokens if not word in TextProcessor.stopWords]

    @staticmethod
    def getTermFrequency(document,features):
        termDictionary = dict.fromkeys(features,0)
        tokens = Counter(TextProcessor.getTokens(document))
        for key,val in termDictionary.items():
            if key in tokens:
                termDictionary[key] = tokens[key]
        totalFeatures = len(features)

        for key in termDictionary:
            termDictionary[key] = round(termDictionary[key] / totalFeatures, 4)
        return termDictionary

    @staticmethod
    def removeContradictions(rawText):
        pass    

    @staticmethod
    def getTF_IDF(termFrequencyDict,inverseDataFrequencyDict):
        tfidfDict = {}

        for word,val in termFrequencyDict.items():
            tfidfDict[word] = val * inverseDataFrequencyDict[word]
        return tfidfDict


class TextSimilarityEngine:

    def __init__(self,docs):
        self._docs = docs
        self._documentTermFrequecy = None 
        self._inverseDataFrequency = None
        self._tfidf = None
    
    def generateFeatureVector(self):
        features = list(set.union(*map(lambda doc:set(TextProcessor.getTokens(doc)),self._docs)))
        return features 

    def generateDocumentTermFrequencies(self):
        features = self.generateFeatureVector()
        self._documentTermFrequecy = list(map(lambda doc: TextProcessor.getTermFrequency(doc,features),self._docs))
        return self._documentTermFrequecy

    def generateInverseDataFrequencyDictionary(self):
        totalDocuments = len(self._docs)
        self._inverseDataFrequency = dict.fromkeys(self._documentTermFrequecy[0].keys(),0)
        for docTFDict in self._documentTermFrequecy:
            for word, val in docTFDict.items():
                if val > 0:
                    self._inverseDataFrequency[word] += 1
        #print(self._inverseDataFrequency)
        for word, val in self._inverseDataFrequency.items():
            self._inverseDataFrequency[word] = 1+math.log10(totalDocuments/float(val))
        return self._inverseDataFrequency
    
    def generateTFIDF(self):
        self._tfidf = []
        for tfDict in self._documentTermFrequecy:
            tfidfDict = TextProcessor.getTF_IDF(tfDict,self._inverseDataFrequency)
            
            self._tfidf.append(tfidfDict)
        return self._tfidf
    
    @staticmethod
    def getNorm(vector):
        summation = 0
        for i in range(len(vector)):
            summation += vector[i]**2
        normVec = math.sqrt(summation)
        return normVec
    @staticmethod
    def getDotProduct(vector1, vector2):
        dotProd = 0
        for i in range(len(vector1)):
            dotProd+= vector1[i]*vector2[i]
        return dotProd
    
    @staticmethod
    def getSimilarity(a,b):
        try:
            cosine = TextSimilarityEngine.getDotProduct(a,b) / (TextSimilarityEngine.getNorm(a)*TextSimilarityEngine.getNorm(b))
        except ZeroDivisionError:
            return 0
        return cosine if cosine <= 1 else 1

"""
DEBUGGING CODE

docs = [doc1,doc2]
textSimilarityObj = TextSimilarityEngine(docs)
textSimilarityObj.generateDocumentTermFrequencies()
textSimilarityObj.generateInverseDataFrequencyDictionary()
textSimilarityObj.generateTFIDF()
doc1_tfidf = list(textSimilarityObj._tfidf[0].values())
doc2_tfidf = list(textSimilarityObj._tfidf[1].values())
cosine_doc1_doc2 = TextSimilarityEngine.getSimilarity(doc1_tfidf,doc2_tfidf)
print(cosine_doc1_doc2)

"""