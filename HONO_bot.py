print("\nHONO Bot> Please wait, while I am loading my dependencies\n")

from nltk import pos_tag,word_tokenize,ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet,stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree
# from DateExtractor import extractDate
import json
import math
import re
import string
# from DocumentRetrievalModel import DocumentRetrievalModel as DRM
# from ProcessedQuestion import ProcessedQuestion as PQ
import re
import sys
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# ClassName : ProcessedQuestion.py
# Description : Takes question as an input and process it to find out question
#   and answer type, also prepare question vector and prepare search query for
#   Information Retrieval process
# Arguments : 
#       Input :
#           question(str) : String of question
#           useStemmer(boolean) : Indicate to use stemmer for question tokens
#           useSynonyms(boolean) : Indicate to use thesaraus for query expansion
#           removeStopwords(boolean) : Indicate to remove stop words from search
#                                      query
#       Output :
#           Instance of ProcessedQuestion with useful following structure
#               qVector(dict) : Key Value pair of word and its frequency
#                               to be used for Information Retrieval and 
#                               similarity calculation
#               question(str) : Raw question
#               qType(str) : Type of question
#               aType(str) : Expected answer type
#                       ["PERSON","LOCATION","DATE","DEFINITION","YESNO"]
#   
class ProcessedQuestion:
    def __init__(self, question, useStemmer = False, useSynonyms = False, removeStopwords = False):
        self.question = question
        self.useStemmer = useStemmer
        self.useSynonyms = useSynonyms
        self.removeStopwords = removeStopwords
        self.stopWords = stopwords.words("english")
        self.stem = lambda k : k.lower()
        if self.useStemmer:
            ps = PorterStemmer()
            self.stem = ps.stem
        self.qType = self.determineQuestionType(question)
        self.searchQuery = self.buildSearchQuery(question)
        self.qVector = self.getQueryVector(self.searchQuery)
        self.aType = self.determineAnswerType(question)
    
    # To determine type of question by analyzing POS tag of question from Penn 
    # Treebank tagset
    #
    # Input:
    #           question(str) : Question string
    # Output:
    #           qType(str) : Type of question among following
    #                   [ WP ->  who
    #                     WDT -> what, why, how
    #                     WP$ -> whose
    #                     WRB -> where ]
    def determineQuestionType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTags = []
        for token in qPOS:
            if token[1] in questionTaggers:
                qTags.append(token[1])
        qType = ''
        if(len(qTags)>1):
            qType = 'complex'
        elif(len(qTags) == 1):
            qType = qTags[0]
        else:
            qType = "None"
        return qType
    
    # To determine type of expected answer depending of question type
    #
    # Input:
    #           question(str) : Question string
    # Output:
    #           aType(str) : Type of answer among following
    #               [PERSON, LOCATION, DATE, ORGANIZATION, QUANTITY, DEFINITION
    #                   FULL]
    def determineAnswerType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTag = None

        for token in qPOS:
            if token[1] in questionTaggers:
                qTag = token[0].lower()
                break
        
        if(qTag == None):
            if len(qPOS) > 1:
                if qPOS[1][1].lower() in ['is','are','can','should']:
                    qTag = "YESNO"
        #who/where/what/why/when/is/are/can/should
        if qTag == "who":
            return "PERSON"
        elif qTag == "where":
            return "LOCATION"
        elif qTag == "when":
            return "DATE"
        elif qTag == "what":
            # Defination type question
            # If question of type whd modal noun? its a defination question
            qTok = self.getContinuousChunk(question)
            #print(qTok)
            if(len(qTok) == 4):
                if qTok[1][1] in ['is','are','was','were'] and qTok[2][0] in ["NN","NNS","NNP","NNPS"]:
                    self.question = " ".join([qTok[0][1],qTok[2][1],qTok[1][1]])
                    #print("Type of question","Definition",self.question)
                    return "DEFINITION"

            # ELSE USE FIRST HEAD WORD
            for token in qPOS:
                if token[0].lower() in ["city","place","country"]:
                    return "LOCATION"
                elif token[0].lower() in ["company","industry","organization"]:
                    return "ORGANIZATION"
                elif token[1] in ["NN","NNS"]:
                    return "FULL"
                elif token[1] in ["NNP","NNPS"]:
                    return "FULL"
            return "FULL"
        elif qTag == "how":
            if len(qPOS)>1:
                t2 = qPOS[2]
                if t2[0].lower() in ["few","great","little","many","much"]:
                    return "QUANTITY"
                elif t2[0].lower() in ["tall","wide","big","far"]:
                    return "LINEAR_MEASURE"
            return "FULL"
        else:
            return "FULL"
    
    # To build search query by dropping question word
    #
    # Input:
    #           question(str) : Question string
    # Output:
    #           searchQuery(list) : List of tokens
    def buildSearchQuery(self, question):
        qPOS = pos_tag(word_tokenize(question))
        searchQuery = []
        questionTaggers = ['WP','WDT','WP$','WRB']
        for tag in qPOS:
            if tag[1] in questionTaggers:
                continue
            else:
                searchQuery.append(tag[0])
                if(self.useSynonyms):
                    syn = self.getSynonyms(tag[0])
                    if(len(syn) > 0):
                        searchQuery.extend(syn)
        return searchQuery
    
    # To build query vector
    #
    # Input:
    #       searchQuery(list) : List of tokens from buildSearchQuery method
    # Output:
    #       qVector(dict) : Dictionary of words and their frequency
    def getQueryVector(self, searchQuery):
        vector = {}
        for token in searchQuery:
            if self.removeStopwords:
                if token in self.stopWords:
                    continue
            token = self.stem(token)
            if token in vector.keys():
                vector[token] += 1
            else:
                vector[token] = 1
        return vector
    
    # To get continuous chunk of similar POS tags.
    # E.g.  If two NN tags are consequetive, this method will merge and return
    #       single NN with combined value.
    #       It is helpful in detecting name of single person like John Cena, 
    #       Steve Jobs
    # Input:
    #       question(str) : question string
    # Output:
    #       
    def getContinuousChunk(self,question):
        chunks = []
        answerToken = word_tokenize(question)
        nc = pos_tag(answerToken)

        prevPos = nc[0][1]
        entity = {"pos":prevPos,"chunk":[]}
        for c_node in nc:
            (token,pos) = c_node
            if pos == prevPos:
                prevPos = pos       
                entity["chunk"].append(token)
            elif prevPos in ["DT","JJ"]:
                prevPos = pos
                entity["pos"] = pos
                entity["chunk"].append(token)
            else:
                if not len(entity["chunk"]) == 0:
                    chunks.append((entity["pos"]," ".join(entity["chunk"])))
                    entity = {"pos":pos,"chunk":[token]}
                    prevPos = pos
        if not len(entity["chunk"]) == 0:
            chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    # To get synonyms of word in order to improve query by using query
    # expanision technique
    # Input:
    #       word(str) : Word token
    # Output:
    #       synonyms(list) : List of synonyms of given word
    def getSynonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name().lower()
                synonyms.extend(w.split("_"))
        return list(set(synonyms))
    
    # String representation of this class
    def __repr__(self):
        msg = "Q: " + self.question + "\n"
        msg += "QType: " + self.qType + "\n"
        msg += "QVector: " + str(self.qVector) + "\n"
        return msg
    
    
# ClassName : DocumentRetrievalModel.py
# Description : Script preprocesses article and paragraph to computer TFIDF.
#               Additionally, helps in answer processing 
# Arguments : 
#       Input :
#           paragraphs(list)        : List of paragraphs
#           useStemmer(boolean)     : Indicate to use stemmer for word tokens
#           removeStopWord(boolean) : Indicate to remove stop words from 
#                                     paragraph in order to keep relevant words
#       Output :
#           Instance of DocumentRetrievalModel with following structure
#               query(function) : Take instance of processedQuestion and return
#                                 answer based on IR and Answer Processing
#                                 techniques

# Importing Library


class DocumentRetrievalModel:
    def __init__(self,paragraphs,removeStopWord = False,useStemmer = False):
        self.idf = {}               # dict to store IDF for words in paragraph
        self.paragraphInfo = {}     # structure to store paragraphVector
        self.paragraphs = paragraphs
        self.totalParas = len(paragraphs)
        self.stopwords = stopwords.words('english')
        self.removeStopWord = removeStopWord
        self.useStemmer = useStemmer
        self.vData = None
        self.stem = lambda k:k.lower()
        if(useStemmer):
            ps = PorterStemmer()
            self.stem = ps.stem
            
        # Initialize
        self.computeTFIDF()
        
    # Return term frequency for Paragraph
    # Input:
    #       paragraph(str): Paragraph as a whole in string format
    # Output:
    #       wordFrequence(dict) : Dictionary of word and term frequency
    def getTermFrequencyCount(self,paragraph):
        sentences = sent_tokenize(paragraph)
        wordFrequency = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if self.removeStopWord == True:
                    if word.lower() in self.stopwords:
                        #Ignore stopwords
                        continue
                    if not re.match(r"[a-zA-Z0-9\-\_\\/\.\']+",word):
                        continue
                #Use of Stemmer
                if self.useStemmer:
                    word = self.stem(word)
                    
                if word in wordFrequency.keys():
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
        return wordFrequency
    
    # Computes term-frequency inverse document frequency for every token of each
    # paragraph
    # Output:
    #       paragraphInfo(dict): Dictionary for every paragraph with following 
    #                            keys
    #                               vector : dictionary of TFIDF for every word
    def computeTFIDF(self):
        # Compute Term Frequency
        self.paragraphInfo = {}
        for index in range(0,len(self.paragraphs)):
            wordFrequency = self.getTermFrequencyCount(self.paragraphs[index])
            self.paragraphInfo[index] = {}
            self.paragraphInfo[index]['wF'] = wordFrequency
        
        wordParagraphFrequency = {}
        for index in range(0,len(self.paragraphInfo)):
            for word in self.paragraphInfo[index]['wF'].keys():
                if word in wordParagraphFrequency.keys():
                    wordParagraphFrequency[word] += 1
                else:
                    wordParagraphFrequency[word] = 1
        
        self.idf = {}
        for word in wordParagraphFrequency:
            # Adding Laplace smoothing by adding 1 to total number of documents
            self.idf[word] = math.log((self.totalParas+1)/wordParagraphFrequency[word])
        
        #Compute Paragraph Vector
        for index in range(0,len(self.paragraphInfo)):
            self.paragraphInfo[index]['vector'] = {}
            for word in self.paragraphInfo[index]['wF'].keys():
                self.paragraphInfo[index]['vector'][word] = self.paragraphInfo[index]['wF'][word] * self.idf[word]
    

    # To find answer to the question by first finding relevant paragraph, then
    # by finding relevant sentence and then by procssing sentence to get answer
    # based on expected answer type
    # Input:
    #           pQ(ProcessedQuestion) : Instance of ProcessedQuestion
    # Output:
    #           answer(str) : Response of QA System
    def query(self,pQ):
        
        # Get relevant Paragraph
        relevantParagraph = self.getSimilarParagraph(pQ.qVector)

        # Get All sentences
        sentences = []
        for tup in relevantParagraph:
            if tup != None:
                p2 = self.paragraphs[tup[0]]
                sentences.extend(sent_tokenize(p2))
        
        # Get Relevant Sentences
        if len(sentences) == 0:
            return "Oops! Unable to find answer"

        # Get most relevant sentence using unigram similarity
        relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)

        # AnswerType
        aType = pQ.aType
        
        # Default Answer
        answer = relevantSentences[0][0]

        ps = PorterStemmer()
        # For question type looking for Person
        if aType == "PERSON":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "PERSON":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    # If any entity is already in question
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "LOCATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "GPE":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    # If any entity is already in question
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "ORGANIZATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "ORGANIZATION":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    # If any entity is already in question
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "DATE":
            allDates = []
            for s in relevantSentences:
                allDates.extend(extractDate(s[0]))
            if len(allDates)>0:
                answer = allDates[0]
        elif aType in ["NN","NNP"]:
            candidateAnswers = []
            ne = self.getContinuousChunk([s[0] for s in relevantSentences])
            for entity in ne:
                if aType == "NN":
                    if entity[0] == "NN" or entity[0] == "NNS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
                elif aType == "NNP":
                    if entity[0] == "NNP" or entity[0] == "NNPS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
        elif aType == "DEFINITION":
            relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)
            answer = relevantSentences[0][0]
        return answer
        
    # Get top 3 relevant paragraph based on cosine similarity between question 
    # vector and paragraph vector
    # Input :
    #       queryVector(dict) : Dictionary of words in question with their 
    #                           frequency
    # Output:
    #       pRanking(list) : List of tuple with top 3 paragraph with its
    #                        similarity coefficient
    def getSimilarParagraph(self,queryVector):    
        queryVectorDistance = 0
        for word in queryVector.keys():
            if word in self.idf.keys():
                queryVectorDistance += math.pow(queryVector[word]*self.idf[word],2)
        queryVectorDistance = math.pow(queryVectorDistance,0.5)
        if queryVectorDistance == 0:
            return [None]
        pRanking = []
        for index in range(0,len(self.paragraphInfo)):
            sim = self.computeSimilarity(self.paragraphInfo[index], queryVector, queryVectorDistance)
            pRanking.append((index,sim))
        
        return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]
    
    # Compute cosine similarity betweent queryVector and paragraphVector
    # Input:
    #       pInfo(dict)         : Dictionary containing wordFrequency and 
    #                             paragraph Vector
    #       queryVector(dict)   : Query vector for question
    #       queryDistance(float): Distance of queryVector from origin
    # Output:
    #       sim(float)          : Cosine similarity coefficient
    def computeSimilarity(self, pInfo, queryVector, queryDistance):
        # Computing pVectorDistance
        pVectorDistance = 0
        for word in pInfo['wF'].keys():
            pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
        pVectorDistance = math.pow(pVectorDistance,0.5)
        if(pVectorDistance == 0):
            return 0

        # Computing dot product
        dotProduct = 0
        for word in queryVector.keys():
            if word in pInfo['wF']:
                q = queryVector[word]
                w = pInfo['wF'][word]
                idf = self.idf[word]
                dotProduct += q*w*idf*idf
        
        sim = dotProduct / (pVectorDistance * queryDistance)
        return sim
    
    # Get most relevant sentences using unigram similarity between question
    # sentence and sentence in paragraph containing potential answer
    # Input:
    #       sentences(list)      : List of sentences in order of occurance as in
    #                              paragraph
    #       pQ(ProcessedQuestion): Instance of processedQuestion
    #       nGram(int)           : Value of nGram (default 3)
    # Output:
    #       relevantSentences(list) : List of tuple with sentence and their
    #                                 similarity coefficient
    def getMostRelevantSentences(self, sentences, pQ, nGram=3):
        relevantSentences = []
        for sent in sentences:
            sim = 0
            if(len(word_tokenize(pQ.question))>nGram+1):
                sim = self.sim_ngram_sentence(pQ.question,sent,nGram)
            else:
                sim = self.sim_sentence(pQ.qVector, sent)
            relevantSentences.append((sent,sim))
        
        return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)
    
    # Compute ngram similarity between a sentence and question
    # Input:
    #       question(str)   : Question string
    #       sentence(str)   : Sentence string
    #       nGram(int)      : Value of n in nGram
    # Output:
    #       sim(float)      : Ngram Similarity Coefficient
    def sim_ngram_sentence(self, question, sentence,nGram):
        #considering stop words as well
        ps = PorterStemmer()
        getToken = lambda question:[ ps.stem(w.lower()) for w in word_tokenize(question) ]
        getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]
        qToken = getToken(question)
        sToken = getToken(sentence)

        if(len(qToken) > nGram):
            q3gram = set(getNGram(qToken,nGram))
            s3gram = set(getNGram(sToken,nGram))
            if(len(s3gram) < nGram):
                return 0
            qLen = len(q3gram)
            sLen = len(s3gram)
            sim = len(q3gram.intersection(s3gram)) / len(q3gram.union(s3gram))
            return sim
        else:
            return 0
    
    # Compute similarity between sentence and queryVector based on number of 
    # common words in both sentence. It doesn't consider occurance of words
    # Input:
    #       queryVector(dict)   : Dictionary of words in question
    #       sentence(str)       : Sentence string
    # Ouput:
    #       sim(float)          : Similarity Coefficient    
    def sim_sentence(self, queryVector, sentence):
        sentToken = word_tokenize(sentence)
        ps = PorterStemmer()
        for index in range(0,len(sentToken)):
            sentToken[index] = ps.stem(sentToken[index])
        sim = 0
        for word in queryVector.keys():
            w = ps.stem(word)
            if w in sentToken:
                sim += 1
        return sim/(len(sentToken)*len(queryVector.keys()))
    
    # Get Named Entity from the sentence in form of PERSON, GPE, & ORGANIZATION
    # Input:
    #       answers(list)       : List of potential sentence containing answer
    # Output:
    #       chunks(list)        : List of tuple with entity and name in ranked 
    #                             order
    def getNamedEntity(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            nc = ne_chunk(pos_tag(answerToken))
            entity = {"label":None,"chunk":[]}
            for c_node in nc:
                if(type(c_node) == Tree):
                    if(entity["label"] == None):
                        entity["label"] = c_node.label()
                    entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])
                else:
                    (token,pos) = c_node
                    if pos == "NNP":
                        entity["chunk"].append(token)
                    else:
                        if not len(entity["chunk"]) == 0:
                            chunks.append((entity["label"]," ".join(entity["chunk"])))
                            entity = {"label":None,"chunk":[]}
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["label"]," ".join(entity["chunk"])))
        return chunks
    
    # To get continuous chunk of similar POS tags.
    # E.g.  If two NN tags are consequetive, this method will merge and return
    #       single NN with combined value.
    #       It is helpful in detecting name of single person like John Cena, 
    #       Steve Jobs
    # Input:
    #       answers(list) : list of potential sentence string
    # Output:
    #       chunks(list)  : list of tuple with entity and name in ranked order
    def getContinuousChunk(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            if(len(answerToken)==0):
                continue
            nc = pos_tag(answerToken)
            
            prevPos = nc[0][1]
            entity = {"pos":prevPos,"chunk":[]}
            for c_node in nc:
                (token,pos) = c_node
                if pos == prevPos:
                    prevPos = pos       
                    entity["chunk"].append(token)
                elif prevPos in ["DT","JJ"]:
                    prevPos = pos
                    entity["pos"] = pos
                    entity["chunk"].append(token)
                else:
                    if not len(entity["chunk"]) == 0:
                        chunks.append((entity["pos"]," ".join(entity["chunk"])))
                        entity = {"pos":pos,"chunk":[token]}
                        prevPos = pos
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getqRev(self, pq):
        if self.vData == None:
            # For testing purpose
            self.vData = json.loads(open("validatedata.py","r").readline())
        revMatrix = []
        for t in self.vData:
            sent = t["q"]
            revMatrix.append((t["a"],self.sim_sentence(pq.qVector,sent)))
        return sorted(revMatrix,key=lambda tup:(tup[1],tup[0]),reverse=True)[0][0]
        
    def __repr__(self):
        msg = "Total Paras " + str(self.totalParas) + "\n"
        msg += "Total Unique Word " + str(len(self.idf)) + "\n"
        msg += str(self.getMostSignificantWords())
        return msg
    
# Predefined strings.
numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september|october|november|december)"
dmy = "(year|day|week|month)"
rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
exp1 = "(before|after|earlier|later|ago)"
exp2 = "(this|next|last)"
iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
year = "((?<=\s)\d{4}|^\d{4})"
regxp1 = "((\d+|(" + numbers + "[-\s]?)+) " + dmy + "s? " + exp1 + ")"
regxp2 = "(" + exp2 + " (" + dmy + "|" + week_day + "|" + month + "))"

date = "([012]?[0-9]|3[01])"
regxp3 = "(" + date + " " + month + " " + year + ")"
regxp4 = "(" + month + " " + date + "[th|st|rd]?[,]? " + year + ")"

reg1 = re.compile(regxp1, re.IGNORECASE)
reg2 = re.compile(regxp2, re.IGNORECASE)
reg3 = re.compile(rel_day, re.IGNORECASE)
reg4 = re.compile(iso)
reg5 = re.compile(year)
reg6 = re.compile(regxp3, re.IGNORECASE)
reg7 = re.compile(regxp4, re.IGNORECASE)

def extractDate(text):

    # Initialization
    timex_found = []

    # re.findall() finds all the substring matches, keep only the full
    # matching string. Captures expressions such as 'number of days' ago, etc.
    found = reg1.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    # Variations of this thursday, next year, etc
    found = reg2.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    # today, tomorrow, etc
    found = reg3.findall(text)
    for timex in found:
        timex_found.append(timex)

    # ISO
    found = reg4.findall(text)
    for timex in found:
        timex_found.append(timex)

    # Dates
    found = reg6.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    found = reg7.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    # Year
    found = reg5.findall(text)
    for timex in found:
        timex_found.append(timex)
    # Tag only temporal expressions which haven't been tagged.
    #for timex in timex_found:
    #    text = re.sub(timex + '(?!</TIMEX2>)', '<TIMEX2>' + timex + '</TIMEX2>', text)

    return timex_found


## main execution

if len(sys.argv) == 1:
	print("HONO Bot> I need some reference to answer your question")
	print("HONO Bot> Please! Rerun me using following syntax")
	print("\t\t$ python3 P2.py <datasetName>")
	print("HONO Bot> You can find dataset name in \"dataset\" folder")
	print("HONO Bot> Thanks! Bye")
	exit()

datasetName = sys.argv[1]
# Loading Dataset
try:
	datasetFile = open(datasetName,"r")
except FileNotFoundError:
	print("HONO Bot> Oops! I am unable to locate \"" + datasetName + "\"")
	exit()

# Retrieving paragraphs : Assumption is that each paragraph in dataset is
# separated by new line character
paragraphs = []
for para in datasetFile.readlines():
	if(len(para.strip()) > 0):
		paragraphs.append(para.strip())

# Processing Paragraphs
DocumentRetrievalModel = DocumentRetrievalModel(paragraphs,True,True)

print("\nHONO Bot> Hey! I am ready. Ask me factoid based questions only :P")
print("HONO Bot> You can exit anytime, just type 'bye'")

# Greet Pattern
greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)

isActive = True
while isActive:
	userQuery = input("You> ")
	if(not len(userQuery)>0):
		response = "You need to ask something"

	elif greetPattern.findall(userQuery):
		response = "Hello!"
	elif userQuery.strip().lower() == "bye":
		response = "Bye Bye!"
		isActive = False
	else:
		# Proocess Question
		pq = ProcessedQuestion(userQuery,True,False,True)

		# Get Response From Bot
		response =DocumentRetrievalModel.query(pq)
	print("HONO Bot> ",response)
