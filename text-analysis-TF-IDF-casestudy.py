#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizerModel, CountVectorizer

from pyspark.ml.linalg import SparseVector, Vector

spark = SparkSession.builder.appName("PySpark Text Analysis Example").getOrCreate()

sc = spark.sparkContext

# read all the files
wikif = sc.wholeTextFiles("file:///home/hduser/Downloads/sharedfolder/mlexamples/TFIDF-CaseStudy-4-PySpark/wikidocs/*.txt")
# wikif = sc.wholeTextFiles("mllib/wikidocs/*.txt")

wikif.first()

# Remove path from the file name
# wikif.first()[0].split('/')[-1]
# (wikif.first()[0].split('/')[-1],wikif.first()[1])

wiki=wikif.map(lambda rec:((rec[0].split('/')[-1]),rec[1]))

wiki.first()

# convert to dataframe
filedata=spark.createDataFrame(wiki).toDF('label','doc')
filedata.printSchema()

filedata.show(2)

# convert document to list of words
tokenizer = RegexTokenizer(inputCol="doc", outputCol="allwords", pattern="\\W")
allWordsData = tokenizer.transform(filedata)
allWordsData.printSchema()

allWordsData.show(2)

# remove the stop words
remover = StopWordsRemover(inputCol="allwords", outputCol="words")
wordsData=remover.transform(allWordsData)
wordsData.printSchema()

wordsData.show(2)

wordsData.select('allwords').show(1,False)

wordsData.select('words').show(1,False)

# Get the required columns 
nwordsData=wordsData.select('label','words')
nwordsData.printSchema()

nwordsData.show(2)

# Build a term frequency matrix. Also check the vocabulary
# val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("rawFeatures").fit(nwordsData)
# cvModel.vocabulary.length
# val cvm = cvModel.transform(nwordsData)
cvModel=CountVectorizer(inputCol='words',outputCol='rawFeatures').fit(nwordsData)
len(cvModel.vocabulary)

type(cvModel.vocabulary)

cvModel.vocabulary

cvm = cvModel.transform(nwordsData)
type(cvm)

cvm.printSchema()

cvm.show(2)

# Build the Inverse document frequency
idf=IDF(inputCol="rawFeatures", outputCol="features").fit(cvm)
tfIDFData = idf.transform(cvm)

# Apply the function to extract the top words.
# Top 10 is specified below but you can change that.
tfIDFData.printSchema()

tfIDFData.show(2)

tfIDFData.count()

tfIDFRDD=tfIDFData.select("label","features").rdd

tfIDFRDD.count()

type(tfIDFRDD.first())

tfIDFRDD.first()

tfIDFRDD.first().features.indices

len(tfIDFRDD.first().features.indices)

tfIDFRDD.first().features.values

len(tfIDFRDD.first().features.values)

type(tfIDFRDD.first().features.values[0])

type(tfIDFRDD.first().features.indices[0])

tfIDFRDD.first().label

def findf2(vocab,words,idfs):
    dict1={}
    for i in range(len(words)):
        if idfs[i] in dict1:
            dict1[idfs[i]].append(words[i])
        else:
            dict1[idfs[i]]=[words[i]]
    list1=list(dict1.keys())
    list1.sort(reverse=True)
    kwords=[]
    for j in list1[:10]:
        for k in range(len(dict1[j])):
            kwords.append(vocab[dict1[j][k]])
    return kwords[:10]
 

for rec in tfIDFRDD.take(tfIDFRDD.count()):
    kwords10=(rec.label, findf2(cvModel.vocabulary,rec.features.indices, rec.features.values))
    print(kwords10)
    

