from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import pipeline

print 'Start data preprocessing...'
df = sqlContext.read.load('./train.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

df = df.cache()
df = df.withColumnRenamed('is_duplicate', 'label')
# indexers = [Tokenizer(inputCol=column, outputCol="words").transform(df) for column in ['question1', 'question2']]
tokenizer = Tokenizer(inputCol='question1', outputCol='words')
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
# df = pipeline.fit(df)
df = tokenizer.transform(df)
df = remover.transform(df)
df = hashingTF.transform(df)
df = idf.fit(df).transform(df)
df.select("features").show()

lr = LogisticRegression(maxIter=10, regParam=0.001)
