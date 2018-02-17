
pyspark --packages com.databricks:spark-csv_2.11:1.5.0
hdfs dfs -put titanic_train.csv


from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer, VectorAssembler

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

titanic = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('titanic_train.csv')


#Question 2

titanic.columns
titanic.count()
titanic.show(10)
titanic.describe().show()


# titanic.groupBy('Sex').count().sort('Sex', desc = True).show(10)
from pyspark.sql.functions import desc

sex = titanic.groupBy('Sex').count()
sex = sex.sort(desc('count'))
sex.show(10)

+------+-----+
|   Sex|count|
+------+-----+
|female|  314|
|  male|  577|
+------+-----+

titanic.groupBy('Cabin').count().sort('Cabin', desc = True).show(10)

cabin = titanic.groupBy('Cabin').count()
cabin = cabin.sort(desc('count'))
cabin.show(10)


+-----+-----+
|Cabin|count|
+-----+-----+
|     |  687|
|  A10|    1|
|  A14|    1|
|  A16|    1|
|  A19|    1|
|  A20|    1|
|  A23|    1|
|  A24|    1|
|  A26|    1|
|  A31|    1|
+-----+-----+


titanic.groupBy('Embarked').count().sort('Embarked', desc = True).show(10)


embarked = titanic.groupBy('Embarked').count()
embarked = embarked.sort(desc('count'))
embarked.show(10)



+--------+-----+
|Embarked|count|
+--------+-----+
|        |    2|
|       C|  168|
|       Q|   77|
|       S|  644|
+--------+-----+


# Question 3


titanic2 = titanic

titanic2 = titanic2.drop("PassengerId")
titanic2 = titanic2.drop("Name")
titanic2 = titanic2.drop("Embarked")
titanic2 = titanic2.drop("Cabin")
titanic2 = titanic2.drop("Ticket")
titanic2 = titanic2.drop("Parch")

from pyspark.sql.types import DoubleType

titanic2 = titanic2.withColumn("Age", titanic2["Age"].cast(DoubleType()))
titanic2 = titanic2.withColumn("Fare", titanic2["Fare"].cast(DoubleType()))
titanic2 = titanic2.withColumn("Survived", titanic2["Survived"].cast(DoubleType()))
titanic2 = titanic2.withColumn("SibSp", titanic2["SibSp"].cast(DoubleType()))
titanic2 = titanic2.withColumn("Pclass", titanic2["Pclass"].cast(DoubleType()))


# titanic2 = titanic2.withColumn("Parch", titanic2["Parch"].cast(DoubleType()))


# titanic2 = titanic2.withColumn("Age", titanic2["Age"].cast(DoubleType()))


# titanic = titanic.withColumn("Survived", titanic["Survived"].cast(DoubleType()))
# titanic = titanic.withColumn("SibSp", titanic["SibSp"].cast(DoubleType()))
# titanic = titanic.withColumn("Parch", titanic["Parch"].cast(DoubleType()))
# titanic = titanic.withColumn("Pclass", titanic["Pclass"].cast(DoubleType()))


from pyspark.sql.functions import avg
titanic2 = titanic2.fillna(titanic2.select(avg("Age")).head()[0], subset="Age")

titanic2 = titanic

# titanic2 = titanic2.fillna(titanic2.select(avg("Parch")).head()[0], subset="Parch")




titanic2 = titanic2.withColumn("AgeNA", titanic2.Age)

from pyspark.sql.functions import *
titanic2 = titanic2\
.withColumn('AgeNA',when(titanic2.Age == titanic2.select(avg("Age")).head()[0] ,1).otherwise(0))\
.drop(titanic2.AgeNA)





# Question 4


# indexing and encoding

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# string indexer
si1 = StringIndexer(inputCol="Sex",outputCol="Sex_indexer")


# OneHotEncoder
oh1 = OneHotEncoder(inputCol="Sex_indexer",outputCol="Sex_encoder")
oh4 = OneHotEncoder(inputCol="Pclass",outputCol="Pclass_encoder")
oh5 = OneHotEncoder(inputCol="SibSp",outputCol="SibSp_encoder")





# input columns and VectorAssembler. Sex_indexer is already in binary format
inputcolumns = ['Age', 'AgeNA2' , 'Sex_indexer','Pclass_encoder', 'SibSp_encoder', 'Fare']

va = VectorAssembler(inputCols = inputcolumns, outputCol = 'features')

# LogisticRegression

#LogisticRegression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol="Survived")

#assemble the pipeline

from pyspark.ml import Pipeline
steps = [si1, oh4, oh5, va, lr]
pl = Pipeline(stages=steps)

#traintestsplit
# from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
df = titanic2
# df = df.withColumn("Pclass", df["Pclass"].cast(DoubleType()))

train, test = df.randomSplit([0.7, 0.3], seed=42)



#fitting the pipeline

plmodel = pl.fit(train)

#transforming and predicting on the test data using the pipeline
predictions = plmodel.transform(test)

print("Coefficients: " + str(plmodel.stages[-1].coefficients))
print("Intercept: " + str(plmodel.stages[-1].intercept))

# Step 10

# Evaluate model performance
# print 5 rows
predictions.show(5)


predictions.select("Survived", "prediction").show(5)

#cacluate AUC

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol='Survived')
evaluator.evaluate(predictions)
