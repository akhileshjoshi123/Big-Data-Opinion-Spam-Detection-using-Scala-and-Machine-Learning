// Databricks notebook source
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer , CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.mllib.linalg.Vector

// COMMAND ----------

val data = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/p43vlj021492792905041/Reviews.csv")
data.createOrReplaceTempView("DATA")
data.show()

// COMMAND ----------

data.printSchema()

// COMMAND ----------

val AFINN = sc.textFile("/FileStore/tables/48d7z1nh1493504053511/AFINN_111-47bc9.txt").map(x=> x.split("\t")).map(x=>(x(0).toString,x(1).toInt))

// COMMAND ----------

val scalaMap = AFINN.collectAsMap.toMap
val b = sc.broadcast(scalaMap)

// COMMAND ----------

data.createOrReplaceTempView("Reviews")

// COMMAND ----------

val extracted_reviews = sql("select * from Reviews")
extracted_reviews.show()

// COMMAND ----------

val reviewsSenti = extracted_reviews.map(reviewText => {
 
val reviewWordsSentiment = reviewText(9).toString.split(" ").map(word => {


val senti : Int = b.value.getOrElse(word.toLowerCase(),0)


 
senti;
 
});
 
val reviewSentiment = reviewWordsSentiment.sum
 

(reviewText(0).toString,reviewText(1).toString,reviewText(2).toString,reviewText(3).toString,reviewText(4).toString,reviewText(5).toString,reviewText(6).toString,reviewText(7).toString,reviewText(8).toString,reviewText(9).toString,reviewSentiment)
 
})

// COMMAND ----------

reviewsSenti.show()

// COMMAND ----------

reviewsSenti.createOrReplaceTempView("ReviewsSentiment")

val something = sql("select CAST(_1 AS INT)AS ID, _2 AS productID ,_3 AS userID ,_4 AS profileName, CAST(_5 AS INT) AS HelpfulnessNum, CAST(_6 AS INT) AS HelpfulnessDen,CAST(_7 AS INT) AS Score,CAST(_8 AS LONG) AS time,_9 AS Summary,_10 AS Text,CAST(_11 AS INT) AS SentiScore from ReviewsSentiment")


// COMMAND ----------

something.groupBy("userID","SentiScore").sum().collect().foreach(println)

// COMMAND ----------

something.createOrReplaceTempView("ReviewsWithSentiment")

// COMMAND ----------

val xyz = sqlContext.sql("select userID, avg(SentiScore) from ReviewsWithSentiment group by userID having avg(SentiScore)>20")

// COMMAND ----------

var avg_senti_score = sqlContext.sql("select avg(SentiScore) as score from ReviewsWithSentiment ")
avg_senti_score.show()

// COMMAND ----------

import math._
import org.apache.spark.sql.functions._
val BinarySummary =something.withColumn("SentiScore", when($"SentiScore">=5.20, 1).otherwise(0))
BinarySummary.show()
BinarySummary.createOrReplaceTempView("HELP")

// COMMAND ----------

print("Distinct Product : ")
sqlContext.sql("SELECT count(distinct ProductId) as distinct_products FROM DATA ").take(10).foreach(println)


print ("Distinct Users : ")
sqlContext.sql("SELECT count(distinct UserId) as distinct_users FROM DATA ").take(10).foreach(println)

println("Users per Product : ")
sqlContext.sql("SELECT ProductId , count(UserId) as user_count FROM DATA group by ProductId  order by user_count desc").take(10).foreach(println)

val abc = sqlContext.sql("SELECT ProductId , count(UserId) as user_count FROM DATA group by ProductId  order by user_count desc")

println("\nUsers Score Count : ")
sqlContext.sql("SELECT UserId , count(Score) as Score_Count FROM DATA group by UserId order by  Score_Count desc ").take(10).foreach(println)


// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier,LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, IDF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector

// COMMAND ----------

BinarySummary.printSchema()

// COMMAND ----------

val mldatabase = sqlContext.sql("SELECT Id,Text,SentiScore as label FROM HELP")
mldatabase.printSchema()

// COMMAND ----------

//test - train split
val Array(training, test) = mldatabase.randomSplit(Array(0.8, 0.2), seed = 12345)

// COMMAND ----------

println("Original Dataset Records = " + mldatabase.count())
println("Training Recods = " + training.count() + ", " + training.count*100/(mldatabase.count()).toDouble + "%")
println("Test Records = " + test.count() + ", " + test.count*100/(mldatabase.count().toDouble) + "%")

// COMMAND ----------

val tokenizer = new Tokenizer().setInputCol("Text").setOutputCol("words")
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lr))

// COMMAND ----------

println("Logistic Regression Features = " + lr.getFeaturesCol)
println("Logistic Regression Label = " + lr.getLabelCol)
println("Threshold = " + lr.getThreshold)

// COMMAND ----------

val model = pipeline.fit(training)

// COMMAND ----------

val predictions = model.transform(test)

// COMMAND ----------

predictions.select("Id", "probability", "prediction", "label").sample(false,0.01,10L).show(5)

// COMMAND ----------

predictions.sample(false,0.001,10L).show(5)

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
println("Area under the ROC curve = " + evaluator.evaluate(predictions))

// COMMAND ----------

val rf  = new RandomForestClassifier()
    .setNumTrees(100)
    .setFeatureSubsetStrategy("auto")
val pipeline_rf = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, rf))

// COMMAND ----------

val model_rf = pipeline_rf.fit(training)

// COMMAND ----------

val predictions_rf = model_rf.transform(test)
predictions.select("Id", "probability", "prediction", "label").sample(false,0.01,10L).show(5)

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
println("Area under the ROC curve = " + evaluator.evaluate(predictions_rf))

// COMMAND ----------

val paramGrid = new ParamGridBuilder().
  addGrid(lr.regParam, Array(0.01, 0.1, 0.2)).
  addGrid(lr.threshold, Array(0.5, 0.6, 0.7)).
  build()

// COMMAND ----------

//creating cross validator
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)

// COMMAND ----------

val cvModel = cv.fit(training)
println("Area under the ROC curve for best fitted model = " + evaluator.evaluate(cvModel.transform(test)))

// COMMAND ----------

println("Area under the ROC curve for non-tuned model = " + evaluator.evaluate(predictions))
println("Area under the ROC curve for fitted model = " + evaluator.evaluate(cvModel.transform(test)))
println("Improvement = " + "%.2f".format((evaluator.evaluate(cvModel.transform(test)) - evaluator.evaluate(predictions)) *100 / evaluator.evaluate(predictions)) + "%")
