// Databricks notebook source
//Loading of Data as CSV File
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/idbsfrvl1492359409323/Reviews.csv")
//df.createOrReplaceTempView("DATA")

// COMMAND ----------

df.show()

// COMMAND ----------

//AFINN directory File is used for generated sentiment Scores . SRC : 
val AFINN = sc.textFile("/FileStore/tables/klj5l9c81492839291650/AFINN_111-47bc9.txt").map(x=> x.split("\t")).map(x=>(x(0).toString,x(1).toInt))

// COMMAND ----------

//Converting AFINN RDD to brodcast variable as we need to access it within another RDDs transformation. 

val scalaMap = AFINN.collectAsMap.toMap
val b = sc.broadcast(scalaMap)

// COMMAND ----------

//creating views from dataframe.
df.createOrReplaceTempView("Reviews")

// COMMAND ----------

val extracted_reviews = sql("select * from Reviews")
extracted_reviews.show()

// COMMAND ----------

//Sentiment Analysis, Assigning Sentiment Score to each reviews.
val reviewsSenti = extracted_reviews.map(reviewText => {
 
val reviewWordsSentiment = reviewText(9).toString.split(" ").map(word => {


val senti : Int = b.value.getOrElse(word.toLowerCase(),0)


 
senti;
 
});
 
val reviewSentiment = reviewWordsSentiment.sum
 

(reviewText(0).toString,reviewText(1).toString,reviewText(2).toString,reviewText(3).toString,reviewText(4).toString,reviewText(5).toString,reviewText(6).toString,reviewText(7).toString,reviewText(8).toString,reviewText(9).toString,reviewSentiment)
 
})

// COMMAND ----------

reviewsSenti.createOrReplaceTempView("ReviewsSentiment")


// COMMAND ----------

//rename column names
val reviewSenti = sql("select CAST(_1 AS INT)AS ID, _2 AS productID ,_3 AS userID ,_4 AS profileName, CAST(_5 AS INT) AS HelpfulnessNum, CAST(_6 AS INT) AS HelpfulnessDen,CAST(_7 AS INT) AS Score,CAST(_8 AS LONG) AS time,_9 AS Summary,_10 AS Text,CAST(_11 AS INT) AS SentiScore from ReviewsSentiment")

// COMMAND ----------

reviewSenti.show()
reviewSenti.createOrReplaceTempView("ReviewsWithSentiment")

// COMMAND ----------

//checking Average sentiment score.
val avg_senti_score = sqlContext.sql("select avg(SentiScore) from ReviewsWithSentiment ")

avg_senti_score.show()

// COMMAND ----------

//detecting positive spammers with different ranges.
val posSpammer = sqlContext.sql("select userID, avg(SentiScore) as AvgSentiScore from ReviewsWithSentiment group by userID having avg(SentiScore)>5")
posSpammer.count()

// COMMAND ----------

//detecting negative spammers with different ranges.
val negSpammer = sqlContext.sql("select userID, avg(SentiScore) as AvgSentiScore from ReviewsWithSentiment group by userID having avg(SentiScore)<-5")
negSpammer.count()

// COMMAND ----------

display(negSpammer)

// COMMAND ----------

//saving as a CSV file for future references.
posSpammer.select("userID", "AvgSentiScore").write
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("codec", "org.apache.hadoop.io.compress.GzipCodec")
    .save("posSpammer.csv");

// COMMAND ----------

//saving as a CSV file for future references.

negSpammer.select("userID", "AvgSentiScore").write
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("codec", "org.apache.hadoop.io.compress.GzipCodec")
    .save("negSpammer.csv");
