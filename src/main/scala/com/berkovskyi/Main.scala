package com.berkovskyi

import org.apache.spark.sql.expressions.Window

import org.apache.spark.sql.functions._

object Main extends App {
  import org.apache.log4j.Logger
  import org.apache.log4j.Level

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val spark = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Spark CSV Reader")
    .getOrCreate

  val data = getClass.getResource("/data.csv")

  import spark.sqlContext.implicits._

  val df = spark.read
    .option("header", "true")
    .csv(data.toString)

  val w = Window.partitionBy("category", "userId").orderBy("eventTime")
  val ts = unix_timestamp($"eventTime")
  val diff =
    coalesce(ts - unix_timestamp(lag("eventTime", 1, 0).over(w)), lit(0))
  val indicator = (diff > 300).cast("integer")
  val subgroup = sum(indicator).over(w).alias("subgroup")
  val windowDuration = unix_timestamp($"max") - unix_timestamp($"min")
  val grouped = df
    .withColumn("subgroup", subgroup)
    .groupBy($"category", $"userId", $"subgroup")
    .agg(max($"eventTime"), min($"eventTime"))
    .withColumnRenamed("max(eventTime)", "max")
    .withColumnRenamed("min(eventTime)", "min")
    .withColumn("sessionId",
                sha1(concat_ws("", $"subgroup", $"category", $"userId")))
    .withColumn("windowDuration", windowDuration)
    .orderBy($"category")
    .cache()
  grouped.createOrReplaceTempView("grouped")

  println("Enriched data")

  val result = df
    .as("df")
    .join(grouped.as("gr"),
          (df("category") === grouped("category"))
            && (df("eventTime") between (grouped("min"), grouped("max"))),
          "left_outer")
    .select($"df.eventTime",
            $"df.eventType",
            $"df.category",
            $"df.userId",
            $"gr.sessionId",
            $"gr.min".as("start time"),
            $"gr.max".as("end time"))
    .show

  println("mediana")

  spark.sqlContext
    .sql(
      "select category, percentile(windowDuration, 0.5) as median from grouped group by category")
    .show

  println("users spend les than 1 minutes")
  grouped
    .groupBy($"category", $"userId")
    .agg(min("windowDuration"))
    .filter($"min(windowDuration)" < 60)
    .withColumnRenamed("min(windowDuration)", "less than 1 min (sec)")
    .show

  println("users spend more than 5 minutes")

  grouped
    .groupBy($"category", $"userId")
    .agg(max("windowDuration"))
    .filter($"max(windowDuration)" > 300)
    .withColumnRenamed("max(windowDuration)", "more than 1 min (sec)")
    .show

  println("users spend 1 to 5 minutes")

  grouped
    .filter($"windowDuration" > 60)
    .filter($"windowDuration" < 300)
    .groupBy($"category", $"userId")
    .agg(max("windowDuration"))
    .withColumnRenamed("max(windowDuration)", "1 to 5 minutes (sec)")
    .show

  val runkW = Window.partitionBy("userId").orderBy("eventTime")

  val isAnotherProduct =
    coalesce($"product" === lag("product", 1).over(runkW), lit(true))
  val userIndicator = (!isAnotherProduct).cast("integer")
  val usersSubGroup = sum(userIndicator).over(runkW).alias("prodSubgroup")

  println("top 10 ranked by time spent by users on product p")

  df.withColumn("prodSubgroup", usersSubGroup)
    .groupBy($"userId", $"product", $"prodSubgroup")
    .agg(max($"eventTime"), min($"eventTime"))
    .withColumnRenamed("max(eventTime)", "max")
    .withColumnRenamed("min(eventTime)", "min")
    .withColumn("windowDuration", windowDuration)
    .groupBy($"product")
    .sum("windowDuration")
    .withColumnRenamed("sum(windowDuration)", "totalDuration")
    .orderBy(desc("totalDuration"))
    .show(10)

}
