import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.ml.feature.RobustScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scopt.OptionParser
import scala.reflect.runtime.universe._
import scala.collection.mutable
import org.apache.spark.rdd.RDD

object LastFMRecommender {

  /*

    TODO: 
    1. implement a better evaluation method
    2. show recommended outputs for K users

  */

  abstract class AbstractParams[T: TypeTag] {

  private def tag: TypeTag[T] = typeTag[T]

  /**
   * Finds all case class fields in concrete class instance, and outputs them in JSON-style format:
   * {
   *   [field name]:\t[field value]\n
   *   [field name]:\t[field value]\n
   *   ...
   * }
   */
  override def toString: String = {
    val tpe = tag.tpe
    val allAccessors = tpe.decls.collect {
      case m: MethodSymbol if m.isCaseAccessor => m
    }
    val mirror = runtimeMirror(getClass.getClassLoader)
    val instanceMirror = mirror.reflect(this)
    allAccessors.map { f =>
      val paramName = f.name.toString
      val fieldMirror = instanceMirror.reflectField(f)
      val paramValue = fieldMirror.get
      s"  $paramName:\t$paramValue"
      }.mkString("{\n", ",\n", "\n}")
    }
  }

  case class Params(
    // default parameter definitions for the ALS model. 
    // user can update these parameters when calling the scala job 

    input: String = null,
    item: String = "track",
    alpha: Double = 0.01,
    numIterations: Int = 10,
    numItemsToRecommend: Int = 10,
    dataLimit: Int = 5000,
    rank: Int = 10,
    implicitPrefs: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
      /*
        parses user parameters and runs the als model with these parameters 
        required: input parameter to indicate where the ALS dataset is located. 
      */
      val defaultParams = Params()
      val parser = new OptionParser[Params]("MovieLensALS") {
      head("ALSLastFM: Basic usage of ALS to create a song recommender with the LASTFM dataset")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numItemsToRecommend")
        .text(s"number of iterations, default: ${defaultParams.numItemsToRecommend}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("dataLimit")
        .text(s"Reduce dataset size to this number, default: ${defaultParams.dataLimit} (auto)")
        .action((x, c) => c.copy(dataLimit = x))
      opt[String]("item")
        .text(s"Choose weather to recommend artists or tracks, default: ${defaultParams.item} (auto)")
        .action((x, c) => c.copy(item = x))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
        """.stripMargin)
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }

      }

  def run(params : Params): Unit = {
      /*
        runs ALS algorithm on the LASTFM dataset. 
        params: 
          input - path to LASTFM dataset
          rank - the rank of the ALS model (5,10,15 generally)
          numIterations - the number of iterations the ALS model should train 
          dataLimit - if running locally, might want to limit data to minimise memory issues
          implicitPrefs - enable if including ratings that aren't explicit user feedback
          item - whether to recoend artists or tracks, choose between "artist" or "track"
        procedure: 
          filters and aggregates dataset, uses count as a proxy for rating
          StringIndexer to index users and tracks
          RobustScaler to scale ratings while staying robust to outliers 
          runs ALS model using the scaled ratings as labels
          displays rmse error 
      */
      val spark = SparkSession.builder() 
                          .master("local") 
                          .appName("ALSLastFM") 
                          .getOrCreate()
      import spark.implicits._
      
      // "../resources/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
      var data_path:String = params.input
      var agg_col:String = params.item + "_id"
      var agg_index:String = params.item + "_index"

      val schema = new StructType()
              .add("user_id", StringType, true)
              .add("timestamp", StringType, true)
              .add("artist_id", StringType, true)
              .add("artist_name", StringType, true)
              .add("track_id", StringType, true)
              .add("track_name", StringType, true)
      val df_filtered = spark.read.option("header", false)
              .schema(schema)
              .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
              .option("sep", "\t")
              .load(data_path).drop("timestamp").na.drop()
      val df_agg_filtered = df_filtered.select("user_id", agg_col)
              .groupBy("user_id", agg_col)
              .agg(count("*").alias("count")).orderBy("user_id").limit(params.dataLimit)


      val Array(training, test) = df_agg_filtered.randomSplit(Array[Double](0.8, 0.2), 18)

      //revisit to make more efficient

      val feat = df_agg_filtered.columns.filter(_ .contains("id"))
      val inds = feat.map { colName =>
          new StringIndexer()
          .setInputCol(colName)
          .setOutputCol(colName.replace("id", "index"))
          .fit(df_agg_filtered)
          .setHandleInvalid("keep")
      }

      val va = new VectorAssembler()
          .setInputCols(Array("count"))
          .setOutputCol("count_assembled")

      val scaler = new RobustScaler()
        .setInputCol("count_assembled")
        .setOutputCol("rating")

      val pipeline = new Pipeline()
        .setStages(inds.toArray ++ Array(va, scaler))

      val tr_s = pipeline.fit(training).transform(training)
      val ts_s = pipeline.fit(training).transform(test)

      val tr_final = tr_s.withColumn("rating_as_array", vector_to_array(tr_s("rating"))
                      .getItem(0))
                      .select("user_index", agg_index,"rating_as_array")
                      .orderBy("user_index")
      val ts_final = ts_s.withColumn("rating_as_array", vector_to_array(ts_s("rating"))
                      .getItem(0))
                      .select("user_index", agg_index,"rating_as_array")
                      .orderBy("user_index")

      val als = new ALS()
          .setRank(params.rank)
          .setAlpha(params.alpha)
          .setMaxIter(params.numIterations)
          .setUserCol("user_index")
          .setItemCol(agg_index)
          .setRatingCol("rating_as_array")

      val model = als.fit(tr_final)
      model.setColdStartStrategy("drop")

      val predictions = model.transform(ts_final)

      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating_as_array")
        .setPredictionCol("prediction")

      val rmse = evaluator.evaluate(predictions)

      model.save("models/als_lastfm")
      println(s"Root-mean-square error = $rmse") 
      
      // show user recommendations 
      
      val userRecs = model.recommendForAllUsers(params.numItemsToRecommend)
      val firstRec = userRecs
                        .withColumn("columns",expr("struct(recommendations[0] as rec1) as columns"))
                        .select("user_index","columns.*").select("user_index", "rec1.*")

  
      val users = df_agg_filtered.select("user_id").map(_.getString(0)).distinct().collectAsList().toArray().map(_.asInstanceOf[String])
      val usermap = new IndexToString()
          .setInputCol("user_index")
          .setOutputCol("user_id")
          .setLabels(users)

      val userout = usermap.transform(firstRec)

      val tracks = df_agg_filtered.select("track_id").map(_.getString(0)).distinct().collectAsList().toArray().map(_.asInstanceOf[String])
      val trackmap = new IndexToString()
          .setInputCol("track_index")
          .setOutputCol("track_id")
          .setLabels(tracks)

      val trackout = trackmap.transform(userout)

      val final_result = trackout.as("results")
        .join(df_filtered.as("in"), $"results.track_id" === $"in.track_id")
        .select("results.user_id", "in.track_name", "in.artist_name", "results.rating").distinct()

      println(final_result.show())

      }
      
  }
    