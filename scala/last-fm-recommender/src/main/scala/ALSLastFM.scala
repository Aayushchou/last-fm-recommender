import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
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
    input: String = null,
    alpha: Double = 0.01,
    numIterations: Int = 10,
    dataLimit: Int = 5000,
    rank: Int = 10,
    implicitPrefs: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
      val defaultParams = Params()
      val parser = new OptionParser[Params]("MovieLensALS") {
      head("ALSLastFM: Basic usage of ALS to create a song recommender with the LASTFM dataset")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("dataLimit")
        .text(s"Reduce dataset size to this number, default: ${defaultParams.dataLimit} (auto)")
        .action((x, c) => c.copy(dataLimit = x))
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

      val spark = SparkSession.builder() 
                          .master("local") 
                          .appName("ALSLastFM") 
                          .getOrCreate()
                          
      // "../../resources/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
      var data_path:String = params.input
      val schema = new StructType()
              .add("user_id", StringType, true)
              .add("timestamp", StringType, true)
              .add("artist_id", StringType, true)
              .add("artist_name", StringType, true)
              .add("track_id", StringType, true)
              .add("track_name", StringType, true)
      val listener_data = spark.read.option("header", false)
              .schema(schema)
              .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
              .option("sep", "\t")
              .load(data_path)
      val df_filtered = listener_data.drop("timestamp").na.drop()
      val df_agg = df_filtered.select("user_id", "track_id")
              .groupBy("user_id", "track_id")
              .agg(count("*").alias("count")).orderBy("user_id")
      val df_agg_filtered = df_agg.limit(params.dataLimit)


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

      val tr_full = tr_s.withColumn("rating_as_array", vector_to_array(tr_s("rating")).getItem(0))
      val ts_full = ts_s.withColumn("rating_as_array", vector_to_array(ts_s("rating")).getItem(0))

      val tr_final = tr_full.select("user_index", "track_index", "count","rating_as_array").orderBy("user_index")
      val ts_final = ts_full.select("user_index", "track_index", "count", "rating_as_array").orderBy("user_index")

      val als = new ALS()
          .setRank(params.rank)
          .setAlpha(params.alpha)
          .setMaxIter(params.numIterations)
          .setUserCol("user_index")
          .setItemCol("track_index")
          .setRatingCol("rating_as_array")

      val model = als.fit(tr_final)
      model.setColdStartStrategy("drop")

      val predictions = model.transform(ts_final)

      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating_as_array")
        .setPredictionCol("prediction")

      val rmse = evaluator.evaluate(predictions)
      println(s"Root-mean-square error = $rmse") 
      }
    }
    