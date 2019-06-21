/*
 Tools
 @author: Catalina CHIRCU
 */
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD

object Tools {
  val encodeIntToDouble    = udf[Double, Int](_.toDouble)

  val toFeaturesPoly = udf[Vector, Double] { (b) =>
    Vectors.dense(b)
  }

  def featureNormalizationSVDForDataFrame(sco:SparkContext, X:DataFrame,featureName:String):DataFrame={
    val rddCollection:RDD[Double] = convertDataFrameToRDD(X,featureName)
    val mean:Double = rddCollection.mean()
    val stddev:Double = rddCollection.stdev()
    val tranformToNormalized = udf[Double, Double] {
      e => (e - mean) / stddev
    }

    val resultingMatrix:DataFrame = X
      .withColumn("featNorm", tranformToNormalized(encodeIntToDouble(X(featureName))))
    resultingMatrix
  }

  def convertDataFrameToRDD(df:DataFrame, featureName:String):RDD[Double] = {
    val r = df.withColumn(featureName, encodeIntToDouble(df(featureName)))
      .select(featureName).rdd
      .map(e => e(0).asInstanceOf[Double])
    r
  }

  def leastSquaresError(result:DataFrame):Double = {
    val rm:RegressionMetrics = new RegressionMetrics(
      result
        .select("label","prediction")
        .rdd.
        map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    Math.sqrt(rm.meanSquaredError)
  }

}
