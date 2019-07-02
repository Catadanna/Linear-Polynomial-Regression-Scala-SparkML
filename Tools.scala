/*
 Tools
 @author: Catalina CHIRCU
 */
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
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

  /**
    * Transforms a DataFrame into a DenseMatrix
    * @param featuresDF
    * @return
    */
  def getDenseMatrixFromDF(featuresDF:DataFrame):DenseMatrix = {
    val featuresTrain = featuresDF.columns
    val rows = featuresDF.count().toInt

    val newFeatureArray:Array[Double] =
      featuresTrain
        .indices
        .flatMap(i => featuresDF
          .select(featuresTrain(i))
          .collect())
        .map(r => r.toSeq.toArray).toArray.flatten.flatMap(_.asInstanceOf[org.apache.spark.ml.linalg.DenseVector].values)

    val flatArray = newFeatureArray
    val newCols = flatArray.length / rows
    val denseMat:DenseMatrix = new DenseMatrix(rows, newCols, flatArray, isTransposed=false)
    denseMat
  }

  /**
    * Transforms a DataFrame into a DenseVector
    * @param featuresDF
    * @return
    */

  def getDenseVectorFromDF(featuresDF:DataFrame):DenseVector = {
    val featuresTrain = featuresDF.columns
    val cols = featuresDF.columns.length

    cols match {
      case i if i>1 => throw new IllegalArgumentException
      case _ => {
        def addArray(acc:Array[Array[Double]], cur:Array[Double]):Array[Array[Double]] = {
          acc :+ cur
        }

        val newFeatureArray:Array[Double] = featuresTrain
          .indices
          .flatMap(i => featuresDF
            .select(featuresTrain(i))
            .collect())
          .map(r => r.toSeq.toArray.map(e => e.asInstanceOf[Double])).toArray.flatten


        val denseVec:DenseVector = new DenseVector(newFeatureArray)
        denseVec
      }
    }
  }



}
