import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{PolynomialExpansion, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

class PolyLinearRegression {
  def getDataPolynomial(currentfile:String, sc:SparkSession, sco:SparkContext, degree:Int):DataFrame = {
    val df_rough:DataFrame = sc.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("mode", "DROPMALFORMED")
      .option("inferSchema", true)
      .load(currentfile)
      .toDF("Annee", "VoyTr")
      .orderBy("Annee")


    val df:DataFrame = Tools.featureNormalizationSVDForDataFrame(sco,df_rough,"Annee")
    val df1 = df
      .withColumn("featNormTemp", Tools.toFeaturesPoly(df("featNorm")))
      .withColumn("label", Tools.encodeIntToDouble(df_rough("VoyTr")))

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("featNormTemp")
      .setOutputCol("polyFeatures")
      .setDegree(degree)

    val polyDF:DataFrame = polyExpansion.transform(df1.select("featNormTemp"))

    val datafixedWithFeatures:DataFrame = polyDF
      .withColumn("features", polyDF("polyFeatures"))

    val datafixedWithFeaturesLabel = datafixedWithFeatures
      .join(df1,df1("featNormTemp") === datafixedWithFeatures("fn"))
      .select("label", "polyFeatures")

    datafixedWithFeaturesLabel
  }

  def parseLRForPolynomial(training:DataFrame, test:DataFrame, maxIter:Int, lambda:Double, alpha:Double):Double = {

    /* Assembler for Pipeline */
    val assembler = new VectorAssembler()
      .setInputCols(Array("polyFeatures"))
      .setOutputCol("features2")

    /* Linear Regression object */
    val lr = new LinearRegression()
      .setMaxIter(maxIter)
      .setRegParam(lambda)
      .setElasticNetParam(alpha)
      .setFeaturesCol("features2")
      .setLabelCol("label")

    /* Fit the model : */
    val pipeline:Pipeline = new Pipeline().setStages(Array(assembler,lr))
    val lrModel:PipelineModel = pipeline.fit(training)

    /* Predict on the test data : */
    val result:DataFrame = lrModel.transform(test)

    /* Metrics, mean squared error :  */
    val mse:Double = Tools.leastSquaresError(result)
    mse
  }
}

