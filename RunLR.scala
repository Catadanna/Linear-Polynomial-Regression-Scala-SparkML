import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

object RunLR {
  def main(args:Array[String]):Unit = {
    // Spark  Context instance :
    val conf = new SparkConf().setAppName("test").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Spark Session instance :
    val ss = org.apache.spark.sql
      .SparkSession.builder()
      .master("local")
      .appName("Read CSV")
      .enableHiveSupport()
      .getOrCreate()

    val f_train = "./src/main/inc/tv_train.csv"
    val f_test = "./src/main/inc/tv_test.csv"

    val lr2 = new LR2

    val maxIter=10
    val lambda = 0.0
    val alpha = 0.3
    var s = ""
    val degree:Int = 6

    val train:DataFrame = lr2.getDataPolynomial(f_train,ss,sc,degree)
    val test:DataFrame = lr2.getDataPolynomial(f_test,ss,sc,degree)
    val result = lr2.parseLRForPolynomial(train,test, maxIter, lambda, alpha)
    println("Error on test for LR : " +result)

    }


}
