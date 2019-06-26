import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix


/**
  *
  * @param sc
  */
class NormalEquation(sc:SparkContext) {
  /**
    * Computes the inverse of the input matrix X
    *
    * @param X
    * @return : the inverse of X, or an error if X is not invertible
    */
    def computeInverse(X: RowMatrix): DenseMatrix = {
      val nCoef = X.numCols.toInt

      val svd = X.computeSVD(nCoef, computeU = true, rCond = 1e-200)

      if (svd.s.size < nCoef) {
        sys.error(s"RowMatrix.computeInverse called on singular matrix.")
      }

      // Create the inv diagonal matrix from S
      val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x,-1))))

      // U cannot be a RowMatrix
      val U = new DenseMatrix(svd.U.numRows().toInt,svd.U.numCols().toInt,svd.U.rows.collect.flatMap(x => x.toArray))

      // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
      val V:Matrix = svd.V
      // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
      (V.multiply(invS)).multiply(U)
    }

  /**
    * Computes normal equation
    * @param X
    * @param Y
    * @param lambda
    * @return
    */

    def normalEquation(X:DenseMatrix, Y:Vector, lambda:Double):DenseVector = {
      val xT:DenseMatrix = X.transpose
      val xTx:DenseMatrix = xT.multiply(X)

      val lengthxTx:Int = Math.sqrt(xTx.values.length).intValue()

      // Mulyiply with lambda :
      val allLambda = DenseMatrix.eye(lengthxTx).values.map(_*lambda)

      // Set first element to zero :
      val xTxLambdaIndices:Array[Double] = xTx.values.indices.map(i => xTx.values(i)+allLambda(i)).toArray
      val xTxLambdaLestFirst:Array[Double] = xTxLambdaIndices.indices.map(i => {if (i==0) 0.0 else xTxLambdaIndices(i)}).toArray

      // New Matrix with Lambda
      val xTxL:DenseMatrix = new DenseMatrix(lengthxTx,lengthxTx,xTxLambdaLestFirst,isTransposed = true)

      val rm = matrixToRowMatrix(xTxL,sc)

      // Compute inverse, multiply with xT and then with Y :
      val theta:DenseVector = computeInverse(rm).multiply(xT).multiply(Y)
      theta
    }

    def predict(X:DenseMatrix,theta:DenseVector):DenseVector = {
      X.multiply(theta)
    }

  /**
    * Transforms a Matrix into a rowMatrix
    * @param m
    * @param sc
    * @return
    */

  def matrixToRowMatrix(m:Matrix, sc:SparkContext):RowMatrix = {
    val s = m.rowIter.toVector
    val rows = sc.parallelize(s)
    val newRowMatrix: RowMatrix = new RowMatrix(rows)
    newRowMatrix
  }
}





