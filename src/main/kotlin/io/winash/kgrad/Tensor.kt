package io.winash.kgrad

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.sqrt

/**
 * A tensor is a wrapper around an INDArray
 * Scalars are represented as INDArrays with length 1  - alo called a rank 0 tensor
 * Vectors are represented as INDArrays with length > 1 - also called a rank 1 tensor
 * Matrices are represented as INDArrays with shape [n, m] where n > 1 and m > 1 - also called a rank 2 tensor
 * Learning materials:
 * https://d2l.ai/chapter_preliminaries/linear-algebra.html
 */
class Tensor(var data: INDArray) {
    var grad: INDArray? = null
    var ctx: Function? = null

    fun backward(allowFill: Boolean = true) {
        if (ctx == null) return

        // If grad is null and this is the end of the backward pass (usually the loss tensor), fill with ones
        if (grad == null && allowFill) {
            grad = Nd4j.onesLike(data)
        }

        val gradients = ctx!!.backward(grad!!)
        for ((parentTensor, gradContribution) in ctx!!.parents.zip(gradients)) {
            // Make sure the shapes are compatible for accumulation
            assert(gradContribution.shape().contentEquals(parentTensor.data.shape()))
            // If parent tensor's grad is not initialized, simply assign the new gradContribution
            if (parentTensor.grad == null) {
                parentTensor.grad = gradContribution
            } else {
                // Otherwise, accumulate the gradients
                parentTensor.grad!!.addi(gradContribution)
            }
            // Recursive call on parent tensor
            parentTensor.backward(false)
        }
    }

    fun shape() = data.shape()


    fun resetGrad() {
        grad = null
    }

    companion object {
        @JvmStatic
        fun zeros(vararg shape: Long) = Tensor(Nd4j.zeros(*shape))

        @JvmStatic
        fun ones(vararg shape: Long) = Tensor(Nd4j.ones(*shape))

        @JvmStatic
        fun fillNum(data:Double,vararg shape: Long) = Tensor(Nd4j.ones(*shape).mul(data))

        @JvmStatic
        fun randn(vararg shape: Long) = Tensor(Nd4j.randn(*shape))

        @JvmStatic
        fun rand(vararg shape: Long) = Tensor(Nd4j.rand(*shape))

        @JvmStatic
        fun xavierUniform(vararg shape: Long) = Tensor(Nd4j.rand(*shape).sub(0.5).mul(2.0).mul(sqrt(6.0 / (shape[0] + shape[1]))))
    }
}


// Tensor operations

fun Tensor.mul(other: Tensor) = Function.apply({ Functions.Mul(this, other) }, this, other)
fun Tensor.add(other: Tensor) = Function.apply({ Functions.Add(this, other) }, this, other)
fun Tensor.relu() = Function.apply({ p -> Functions.ReLU(*p) }, this)
fun Tensor.dot(other: Tensor) = Function.apply({ p -> Functions.Dot(*p) }, this, other)
fun Tensor.sum() = Function.apply({ p -> Functions.Sum(*p) }, this)
fun Tensor.logSoftmax() = Function.apply({ p -> Functions.LogSoftmax(*p) }, this)
fun Tensor.mean() = Function.apply({ p -> Functions.Mean(*p) }, this)
fun Tensor.norm() = Function.apply({ p -> Functions.Norm(*p) }, this)
fun Tensor.abs() = Function.apply({ p -> Functions.Abs(*p) }, this)
fun Tensor.div(other: Tensor) = Function.apply({ Functions.Div(this, other) }, this, other)
fun Tensor.mmul(other: Tensor) = Function.apply({ Functions.MMul(this, other) }, this, other)
fun Tensor.neg() = Function.apply({ p -> Functions.Neg(*p) }, this)
fun Tensor.sigmoid() = Function.apply({ p -> Functions.Sigmoid(*p) }, this)

fun Tensor.reshape(vararg shape: Long) = Tensor(data.reshape(*shape))
fun Tensor.transpose() = Tensor(data.transpose())


fun Tensor.print() {
    println(data)
}

fun Tensor.op(op: (INDArray) -> INDArray, back: (INDArray, Array<out Tensor>) -> List<INDArray>) =
    Function.apply({ p -> Functions.Op(op, back, *p) }, this)







