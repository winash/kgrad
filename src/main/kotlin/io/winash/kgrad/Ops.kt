package io.winash.kgrad

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object Functions {

    class Mul(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val (x, y) = parents.map { it.data }
            saveForBackward(x, y)
            return x.mul(y)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val (x, y) = savedTensors
            return listOf(y.mul(gradOutput), x.mul(gradOutput))
        }
    }

    class MMul(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val (x, y) = parents.map { it.data }
            saveForBackward(x, y)
            return x.mmul(y)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val (x, y) = savedTensors
            return listOf(gradOutput.mmul(y.transpose()), x.transpose().mmul(gradOutput))
        }
    }


    class Add(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val (x, y) = parents.map { it.data }
            return x.add(y)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            // Addition, the gradient is distributed equally to both operands.
            return listOf(gradOutput, gradOutput)
        }
    }

    class ReLU(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Transforms.relu(input, false)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            val gradInput = gradOutput.mul(input.gt(0).castTo(input.dataType()))
            return listOf(gradInput)
        }
    }

    class Sigmoid(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Transforms.sigmoid(input)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            val gradInput = gradOutput.mul(Transforms.sigmoid(input).mul(Transforms.sigmoid(input).rsub(1)))
            return listOf(gradInput)
        }
    }

    class Tanh(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Transforms.tan(input)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            val gradInput = gradOutput.mul(Transforms.tanh(input).mul(Transforms.tanh(input).rsub(1)))
            return listOf(gradInput)
        }
    }


    class Dot(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val (x, y) = parents.map { it.data }
            saveForBackward(x, y)
            val mmul = x.mmul(y)
            if (x.rank() == 1 && y.rank() == 1) {
                // If x and y are vectors, mmul is a scalar
                // Reshape to a vector
                return mmul.reshape(1)
            }
            return mmul
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val (x, y) = savedTensors
            // Check if both inputs are vectors for a vector dot product
            if (x.rank() == 1 && y.rank() == 1) {
                if (x == y) {
                    // If x and y are the same vector, the dot product is the square of the norm
                    // The gradient of the square of the norm is 2x
                    return listOf(x.mul(2), x.mul(2))
                }
                // Make sure gradOutput is a scalar or has the same shape as x and y
                // If it's not, it suggests a problem with how gradOutput is being computed
                // In a vector dot product, gradOutput should be a scalar or have the same length as x and y
                val gradOutputVector = if (gradOutput.length() == 1L) {
                    gradOutput.broadcast(*x.shape())
                } else {
                    gradOutput
                }

                // Gradients for vector dot product
                return listOf(gradOutputVector.mul(y), gradOutputVector.mul(x))
            } else {
                // Handle matrix multiplication
                val gradInput = gradOutput.mmul(y.transpose())
                val gradWeight = x.transpose().mmul(gradOutput)

                return listOf(gradInput, gradWeight)
            }
        }
    }

    class Sum(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Nd4j.scalar(input.sumNumber())
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            return listOf(Nd4j.onesLike(input).mul(gradOutput))
        }
    }


    class LogSoftmax(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            val maxInput = input.max(1).reshape(-1, 1)
            val logsumexp = if (input.isMatrix) {
                Transforms.log(Transforms.exp(input.sub(maxInput)).sum(1)).reshape(-1, 1)
            } else {
                Transforms.log(Transforms.exp(input.sub(maxInput)).sum(0))
            }
            val output = input.sub(logsumexp)
            saveForBackward(output)
            return output
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val output = savedTensors[0]
            val gradInput = gradOutput.sub(Transforms.exp(output).mul(gradOutput.sum(1).reshape(-1, 1)))
            return listOf(gradInput)
        }
    }


    class Mean(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Nd4j.scalar(input.meanNumber())
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            return listOf(Nd4j.onesLike(input).mul(gradOutput).div(input.length()))
        }
    }


    class Norm(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Nd4j.scalar(input.norm2Number())
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            return listOf(input.div(input.norm2Number()).mul(gradOutput))
        }
    }

    class Op(
        val op: (INDArray) -> INDArray,
        val back: (INDArray, Array<out Tensor>) -> List<INDArray>,
        vararg parents: Tensor
    ) :
        Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return op(input)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            return back(gradOutput, parents)
        }
    }

    class Abs(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            saveForBackward(input)
            return Transforms.abs(input)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val input = savedTensors[0]
            val gradInput = gradOutput.mul(Transforms.abs(input).div(input))
            return listOf(gradInput)
        }
    }

    class Div(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val (x, y) = parents.map { it.data }
            saveForBackward(x, y)
            return x.div(y)
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            val (x, y) = savedTensors
            return listOf(gradOutput.div(y), gradOutput.div(x))
        }
    }

    class Neg(vararg parents: Tensor) : Function(*parents) {
        override fun forward(): INDArray {
            val input = parents[0].data
            return input.neg()
        }

        override fun backward(gradOutput: INDArray): List<INDArray> {
            return listOf(gradOutput.neg())
        }
    }


    fun sparseCategoricalCrossEntropy(logits: Tensor, target: Tensor, ignoreIndex: Int = -1): Tensor {
        val lossMask = target.data.neq(ignoreIndex).castTo(Nd4j.dataType())
        val yFlat = target.data.ravel().reshape(-1, 1)

        val yOneHot = yFlat.eq(Nd4j.arange(logits.data.size(logits.data.rank() - 1).toDouble())
            .reshape(1, logits.data.size(logits.data.rank() - 1)))
            .castTo(Nd4j.dataType()).mul(lossMask.reshape(-1, 1))
            .reshape(*target.shape(), logits.data.size(logits.data.rank() - 1))


        val losses = logits.neg().mul(Tensor(yOneHot))
        val sum = losses.data.sum(-1)
        losses.data = sum
        return losses.sum().div(Tensor(lossMask.sum()))
    }


}



