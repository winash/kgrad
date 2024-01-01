package io.winash.kgrad

import org.nd4j.linalg.api.ndarray.INDArray


typealias Callable = (Tensor) -> Tensor

abstract class Function(vararg val parents: Tensor) {
    protected val savedTensors = mutableListOf<INDArray>()

    protected fun saveForBackward(vararg tensors: INDArray) {
        savedTensors.addAll(tensors)
    }

    abstract fun forward(): INDArray
    abstract fun backward(gradOutput: INDArray): List<INDArray>

    companion object {
        fun <T : Function> apply(factory: (Array<out Tensor>) -> T, vararg parents: Tensor): Tensor {
            val ctx = factory(parents)
            val outData = ctx.forward()
            val output = Tensor(outData)
            output.ctx = ctx
            return output
        }
    }
}

