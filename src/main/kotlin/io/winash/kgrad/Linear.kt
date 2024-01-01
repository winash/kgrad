package io.winash.kgrad

class Linear(private val weight: Tensor, private val bias: Tensor? = null) : Module() {

    override fun forward(input: Tensor): Tensor {
        val output = if (weight.data.shape().size == 1) input.mul(weight) else input.dot(weight)
        return if (bias != null) output.add(bias) else output
    }

}


class LazyLinear(private val outFeatures: Int, private val bias: Boolean = true) : Module() {
    private var weight: Tensor? = null
        get() = field
    private var biasTensor: Tensor? = null
    private var inferredInFeatures: Int = 0

    override fun forward(input: Tensor): Tensor {
        if (weight == null) {
            inferredInFeatures = input.data.shape().last().toInt()
            weight = Tensor.randn(inferredInFeatures.toLong(), outFeatures.toLong())
            if (bias) {
                biasTensor = Tensor.randn(outFeatures.toLong())
            }
        }
        val output = input.data.mmul(weight!!.data)
        return if (biasTensor != null) Tensor(output.add(biasTensor!!.data)) else Tensor(output)
    }
}
