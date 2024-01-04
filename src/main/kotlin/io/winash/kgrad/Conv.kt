package io.winash.kgrad

import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex


fun corr2d(input: Tensor, kernel: Tensor): Tensor {
    val kernelShape = kernel.shape()
    val output = Tensor.zeros(input.shape()[0] - kernelShape[0] + 1, input.shape()[1] - kernelShape[1] + 1)
    for (i in 0..<output.shape()[0]) {
        for (j in 0..<output.shape()[1]) {
            output.data.putScalar(
                i,
                j,
                input.data.get(
                    NDArrayIndex.interval(i, i + kernelShape[0]),
                    NDArrayIndex.interval(j, j + kernelShape[1])
                ).mul(kernel.data).sumNumber().toDouble()
            )
        }
    }
    return output
}

/**
 * A simple 2D convolution layer for learning
 * https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#convolutional-layers
 */
class SimpleConv2D(private val weight: Tensor, private val bias: Tensor? = null) : Module() {

    constructor(kernelSize: Int) : this(
        Tensor.rand(kernelSize.toLong()),
        Tensor.zeros(1)
    )


    override fun forward(input: Tensor): Tensor {
        val output = corr2d(input, weight)
        return if (bias != null) output.add(bias) else output
    }

    override fun name(): String {
        return "SimpleConv2D"
    }

}


/**
 * 2D convolution layer
 */
class Conv2D(
    private val outChannels: Int,
    private val kernelSize: Pair<Int, Int> = Pair(1, 1),
    private val padding: Pair<Int, Int> = Pair(0, 0),
    private val strides: Pair<Int, Int> = Pair(1, 1),
    private val dilation: Pair<Int, Int> = Pair(1, 1),
    private val paddingMode: PaddingMode = PaddingMode.SAME,
    private val groups: Int = 1,
    private val device: Device = Device.CPU,
    private val bias: Boolean = false
) : Module() {
    var weight: Tensor? = null
    private var biasTensor: Tensor? = null

    override fun forward(input: Tensor): Tensor {
        if (weight == null) {
            val inChannels = input.shape()[1]
            val kernelHeight = kernelSize.first.toLong()
            val kernelWidth = kernelSize.second.toLong()
            weight = Tensor.xavierUniform(kernelHeight, kernelWidth, inChannels, outChannels.toLong())
            if (bias) {
                biasTensor = Tensor.randn(outChannels.toLong())
            }
        }

        // Just use the ND4J implementation for now
        // TODO: Implement this ourselves
        val conv2DConfig = Conv2DConfig.builder()
            .kH(kernelSize.first.toLong())
            .kW(kernelSize.second.toLong())
            .sH(strides.first.toLong())
            .sW(strides.second.toLong())
            .pH(padding.first.toLong())
            .pW(padding.second.toLong())
            .dH(dilation.first.toLong())
            .dW(dilation.second.toLong())
            .paddingMode(paddingMode)
            .build()

        return Tensor(Nd4j.exec(Conv2D(input.data, weight!!.data, biasTensor?.data, conv2DConfig))[0])

        // Perform the convolution operation

    }

    override fun name(): String {
        return "Conv2D"
    }

    override fun getWeights(): List<Tensor?> {
        return listOf(weight)
    }



}



