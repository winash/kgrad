package io.winash.kgrad

import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D
import org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig
import org.nd4j.linalg.factory.Nd4j

class Pool2D(
    private val kernelSize: Pair<Int, Int>,
    private val mode: PoolMode = PoolMode.MAX,
    private val stride: Pair<Int, Int> = Pair(1, 1),
    private val padding: Pair<Int, Int> = Pair(0, 0)
) : Module() {
    override fun forward(input: Tensor): Tensor {
        // reshape input to rank 4
        val kH = kernelSize.first
        val kW = kernelSize.second
        val sH = stride.first
        val sW = stride.second
        val pH = padding.first
        val pW = padding.second
        val pooling2DConfig = Pooling2DConfig.builder()
            .kH(kH.toLong())
            .kW(kW.toLong())
            .sH(sH.toLong())
            .sW(sW.toLong())
            .pH(pH.toLong())
            .pW(pW.toLong())
            .build()

        // Just use the ND4J implementation for now
        // TODO: Implement this ourselves
        return if (mode == PoolMode.MAX) {
            Tensor(Nd4j.exec(MaxPooling2D(input.data, pooling2DConfig))[0])
        } else {
            Tensor(Nd4j.exec(AvgPooling2D(input.data, pooling2DConfig))[0])
        }

    }

    override fun name(): String {
        return "Pool2D-$mode"
    }
}

enum class PoolMode {
    MAX, AVG
}
