package io.winash.kgrad

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex


class ConvTest {

    /*
     * https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#the-cross-correlation-operation
     */
    @Test
    fun testCorr2d() {
        val x =
            Tensor(Nd4j.create(floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f), intArrayOf(3, 3)))
        val k = Tensor(Nd4j.create(floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f), intArrayOf(2, 2)))
        corr2d(x, k).print()

    }


    /**
     * https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#convolutional-layers
     */
    @Test
    fun testConv2D() {
        val x = Tensor.ones(6, 8)
        x.data.put(arrayOf<INDArrayIndex>(NDArrayIndex.all(), NDArrayIndex.interval(2, 6)), Nd4j.zeros(6, 4));
        x.print()
        val k = Tensor(Nd4j.create(doubleArrayOf(1.0, -1.0))).reshape(1, 2)
        k.print()
        val y = corr2d(x, k)
        y.print()
        corr2d(x.transpose(), k).print()

    }


}
