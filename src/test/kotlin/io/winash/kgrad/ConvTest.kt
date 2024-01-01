package io.winash.kgrad

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j


class ConvTest {
    @Test
    fun testCorr2d() {
        val x = Tensor(Nd4j.create(floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f), intArrayOf(3, 3)))
        val k = Tensor(Nd4j.create(floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f), intArrayOf(2, 2)))
        corr2d(x, k).print()

    }

}
