package io.winash.kgrad

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.Ignore

class TensorTest {

    @Test
    fun testAdd() {
        val x = Tensor(Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = Tensor(Nd4j.create(doubleArrayOf(4.0, 5.0, 6.0)))
        val z = x.add(y)
        val expected = Nd4j.create(doubleArrayOf(5.0, 7.0, 9.0))
        assertEquals(expected, z.data)

    }

    @Test
    fun testMul() {
        val x = Tensor(Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = Tensor(Nd4j.create(doubleArrayOf(4.0, 5.0, 6.0)))
        val z = x.mul(y)
        val expected = Nd4j.create(doubleArrayOf(4.0, 10.0, 18.0))
        assertEquals(expected, z.data)
    }


    @Test
    fun testSum() {
        val x = Tensor(Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = x.sum()
        val expected = Nd4j.create(doubleArrayOf(6.0))
        assertEquals(expected, y.data.reshape(1))
    }

    @Test
    fun testDot() {
        val x = Tensor(Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = Tensor(Nd4j.create(doubleArrayOf(4.0, 5.0, 6.0)))
        val z = x.dot(y)
        val expected = Nd4j.create(doubleArrayOf(32.0)).reshape(1,1)
        assertEquals(expected, z.data)
    }

    @Test
    fun testLogSoftmax() {
        val x = Tensor(Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = x.logSoftmax()
        val expected = Nd4j.create(doubleArrayOf(-2.4076061, -1.4076061, -0.40760604))
        assertEquals(expected, y.data)
    }

    @Test
    fun testReLU() {
        val x = Tensor(Nd4j.create(doubleArrayOf(-1.0, 2.0, -3.0)))
        val y = x.relu()
        val expected = Nd4j.create(doubleArrayOf(0.0, 2.0, 0.0))
        assertEquals(expected, y.data)
    }


    // Test exact values with Pytorch to make sure it works well
    @Test
    @Ignore
    fun testGrad() {
        val xInit = Nd4j.create(arrayOf(floatArrayOf(0.49671415f, -0.1382643f, 0.64768857f)))
        val wInit = Nd4j.create(arrayOf(
            floatArrayOf(1.5230298f, -0.23415338f, -0.23413695f),
            floatArrayOf(1.5792128f, 0.7674347f, -0.46947438f),
            floatArrayOf(0.54256004f, -0.46341768f, -0.46572974f)
        ))
        val mInit = Nd4j.create(arrayOf(floatArrayOf(0.24196227f, -1.9132802f, -1.7249179f)))


        val x = Tensor(xInit)
        val W = Tensor(wInit)
        val m = Tensor(mInit)

        val out = x.dot(W)
        val outr = out.relu()
        val outl = outr.logSoftmax()
        val outm = outl.mul(m)
        val outa = outm.add(m)
        val outx = outa.sum()

        outx.backward()

        println(Triple(outx.data, x.grad, W.grad))
    }

}
