package io.winash.kgrad

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j.*
import org.nd4j.linalg.indexing.NDArrayIndex


/**
 * Learn some linear algebra
 * https://d2l.ai/chapter_preliminaries/linear-algebra.html#
 */
class Basics {

    @Test
    fun scalar() {
        val x = Tensor(scalar(2.0))
        val y = Tensor(scalar(3.0))
        val z = x.mul(y)
        val expected = Tensor(scalar(6.0))
        assert(z.data == expected.data)
    }


    @Test
    fun vector() {
        val x = Tensor(arange(3.0))
        // get the second element from x
        val y = x.data[NDArrayIndex.point(1)]
        assert(y.getDouble(0) == 1.0)
        assert(x.data.length() == 3L)
        val z = x.mul(x)

        assert(z.data.getDouble(2) == 4.0)

    }

    /**
     * A matrix is a 2D tensor
     */
    @Test
    fun matrix() {
        val x = Tensor(arange(6.0).reshape(2, 3))
        val y = x.data[NDArrayIndex.point(1), NDArrayIndex.point(1)]
        assert(y.getDouble(0) == 4.0)
        assert(x.data.shape().contentEquals(longArrayOf(2, 3)))
        val z = x.mul(x)
        assert(z.data.getDouble(1) == 9.0)

        // Transpose x
        val xT = Tensor(x.data.transpose())
        assert(xT.data.shape().contentEquals(longArrayOf(3, 2)))

    }

    @Test
    fun basicTensorArithmetic() {
        val x = Tensor(arange(6.0).reshape(2, 3))
        val y = Tensor(arange(6.0).reshape(2, 3))
        val z = x.add(y)
        assert(z.data.getDouble(1, 0) == 6.0)
        val w = x.mul(y)
        assert(w.data.getDouble(1, 0) == 9.0)

        val a = Tensor(arange(24.0).reshape(2, 3, 4))
        val b = Tensor(arange(24.0).reshape(2, 3, 4))
        val c = a.add(b)
        println(c.data)
        assert(c.data.getDouble(1, 0, 0) == 24.0)

    }


    @Test
    fun tensorReduction() {
        val x = Tensor(arange(6.0).reshape(2, 3))
        val y = x.sum()
        assert(y.data.getDouble(0) == 15.0)

        val a = Tensor(arange(24.0).reshape(2, 3, 4))
        val b = a.sum()
        assert(b.data.getDouble(0) == 276.0)


    }


    @Test
    fun testNonReduction() {
        val op = Tensor(arange(6.0).reshape(2, 3)).op({
            val z = zeros(2, 1)
            it.sum(z, true, 1)
            z
        },
            { _, _ ->
                listOf()
            })

        op.print()
        assert(op.data.getDouble(0, 0) == 3.0)

    }

    @Test
    fun testDot() {
        val ones = Tensor.ones(3)
        val x = Tensor.ones(3)
        ones.print()
        x.print()
        val z = ones.dot(x)
        z.print()
        assert(z.data.getDouble(0) == 3.0)

    }

    /**
     * https://d2l.ai/chapter_preliminaries/linear-algebra.html#norms
     */
    @Test
    fun testNorm() {
        val x = Tensor.ones(4, 9)
        val y = x.norm()
        y.print()
        assert(y.data.getDouble(0) == 6.0)

        val t = Tensor(create(doubleArrayOf(3.0, -4.0)))
        val u = t.norm()
        u.print()
        assert(u.data.getDouble(0) == 5.0)

    }


}
