package io.winash.kgrad

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertNull

/**
 * Learn some automatic differentiation
 * https://d2l.ai/chapter_preliminaries/autograd.html
 */
class Autograd {

    @Test
    fun testSimpleFunction() {
        val tensor = Tensor(Nd4j.arange(4.0))
        assertNull(tensor.grad)
        val newTensor = tensor.dot(tensor).op(
            { p ->
                p.mul(2)

            },
            { grad, _ ->
                listOf(grad.mul(2))
            }
        )
        newTensor.print()
        newTensor.backward()
        println(tensor.grad)
        assert(tensor.grad == tensor.data.mul(4))
        tensor.resetGrad()
        val y = tensor.sum()
        y.backward()
        println(tensor.grad)

    }


    @Test
    fun testMulByItself() {
        val tensor = Tensor(Nd4j.arange(4.0))
        val y = tensor.mul(tensor)
        y.sum().backward()
        println(tensor.grad)
        assert(tensor.grad == tensor.data.mul(2))
    }


    @Test
    fun testControlFlow() {
        val tensor = Tensor(Nd4j.arange(4.0))
        val y = compute(tensor)
        y.print()
        y.backward()
        println(y.data.div(tensor.data))
    }

    private fun compute(a: Tensor): Tensor {
        var b = a.op(
            { p ->
                p.mul(2)

            },
            { grad, _ ->
                listOf(grad.mul(2))
            })
        while (b.norm().data.getDouble(0) < 1000) {
            b = b.op(
                { p ->
                    p.mul(2)

                },
                { grad, _ ->
                    listOf(grad.mul(2))
                })
        }
        var c: Tensor
        if (b.sum().data.getDouble(0) > 0) {
            c = b
        } else {
            c = b.op(
                { p ->
                    p.mul(100)

                },
                { grad, _ ->
                    listOf(grad.mul(100))
                })
        }
        return c


    }


}
