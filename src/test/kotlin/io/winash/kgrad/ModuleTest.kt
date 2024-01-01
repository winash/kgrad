package io.winash.kgrad

import org.junit.jupiter.api.Test

class ModuleTest {

    @Test
    fun testSequential() {
        val model = Sequential(LazyLinear(256), ReLU(), LazyLinear(10))
        val tensor = model.forward(Tensor.rand(2, 20))
        println(tensor.data.shape().asList())

    }


    @Test
    fun testMLP() {
        class MLP : Module() {
            private val hidden = LazyLinear(256)
            private val out = LazyLinear(10)
            override fun forward(input: Tensor): Tensor {
                return out.forward(hidden.forward(input).relu())
            }
        }

        val model = MLP()
        val tensor = model.forward(Tensor.rand(2, 20))
        println(tensor.data.shape().asList())

    }

    @Test
    fun testFixHiddenMLP() {
        class MLP : Module() {
            private val weight = Tensor.rand(20, 20)
            private val linear = LazyLinear(20)

            override fun forward(input: Tensor): Tensor {
                var forward = linear.forward(input)
                forward = forward.mmul(weight)
                forward = forward.add(Tensor.ones(2, 20))
                var fw = forward.relu()
                fw = linear.forward(fw)
                while (fw.abs().sum().data.getDouble(0) > 1) {
                    fw = fw.div(Tensor.fillNum(2.0, *fw.data.shape()))
                }
                return fw.sum()
            }
        }

        val model = MLP()
        val tensor = model.forward(Tensor.rand(2, 20))
        println(tensor.data.getDouble(0))

    }


    @Test
    fun parameterTest(){
        val sequential = Sequential(LazyLinear(8), ReLU(), LazyLinear(1))
        val tensor = sequential.forward(Tensor.rand(2, 4))
        println(tensor.data.shape().asList())
    }





}
