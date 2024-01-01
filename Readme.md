A simple Neural network framework written in Kotlin. This is largely a learning exercise for me, but I hope it can be useful to others as well, who would like a more in-depth understanding of how neural networks work.

Most of the code in the project is inspired from Torch, Micro-grad, TinyGrad and many other sources. It also serves as a companion to https://d2l.ai/ (Dive into Deep Learning) which is a great resource for learning about deep learning.
Start at the Tensor class and work your way up from there. 

In the code there are many references to the book eg: Here is the MLP implementation from the book in Kotlin https://d2l.ai/chapter_builders-guide/model-construction.html
```kotlin
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
```



