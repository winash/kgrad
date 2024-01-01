package io.winash.kgrad


abstract class Module {

    abstract fun forward(input: Tensor): Tensor
}

class Sequential(private vararg val module: Module) : Module() {

    override fun forward(input: Tensor): Tensor {
        var output = input
        for (m in module) {
            output = m.forward(output)
        }
        return output
    }

}




