package io.winash.kgrad


abstract class Module {
    abstract fun forward(input: Tensor): Tensor
    abstract fun name(): String

    open fun getWeights(): List<Tensor?> {
        return emptyList()
    }
}

class Sequential(private vararg val module: Module) : Module() {


    override fun forward(input: Tensor): Tensor {
        var output = input
        for (m in module) {
            output = m.forward(output)
        }

        return output
    }

    override fun name(): String {
        return "Sequential"
    }

    override fun getWeights(): List<Tensor?> {
        return module.flatMap { it.getWeights() }
    }


}




