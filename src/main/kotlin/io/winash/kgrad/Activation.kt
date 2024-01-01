package io.winash.kgrad


class ReLU : Module() {

    override fun forward(input: Tensor): Tensor {
        return input.relu()
    }

}
