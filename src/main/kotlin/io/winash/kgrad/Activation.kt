package io.winash.kgrad


class ReLU : Module() {

    override fun forward(input: Tensor): Tensor {
        return input.relu()
    }

    override fun name(): String {
        return "ReLU"
    }

}

class Sigmoid : Module() {
    override fun forward(input: Tensor): Tensor {
        return input.sigmoid()
    }

    override fun name(): String {
        return "Sigmoid"
    }

}


class Flatten : Module() {
    override fun forward(input: Tensor): Tensor {
        return input.reshape(input.shape()[0], -1)
    }

    override fun name(): String {
        return "Flatten"
    }

}

