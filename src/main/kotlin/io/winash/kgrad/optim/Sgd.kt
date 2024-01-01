package io.winash.kgrad.optim

import io.winash.kgrad.Tensor

class Sgd(private val tensors: List<Tensor>, private val lr: Double = 0.01) {

    fun step() {
        tensors.forEach { t ->
            if (t.grad != null) {
                t.data.subi(t.grad!!.mul(lr))
            }
        }
    }
}

