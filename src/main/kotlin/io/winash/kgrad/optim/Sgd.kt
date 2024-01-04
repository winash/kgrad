package io.winash.kgrad.optim

import io.winash.kgrad.Tensor

class Sgd(private var tensors: List<Tensor?>, private val lr: Double = 0.1) {

    fun zeroGrad() {
        tensors.forEach { t ->
            t?.grad = null
        }
    }

    fun step() {
        tensors.forEach { t ->
            if (t?.grad != null) {
                t.data.subi(t.grad!!.mul(lr))
            }
        }
    }
}







