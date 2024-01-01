package io.winash.kgrad

import org.nd4j.linalg.indexing.NDArrayIndex


fun corr2d(input: Tensor, kernel: Tensor): Tensor {
    val kernelShape = kernel.shape()
    val output = Tensor.zeros(input.shape()[0] - kernelShape[0] + 1, input.shape()[1] - kernelShape[1] + 1)
    for (i in 0..<output.shape()[0]) {
        for (j in 0..<output.shape()[1]) {
            output.data.putScalar(
                i,
                j,
                input.data.get(
                    NDArrayIndex.interval(i, i + kernelShape[0]),
                    NDArrayIndex.interval(j, j + kernelShape[1])
                ).mul(kernel.data).sumNumber().toDouble()
            )
        }
    }
    return output
}
