package io.winash.kgrad

import io.winash.kgrad.optim.Sgd
import org.junit.jupiter.api.Test
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.sqrt


class MinstTest {

    @Test
    fun test() {
        val trainImages = this::class.java.classLoader.getResourceAsStream("train-images.idx3-ubyte")?.readBytes()
        val trainLabels = this::class.java.classLoader.getResourceAsStream("train-labels.idx1-ubyte")?.readBytes()
        val images = this::class.java.classLoader.getResourceAsStream("t10k-images.idx3-ubyte")?.readBytes()
        val labels = this::class.java.classLoader.getResourceAsStream("t10k-labels.idx1-ubyte")?.readBytes()

        val xTrainImages = Nd4j.create      (trainImages!!.copyOfRange(16, trainImages.size).map { it.toUByte().toInt() }).reshape(-1,28,28)
        val xImages = Nd4j.create(images!!.copyOfRange(16, images.size).map { it.toUByte().toInt() }).reshape(-1,28,28)
        val xTrainLabels = Nd4j.create(trainLabels!!.copyOfRange(8, trainLabels.size).map { it.toUByte().toInt() })
        val xLabels = Nd4j.create(labels!!.copyOfRange(8, labels.size).map { it.toUByte().toInt() })

        val model = TwoLayerNet()
        val optim = Sgd(listOf(model.l1, model.l2))

        val batchSize = 128
        val epochs = 1000
        val lossHistory = mutableListOf<Double>()
        val trainAccHistory = mutableListOf<Double>()
        val testAccHistory = mutableListOf<Double>()

        for (epoch in 0..<epochs) {
            val numRows = xTrainImages.size(0)
            val indices = Nd4j.arange(numRows * 1.0) // Create an array of indices
            Nd4j.shuffle(indices, 0) // Shuffle along the first dimension
            val randomIndices =
                indices.get(NDArrayIndex.interval(0, batchSize.toLong())) // Take the first batchSize elements
            val xBatch = xTrainImages.get(randomIndices)
            val imageTrainTensor = Tensor(xBatch.reshape(-1, 28 * 28))
            val labelTrainTensor = xTrainLabels.get(randomIndices)

            val zeros = Nd4j.zeros(randomIndices.length(), 10)
            for (i in 0..<zeros.rows()) {
                val columnIndex: Int = labelTrainTensor.getInt(i) // Get the column index from Y
                zeros.putScalar(i.toLong(), columnIndex.toLong(), -1.0) // Set -1.0 at the specified position
            }
            val y = Tensor(zeros)


            val yPred = model.forward(imageTrainTensor)
            val loss = yPred.mul(y).mean()
            loss.backward()
            optim.step()
            lossHistory.add(loss.data.getDouble(0))

            val trainAcc = (yPred.data.argMax(1).castTo(DataType.FLOAT).eq(labelTrainTensor)).castTo(DataType.INT32).mean()

            println("Epoch: $epoch, Loss: ${loss.data.getDouble(0)}, Train Acc: $trainAcc")
            trainAccHistory.add(trainAcc.data().getDouble(0))
        }

        println("Done training")

        Tensor(xImages.reshape(-1, 28 * 28)).also { imageTestTensor ->
            val yPred = model.forward(imageTestTensor)
            val testAcc = (yPred.data.argMax(1).castTo(DataType.FLOAT).eq(xLabels)).castTo(DataType.INT32).mean()
            println("Test Acc: $testAcc")
            testAccHistory.add(testAcc.data().getDouble(0))
        }

println("Done testing")


    }


    class TwoLayerNet(val l1: Tensor, val l2: Tensor) {
        constructor() : this(
            Tensor(layerInit(784, 128)),
            Tensor(layerInit(128, 10)),
        )

        fun forward(x: Tensor): Tensor {
            val h1 = x.dot(l1).relu()
            val h2 = h1.dot(l2)
            return h2.logSoftmax()
        }

    }


}

fun layerInit(m: Int, h: Int): INDArray {
    val ret: INDArray = Nd4j.rand(UniformDistribution(-1.0, 1.0), m.toLong(), h.toLong())
    val scale = sqrt(1.0 / (m * h))
    ret.muli(scale)
    return ret
}
