package io.winash.kgrad.nets

import io.winash.kgrad.*
import io.winash.kgrad.optim.Sgd
import org.junit.jupiter.api.Test
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.test.Ignore

class LeNet() : Module() {

    private val net = Sequential(
        Conv2D(6, Pair(5, 5), Pair(2, 2)),
        Sigmoid(),
        Pool2D(Pair(2, 2), PoolMode.AVG, Pair(2, 2)),
        Conv2D(16, kernelSize = Pair(5, 5), paddingMode = PaddingMode.VALID),
        Sigmoid(),
        Pool2D(Pair(2, 2), PoolMode.AVG, Pair(2, 2)),
        Flatten(),
        LazyLinear(120), Sigmoid(),
        LazyLinear(84), Sigmoid(),
        LazyLinear(10)
    )

    override fun forward(input: Tensor): Tensor {
        return net.forward(input)
    }

    override fun name(): String {
        return "LeNet"
    }

    override fun getWeights(): List<Tensor?> {
        return net.getWeights()
    }


}

class LeNetTest() {
    @Test
    fun testLeNet() {
        val model = LeNet()
        val tensor = model.forward(Tensor.randn(1, 1, 28, 28))
        println(tensor.data.shape().asList())

    }
}

class MinstLeNetTest {
    @Test
    @Ignore("Does not work yet")
    fun testLeNet() {
        val trainImages = this::class.java.classLoader.getResourceAsStream("train-images.idx3-ubyte")?.readBytes()
        val trainLabels = this::class.java.classLoader.getResourceAsStream("train-labels.idx1-ubyte")?.readBytes()
        val images = this::class.java.classLoader.getResourceAsStream("t10k-images.idx3-ubyte")?.readBytes()
        val labels = this::class.java.classLoader.getResourceAsStream("t10k-labels.idx1-ubyte")?.readBytes()

        val xTrainImages = Nd4j.create      (trainImages!!.copyOfRange(16, trainImages.size).map { it.toUByte().toInt() }).reshape(-1,1,28,28)
        val xImages = Nd4j.create(images!!.copyOfRange(16, images.size).map { it.toUByte().toInt() }).reshape(-1,1,28,28)
        val xTrainLabels = Nd4j.create(trainLabels!!.copyOfRange(8, trainLabels.size).map { it.toUByte().toInt() })
        val xLabels = Nd4j.create(labels!!.copyOfRange(8, labels.size).map { it.toUByte().toInt() })


        val model = LeNet()

        val optim by lazy { Sgd(model.getWeights()) }

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

            val imageTrainTensor = Tensor(xBatch.reshape(-1,1, 28,28))
            val labelTrainTensor = xTrainLabels.get(randomIndices)

            val yPred = model.forward(imageTrainTensor)
            val zeros = Nd4j.zeros(randomIndices.length(), 10)
            for (i in 0..<zeros.rows()) {
                val columnIndex: Int = labelTrainTensor.getInt(i) // Get the column index from Y
                zeros.putScalar(i.toLong(), columnIndex.toLong(), -1.0) // Set -1.0 at the specified position
            }
            val y = Tensor(zeros)


            val loss = Functions.sparseCategoricalCrossEntropy(yPred, y)

            optim.zeroGrad()

            loss.backward()

            optim.step()

            lossHistory.add(loss.data.getDouble(0))
            val trainAcc = (yPred.data.argMax(1).castTo(DataType.FLOAT).eq(labelTrainTensor)).castTo(DataType.INT32).mean()

            println("Epoch: $epoch, Loss: ${loss.data.getDouble(0)}, Train Acc: $trainAcc")
            trainAccHistory.add(trainAcc.data().getDouble(0))
        }










    }
}



