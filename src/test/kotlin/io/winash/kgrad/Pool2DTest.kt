package io.winash.kgrad

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j

class Pool2DTest {

    @Test
    fun testPool2DMax() {
        val tensor = Tensor(
            Nd4j.create(
                arrayOf(
                    doubleArrayOf(0.0, 1.0, 2.0),
                    doubleArrayOf(3.0, 4.0, 5.0),
                    doubleArrayOf(6.0, 7.0, 8.0)
                )
            )
        )
        val pool = Pool2D(Pair(2, 2))
        pool.forward(tensor).print()

    }

    @Test
    fun testPool2DAvg() {
        val tensor = Tensor(
            Nd4j.create(
                arrayOf(
                    doubleArrayOf(0.0, 1.0, 2.0),
                    doubleArrayOf(3.0, 4.0, 5.0),
                    doubleArrayOf(6.0, 7.0, 8.0)
                )
            )
        )
        val pool = Pool2D(Pair(2, 2),PoolMode.AVG)
        pool.forward(tensor).print()

    }


}


