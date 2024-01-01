import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun main(args: Array<String>) {
    println("Hello World!")

    // Try adding program arguments via Run/Debug configuration.
    // Learn more about running applications: https://www.jetbrains.com/help/idea/running-applications.html.
    val zeros: INDArray = Nd4j.zeros(2, 3)
    println("Zeros: \n$zeros")

    // Create a 2x3 matrix filled with random numbers

    // Create a 2x3 matrix filled with random numbers
    val randomMatrix: INDArray = Nd4j.rand(2, 3)
    println("Random Matrix: \n$randomMatrix")

    // Matrix Operations: Add, Subtract, Multiply, etc.

    // Matrix Operations: Add, Subtract, Multiply, etc.
    val sum: INDArray = zeros.add(randomMatrix)
    println("Sum: \n$sum")
}
