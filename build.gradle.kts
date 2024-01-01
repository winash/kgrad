plugins {
    kotlin("jvm") version "1.9.21"
    application
}

group = "io.winash"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")

}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(11)
}

application {
    mainClass.set("MainKt")
}
