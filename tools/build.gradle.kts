plugins {
    java
    application
}

application {
    mainClass.set("tools.SaveToJson")
}

dependencies {
    implementation(project(":game-app:game-core"))
}
