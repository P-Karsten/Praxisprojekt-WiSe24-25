package cga.exercise.components.geometry

import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.texture.TextureCubeMap

class SkyboxMaterial(private val cubeMap: TextureCubeMap) {
    fun bind(shaderProgram: ShaderProgram) {
        // Binde die Cubemap-Textur an die Textur-Einheit 0
        cubeMap.bind(0)
        shaderProgram.setUniform("skybox", 0) // Setze den Uniform-Wert "skybox"
    }
}
