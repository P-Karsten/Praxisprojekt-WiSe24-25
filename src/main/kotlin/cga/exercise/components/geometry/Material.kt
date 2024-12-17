package cga.exercise.components.geometry

import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.texture.Texture2D
import org.joml.Vector2f

class Material(
    var diff: Texture2D? = null,
    var emit: Texture2D? = null,
    var specular: Texture2D? = null,
    var shininess: Float = 50.0f,
    var tcMultiplier: Vector2f = Vector2f(1.0f)
) {
    fun bind(shaderProgram: ShaderProgram) {

        emit?.bind(1)
        shaderProgram.setUniform("material_emissive", 1)
        diff?.bind(2)
        shaderProgram.setUniform("material_diffuse", 2)
        specular?.bind(3)
        shaderProgram.setUniform("material_specular", 3)

        shaderProgram.setUniform("shininess", shininess)
        shaderProgram.setUniform("tcMultiplier", tcMultiplier)
    }
}
