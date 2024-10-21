package cga.exercise.components.geometry

import cga.exercise.components.shader.ShaderProgram
import org.joml.Matrix4f
import org.joml.Vector3f

class Renderable(private val meshes: MutableList<Mesh>) : Transformable() , IRenderable {

    fun cleanup() {
        for (mesh in meshes) {
            mesh.cleanup()
        }
    }

    override fun render(shaderProgram: ShaderProgram,emitcol:Vector3f) {

        shaderProgram.use()
        shaderProgram.setUniform("model_matrix", getWorldModelMatrix())
        shaderProgram.setUniform("emit_col", emitcol)
        for (mesh in meshes) {

            mesh.render(shaderProgram)
        }

    }
}