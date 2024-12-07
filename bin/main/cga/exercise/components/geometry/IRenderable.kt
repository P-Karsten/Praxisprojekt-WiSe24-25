package cga.exercise.components.geometry

import cga.exercise.components.shader.ShaderProgram
import org.joml.Vector3f

interface IRenderable {
    fun render(shaderProgram: ShaderProgram,emitcol:Vector3f)
}