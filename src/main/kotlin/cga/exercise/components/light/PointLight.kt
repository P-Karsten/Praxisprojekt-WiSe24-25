package cga.exercise.components.light

import cga.exercise.components.geometry.Transformable
import cga.exercise.components.shader.ShaderProgram
import org.joml.Matrix4f
import org.joml.Vector3f
import org.joml.Vector4f
import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL20

open class PointLight(initialPosition: Vector3f,  val lightColor: Vector3f) : Transformable() , IPointLight {

    init {
        translate(initialPosition)
    }

    override fun bind(shaderProgram: ShaderProgram,viewMatrix: Matrix4f,i:Int) {
        val worldPosition = getWorldPosition()
        val lightColorUniform = "pointLight.lightColor[$i]"
        val lightPositionUniform = "pointLight.position[$i]"

        val color = Vector3f(lightColor)
        val position = Vector3f(worldPosition)


        shaderProgram.setUniform(lightColorUniform, color)
        var test = Vector4f(position,1.0f).mul(viewMatrix)
        shaderProgram.setUniform(lightPositionUniform, Vector3f(test.x,test.y,test.z))
    }
}