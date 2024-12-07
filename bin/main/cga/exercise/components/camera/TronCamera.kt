package cga.exercise.components.camera

import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.geometry.Transformable
import org.joml.Matrix4f
import org.joml.Vector3f

class TronCamera : Transformable(), ICamera {
    private val fov: Float = Math.toRadians(90.0).toFloat() // Field of View
    private val aspectRatio: Float = 16.0f / 9.0f // Seitenverhältnis (horizontaler Öffnungswinkel)
    private val nearPlane: Float = 0.01f // Near Plane
    private val farPlane: Float = 10000.0f // Far Plane

    private var viewMatrix: Matrix4f = Matrix4f() // View-Matrix
    private var projectionMatrix: Matrix4f = Matrix4f() // Projection-Matrix

    init {
        //initialisiert Zustand einer Klasse
    }

    override fun getCalculateViewMatrix(): Matrix4f {
        return viewMatrix   //Kameramatrix, simuliert Kameraperspektive
    }

    override fun getCalculateProjectionMatrix(): Matrix4f {
        return projectionMatrix  //Perspektive und Sichtbereich
    }

    private fun calculateViewMatrix() {
        viewMatrix = Matrix4f().lookAt(getWorldPosition(), getWorldPosition().add(getWorldZAxis().negate()), getWorldYAxis())

    }

    private fun calculateProjectionMatrix() {
        projectionMatrix = Matrix4f().perspective(fov, aspectRatio, nearPlane, farPlane)
    }

    fun updateViewMatrix() {
        calculateViewMatrix()
    }

    fun updateProjectionMatrix() {
        calculateProjectionMatrix()
    }
    override fun bind(shader: ShaderProgram) {
        shader.setUniform("view_matrix", viewMatrix)
        shader.setUniform("proj_matrix", projectionMatrix)
    }

}

/*package cga.exercise.components.camera
import cga.exercise.components.geometry.Transformable
import cga.exercise.components.shader.ShaderProgram


import org.joml.Matrix4f
import org.joml.Vector3f


class TronCamera : Transformable(), ICamera {
    private var fieldOfView: Float = Math.toRadians(90.0).toFloat()
    private var aspectRatio: Float = 16.0f / 9.0f
    private var nearPlane: Float = 0.1f
    private var farPlane: Float = 100.0f



    override fun getCalculateViewMatrix(): Matrix4f {
        val viewMatrix = Matrix4f()
        viewMatrix.lookAt(getPosition(), getPosition().add(getFront(), Vector3f()), getUp())
        return viewMatrix
    }

    fun getFront(): Vector3f {
        return getZAxis().negate()
    }
    fun getUp(): Vector3f {
        return getYAxis().negate()
    }

    override fun getCalculateProjectionMatrix(): Matrix4f {
        val projectionMatrix = Matrix4f()
        projectionMatrix.perspective(fieldOfView, aspectRatio, nearPlane, farPlane)
        return projectionMatrix
    }

    override fun bind(shader: ShaderProgram) {
        shader.setUniform("proj_matrix", getCalculateProjectionMatrix())
        shader.setUniform("view_matrix", getCalculateViewMatrix())
    }
}*/