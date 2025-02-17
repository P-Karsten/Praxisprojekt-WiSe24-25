package cga.exercise.components.geometry

import cga.exercise.game.Scene
import org.joml.Math.atan2
import org.joml.Matrix4f
import org.joml.Quaternionf
import org.joml.Vector3f
open class Transformable(private var modelMatrix: Matrix4f = Matrix4f(), var parent: Transformable? = null) {


    fun setPosition(targetPosition: Vector3f) {

        val currentWorldPosition = getWorldPosition()
        val offset = Vector3f(targetPosition).sub(currentWorldPosition)
        translate(offset)
    }

    fun setRay(target: Matrix4f) {
        modelMatrix.add(target)
        updateModelMatrix()
    }

    fun rayToSpaceship(target: Matrix4f) {
        modelMatrix.identity()
        modelMatrix.add(target)
        updateModelMatrix()
    }

    /**
     * Returns copy of object model matrix
     * @return modelMatrix
     */
    fun getModelMatrix(): Matrix4f {
        // todo
        return Matrix4f(modelMatrix)
        //throw NotImplementedError()
    }

    /**
     * Returns multiplication of world and object model matrices.
     * Multiplication has to be recursive for all parents.
     * Hint: scene graph
     * @return world modelMatrix
     */
    fun getWorldModelMatrix(): Matrix4f {
        var temp = Matrix4f()
        val worldModelMatrix = Matrix4f(modelMatrix)

        parent?.let {
            temp.set(it.getWorldModelMatrix())
            temp.mul(worldModelMatrix)
            worldModelMatrix.set(temp)
        }

        return worldModelMatrix
    }

    /**
     * Rotates object around its own origin.
     * @param pitch radiant angle around x-axis ccw
     * @param yaw radiant angle around y-axis ccw
     * @param roll radiant angle around z-axis ccw
     */
    private var quaternion = Quaternionf()
    fun transformVector(vector: Vector3f): Vector3f {
        val rotationQuat = getRotationQuaternion()
        return rotationQuat.transform(vector)
    }
    fun transformToLocal(vector: Vector3f): Vector3f {
        val rotationQuat = getRotationQuaternion().invert() // Invertiert das Quaternion
        return rotationQuat.transform(vector) // Transformiert den Zielvektor
    }
    fun getRotationQuaternion(): Quaternionf {
        return quaternion
    }
    fun rotate(pitch: Float, yaw: Float, roll: Float) {
        // Set roll to 0


        val pitchQuat = Quaternionf().rotateX(pitch) // Pitch (rotation around X-axis)
        val yawQuat = Quaternionf().rotateY(yaw)     // Yaw (rotation around Y-axis)
        val rollQuat = Quaternionf().rotateZ(roll)   // Set roll to 0

        // Combine rotations (quaternion multiplication: yaw * pitch * roll)
        quaternion = yawQuat.mul(pitchQuat).mul(rollQuat)

        // Update the model matrix with the rotation
        updateModelMatrix()
    }

    private fun updateModelMatrix() {
        //modelMatrix.identity()  // Reset to identity
        modelMatrix.rotate(quaternion)  // Apply the quaternion rotation
    }
    fun setRotation(pitch: Float, yaw: Float, roll: Float) {
        // Reset the quaternion to identity
        quaternion.identity()

        // Create quaternions for each axis
        val pitchQuat = Quaternionf().rotateX(pitch)
        val yawQuat = Quaternionf().rotateY(yaw)
        val rollQuat = Quaternionf().rotateZ(roll)

        // Combine rotations (yaw * pitch * roll)
        quaternion = yawQuat.mul(pitchQuat).mul(rollQuat)

        // Update the model matrix to reflect the new rotation
        modelMatrix.identity()
        updateModelMatrix()
    }
    /**
     * Rotates object around given rotation center.
     * @param pitch radiant angle around x-axis ccw
     * @param yaw radiant angle around y-axis ccw
     * @param roll radiant angle around z-axis ccw
     * @param altMidpoint rotation center
     */
    fun rotateAroundPoint(pitch: Float, yaw: Float, roll: Float, altMidpoint: Vector3f) {
        // todo
        val translationToOrigin = Matrix4f().translate(-altMidpoint.x, -altMidpoint.y, -altMidpoint.z)
        val rotation = Matrix4f().rotateXYZ(pitch, yaw, roll)
        val translationBack = Matrix4f().translate(altMidpoint.x, altMidpoint.y, altMidpoint.z)


        val tempMatrix = Matrix4f()


        tempMatrix.identity()
        tempMatrix.mul(translationBack)
        tempMatrix.mul(rotation)
        tempMatrix.mul(translationToOrigin)


        modelMatrix = tempMatrix.mul(modelMatrix)
        //throw NotImplementedError()
    }

    /**
     * Translates object based on its own coordinate system.
     * @param deltaPos delta positions
     */
    fun translate(deltaPos: Vector3f) {
        // todo
        modelMatrix.translate(deltaPos)
        //throw NotImplementedError()
    }


    /**
     * Translates object based on its parent coordinate system.
     * Hint: this operation has to be left-multiplied
     * @param deltaPos delta positions (x, y, z)
     */
    fun preTranslate(deltaPos: Vector3f) {
        // todo
        val translation = Matrix4f().translate(deltaPos)
        modelMatrix = translation.mul(modelMatrix)
        //throw NotImplementedError()
    }

    /**
     * Scales object related to its own origin
     * @param scale scale factor (x, y, z)
     */
    fun scale(scale: Vector3f) {
        // todo
        modelMatrix.scale(scale)
        //throw NotImplementedError()
    }

    /**
     * Returns position based on aggregated translations.
     * Hint: last column of model matrix
     * @return position
     */
    fun getPosition(): Vector3f {
        // todo
        val position = Vector3f()
        modelMatrix.getColumn(3, position)
        return position
        //throw NotImplementedError()
    }

    /**
     * Returns position based on aggregated translations incl. parents.
     * Hint: last column of world model matrix
     * @return position
     */
    fun getWorldPosition(): Vector3f {
        // todo
        val worldModelMatrix = getWorldModelMatrix()
        val worldPosition = Vector3f()
        worldModelMatrix.getColumn(3, worldPosition)
        return worldPosition
        //throw NotImplementedError()
    }

    /**
     * Returns x-axis of object coordinate system
     * Hint: first normalized column of model matrix
     * @return x-axis
     */
    fun getXAxis(): Vector3f {
        // todo
        val xAxis = Vector3f()
        modelMatrix.getColumn(0, xAxis)
        return xAxis.normalize()
        //throw NotImplementedError()
    }

    /**
     * Returns y-axis of object coordinate system
     * Hint: second normalized column of model matrix
     * @return y-axis
     */
    fun getYAxis(): Vector3f {
        // todo
        val yAxis = Vector3f()
        modelMatrix.getColumn(1, yAxis)
        return yAxis.normalize()
       // throw NotImplementedError()
    }

    /**
     * Returns z-axis of object coordinate system
     * Hint: third normalized column of model matrix
     * @return z-axis
     */
    fun getZAxis(): Vector3f {
        // todo
        val zAxis = Vector3f()
        modelMatrix.getColumn(2, zAxis)
        return zAxis.normalize()
       // throw NotImplementedError()
    }

    /**
     * Returns x-axis of world coordinate system
     * Hint: first normalized column of world model matrix
     * @return x-axis
     */
    fun getWorldXAxis(): Vector3f {
        // todo
        val worldModelMatrix = getWorldModelMatrix()
        val worldXAxis = Vector3f()
        worldModelMatrix.getColumn(0, worldXAxis)
        return worldXAxis.normalize()
        //throw NotImplementedError()
    }

    /**
     * Returns y-axis of world coordinate system
     * Hint: second normalized column of world model matrix
     * @return y-axis
     */
    fun getWorldYAxis(): Vector3f {
        // todo
        val worldModelMatrix = getWorldModelMatrix()
        val worldYAxis = Vector3f()
        worldModelMatrix.getColumn(1, worldYAxis)
        return worldYAxis.normalize()
        //throw NotImplementedError()
    }

    /**
     * Returns z-axis of world coordinate system
     * Hint: third normalized column of world model matrix
     * @return z-axis
     */
    fun getWorldZAxis(): Vector3f {
        // todo
        val worldModelMatrix = getWorldModelMatrix()
        val worldZAxis = Vector3f()
        worldModelMatrix.getColumn(2, worldZAxis)
        return worldZAxis.normalize()
       // throw NotImplementedError()
    }

    /*fun getRotation(): Scene.Vector3f {
        val xAxis = getXAxis()
        val yAxis = getYAxis()
        val zAxis = getZAxis()

        // Berechne die Yaw (Rotation um die y-Achse)
        val yaw = Math.atan2(xAxis.z.toDouble(), xAxis.x.toDouble()).toFloat()
        // Berechne die Pitch (Rotation um die x-Achse)
        val pitch = Math.asin(-xAxis.y.toDouble()).toFloat()

        // Berechne die Roll (Rotation um die z-Achse)
        val roll = Math.atan2(yAxis.y.toDouble(), zAxis.y.toDouble()).toFloat()
        //val roll = Math.atan2(zAxis.x.toDouble(), zAxis.z.toDouble()).toFloat()
        return Scene.Vector3f(pitch, yaw, roll)
    }*/
    fun getRotation(): Scene.Vector3f {
        val xAxis = getXAxis()
        val yAxis = getYAxis()
        val zAxis = getZAxis()

        // Calculate Yaw (rotation around the Y-axis)
        val yaw = Math.atan2(xAxis.z.toDouble(), xAxis.x.toDouble()).toFloat()

        // Calculate Pitch (rotation around the X-axis)
        val roll = Math.asin(xAxis.y.toDouble()).toFloat()

        // Calculate Roll (rotation around the Z-axis)
        val pitch = Math.atan2(yAxis.y.toDouble(), zAxis.y.toDouble()).toFloat()
        //val roll = Math.atan2(yAxis.x.toDouble(), zAxis.x.toDouble()).toFloat()

        return Scene.Vector3f(pitch, yaw, roll)
    }
}