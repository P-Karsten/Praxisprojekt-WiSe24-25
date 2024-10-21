import org.joml.Vector3f
import org.joml.Vector4f
import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.light.IPointLight
import cga.exercise.components.light.ISpotLight
import cga.exercise.components.light.PointLight
import org.joml.Matrix3f
import org.joml.Matrix4f
import kotlin.math.cos

open class SpotLight(
        position: Vector3f,
        color: Vector3f,
        private val innerConeAngle: Float,
        private val outerConeAngle: Float

) : PointLight(position, color), ISpotLight {

    override fun bind(shaderProgram: ShaderProgram,viewMatrix: Matrix4f) {


        // Übergeben Sie die Kegeldefinitionen und den Richtungsvektor an den Fragment-Shader
        shaderProgram.use()
        shaderProgram.setUniform("spotLight.innerConeAngle", cos(innerConeAngle))
        shaderProgram.setUniform("spotLight.outerConeAngle", cos(outerConeAngle))
        shaderProgram.setUniform("spotLight.direction",  getWorldZAxis().negate().mul(Matrix3f(viewMatrix)))
        var test =Vector4f(getWorldPosition(),1.0f).mul(viewMatrix)
        shaderProgram.setUniform("spotLight.position", Vector3f(test.x,test.y,test.z))

        shaderProgram.setUniform("spotLight.lightColor",lightColor)
    }




}