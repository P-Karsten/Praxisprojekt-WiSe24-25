package cga.exercise.components.texture

import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL11.*
import org.lwjgl.opengl.GL13.*
import org.lwjgl.opengl.GL30.*
import org.lwjgl.stb.STBImage
import java.nio.ByteBuffer

class TextureCubeMap(
    right: String,
    left: String,
    top: String,
    bottom: String,
    front: String,
    back: String
) {
    private val texID: Int = glGenTextures()

    init {
        val faces = arrayOf(right, left, top, bottom, front, back)
        glBindTexture(GL_TEXTURE_CUBE_MAP, texID)

        for (i in faces.indices) {
            val imageData = loadTexture(faces[i])
            if (imageData != null) {
                println("Loaded face: ${faces[i]} (Width: ${imageData.width}, Height: ${imageData.height})")
                glTexImage2D(
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA,
                    imageData.width, imageData.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData.data
                )
                STBImage.stbi_image_free(imageData.data) // Speicher freigeben
            } else {
                throw Exception("Failed to load cube map texture: ${faces[i]}")
            }
        }

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_CUBE_MAP, 0) // Unbind
    }

    fun bind(textureUnit: Int) {
        glActiveTexture(GL_TEXTURE0 + textureUnit)
        glBindTexture(GL_TEXTURE_CUBE_MAP, texID)
    }

    fun unbind() {
        GL11.glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    }

    private data class ImageData(val data: ByteBuffer, val width: Int, val height: Int)

    private fun loadTexture(path: String): ImageData? {
        val widthBuffer = org.lwjgl.BufferUtils.createIntBuffer(1)
        val heightBuffer = org.lwjgl.BufferUtils.createIntBuffer(1)
        val channelsBuffer = org.lwjgl.BufferUtils.createIntBuffer(1)

        STBImage.stbi_set_flip_vertically_on_load(false)
        val data = STBImage.stbi_load(path, widthBuffer, heightBuffer, channelsBuffer, 4)
        return if (data != null) {
            ImageData(data, widthBuffer.get(), heightBuffer.get())
        } else {
            null
        }
    }
}
