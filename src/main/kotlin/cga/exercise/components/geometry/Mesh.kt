package cga.exercise.components.geometry

import cga.exercise.components.shader.ShaderProgram
import org.lwjgl.opengl.GL15
import org.lwjgl.opengl.GL20
import org.lwjgl.opengl.GL30.*

/**
 * Creates a Mesh object from vertexdata, intexdata and a given set of vertex attributes
 *
 * @param vertexdata plain float array of vertex data
 * @param indexdata  index data
 * @param attributes vertex attributes contained in vertex data
 * @throws Exception If the creation of the required OpenGL objects fails, an exception is thrown
 *
 * Created 29.03.2023.
 */
class Mesh(vertexdata: FloatArray, indexdata: IntArray, attributes: Array<VertexAttribute>, material: Material?=null) {
    //private data
    private var vaoId = 0
    private var vboId = 0
    private var iboId = 0
    private var indexcount = indexdata.size
    private var meshMaterial: Material? = material

    init {
        val vertices = vertexdata
        val indices = indexdata
        // todo Aufgabe 1.2.2 (shovel geometry data to GPU and tell OpenGL how to interpret it)
        vaoId = glGenVertexArrays()
        glBindVertexArray(vaoId)

        vboId = glGenBuffers()
        glBindBuffer(GL_ARRAY_BUFFER,vboId)
        glBufferData(GL_ARRAY_BUFFER,vertices, GL_STATIC_DRAW)
        var i=0
        for (item in attributes ){
            glEnableVertexAttribArray(i)

            GL20.glVertexAttribPointer(i,item.n, GL_FLOAT,false,item.stride,item.offset)
            i++
        }

        iboId = glGenBuffers()
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,iboId)
        GL15.glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices, GL_STATIC_DRAW)

        // Unbind VBO and VAO
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
    }

    /**
     * Renders the mesh
     */
    fun render(shaderProgram: ShaderProgram) {
        // todo

        glBindVertexArray(vaoId)
        shaderProgram.use()
        meshMaterial?.bind(shaderProgram)
        glDrawElements(GL_TRIANGLES,indexcount, GL_UNSIGNED_INT,0)
        glBindVertexArray(0)

    }

    /**
     * Deletes the previously allocated OpenGL objects for this mesh
     */
    fun cleanup() {
        if (iboId != 0) glDeleteBuffers(iboId)
        if (vboId != 0) glDeleteBuffers(vboId)
        if (vaoId != 0) glDeleteVertexArrays(vaoId)
    }
}