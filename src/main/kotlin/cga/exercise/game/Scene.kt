package cga.exercise.game

import SpotLight
import cga.exercise.components.camera.TronCamera
import cga.exercise.components.geometry.*
import cga.exercise.components.light.PointLight
import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.texture.Texture2D
import cga.exercise.components.texture.TextureCubeMap
import cga.framework.GLError
import cga.framework.GameWindow
import cga.framework.ModelLoader
import cga.framework.OBJLoader.loadOBJ
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.joml.*
import org.joml.Math.atan2
import org.joml.Vector3f
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL30.*
import java.util.Random
import kotlin.collections.ArrayList
import kotlin.math.*
import org.joml.Vector3f as Vector3f1

/**
 * Created 29.03.2023.
 */
class Scene(private val window: GameWindow) {
    private val staticShader: ShaderProgram = ShaderProgram("assets/shaders/tron_vert.glsl", "assets/shaders/tron_frag.glsl")
    private val skyboxShader: ShaderProgram = ShaderProgram("assets/shaders/skybox_vert.glsl", "assets/shaders/skybox_frag.glsl")

    data class DataModel(val id: Int, val message: String)

    var predict =true

    private var camera: TronCamera
    private var camera_fp: TronCamera
    lateinit var astmesh: Mesh
    //var rayPos=Vector3f()
    //var rayMat=Matrix4f()

    var vmaxa=0.01f
    var fps = 0
    var lastTime = System.currentTimeMillis()
    var frames = 0
    var prevleng=10000f
    var vmaxa2=0.00017f
    var score =0f
    var pause =true
    var rayl= 0
    var rayPos=Vector3f()
    var rayMat=Matrix4f()
    var shortyaw=java.lang.Math.PI.toFloat()
    var shortpitch=java.lang.Math.PI.toFloat()
    var inputkey= ""
    var cAsteroid=Vector3f(10000f,10000f,10000f)
    var cdasteroid=Vector3f(10000f,10000f,10000f)
    var sendcd =0
    var speed = -0.25f
    var shoot =false
    var cammode =0f
    var camselect=0f
    var tempshader=1f
    var hit=false
    var previousdis=java.lang.Math.PI*2
    var asteroidlist = mutableListOf<Renderable>()
    var asteroidlist2 = mutableListOf<Renderable>()
    var meshlist = mutableListOf<Mesh>()
    var renderable = Renderable(meshlist)
    var Moon = Renderable(meshlist)
    var skyboxExp = Renderable(meshlist)
    var spaceship = Renderable(meshlist)
    var ray = Renderable(meshlist)
    var ray3: Renderable? = Renderable(meshlist)
    val stride = 8 * 4
    val atr1 = VertexAttribute(3, GL_FLOAT, stride, 0)     //position attribute
    val atr2 = VertexAttribute(3, GL_FLOAT, stride, 3 * 4) //texture coordinate attribute
    val atr3 = VertexAttribute(3, GL_FLOAT, stride, 5 * 4) //normal attribute
    val vertexAttributes = arrayOf(atr1, atr2, atr3)
    var astobj =loadOBJ("assets/a2/rock_by_dommk.obj",true,true)
    var astdiff = Texture2D("assets/a2/rock_Base_Color.png",true)
    var astemit = Texture2D("assets/a2/rock_Height.png",true)
    var astspec = Texture2D("assets/a2/rock_by_dommk_nmap.tga",true)
    val astmat=Material(
        astdiff,
        astemit,
        astspec,
        2.0f,
        Vector2f(1.0f, 1.0f)

    )
    val skyboxNewTexture = TextureCubeMap(
        "assets/skybox/right.png",
        "assets/skybox/left.png",
        "assets/skybox/top.png",
        "assets/skybox/bottom.png",
        "assets/skybox/front.png",
        "assets/skybox/back.png"
    )
    var skyboxMaterial2: SkyboxMaterial
    var laserDirX = 0f
    var laserDirTmpX = 0f
    var laserDirY = 0f
    var laserDirTmpY = 0f
    var laserTranslate = 0f
    var spaceshipPosOld = Vector3f1(0f,0f,0f)
    val astSpawns = arrayOf(
        Vector3f1(-45f, 23f, -67f),
        Vector3f1(12f, -89f, 56f),
        Vector3f1(-73f, 48f, -21f),
        Vector3f1(88f, -32f, 99f),
        Vector3f1(-7f, 65f, -43f),
        Vector3f1(54f, -76f, 34f),
        Vector3f1(-100f, 21f, -58f),
        Vector3f1(83f, -44f, 9f),
        Vector3f1(-33f, 57f, -87f),
        Vector3f1(19f, -12f, 74f),
        Vector3f1(61f, -99f, 22f),
        Vector3f1(-28f, 36f, -65f),
        Vector3f1(70f, -80f, 15f),
        Vector3f1(-91f, 48f, -39f),
        Vector3f1(44f, -53f, 67f),
        Vector3f1(-19f, 62f, -84f),
        Vector3f1(29f, -24f, 33f),
        Vector3f1(85f, -67f, -18f),
        Vector3f1(-47f, 50f, -100f),
        Vector3f1(39f, -35f, 72f),
        Vector3f1(-64f, 15f, -49f),
        Vector3f1(92f, -88f, 60f),
        Vector3f1(-8f, 76f, -42f),
        Vector3f1(56f, -70f, 19f),
        Vector3f1(-77f, 21f, -11f),
        Vector3f1(48f, -60f, 38f),
        Vector3f1(-95f, 35f, -29f),
        Vector3f1(30f, -45f, 83f),
        Vector3f1(-22f, 66f, -53f),
        Vector3f1(62f, -78f, 27f),
        Vector3f1(-39f, 12f, -91f),
        Vector3f1(81f, -54f, 14f),
        Vector3f1(-88f, 42f, -31f),
        Vector3f1(45f, -65f, 69f),
        Vector3f1(-16f, 24f, -92f),
        Vector3f1(36f, -46f, 55f),
        Vector3f1(-72f, 18f, -41f),
        Vector3f1(94f, -82f, 63f),
        Vector3f1(-10f, 79f, -23f),
        Vector3f1(58f, -89f, 47f),
        Vector3f1(-85f, 29f, -34f),
        Vector3f1(40f, -51f, 90f),
        Vector3f1(-30f, 67f, -71f),
        Vector3f1(67f, -73f, 24f),
        Vector3f1(-49f, 13f, -99f),
        Vector3f1(79f, -61f, 37f),
        Vector3f1(-96f, 47f, -27f),
        Vector3f1(25f, -38f, 84f),
        Vector3f1(-24f, 55f, -44f),
        Vector3f1(64f, -92f, 18f)
    )
    /*
    val astSpawns = arrayOf(
        Vector3f1(50f,10f,10f),
        Vector3f1(10f,50f,-56f),
        Vector3f1(20f,-30f,73f),
        Vector3f1(-50f,10f,-75f),
        Vector3f1(25f,-22f,26f),
        Vector3f1(-90f,-87f,-29f),
        Vector3f1(-50f,35f,10f),
        Vector3f1(80f,-55f,-9f),
        Vector3f1(-20f,34f,58f),
        Vector3f1(65f,95f,21f),
        Vector3f1(-15f,-76f,-34f),
        Vector3f1(80f,12f,-79f),
        Vector3f1(-33f,-66f,-2f),
        Vector3f1(-95f,-81f,8f),
        Vector3f1(20f,53f,26f),
        Vector3f1(87f,-23f,66f),
        Vector3f1(17f,33f,-26f)
    )
    var nextAstDistance = 0f
    )*/
    var nextAstDistance = 0f
    var astSpawnCounter = 0
    var counter = 0
    var astHit = false
    var errorCount = 0
    val desiredGammaValue = 2.2f // Beispielwert für den gewünschten Gammawert
    val lightPosition = Vector3f1(0f, 5f, 0f) // Anpassen der Lichtposition
    val lightColor = Vector3f1(0.11f, 0.11f, 0.11f) // Anpassen der Lichtfarbe (hier: Weiß)
    var pointLight = PointLight(lightPosition, lightColor)
    var pointLight2 = PointLight(Vector3f1(0f,1f,0f), Vector3f1(0.0f,0.0f,0.0f))
    var pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(0.0f,0.0f,0.0f))
    val spotLight = SpotLight(Vector3f1(0f,2f,0f), Vector3f1(500f,500f,500f),Math.toRadians(1f),org.joml.Math.toRadians(2f))
    val pointLight5 = PointLight(Vector3f1(0f,0f,0f), Vector3f1(500.0f,500.0f,500.0f))
    private val initialSpaceshipPosition = Vector3f1(0.0f, 0.0f, 0.0f)
    private var currentSpaceshipPosition = Vector3f1(initialSpaceshipPosition)
    private var collisionCheckTimer: Float = 0f
    private val collisionCheckInterval: Float = 0.1f


    @Serializable
    data class Action(var action:Int)
    @Serializable
    sealed class Vector3fc

    @Serializable
    data class Vector3f(var x: Float, var y: Float, var z: Float) : Vector3fc()
    @Serializable
    data class GameData(
        val spaceshipPosition: List<Float>,
        val pitch: Float,
        val yaw : Float,
        val hit: Boolean,
        val alive: Boolean,
        val counter: Float,
        val posDistance: Float,
    )
    var action= 11
    var gameDataset = mutableListOf<GameData>()

    fun collectData(
        spaceshipPos: Vector3f1,
        pitch: Float,
        yaw : Float,
        hit: Boolean,
        alive: Boolean,
        counter: Float,
        posDistance: Float
    ) {                             //rotation spaceship
        val data = GameData(
            spaceshipPosition = listOf(spaceshipPos.x, spaceshipPos.y, spaceshipPos.z),
            pitch= pitch,
            yaw = yaw,
            hit = hit,
            alive = alive,
            counter = counter,
            posDistance = posDistance
        )
        gameDataset.add(data)
    }

    //scene setup
    suspend fun testapi() {
        // Create HttpClient with JSON support
        val client = HttpClient(CIO) {
            install(ContentNegotiation) {
                json(Json { prettyPrint = true; isLenient = true; ignoreUnknownKeys = true })
            }
        }
        // Prepare the data to send
        val dataToSend = gameDataset.last()
        if (sendcd >=0) {
            try {
                // Sending a POST request to the FastAPI server
                val postResponse: Action = client.post("http://127.0.0.1:8000/send/") {
                    contentType(ContentType.Application.Json)
                    setBody(dataToSend)  // Send GameData as request body
                }.body() // Extract the response body as GameData


                //println("POST Response: ${postResponse}")
                action = postResponse.action
                //println(dataToSend.spaceshipRotation.y)

            } catch (e: Exception) {
                errorCount++
                println("Error sending request: ${e.localizedMessage} - - - - - - ${errorCount}")
            } finally {
                // Close the client after use
                client.close()
            }
            sendcd=0
        }
        sendcd++

    }
    fun normalizeAngle(angle: Double, rangeStart: Double = -PI, rangeEnd: Double = PI): Double {
        val rangeWidth = rangeEnd - rangeStart
        return rangeStart + ((angle - rangeStart) % rangeWidth + rangeWidth) % rangeWidth
    }

    // Function to calculate the angular distance between two angles
    fun angularDistance(yaw1: Double, yaw2: Double): Double {
        val diff = yaw1 - yaw2
        return normalizeAngle(diff)
    }

    // Function to calculate the shortest distance between two yaw angles
    fun yawDistance(yaw1: Double, yaw2: Double): Double {
        // Normalize to [-PI, PI)
        val normalizedYaw1 = normalizeAngle(yaw1)
        val normalizedYaw2 = normalizeAngle(yaw2)

        // Compute the difference and wrap around
        val diff = normalizedYaw1 - normalizedYaw2
        return normalizeAngle(diff)
    }

    fun posDistance(v1: org.joml.Vector3f, v2: org.joml.Vector3f): Float {
        val dx = v2.x - v1.x
        val dy = v2.y - v1.y
        val dz = v2.z - v1.z

        return sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2))
    }

    fun normalize(vector: Vector3f): Vector3f {
        val length = Math.sqrt(
            (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).toDouble()
        ).toFloat()

        return if (length != 0f) {
            Vector3f(vector.x / length, vector.y / length, vector.z / length)
        } else {
            Vector3f(0f, 0f, 0f)  // Return a zero vector if length is 0 to avoid division by zero
        }
    }
    fun dotProduct(a: Vector3f, b: Vector3f): Float {
        return a.x * b.x + a.y * b.y + a.z * b.z
    }
    fun updateFPS() {
        val currentTime = System.currentTimeMillis()
        frames++

        if (currentTime - lastTime >= 1000) {
            fps = frames
            frames = 0
            lastTime = currentTime
            println("FPS: $fps") // Ausgabe der FPS in der Konsole
        }
    }
    init {

        camera = TronCamera()
        camera.rotate(-0.110865f,0f,0f)
        camera.translate(Vector3f1(0.0f, 0.0f, 14.0f))

        camera_fp = TronCamera()
        camera_fp.rotate(-1.57f,0f,0f)
        camera_fp.translate(Vector3f1(0.0f, 200f, -15f))

        enableDepthTest(GL_LESS) //Tiefentest, werden Pixel in der richtigen Reihenfolge gerendert
        //enableFaceCulling(GL_CCW, GL_FRONT)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); GLError.checkThrow() //schwarze hintergrundfarbe, alpha1.0f völlige deckkraft

        var raytex = Texture2D("assets/textures/raytex.png", true)
        val rayMaterial = Material(

            raytex,
            raytex,
            raytex,
            600.0f,
            Vector2f(1.0f, 1.0f)
        )

        val skyboxVertices = floatArrayOf(
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f, -1.0f,
            1.0f,  1.0f, -1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            1.0f, -1.0f,  1.0f
        )

        val vertexAttributesSkybox = arrayOf(VertexAttribute(3, 0, 3 * 4, 0))
        val indexData = IntArray(skyboxVertices.size / 3) { it }

        val skyboxMesh = Mesh(skyboxVertices, indexData, vertexAttributesSkybox)
        skyboxExp = Renderable(mutableListOf(skyboxMesh))
        skyboxMaterial2 = SkyboxMaterial(skyboxNewTexture)

        spaceship= ModelLoader.loadModel("assets/starsparrow/StarSparrow01.obj", 0f, Math.toRadians(180f), 0f)!!

        camera.parent = spaceship
        camera_fp.parent = spaceship
        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)

        skyboxExp.translate(spaceship.getWorldPosition())
        skyboxExp.scale(Vector3f1(5000f,5000f,5000f))



        Moon = ModelLoader.loadModel("assets/Moon_3D_Model/moon.obj", -1.5708f, 1.5708f, 0f)!!
        Moon.scale(Vector3f1(0.5f,0.5f,0.5f))
        Moon.translate(Vector3f1(-500f,1100f,0f))
        renderable.scale(Vector3f1(25.7f, 25.7f, 25.7f))
        pointLight5.parent=Moon


        //Laser
        var ras = loadOBJ("assets/models/model.obj",true,true)
        var raymesh = Mesh(ras.objects[0].meshes[0].vertexData,ras.objects[0].meshes[0].indexData,vertexAttributes,rayMaterial)
        ray = Renderable(mutableListOf(raymesh))
        ray.rotate(0f,1.5708f,0f)


        spotLight.rotate(Math.toRadians(-2f),0f,0f)
        spotLight.parent = spaceship
        ray.parent = spaceship

        pointLight4.parent = ray

        pointLight.parent = spaceship
/*
        for(i in 1..25)//random asteroid spawn
        {
            var rendertemp = ModelLoader.loadModel("assets/10464_Asteroid_L3.123c72035d71-abea-4a34-9131-5e9eeeffadcb/10464_Asteroid_v1_Iterations-2.obj", -1.5708f, 1.5708f, 0f)!!
            var ascale=Random().nextFloat(0.005f,0.01f)

            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-10000f,10000f),Random().nextFloat(0f,1f),Random().nextFloat(-10000f,10000f)))
            asteroidlist.add(rendertemp)
        }
*/
        astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)

        camera.updateViewMatrix()
        camera.updateProjectionMatrix()
        val cameraPosition = camera.getWorldPosition()
        val viewMatrixSkybox = Matrix4f(camera.viewMatrix)
        viewMatrixSkybox.m30(0f)
        viewMatrixSkybox.m31(0f)
        viewMatrixSkybox.m32(0f)
        val viewMatrixSkyboxNoTranslation = Matrix4f().lookAt(cameraPosition, org.joml.Vector3f(0f, 0f, 0f), org.joml.Vector3f(0f, 1f, 0f))
        skyboxShader.setUniform("view", viewMatrixSkyboxNoTranslation)
        skyboxShader.setUniform("projection", camera.projectionMatrix)

    }


    fun render(dt: Float, t: Float)= runBlocking {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        renderSkybox()
        staticShader.use()
        staticShader.setUniform("gammaValue", desiredGammaValue)

        camera.updateViewMatrix()
        camera.updateProjectionMatrix()
        camera.bind(staticShader)

        updateFPS()
        pointLight.bind(staticShader,camera.getCalculateViewMatrix(),0)
        spotLight.bind(staticShader, camera.getCalculateViewMatrix())

        spaceship.render(staticShader, Vector3f1(1.2f,1.2f,1.2f))
        Moon.render(staticShader, Vector3f1(1f,1f,1f))


        if(shoot) {
            ray.render(staticShader, Vector3f1(10f, 0.1f, 0.1f))
        }


        if(pause)
        {
            for(i in 0..asteroidlist.lastIndex-1)
            {

                asteroidlist[i].translate(spaceship.getWorldPosition().sub(asteroidlist[i].getWorldPosition(),Vector3f1()).mul(Vector3f1(vmaxa,vmaxa,vmaxa)))

                asteroidlist[i].render(staticShader,Vector3f1(0.2f,0.2f,0.2f))
            }
            for(i in 0..asteroidlist2.lastIndex)
            {

                asteroidlist2[i].translate(spaceship.getWorldPosition().sub(asteroidlist2[i].getWorldPosition(),Vector3f1()).mul(Vector3f1(vmaxa2,vmaxa2,vmaxa2)))

                asteroidlist2[i].render(staticShader,Vector3f1(0.2f,0.15f,0.15f))
            }


            score+=1f
           /* if(score.toInt()%100==0)
            {
                //println(vmaxa)
                vmaxa*=1.1f
                vmaxa2*=1.01f
            }*/
        /*if(score.toInt()%100==0){


            astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
            var rendertemp = Renderable(mutableListOf(astmesh))


            var ascale=Random().nextFloat(6f,10f)

            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(0f,0.1f),Random().nextFloat(-100f,100f)))
            asteroidlist2.add(rendertemp)

        }*/
        }

        else{
            pointLight = PointLight(Vector3f1(0f, 5f, 0f), Vector3f1(0.11f, 0.11f, 0.11f))
            pointLight.parent = spaceship
        }

        for(i in 0 .. asteroidlist2.lastIndex) {


            var direction = Vector3f1(
                asteroidlist2[i].getWorldPosition().x - spaceship.getWorldPosition().x.toFloat(),
                asteroidlist2[i].getWorldPosition().y- spaceship.getWorldPosition().y.toFloat(),
                asteroidlist2[i].getWorldPosition().z - spaceship.getWorldPosition().z.toFloat()
            )
            val directionLocal = spaceship.transformToLocal(direction)
            directionLocal.normalize()
            val yaw = atan2(directionLocal.x, -directionLocal.z)
            val pitch = atan2(directionLocal.y, sqrt(directionLocal.x * directionLocal.x + directionLocal.z * directionLocal.z))

            var yawdistances = yawDistance(yaw.toDouble(),spaceship.getRotation().y.toDouble())
            var pitchdistances= yawDistance(pitch.toDouble(),spaceship.getRotation().x.toDouble()-1.5707964f)
            //Train
            if(predict==false)
            if ((abs(yawdistances)+abs(pitchdistances)< previousdis )||50>spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())) {
                previousdis = abs(yawdistances) + abs(pitchdistances)
                cdasteroid = Scene.Vector3f(
                    asteroidlist2[i].getWorldPosition().x,
                    asteroidlist2[i].getWorldPosition().y,
                    asteroidlist2[i].getWorldPosition().z
                )
                shortyaw=yawdistances.toFloat()
                shortpitch=pitchdistances.toFloat()

                if(spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())<50 && spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())<prevleng)
                {
                    cdasteroid = Scene.Vector3f(
                        asteroidlist2[i].getWorldPosition().x,
                        asteroidlist2[i].getWorldPosition().y,
                        asteroidlist2[i].getWorldPosition().z
                    )
                    prevleng=spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())
                    shortyaw=yawdistances.toFloat()
                    shortpitch=pitchdistances.toFloat()
                    break
                }
            }
            //predict
            if ((abs(yawdistances)+abs(pitchdistances)< previousdis &&600>spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition()))||50>spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())&&predict==true) {
                previousdis = abs(yawdistances) + abs(pitchdistances)
                cdasteroid = Scene.Vector3f(
                    asteroidlist2[i].getWorldPosition().x,
                    asteroidlist2[i].getWorldPosition().y,
                    asteroidlist2[i].getWorldPosition().z
                )
                shortyaw=yawdistances.toFloat()
                shortpitch=pitchdistances.toFloat()

                if(spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())<50 && spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())<prevleng)
                {
                    cdasteroid = Scene.Vector3f(
                        asteroidlist2[i].getWorldPosition().x,
                        asteroidlist2[i].getWorldPosition().y,
                        asteroidlist2[i].getWorldPosition().z
                    )
                    prevleng=spaceship.getWorldPosition().distance(asteroidlist2[i].getWorldPosition())
                    shortyaw=yawdistances.toFloat()
                    shortpitch=pitchdistances.toFloat()
                    break
                }
            }

        }

        var direction=Vector3f1(cAsteroid.x-spaceship.getWorldPosition().x.toFloat() , cAsteroid.y-spaceship.getWorldPosition().y.toFloat() ,cAsteroid.z-spaceship.getWorldPosition().z.toFloat())
        val directionLocal = spaceship.transformToLocal(direction)
        directionLocal.normalize()
        val yaw = atan2(directionLocal.x, -directionLocal.z)
        val pitch = atan2(directionLocal.y, sqrt(directionLocal.x * directionLocal.x + directionLocal.z * directionLocal.z))
        var hit = checkCollisionAsteroid()
        var alive = checkCollisionSpaceship()


        var yawdistance = yawDistance(yaw.toDouble(),spaceship.getRotation().y.toDouble())
        var pitchdistance= yawDistance(pitch.toDouble(),spaceship.getRotation().x.toDouble()-1.5707964f)
        var distance = spaceship.getWorldPosition().distance(Vector3f1(cAsteroid.x,cAsteroid.y,cAsteroid.z))
        //println(abs(pitchdistance)+abs(yawdistance))
        //println(pitchdistance)
        //ray.render(staticShader,Vector3f1(1f,1f,1f))
        //ray.parent = spaceship
        //ray.setRotation(0f,-yaw.toFloat()+1.5707f,pitch.toFloat())
        //spaceship.setRotation(pitch.toFloat(),yaw.toFloat(),0f)
        //ray.setRotation(0f,-spaceship.getRotation().y-1.5707964f,-spaceship.getRotation().x+1.570796f)
        //collectData(Vector3f1(spaceship.getRotation().x,spaceship.getRotation().y,spaceship.getRotation().z),pitchdistance.toFloat(), yawdistance.toFloat(), hit, alive, counter, distance)//score,ChronoUnit.MILLIS.between(starttime,LocalDateTime.now())/1000f)
        //println("yaw"+shortyaw+"pit"+shortpitch+"  leng"+prevleng+" count "+counter)
        collectData(Vector3f1(spaceship.getRotation().x,spaceship.getRotation().y,spaceship.getRotation().z),shortpitch.toFloat(), shortyaw.toFloat(), hit, alive,counter.toFloat(), distance)//score,ChronoUnit.MILLIS.between(starttime,LocalDateTime.now())/1000f)
        testapi()
        previousdis=java.lang.Math.PI*2
        if(counter%49==0&&counter>0&&predict==true){
        //if(counter==15){
            print(counter)
            astSpawnCounter=0
            //counter=0
            //setSpaceshipPositionToStart()
        }
        if(counter>=15&&predict==false){
            print(counter)
            astSpawnCounter=0
            counter=0
            setSpaceshipPositionToStart()
        }
        if(counter>=1500){
            setSpaceshipPositionToStart()
        }
        if(shoot) {
            ray.render(staticShader, Vector3f1(10f, 0.1f, 0.1f))
        }
        //todo graphics koppeln mit predict/train all in one setup
    }


    fun update(dt: Float, t: Float) {
        //RL-Controls
        /*if (!shoot) {
            ray.rayToSpaceship(spaceship.getModelMatrix())
        }*/

        //astHit = checkCollisionAsteroid()
        //println(ray.getWorldPosition())

        when(action) {
            0 -> {spaceship.rotate(0.0f, -0.009f, 0.0f) } //D
            1 -> {spaceship.rotate(0.0f, 0.009f, 0.00f) }//A
            2 -> shoot=true                                             //P
            3 -> spaceship.rotate(-0.009f, 0f, 0f)  //pitch down
            4 -> spaceship.rotate(0.009f, 0f, 0f)    //pitch up
            5 -> spaceship.translate(Vector3f1(0f, 0f, 0.25f))  //S  z=0.2f
            6 -> spaceship.translate(Vector3f1(0f, 0f, speed))    //W
            10 -> setSpaceshipPositionToStart()                    //Game reset
        }

        if(spaceship.getRotation().z>0)
            spaceship.rotate(0f,0f,-spaceship.getRotation().z)
        else
            spaceship.rotate(0f,0f,abs(spaceship.getRotation().z))

        if(shoot==true){
            ray.render(staticShader, Vector3f1(10f,0.1f,0.1f))
            pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(5.0f,0.0f,0.0f))
            pointLight4.parent=ray
            pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
            ray.translate(Vector3f1(7f,0f,0f))
            rayl++

            if(rayl>=90){
                ray.translate(Vector3f1(-630f,0f,0f))
                pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(0.0f,0.0f,0.0f))
                pointLight4.parent=ray
                pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
                //ray.rotate(0f,rayrotang,0f)
                ray.parent=spaceship
                rayl=0
                shoot= false
            }
        }
        /*if(ray.getRotation().x>0)
            ray.rotate(-ray.getRotation().x,0f,0f)
        else
            ray.rotate(abs(spaceship.getRotation().x),0f,0f)
        */
        collisionCheckTimer += dt
        checkCollisionSpaceship()

        if (collisionCheckTimer >= collisionCheckInterval) {
            checkCollisionSpaceship()
            collisionCheckTimer = 0f // Setze den Timer zurück
        }

        action=11

        if (window.getKeyState(GLFW_KEY_O) == true) {
            action =10

        }
        if (window.getKeyState(GLFW_KEY_W) == true) {
            inputkey="W"
            val forward = Vector3f1(0f, 0f, speed)
            spaceship.translate(forward)
        }
        if (window.getKeyState(GLFW_KEY_D) == true) {
            inputkey="D"
            if(cammode<=1f){
                spaceship.rotate(0.0f, -0.01f, 0.0f)
                }
            else{
                spaceship.rotate(0.0f, -0.01f, 0.0f)
            }
        }
        if (window.getKeyState(GLFW_KEY_S) == true) {
            inputkey="S"
            val backward = Vector3f1(0f, 0f, 0.25f)
            spaceship.translate(backward)
        }
        if (window.getKeyState(GLFW_KEY_A) == true) {
            inputkey="A"
            if(cammode<=1f){
                spaceship.rotate(0.0f, 0.01f, 0.00f)

                }
            else{
                spaceship.rotate(0.0f, 0.01f, 0.0f)

            }
        }
        if(window.getKeyState(GLFW_KEY_F)==true)
        {
            spaceship.rotate(0.01f,0f,0f)
        }
        if(window.getKeyState(GLFW_KEY_R)==true)
        {
            spaceship.rotate(-0.01f,0f,0f)
        }
        if(window.getKeyState(GLFW_KEY_E)==true)
        {
            spaceship.rotate(0f,0f,0.01f)
        }
        if(window.getKeyState(GLFW_KEY_Q)==true)
        {
            spaceship.rotate(0f,0f,-0.01f)
        }
        if (window.getKeyState(GLFW_KEY_L) == true) {
            tempshader=tempshader+0.1f
            if(tempshader>=3f){
                tempshader=0f
            }
            staticShader.setUniform("shader",tempshader)
        }

        if (window.getKeyState(GLFW_KEY_P) == true) {
            shoot=true
            inputkey="P"
            checkCollisionAsteroid()
        }
        if (window.getKeyState(GLFW_KEY_C) == true) {
            cammode=cammode+0.1f
            if (cammode>=3f) {
                    cammode=0.0f
                if(cammode>=0.0f&&cammode<0.1){
                    camera.rotate(1.57f,0f,0f)
                    camera.translate(Vector3f1(0f,-200f,0f))
                    camera.parent = spaceship
                }
                }
            if(cammode>=2f) {

                if(cammode>2f&&cammode<=2.1){
                    camera.translate(Vector3f1(0f,200f,0f))
                    camera.rotate(-1.57f, 0f, 0f)
                }
            }

        }
        if (window.getKeyState(GLFW_KEY_B) == true) {

            if(camselect==0f) {
                camera.translate(Vector3f1(0f,1f,-13f))
                camselect=1f
            }
            else{
                camera.translate(Vector3f1(0f,-1f,13f))
                camselect=0f
            }
        }
        if (window.getKeyState(GLFW_KEY_LEFT_SHIFT) == true) {

            if(speed>=-0.5f)
            speed-=0.003f

        }
        if (window.getKeyState(GLFW_KEY_LEFT_SHIFT) == false) {

            if(speed<=-0.1f)
                speed+=0.01f
        }

        if (cammode>1&&cammode<=2) {
            if (window.getKeyState(GLFW_KEY_UP) == true) {
                camera_fp.translate(Vector3f1(0f,0f,-0.1f))
            }
            if (window.getKeyState(GLFW_KEY_DOWN) == true) {
                camera_fp.translate(Vector3f1(0f,0f,0.1f))
            }
            if (window.getKeyState(GLFW_KEY_LEFT) == true) {
                camera_fp.translate(Vector3f1(0.1f,0f,0f))
            }
            if (window.getKeyState(GLFW_KEY_RIGHT) == true) {
                camera_fp.translate(Vector3f1(-0.1f,0f,0f))
            }
        }

        checkCollisionSpaceship()
    }

    private fun checkCollisionSpaceship(): Boolean {
        val spaceshipPosition = spaceship.getWorldPosition()
        val iterator = asteroidlist2.iterator()

        while (iterator.hasNext()) {
            val asteroid = iterator.next()
            val asteroidPosition = asteroid.getWorldPosition()
                //.add(Vector3f1(0f,6f,0f))

            val distance = spaceshipPosition.distance(asteroidPosition)
            nextAstDistance = distance

            if (distance < 14.0f) {
                iterator.remove()
                asteroid.cleanup()
                setSpaceshipPositionToStart()
                return false
            }

            if(distance< spaceshipPosition.distance(Vector3f1(cAsteroid.x,cAsteroid.y,cAsteroid.z)))
            {
                cAsteroid.x = asteroidPosition.x
                cAsteroid.y = asteroidPosition.y+3
                cAsteroid.z = asteroidPosition.z
            }
        }
        return true
    }

    fun renderSkybox() {
        glDepthMask(false)
        glDepthFunc(GL_LEQUAL)
        skyboxShader.use()
        val viewMatrix = Matrix4f(camera.viewMatrix).m30(0f).m31(0f).m32(0f)
        skyboxShader.setUniform("view", viewMatrix)
        skyboxShader.setUniform("projection", camera.projectionMatrix)
        skyboxMaterial2.bind(skyboxShader)
        skyboxNewTexture.bind(0)
        skyboxShader.setUniform("skybox", 0)
        skyboxExp.render(skyboxShader, Vector3f1(1f, 1f, 1f))
        skyboxNewTexture.unbind()
        glDepthMask(true)
        glDepthFunc(GL_LESS)
    }



    
    private fun setSpaceshipPositionToStart() {
        if(predict==false){
        for (renderable in asteroidlist2) {
            renderable.cleanup()
        }
        asteroidlist2.clear()
        //astdiff.cleanup()
        //astemit.cleanup()
        //astspec.cleanup()


        //HOHE RAM AUSLASTUNG
        //GL11.glDeleteTextures(0)          //fine
        GL11.glDeleteTextures(1)    //bad
        GL11.glDeleteTextures(2)    //bad
        GL11.glDeleteTextures(3)    //bad
        GL11.glDeleteTextures(4)    //bad
        GL11.glDeleteTextures(5)    //bad

        glDeleteFramebuffers(0)
        glDeleteFramebuffers(1)
        glDeleteFramebuffers(2)
        glDeleteFramebuffers(3)


        spaceship.cleanup()
        ray.cleanup()
        //Moon.cleanup()
        //skyboxExp.cleanup()
        //staticShader.cleanup()
        //skyboxShader.cleanup()
        //glBindBuffer(GL_ARRAY_BUFFER, 0)
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDeleteBuffers(0)
        glDeleteBuffers(1)
        glDeleteBuffers(2)
        //glBindVertexArray(0)
        glDeleteVertexArrays(0)
        glDeleteVertexArrays(1)
        glDeleteVertexArrays(2)

        //glUseProgram(0)
        //glDeleteProgram(staticShader.programID)
        //glDeleteProgram(skyboxShader.programID)

        System.gc()
        }
        astSpawnCounter = 0
        counter = 0
        spaceship.cleanup()
        spaceship= ModelLoader.loadModel("assets/starsparrow/StarSparrow01.obj", 0f, Math.toRadians(180f), 0f)!!
        camera.parent = spaceship
        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)
        asteroidlist.clear()
        asteroidlist2.clear()
        spotLight.parent = spaceship
        ray.parent = spaceship
        pointLight4.parent = ray
        pointLight.parent = spaceship
        score=0f
        vmaxa2=0.00017f
        //spaceship.rotate(0.0f,Random().nextFloat(-3.141f,3.141f),0f)
        println("reset........................................................................${spaceship.getRotation()}")
        clean()
        cAsteroid=Vector3f(10000f,10000f,10000f)
        cdasteroid=Vector3f(10000f,10000f,10000f)
        previousdis=Math.PI*2
        prevleng=10000f
        astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
        var rendertemp = Renderable(mutableListOf(astmesh))
        var ascale=7f
        rendertemp.scale(Vector3f1(ascale,ascale,ascale))
        if(predict==false) {
            rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(-100f,100f),Random().nextFloat(-100f,100f)))
            asteroidlist2.add(rendertemp)
        }
       /* for(i in 1..15)//random asteroid spawn
        {
            astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
            var ascale=7f
            var rendertemp = Renderable(mutableListOf(astmesh))
            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-120f,120f),Random().nextFloat(-120f,120f),Random().nextFloat(-120f,120f)))
            asteroidlist2.add(rendertemp)
        }*/
        if(predict==true) {
            for (i in 1..15)//set asteroid spawn
            {
                astmesh = Mesh(
                    astobj.objects[0].meshes[0].vertexData,
                    astobj.objects[0].meshes[0].indexData,
                    vertexAttributes,
                    astmat
                )
                var ascale = 7f
                var rendertemp = Renderable(mutableListOf(astmesh))
                rendertemp.scale(Vector3f1(ascale, ascale, ascale))
                rendertemp.translate(astSpawns[i])
                asteroidlist2.add(rendertemp)
            }
        }
    }

    private fun checkCollisionAsteroid(): Boolean {
        val shotPosition = ray.getWorldPosition()
        val iterator2 = asteroidlist2.iterator()
        while (iterator2.hasNext()) {
            val asteroid = iterator2.next()

            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,3f,0f))

            val distance = shotPosition.distance(asteroidPosition)
            if (distance < 12.0f) {


                iterator2.remove()
                cAsteroid=Vector3f(10000f,10000f,10000f)
                cdasteroid=Vector3f(10000f,10000f,10000f)
                previousdis=Math.PI*2
                prevleng=10000f
                score+=500f
                astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
                var rendertemp = Renderable(mutableListOf(astmesh))


                var ascale=7f

                rendertemp.scale(Vector3f1(ascale,ascale,ascale))
                if(predict==false) {
                    rendertemp.translate(
                        Vector3f1(
                            Random().nextFloat(-100f, 100f),
                            Random().nextFloat(-100f, 100f),
                            Random().nextFloat(-100f, 100f)
                        )
                    )
                }
                //AstSpawnList conmparing models
                astSpawnCounter++
                counter++
                if(predict==true) {
                    rendertemp.translate(astSpawns[astSpawnCounter])
                }
                println("Ast number: " + astSpawnCounter + " spawned...at"+rendertemp.getPosition())
                asteroidlist2.add(rendertemp)
                return true
            }
        }
        return false
    }
    fun onKey(key: Int, scancode: Int, action: Int, mode: Int) {}

    fun onMouseMove(xpos: Double, ypos: Double) {
        val x_speed = (xpos - window.windowWidth / 2.0).toFloat() * 0.002f
        val y_speed = (ypos - window.windowHeight / 2.0).toFloat() * 0.002f
        glfwSetCursorPos(window.m_window, window.windowWidth / 2.0, window.windowHeight / 2.0)
        if (cammode > 1 && cammode <= 2) {
            camera.rotateAroundPoint(0f, -x_speed, 0f, renderable.getWorldPosition())
        }
    }

    fun onMouseButton(button: Int, action: Int, mode: Int) {
    }

    fun onMouseScroll(xoffset: Double, yoffset: Double) {
        if (yoffset < 0)
        {
            camera.translate(Vector3f1(0.0f, 0.0f, 0.5f))
        }
        if (yoffset > 0)
        {
            camera.translate(Vector3f1(0.0f, 0.0f, -0.5f))
        }
    }

    fun clean() {
        for (asteroid in asteroidlist2) {
            asteroid.cleanup()
            cAsteroid=Vector3f(10000f,10000f,10000f)
        }
    }

    private fun setSpaceshipPosition(position: Vector3f1) {
        currentSpaceshipPosition.set(position)
        spaceship.translate(currentSpaceshipPosition)
    }
    /**
     * enables culling of specified faces
     * orientation: ordering of the vertices to define the front face
     * faceToCull: specifies the face, that will be culled (back, front)
     * Please read the docs for accepted parameters
     */
    fun enableFaceCulling(orientation: Int, faceToCull: Int){
        glEnable(GL_CULL_FACE); GLError.checkThrow()
        glFrontFace(orientation); GLError.checkThrow()
        glCullFace(faceToCull); GLError.checkThrow()
    }
    /**
     * enables depth test
     * comparisonSpecs: specifies the comparison that takes place during the depth buffer test
     * Please read the docs for accepted parameters
     */
    fun enableDepthTest(comparisonSpecs: Int){
        glEnable(GL_DEPTH_TEST); GLError.checkThrow()
        glDepthFunc(comparisonSpecs); GLError.checkThrow()
    }
}