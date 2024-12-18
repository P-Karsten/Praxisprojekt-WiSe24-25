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
import org.joml.Math
import org.joml.Math.atan2
import org.joml.Matrix4f
import org.joml.Vector2f
import org.joml.Vector3fc
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.opengl.GL30.*
import java.io.File
import java.nio.FloatBuffer
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import java.util.*
import kotlin.math.PI
import kotlin.math.sqrt
import org.joml.Vector3f as Vector3f1

/**
 * Created 29.03.2023.
 */
class Scene(private val window: GameWindow) {
    private val staticShader: ShaderProgram = ShaderProgram("assets/shaders/tron_vert.glsl", "assets/shaders/tron_frag.glsl")
    private val skyboxShader: ShaderProgram = ShaderProgram("assets/shaders/skybox_vert.glsl", "assets/shaders/skybox_frag.glsl")

    data class DataModel(val id: Int, val message: String)

    private var camera: TronCamera
    private var camera_fp: TronCamera
    var light_on = false
    var light_last = 0f
    val light_int = 0.35f
    var b_menu = false
    var astmesh: Mesh
    var vmaxa=0.01f
    var vmaxa2=0.002f
    var shoot2=false
    var score =0f
    var pause =true
    var rayl= 0
    var inputkey= ""
    var cAsteroid=Vector3f(10000f,10000f,10000f)
    var sendcd =0
    var speed = -0.1f
    var shoot =false
    var cammode =0f
    var camselect=0f
    var tempshader=1f
    var asteroidlist = mutableListOf<Renderable>()
    var asteroidlist2 = mutableListOf<Renderable>()
    var meshlist = mutableListOf<Mesh>()
    var renderable = Renderable(meshlist)
    var renderable2 = Renderable(meshlist)
    var Moon = Renderable(meshlist)
    var Moon2 = Renderable(meshlist)
    var skyboxExp = Renderable(meshlist)
    var spaceship = Renderable(meshlist)
    var game_over = Renderable(meshlist)
    var end_game = Renderable(meshlist)
    var reset_game = Renderable(meshlist)
    val starttime = LocalDateTime.now()
    var ray = Renderable(meshlist)
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
    var laserDir = 0f
    var laserDirTmp = 0f

    var counter = 0


    val desiredGammaValue = 2.2f // Beispielwert für den gewünschten Gammawert

    val lightPosition = Vector3f1(0f, 5f, 0f) // Anpassen der Lichtposition
    val lightColor = Vector3f1(0.11f, 0.11f, 0.11f) // Anpassen der Lichtfarbe (hier: Weiß)

    var pointLight = PointLight(lightPosition, lightColor)

    var pointLight2 = PointLight(Vector3f1(0f,1f,0f), Vector3f1(0.0f,0.0f,0.0f))

    var pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(0.0f,0.0f,0.0f))
    val spotLight = SpotLight(Vector3f1(0f,2f,0f), Vector3f1(500f,500f,500f),Math.toRadians(1f),org.joml.Math.toRadians(2f))
    val pointLight5 = PointLight(Vector3f1(0f,0f,0f), Vector3f1(500.0f,500.0f,500.0f))

    private val initialSpaceshipPosition = Vector3f1(0.0f, 1.0f, 0.0f)
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
        val spaceshipRotation: Vector3f,
        val yaw : Float,
        val hit: Boolean,
        val alive: Boolean,
        val hitsCounter: Int,


    )
    var action= 6
    var gameDataset = mutableListOf<GameData>()

    fun collectData(
        spaceshipPos: Vector3f1,
        spaceshipRotation: Vector3f,
        yaw : Float,
        hit: Boolean,
        alive: Boolean,
        hitsCounter: Int

    ) {                             //rotation spaceship
        val data = GameData(
            spaceshipPosition = listOf(spaceshipPos.x, spaceshipPos.y, spaceshipPos.z),
            spaceshipRotation = spaceship.getRotation(),
            yaw = yaw,
            hit = hit,
            alive = alive,
            hitsCounter = hitsCounter
        )
        gameDataset.add(data)

        //println(gameDataset)
    }
    fun saveDataset(dataset: List<GameData>, filename: String) {
        val jsonData = Json.encodeToString(gameDataset)
       File(filename).writeText(jsonData)
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


                //println("POST Response: ${postResponse}"
                action = postResponse.action
                //println(dataToSend.spaceshipRotation.y)

            } catch (e: Exception) {
                //println("Error sending request: ${e.localizedMessage}")
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
    init {

        camera = TronCamera()
        camera.rotate(-0.110865f,0f,0f)
        camera.translate(Vector3f1(0.0f, 0.0f, 14.0f))

        camera_fp = TronCamera()
        camera_fp.rotate(-1.57f,0f,0f)
        camera_fp.translate(Vector3f1(0.0f, 200f, -15f))

        //Menu
        val end = loadOBJ("assets/models/menu/beenden.obj", true, true)
        val ga_ov = loadOBJ("assets/models/menu/game_over.obj", true, true)
        val reset = loadOBJ("assets/models/menu/neustart.obj", true, true)

        enableDepthTest(GL_LESS) //Tiefentest, werden Pixel in der richtigen Reihenfolge gerendert
        //enableFaceCulling(GL_CCW, GL_FRONT)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); GLError.checkThrow() //schwarze hintergrundfarbe, alpha1.0f völlige deckkraft

        val spec = Texture2D("assets/textures/ground_diff.png", true)
        val diff = Texture2D("assets/textures/ground_diff.png", true)
        var raytex = Texture2D("assets/textures/raytex.png", true)
        var fontMat = Texture2D("assets/textures/menu_font.png", true)


        fontMat.bind(0)
        fontMat.setTexParams(GL_CLAMP, GL_CLAMP, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR)
        fontMat.unbind()


        val rayMaterial = Material(

            raytex,
            raytex,
            raytex,
            600.0f,
            Vector2f(1.0f, 1.0f)
        )
        val FontMaterial = Material(

                diff,
                fontMat,
                spec,
                60.0f,
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
        //skyboxExp.render(skyboxShader, Vector3f1(500.0f, 500.0f, 500.0f))


        //Menu
        Moon2 = ModelLoader.loadModel("assets/Moon_3D_Model/moon.obj", -1.5708f, 1.5708f, 0f)!!
        var menu_overMesh = Mesh(ga_ov.objects[0].meshes[0].vertexData,ga_ov.objects[0].meshes[0].indexData,vertexAttributes, FontMaterial)
        game_over = Renderable(mutableListOf(menu_overMesh))
        var menu_reset = Mesh(reset.objects[0].meshes[0].vertexData,reset.objects[0].meshes[0].indexData,vertexAttributes, FontMaterial)
        reset_game = Renderable(mutableListOf(menu_reset))
        var menu_end = Mesh(end.objects[0].meshes[0].vertexData,end.objects[0].meshes[0].indexData,vertexAttributes, FontMaterial)
        end_game = Renderable(mutableListOf(menu_end))


        spaceship= ModelLoader.loadModel("assets/starsparrow/StarSparrow01.obj", 0f, Math.toRadians(180f), 0f)!!

        camera.parent = spaceship
        camera_fp.parent = spaceship
        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)



        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)


        skyboxExp.translate(spaceship.getWorldPosition())
        skyboxExp.scale(Vector3f1(5000f,5000f,5000f))



        //Background
        Moon2.translate(Vector3f1(0f,0f,-100f))
        Moon2.scale(Vector3f1(4f,4f,4f))
        Moon2.translate(Vector3f1(0f,1000f,0f))
        //Game Over
        game_over.translate(Vector3f1(0f,17f,-40f))
        game_over.scale(Vector3f1(22f,22f,22f))
        game_over.translate(Vector3f1(0f,250f,0f))
        //Reset
        reset_game.translate(Vector3f1(0f,0f,-40f))
        reset_game.scale(Vector3f1(15f,15f,15f))
        reset_game.translate(Vector3f1(0f,250f,0f))
        //End
        end_game.translate(Vector3f1(0f,-15f,-40f))
        end_game.scale(Vector3f1(15f,15f,15f))
        end_game.translate(Vector3f1(0f,250f,0f))


        Moon = ModelLoader.loadModel("assets/Moon_3D_Model/moon.obj", -1.5708f, 1.5708f, 0f)!!
        Moon.scale(Vector3f1(0.5f,0.5f,0.5f))
        Moon.translate(Vector3f1(-500f,1100f,0f))
        renderable.scale(Vector3f1(25.7f, 25.7f, 25.7f))
        pointLight5.parent=Moon


        //Laser
        var ras = loadOBJ("assets/models/model.obj",true,true)
        var raymesh = Mesh(ras.objects[0].meshes[0].vertexData,ras.objects[0].meshes[0].indexData,vertexAttributes,rayMaterial)
        ray = Renderable(mutableListOf(raymesh))
        ray.scale(Vector3f1(5f,5f,5f))
        ray.translate(Vector3f1(0f,0f,0f))
        ray.rotate(0f,1.5708f,0f)


        spotLight.rotate(Math.toRadians(-2f),0f,0f)
        spotLight.parent = spaceship
        ray.parent = spaceship
        //ray.translate(initialSpaceshipPosition)

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
        //println("Camera Position: $cameraPosition")
        //println("View Matrix Skybox: $viewMatrixSkybox")
        //println("Projection Matrix: ${camera.projectionMatrix}")
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


        pointLight.bind(staticShader,camera.getCalculateViewMatrix(),0)

        pointLight5.bind(staticShader,camera.getCalculateViewMatrix(),4)

        spotLight.bind(staticShader, camera.getCalculateViewMatrix())

        //Menu
        //menu_backg.render(staticShader, Vector3f1(2f,2f,2f))
        game_over.render(staticShader, Vector3f1(2f,2f,2f))
        reset_game.render(staticShader, Vector3f1(2f,1f,2f))
        end_game.render(staticShader, Vector3f1(2f,2f,2f))

        spaceship.render(staticShader, Vector3f1(1.2f,1.2f,1.2f))
        Moon.render(staticShader, Vector3f1(1f,1f,1f))
        Moon2.render(staticShader, Vector3f1(1f,1f,1f))

        if(shoot==true){
            if (rayl==0) {
                ray.parent = null
                laserDir = spaceship.getRotation().y
                laserDirTmp = ray.getRotation().y
                var a = yawDistance(laserDir.toDouble(), laserDirTmp.toDouble())
                ray.rotate(0f, -a.toFloat() + 1.5708f , 0f)
            }

            ray.render(staticShader, Vector3f1(10f,0.1f,0.1f))
            pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(5.0f,0.0f,0.0f))
            pointLight4.parent=ray
            pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
            ray.translate(Vector3f1(3f,0f,0f))
            rayl++

            if(rayl>=80){
                ray.translate(Vector3f1(-240f,0f,0f))
                pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(0.0f,0.0f,0.0f))
                pointLight4.parent=ray
                pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
                rayl=0
                shoot= false
                ray.parent = spaceship
            }
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

        if(spaceship.getWorldPosition().x>=1700f||spaceship.getWorldPosition().y>=1700f||spaceship.getWorldPosition().z>=1700f||spaceship.getWorldPosition().x<=-1700f||spaceship.getWorldPosition().y<=-1700f||spaceship.getWorldPosition().z<=-1700f) {
            //Flashing
            light_last += dt

            if (light_last >= light_int){
                light_on = !light_on
                light_last = 0f

                if (light_on) {
                    pointLight = PointLight(Vector3f1(0f, 5f, 0f), Vector3f1(1f, 0f, 0f))
                    pointLight.parent = spaceship
                }
                else {
                    pointLight.parent = null
                }
            }


        }
        else{
            pointLight = PointLight(Vector3f1(0f, 5f, 0f), Vector3f1(0.11f, 0.11f, 0.11f))
            pointLight.parent = spaceship
        }
        if(spaceship.getWorldPosition().x>=1800f||spaceship.getWorldPosition().y>=1800f||spaceship.getWorldPosition().z>=1800f||spaceship.getWorldPosition().x<=-1800f||spaceship.getWorldPosition().y<=-1800f||spaceship.getWorldPosition().z<=-1800f) {
            setSpaceshipPositionToStart()
        }

        var direction=Vector3f(spaceship.getWorldPosition().x-cAsteroid.x , spaceship.getWorldPosition().y-cAsteroid.y ,spaceship.getWorldPosition().z-cAsteroid.z )
        var yaw = (atan2(direction.x,direction.z).toDouble())*-1
        var pitch = atan2(direction.y, sqrt(direction.x * direction.x + direction.z * direction.z)).toDouble()
        var hit = checkCollisionAsteroid()
        var alive = checkCollisionSpaceship()

        if (hit) {
            counter++
        }
        //println(cAsteroid)
        //println(direction)
        //println("pitch"+pitch+"yaw"+yaw)
        //println("spaceshiprot"+(spaceship.getRotation().y.toDouble()))
        collectData(spaceship.getWorldPosition(),spaceship.getRotation(), yaw.toFloat(), hit, alive, counter)//score,ChronoUnit.MILLIS.between(starttime,LocalDateTime.now())/1000f)
        testapi()
    }


    fun update(dt: Float, t: Float) {
        //RL-Controls
        when(action) {
            0 -> {spaceship.rotate(0.0f, -0.01f, 0.0f) } //D
            1 -> {spaceship.rotate(0.0f, 0.01f, 0.00f) }//A
            4 -> spaceship.translate(Vector3f1(0f, 0f, 0.0f))  //S  z=0.2f
            3 -> spaceship.translate(Vector3f1(0f, 0f, speed))    //W
            2 -> shoot=true                                             //P
            10 -> setSpaceshipPositionToStart()                    //Game reset
        }
        action=6
        //spaceship.translate(Vector3f1(0f, 0f, speed))
        collisionCheckTimer += dt
        checkCollisionSpaceship()
        if (b_menu ==true){
            checkCollisionMenu()
            //saveDataset(gameDataset,"testdata1")
        }
        if(shoot==true)
            checkCollisionAsteroid()
        if (collisionCheckTimer >= collisionCheckInterval) {
            checkCollisionSpaceship()
            if(shoot==true)
            checkCollisionAsteroid()
            //checkCollisionMenu()
            collisionCheckTimer = 0f // Setze den Timer zurück
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
            val backward = Vector3f1(0f, 0f, 0.2f)
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
        if (window.getKeyState(GLFW_KEY_L) == true) {
            tempshader=tempshader+0.1f
            if(tempshader>=3f){
                tempshader=0f
            }
            staticShader.setUniform("shader",tempshader)
        }
        if (window.getKeyState(GLFW_KEY_N) == true) {
            //pause
        }
        if (window.getKeyState(GLFW_KEY_P) == true) {
            shoot=true
            inputkey="P"
            checkCollisionAsteroid()
        }
        if (window.getKeyState(GLFW_KEY_C) == true) {
            cammode=cammode+0.1f
            println(cammode)
            if (cammode>=3f) {
                    //camera.translate(Vector3f1(0f,-200f,0f))
                    cammode=0.0f
                if(cammode>=0.0f&&cammode<0.1){
                    camera.rotate(1.57f,0f,0f)
                    camera.translate(Vector3f1(0f,-200f,0f))
                    camera.parent = spaceship
                }
                    //camera.parent = camera_fp
                    //camera.rotate(-1.57f,0f,0f)
                }
            if(cammode>=2f) {
                //camera.parent = camera_fp
                if(cammode>2f&&cammode<=2.1){
                    camera.translate(Vector3f1(0f,200f,0f))
                    camera.rotate(-1.57f, 0f, 0f)
                }
            }
                /*else {
                    cammode = 0f
                    //camera.parent = spaceship
                    //camera.rotate(1.57f,0f,0f)
                }*/

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
                //camera_fp.rotate(0f,-0.01f,0f)
                camera_fp.translate(Vector3f1(0.1f,0f,0f))
            }
            if (window.getKeyState(GLFW_KEY_RIGHT) == true) {
                //camera_fp.rotate(0f,0.01f,0f)
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
            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,6f,0f))

            val distance = spaceshipPosition.distance(asteroidPosition)

            if (distance < 12.0f) {
                iterator.remove()
                asteroid.cleanup()
                //GoTo_Menu()
                setSpaceshipPositionToStart()
                return false
            }

            if(distance< spaceshipPosition.distance(Vector3f1(cAsteroid.x,cAsteroid.y,cAsteroid.z)))
            {
                cAsteroid.x = asteroidPosition.x
                cAsteroid.y = asteroidPosition.y
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
        glDepthMask(true)
        glDepthFunc(GL_LESS)
    }


    private fun GoTo_Menu() {
        //Pause / Clean Asteroids
        pause = false
        cleanup()

        //Get to the menu

        Moon2.translate(Vector3f1(0f,-1000f,0f))
        game_over.translate(Vector3f1(0f,-250f,0f))
        reset_game.translate(Vector3f1(0f,-250f,0f))
        end_game.translate(Vector3f1(0f,-250f,0f))
        spaceship.cleanup()
        spaceship= ModelLoader.loadModel("assets/starsparrow/StarSparrow01.obj", 0f, Math.toRadians(180f), 0f)!!
        camera.parent = spaceship
        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)
        spotLight.parent = spaceship
        ray.parent = spaceship
        pointLight4.parent = ray
        pointLight.parent = spaceship

        b_menu = true
    }

    private fun checkCollisionMenu(){
            //saveDataset(gameDataset,"testdata1")
            val shotPosition = ray.getWorldPosition()
            val check_end = end_game.getWorldPosition()
            val check_reset = reset_game.getWorldPosition()

            val end_distance = shotPosition.distance(check_end)
            val reset_distance = shotPosition.distance(check_reset)


            if (end_distance < 3.0f) {
                b_menu = false
                glfwDestroyWindow(1)
            }
            if (reset_distance < 3.0f) {


                pause = true
                b_menu = false

                game_over.translate(Vector3f1(0f,250f,0f))
                reset_game.translate(Vector3f1(0f,250f,0f))
                end_game.translate(Vector3f1(0f,250f,0f))
                Moon2.translate(Vector3f1(0f,1000f,0f))
                setSpaceshipPositionToStart()
            }
        //}
    }
    
    private fun setSpaceshipPositionToStart() {

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
        vmaxa=0.01f
        vmaxa2=0.0001f
        spaceship.rotate(0.0f,Random().nextFloat(-3.141f,3.141f),0.0f)
        print("reset........................................................................${spaceship.getRotation()}")
        cleanup()
        cAsteroid=Vector3f(10000f,10000f,10000f)
        astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
        var rendertemp = Renderable(mutableListOf(astmesh))
        var ascale=Random().nextFloat(6f,10f)
        rendertemp.scale(Vector3f1(ascale,ascale,ascale))
        rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(0f,0.001f),Random().nextFloat(-100f,100f)))
        asteroidlist2.add(rendertemp)
    }

    private fun checkCollisionAsteroid(): Boolean {
        val shotPosition = ray.getWorldPosition()
        val iterator = asteroidlist.iterator()
        val iterator2 = asteroidlist2.iterator()
        while (iterator.hasNext()) {
            val asteroid = iterator.next()

            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,5f,0f))

            val distance = shotPosition.distance(asteroidPosition)
            if (distance < 10.0f) {
                iterator.remove()
                cleanup()
                score+=500f
                cAsteroid=Vector3f(10000f,10000f,10000f)
                astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
                var rendertemp = Renderable(mutableListOf(astmesh))


                var ascale=Random().nextFloat(6f,10f)

                rendertemp.scale(Vector3f1(ascale,ascale,ascale))
                rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(0f,0.001f),Random().nextFloat(-100f,100f)))
                asteroidlist2.add(rendertemp)

                return true
            }
        }
        while (iterator2.hasNext()) {
            val asteroid = iterator2.next()

            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,5f,0f))

            val distance = shotPosition.distance(asteroidPosition)
            if (distance < 10.0f) {
                iterator2.remove()
                cleanup()
                cAsteroid=Vector3f(10000f,10000f,10000f)
                score+=500f
                astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
                var rendertemp = Renderable(mutableListOf(astmesh))


                var ascale=Random().nextFloat(6f,10f)

                rendertemp.scale(Vector3f1(ascale,ascale,ascale))
                rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(0f,0.001f),Random().nextFloat(-100f,100f)))
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
            //spaceship.rotate(-y_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f, 0f)
            //spaceship.rotate(0f, -x_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f)

        }
    }
        //spaceship.rotate(-y_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f, 0f)
        //spaceship.rotate(0f, -x_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f)


    fun onMouseButton(button: Int, action: Int, mode: Int) {
        //shoot=true
        //checkCollisionAsteroid()
    }

    fun onMouseScroll(xoffset: Double, yoffset: Double) {
        if (yoffset < 0)
        {
            //camera.translate(Vector3f1(0.0f, 0.0f, 0.5f))
            camera.translate(Vector3f1(0.0f, 0.0f, 0.5f))
        }
        if (yoffset > 0)
        {
            //camera.translate(Vector3f1(0.0f, 0.0f, -0.5f))
            camera.translate(Vector3f1(0.0f, 0.0f, -0.5f))
        }
    }



    fun cleanup() {
        for (asteroid in asteroidlist) {
            asteroid.cleanup()
            cAsteroid=Vector3f(10000f,10000f,10000f)
        }
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




