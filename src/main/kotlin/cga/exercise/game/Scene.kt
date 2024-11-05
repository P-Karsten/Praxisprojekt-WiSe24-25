package cga.exercise.game

import SpotLight
import cga.exercise.components.camera.TronCamera
import cga.exercise.components.geometry.Material
import cga.exercise.components.geometry.Mesh
import cga.exercise.components.geometry.Renderable
import cga.exercise.components.geometry.VertexAttribute
import cga.exercise.components.light.PointLight
import cga.exercise.components.shader.ShaderProgram
import cga.exercise.components.texture.Texture2D
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
import kotlinx.serialization.json.Json
import org.joml.Math
import org.joml.Vector2f
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.opengl.GL30.*
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import java.util.*
import org.joml.Vector3f as Vector3f1

/**
 * Created 29.03.2023.
 */
class Scene(private val window: GameWindow) {
    private val staticShader: ShaderProgram = ShaderProgram("assets/shaders/tron_vert.glsl", "assets/shaders/tron_frag.glsl")

    data class DataModel(val id: Int, val message: String)

    private var camera: TronCamera
    private var camera_fp: TronCamera
    var light_on = false
    var light_last = 0f
    val light_int = 0.35f
    var b_menu = false
    var astmesh: Mesh
    var vmaxa=0.01f
    var vmaxa2=0.0001f
    var shoot2=false
    var score =0f
    var pause =true
    var rayl= 0
    var inputkey= ""
    var rayl2=0
    var speed = -0.1f
    var shoot =false
    var cammode =0
    var camselect=0f
    var tempshader=1f
    var asteroidlist = mutableListOf<Renderable>()
    var asteroidlist2 = mutableListOf<Renderable>()
    var meshlist = mutableListOf<Mesh>()
    var renderable = Renderable(meshlist)
    var renderable2 = Renderable(meshlist)
    var Moon = Renderable(meshlist)
    var Moon2 = Renderable(meshlist)
    var spaceship = Renderable(meshlist)
    var skybox = Renderable(meshlist)
    var game_over = Renderable(meshlist)
    var end_game = Renderable(meshlist)
    var reset_game = Renderable(meshlist)
    val starttime = LocalDateTime.now()
    var ray = Renderable(meshlist)
    var ray2 = Renderable(meshlist)
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
    data class GameData(
        val spaceshipPosition: List<Float>,
        val spaceshiprotation: Vector3f1,
        val asteroidPositions: List<List<Float>>,
        val action: String,
        val reward: Float,
        val time: Float
    )

    val gameDataset = mutableListOf<GameData>()

    fun collectData(
        spaceshipPos: Vector3f1,
        spaceshiprotation: Vector3f1,
        asteroidPositions: List<Vector3f1>,
        action: String,
        reward: Float,
        time: Float
    ) {                             //rotation spaceship
        val data = GameData(
            spaceshipPosition = listOf(spaceshipPos.x, spaceshipPos.y, spaceshipPos.z),
            spaceshiprotation = spaceship.getRotation(),
            asteroidPositions = asteroidPositions.map { listOf(it.x, it.y, it.z) },
            action = action,
            reward = reward,
            time = time
        )
        gameDataset.add(data)

        println(gameDataset)
    }
    //fun saveDataset(dataset: List<GameData>, filename: String) {
    //    val jsonData = Json.encodeToString(gameDataset)
    //    File(filename).writeText(jsonData)
    //}

    //scene setup
    suspend fun testapi() {
        // Create HttpClient with JSON support
        val client = HttpClient(CIO) {
            install(ContentNegotiation) {
                json(Json { prettyPrint = true; isLenient = true })
            }
        }

        // Prepare the data to send
        val dataToSend = GameData(
            spaceshipPosition = listOf(0.0f, 1.44f, 0.0f),
            spaceshiprotation = Vector3f1(0.0f, 0.0f, 1.571f),
            asteroidPositions = listOf(
                listOf(46.48f, 0.0017f, -9.41f)
                // Add other sample positions as needed
            ),
            action = "",
            reward = 1.0f,
            time = 4.288f
        )

        try {
            // Sending a POST request to the FastAPI server
            val postResponse: GameData = client.get("http://127.0.0.1:8000/get/") {
                contentType(ContentType.Application.Json)
                //setBody(dataToSend)  // Send GameData as request body
            }.body()  // Extract the response body as GameData

            // Print the response (example)
            println("POST Response: spaceshipPosition=${postResponse.spaceshipPosition}, action=${postResponse.action}, reward=${postResponse.reward}")

        } catch (e: Exception) {
            println("Error sending request: ${e.localizedMessage}")
        } finally {
            // Close the client after use
            client.close()
        }
    }
    init {

        camera = TronCamera()
        camera.rotate(-0.110865f,0f,0f)
        camera.translate(Vector3f1(0.0f, 0.0f, 14.0f))

        camera_fp = TronCamera()
        camera_fp.rotate(-0.610865f,0f,0f)
        camera_fp.translate(Vector3f1(0.0f, 3f, -5f))



        //Skybox
        val cu = loadOBJ("assets/models/skybox.obj", true, true)
        //Menu
        val end = loadOBJ("assets/models/menu/beenden.obj", true, true)
        val ga_ov = loadOBJ("assets/models/menu/game_over.obj", true, true)
        val reset = loadOBJ("assets/models/menu/neustart.obj", true, true)

        enableDepthTest(GL_LESS) //Tiefentest, werden Pixel in der richtigen Reihenfolge gerendert
        //enableFaceCulling(GL_CCW, GL_FRONT)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); GLError.checkThrow() //schwarze hintergrundfarbe, alpha1.0f völlige deckkraft

        val spec = Texture2D("assets/textures/ground_diff.png", true)
        val diff = Texture2D("assets/textures/ground_diff.png", true)
        var skybox_emit = Texture2D("assets/textures/skybox.png", true)
        var raytex = Texture2D("assets/textures/ground_diff.png", true)
        var fontMat = Texture2D("assets/textures/menu_font.png", true)


        fontMat.bind(0)
        fontMat.setTexParams(GL_CLAMP, GL_CLAMP, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR)
        fontMat.unbind()


        val rayMaterial = Material(

            diff,
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
        val skyboxMaterial = Material(

                skybox_emit,
                skybox_emit,
                skybox_emit,
                60.0f,
                Vector2f(1.0f, 1.0f)
        )


        //Skybox
        var skyboxmesh = Mesh(cu.objects[0].meshes[0].vertexData,cu.objects[0].meshes[0].indexData,vertexAttributes,skyboxMaterial)
        skybox = Renderable(mutableListOf(skyboxmesh))

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


        //skybox.translate(spaceship.getWorldPosition())
        skybox.scale(Vector3f1(1850f,1850f,1850f))



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
        var ras = loadOBJ("assets/models/newscene.obj",true,true)
        var raymesh = Mesh(ras.objects[0].meshes[0].vertexData,ras.objects[0].meshes[0].indexData,vertexAttributes,rayMaterial)
        ray = Renderable(mutableListOf(raymesh))
        ray.scale(Vector3f1(5f,5f,5f))
        ray.translate(Vector3f1(0f,0f,0f))
        ray.rotate(-1.5708f,1.5708f,0f)

        //2.Laser
        var ras2 = loadOBJ("assets/models/newscene.obj",true,true)
        var raymesh2 = Mesh(ras.objects[0].meshes[0].vertexData,ras.objects[0].meshes[0].indexData,vertexAttributes,rayMaterial)
        ray2 = Renderable(mutableListOf(raymesh2))
        ray2.scale(Vector3f1(5f,5f,5f))
        ray2.translate(Vector3f1(0f,0f,0f))
        ray2.rotate(-1.5708f,1.5708f,0f)

        spotLight.rotate(Math.toRadians(-2f),0f,0f)
        spotLight.parent = spaceship
        ray.parent = spaceship
        ray2.parent=spaceship
        pointLight4.parent = ray
        pointLight2.parent = ray2
        pointLight.parent = spaceship

        for(i in 1..25)//random asteroid spawn
        {
            var rendertemp = ModelLoader.loadModel("assets/10464_Asteroid_L3.123c72035d71-abea-4a34-9131-5e9eeeffadcb/10464_Asteroid_v1_Iterations-2.obj", -1.5708f, 1.5708f, 0f)!!
            var ascale=Random().nextFloat(0.005f,0.01f)

            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-10000f,10000f),Random().nextFloat(0f,1f),Random().nextFloat(-10000f,10000f)))
            asteroidlist.add(rendertemp)
        }

        astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)

    }







    fun render(dt: Float, t: Float)= runBlocking {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        staticShader.use()

        staticShader.setUniform("gammaValue", desiredGammaValue)

        camera.updateViewMatrix()
        camera.updateProjectionMatrix()
        camera.bind(staticShader)


        pointLight.bind(staticShader,camera.getCalculateViewMatrix(),0)

        pointLight5.bind(staticShader,camera.getCalculateViewMatrix(),4)

        spotLight.bind(staticShader, camera.getCalculateViewMatrix())

        skybox.render(staticShader, Vector3f1(1f,1f,1.15f))

        //Menu
        //menu_backg.render(staticShader, Vector3f1(2f,2f,2f))
        game_over.render(staticShader, Vector3f1(2f,2f,2f))
        reset_game.render(staticShader, Vector3f1(2f,1f,2f))
        end_game.render(staticShader, Vector3f1(2f,2f,2f))

        spaceship.render(staticShader, Vector3f1(1.2f,1.2f,1.2f))
        Moon.render(staticShader, Vector3f1(1f,1f,1f))
        Moon2.render(staticShader, Vector3f1(1f,1f,1f))


        if(shoot2==true){
            pointLight2 = PointLight(Vector3f1(0f,1f,0f), Vector3f1(5.0f,0.0f,5.0f))
            pointLight2.parent=ray2
            ray2.render(staticShader, Vector3f1(10f,0.1f,0.1f))
            pointLight2.bind(staticShader,camera.getCalculateViewMatrix(),1)

            ray2.translate(Vector3f1(0f,1f,0f))
            rayl2++

            if(rayl2>=100){

                ray2.translate(Vector3f1(0f,-100f,0f))
                pointLight2 = PointLight(Vector3f1(0f,1f,0f), Vector3f1(0.0f,0.0f,0.0f))
                pointLight2.parent=ray2
                pointLight2.bind(staticShader,camera.getCalculateViewMatrix(),1)
                shoot2= false
                rayl2=0

            }
        }
        if(shoot==true){
            ray.render(staticShader, Vector3f1(10f,0.1f,0.1f))
            pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(5.0f,0.0f,0.0f))
            pointLight4.parent=ray
            pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
            ray.translate(Vector3f1(0f,1f,0f))
            rayl++
            if(rayl==50)
                shoot2=true

            if(rayl>=100){
                ray.translate(Vector3f1(0f,-100f,0f))
                pointLight4 = PointLight(Vector3f1(0f, 1f, 0f), Vector3f1(0.0f,0.0f,0.0f))
                pointLight4.parent=ray
                pointLight4.bind(staticShader,camera.getCalculateViewMatrix(),3)
                shoot= false
                rayl=0

            }
        }






        if(pause)
        {
            for(i in 0..asteroidlist.lastIndex-1)
            {

                asteroidlist[i].translate(spaceship.getWorldPosition().sub(asteroidlist[i].getWorldPosition(),Vector3f1()).mul(Vector3f1(vmaxa,vmaxa,vmaxa)))

                asteroidlist[i].render(staticShader,Vector3f1(0.2f,0.2f,0.2f))
            }
            for(i in 0..asteroidlist2.lastIndex-1)
            {

                asteroidlist2[i].translate(spaceship.getWorldPosition().sub(asteroidlist2[i].getWorldPosition(),Vector3f1()).mul(Vector3f1(vmaxa2,vmaxa2,vmaxa2)))

                asteroidlist2[i].render(staticShader,Vector3f1(0.2f,0.15f,0.15f))
            }

            println(score)
            score+=1f
            if(score.toInt()%100==0)
            {
                println(vmaxa)
                vmaxa*=1.1f
                vmaxa2*=1.01f
            }
        if(score.toInt()%100==0){


            astmesh= Mesh(astobj.objects[0].meshes[0].vertexData,astobj.objects[0].meshes[0].indexData,vertexAttributes,astmat)
            var rendertemp = Renderable(mutableListOf(astmesh))


            var ascale=Random().nextFloat(6f,10f)

            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-100f,100f),Random().nextFloat(0f,1f),Random().nextFloat(-100f,100f)))
            asteroidlist2.add(rendertemp)

        }
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


        collectData(spaceship.getWorldPosition(),spaceship.getRotation(),asteroidPositions = asteroidlist.map { it.getWorldPosition() },inputkey,score,ChronoUnit.MILLIS.between(starttime,LocalDateTime.now())/1000f)



                testapi()



    }


    fun update(dt: Float, t: Float) {
        //skybox.translate(spaceship.getPosition())
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
            if(cammode==0){
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
            if(cammode==0){
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
            if(cammode==0)
            {
                cammode=1
            }
            else
                cammode=0
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

        checkCollisionSpaceship()
    }

    private fun checkCollisionSpaceship() {
        val spaceshipPosition = spaceship.getWorldPosition()

        val iterator = asteroidlist.iterator()
        val iterator2 = asteroidlist2.iterator()
        while (iterator.hasNext()) {
            val asteroid = iterator.next()
            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,6f,0f))

            val distance = spaceshipPosition.distance(asteroidPosition)

            if (distance < 7.0f) {
                iterator.remove()
                asteroid.cleanup()
                GoTo_Menu()
            }
        }
        while (iterator2.hasNext()) {
            val asteroid = iterator2.next()
            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,6f,0f))

            val distance = spaceshipPosition.distance(asteroidPosition)

            if (distance < 12.0f) {
                iterator2.remove()
                asteroid.cleanup()
                GoTo_Menu()
            }
        }
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
        ray2.parent=spaceship
        pointLight4.parent = ray
        pointLight2.parent = ray2
        pointLight.parent = spaceship

        b_menu = true
    }
    private fun checkCollisionMenu(){
            //saveDataset(gameDataset,"testdata1")
            val shotPosition = ray.getWorldPosition()
            val shotPosition2 = ray2.getWorldPosition()
            val check_end = end_game.getWorldPosition()
            val check_reset = reset_game.getWorldPosition()

            val end_distance = shotPosition.distance(check_end)
            val end_distance2 = shotPosition2.distance(check_end)
            val reset_distance = shotPosition.distance(check_reset)
            val reset_distance2 = shotPosition2.distance(check_reset)


            if (end_distance < 3.0f||end_distance2 < 3.0f) {
                b_menu = false
                glfwDestroyWindow(1)
            }
            if (reset_distance < 3.0f||reset_distance2 < 3.0f) {


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

        spaceship.cleanup()
        spaceship= ModelLoader.loadModel("assets/starsparrow/StarSparrow01.obj", 0f, Math.toRadians(180f), 0f)!!
        camera.parent = spaceship
        spaceship.scale(Vector3f1(0.8f, 0.8f, 0.8f))
        spaceship.translate(initialSpaceshipPosition)
        asteroidlist.clear()
        asteroidlist2.clear()
        for(i in 1..25)//random asteroid spawn
        {
            var rendertemp = ModelLoader.loadModel("assets/10464_Asteroid_L3.123c72035d71-abea-4a34-9131-5e9eeeffadcb/10464_Asteroid_v1_Iterations-2.obj", -1.5708f, 1.5708f, 0f)!!
            var ascale=Random().nextFloat(0.005f,0.01f)

            rendertemp.scale(Vector3f1(ascale,ascale,ascale))
            rendertemp.translate(Vector3f1(Random().nextFloat(-10000f,10000f),Random().nextFloat(-10000f,10000f),Random().nextFloat(-10000f,10000f)))
            asteroidlist.add(rendertemp)
        }
        spotLight.parent = spaceship
        ray.parent = spaceship
        ray2.parent=spaceship
        pointLight4.parent = ray
        pointLight2.parent = ray2
        pointLight.parent = spaceship

        score=0f
        vmaxa=0.01f
        vmaxa2=0.0001f

    }

    private fun checkCollisionAsteroid() {
        val shotPosition = ray.getWorldPosition()
        val shotPosition2 = ray2.getWorldPosition()
        val iterator = asteroidlist.iterator()
        val iterator2 = asteroidlist2.iterator()
        while (iterator.hasNext()) {
            val asteroid = iterator.next()

            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,5f,0f))

            val distance = shotPosition.distance(asteroidPosition)
            val distance2 = shotPosition2.distance(asteroidPosition)
            if (distance < 10.0f||distance2 < 10.0f) {
                iterator.remove()
                asteroid.cleanup()
                score+=500f
            }
        }
        while (iterator2.hasNext()) {
            val asteroid = iterator2.next()

            val asteroidPosition = asteroid.getWorldPosition().add(Vector3f1(0f,5f,0f))

            val distance = shotPosition.distance(asteroidPosition)
            val distance2 = shotPosition2.distance(asteroidPosition)
            if (distance < 10.0f||distance2 < 10.0f) {
                iterator2.remove()
                asteroid.cleanup()
                score+=500f
            }
        }
    }
    fun onKey(key: Int, scancode: Int, action: Int, mode: Int) {}

    fun onMouseMove(xpos: Double, ypos: Double) {
        val x_speed = (xpos - window.windowWidth/ 2.0).toFloat() * 0.002f

        val y_speed = (ypos - window.windowHeight/ 2.0).toFloat() * 0.002f

        glfwSetCursorPos(window.m_window, window.windowWidth / 2.0, window.windowHeight/ 2.0)
        if(cammode==0){
        spaceship.rotate(-y_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f, 0f)
        spaceship.rotate(0f, -x_speed.coerceAtMost(0.015f).coerceAtLeast(-0.015f), 0f)

        }
        else
            camera.rotateAroundPoint(0f, -x_speed, 0f, renderable.getWorldPosition())

    }
    fun onMouseButton(button: Int, action: Int, mode: Int) {
        shoot=true
        checkCollisionAsteroid()
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



    fun cleanup() {
        for (asteroid in asteroidlist) {
            asteroid.cleanup()
        }
        for (asteroid in asteroidlist2) {
            asteroid.cleanup()
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