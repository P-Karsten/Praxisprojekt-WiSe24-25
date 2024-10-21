#version 330 core

// todo 2.1.2
layout(location = 0) in vec3 position;
layout(location = 2) in vec3 normal;
layout(location = 1) in vec2 textureCoordinate;

// --- uniforms
// object to world (the default value will be used unless you upload some other matrix using glUniformMatrix4fv)
uniform mat4 model_matrix = mat4(1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0);



 uniform mat4 view_matrix;

 uniform mat4 proj_matrix;
 uniform vec2 tcMultiplier;

struct PointLight {
    vec3 position[10];
    vec3 lightColor[10];
};

struct SpotLight {
    vec3 direction;
    vec3 position;
    vec3 lightColor;
    float innerConeAngle;
    float outerConeAngle;
} ;
uniform SpotLight spotLight;
uniform PointLight pointLight;


out struct VertexData

{
    vec3 color;
    vec2 tc;
    vec3 lightDirpoint[10];
    vec3 lightDirspot;
    vec3 viewDir;
    vec3 normal;

} vertexData;

void main(){
    vertexData.color=normal;
   vec4 viewpos = view_matrix * model_matrix * vec4(position, 1.0);
    vertexData.viewDir = -viewpos.xyz;
    gl_Position = proj_matrix * viewpos ;
    vertexData.normal = transpose(inverse(mat3(view_matrix*model_matrix)))*normal;
    vertexData.tc = textureCoordinate *  tcMultiplier ;
    vertexData.lightDirspot = spotLight.position-viewpos.xyz;
    for (int i = 0; i < 10; i++) {
        vertexData.lightDirpoint[i] = pointLight.position[i] - viewpos.xyz;
    }
}