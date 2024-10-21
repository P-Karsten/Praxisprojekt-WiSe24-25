#version 330 core

in struct VertexData
{
    vec3 color;
    vec2 tc;
    vec3 lightDirpoint[10];
    vec3 lightDirspot;
    vec3 viewDir;
    vec3 normal;
} vertexData;

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
};

uniform SpotLight spotLight;
uniform PointLight pointLight;
uniform sampler2D material_emissive;
uniform sampler2D material_diffuse;
uniform sampler2D material_specular;
uniform float shininess;
uniform vec3 emit_col;
uniform float gammaValue;
uniform float shader;

out vec4 color;

vec3 gamma(vec3 C_linear) {
    float gammaValue = 2.2;  // Gamma value, adjust as needed
    return pow(C_linear, vec3(1.0 / gammaValue));
}

// Inverse Gammakorrektur-Funktion
vec3 invgamma(vec3 value, float gammaValue) {
    return pow(value, vec3(gammaValue));
}

vec3 brdf(vec3 n, vec3 l, vec3 v, vec3 ms, vec3 md, float k,float attenuation)
{
    //vec3 r = reflect(-l, n);
    vec3 h = normalize(l + v);//half-vector blinn-phong
    vec3 c = max(0.0, dot(n, l)) * md * attenuation;
    vec3 d = pow(max(0.0, dot(n, h)), k) * ms * attenuation;

    return c + d;
}

void main()
{

if (shader>=0&&shader<=1){
    vec3 viewDir = normalize(vertexData.viewDir);
    vec3 normal = normalize(vertexData.normal);
    vec3 lightDirspot = normalize(vertexData.lightDirspot);

    vec4 emissiveCol = texture(material_emissive, vertexData.tc)*vec4(emit_col,1.0);
    vec4 diffuseCol = texture(material_diffuse, vertexData.tc);
    vec4 specularCol = texture(material_specular, vertexData.tc);
    color = vec4(0, 0, 0, 1);
    vec4 ambientCol = vec4(0.04, 0.04, 0.04, 1.0);

    // Gammakorrektur für diffuse, specular und emissive Farbwerte
    vec3 linearDiffuseCol = gamma(diffuseCol.xyz);
    vec3 linearSpecularCol = gamma(specularCol.xyz);
    vec3 linearEmissiveCol = gamma(emissiveCol.xyz);

    //ambient color
    color.xyz += vec3(ambientCol);

    //emissive
    color.xyz += linearEmissiveCol;

    //pointlight
    for (int i = 0; i < 10; i++) {
        vec3 lightDirpoint = normalize(vertexData.lightDirpoint[i]);
        float distance = length(vertexData.lightDirpoint[i]);
        float attenuation = 100.0 / (distance * distance);  // attenuation
        color.xyz += brdf(normal, lightDirpoint, viewDir, linearSpecularCol, linearDiffuseCol, shininess, attenuation) * pointLight.lightColor[i];
    }

    vec3 lightDirection = normalize(spotLight.direction);
    float theta = dot(-lightDirspot, lightDirection);
    float gamma = spotLight.outerConeAngle;
    float phi = spotLight.innerConeAngle;

    float intensity = clamp((theta - gamma) / (phi - gamma), 0.0, 1.0);
    color.xyz += brdf(normal, lightDirspot, viewDir, linearSpecularCol, linearDiffuseCol, shininess,100.0/(length(vertexData.lightDirspot)*length(vertexData.lightDirspot))) * spotLight.lightColor * intensity;

    // Inverse Gammakorrektur, um das Ergebnis in sRGB oder Gamma zu konvertieren
    color.xyz = invgamma(color.xyz, gammaValue);

    color.a = 1.0;
}
if(shader>1&&shader<=2)
{
    vec4 diffuseCol = texture(material_diffuse, vertexData.tc);
    vec4 specularCol = texture(material_specular, vertexData.tc);
    vec3 linearDiffuseCol = gamma(diffuseCol.xyz);
    vec3 linearSpecularCol = gamma(specularCol.xyz);
    vec3 intensity;
    vec4 ambientCol = vec4(0.04, 0.04, 0.04,1.0);
    vec4 emissiveCol = texture(material_emissive, vertexData.tc)*vec4(emit_col,1.0);
    intensity= ambientCol.xyz+emissiveCol.xyz;
    vec3 normal = normalize(vertexData.normal);
    vec3 lightDirspot = normalize(vertexData.lightDirspot);
    vec3 lightDirection = normalize(spotLight.direction);
    float theta = dot(-lightDirspot, lightDirection);
    float gamma = spotLight.outerConeAngle;
    float phi = spotLight.innerConeAngle;

    for (int i = 0; i < 10; i++) {
        float distance = length(vertexData.lightDirpoint[i]);
        float attenuation = 1000.0 / (distance * distance);// attenuation
        intensity.xyz += brdf(normal, normalize( vertexData.lightDirpoint[i]), normalize(vertexData.viewDir), linearSpecularCol, linearDiffuseCol, shininess, attenuation*0.8) * pointLight.lightColor[i];
    }
    if (intensity.x > 0.95||intensity.y > 0.95||intensity.z > 0.95)
    color = vec4(1.0,0.5,0.5,1.0);
    else if (intensity.x > 0.35||intensity.y > 0.35||intensity.z > 0.35)
    color = vec4(0.6,0.3,0.3,1.0);
    else if (intensity.x > 0.2||intensity.y > 0.2||intensity.z > 0.2)
    color = vec4(0.4,0.2,0.2,1.0);
    else
    color = vec4(0.1,0.05,0.05,1.0);
}
    if(shader>2&&shader<=3)
    {
        vec3 normalized_normal = normalize(vertexData.color);
        float intensityx = abs(normalized_normal.x);
        float intensityy = abs(normalized_normal.y);
        float intensityz = abs(normalized_normal.z);
        color = vec4(vertexData.color.x, vertexData.color.y, vertexData.color.z, 1.0);
    }
}