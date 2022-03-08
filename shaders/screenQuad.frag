# version 430

in VS_OUT {
    vec2 outTexCoords;
} fs_in;

out vec4 outColor;

layout (binding = 0) uniform usampler2D inputColor;
layout (binding = 1) uniform usampler2D depthTexture;

void main(){

    vec4 tempCol = vec4(texture(inputColor, fs_in.outTexCoords)) / 255.0f;

   // vec4 tempCol = vec4(texture(depthTexture, fs_in.outTexCoords)) / 65535.0f;

    outColor = vec4(tempCol.zyx, 1.0f);
}
