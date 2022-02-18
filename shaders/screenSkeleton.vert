# version 430

//layout (binding = 0, r16ui) readonly uniform uimage2D depthImage; 
layout (binding = 0) uniform usampler2D depthTexture;


uniform mat4 MVP;

layout(std430, binding = 0) buffer kp
{
    float keypoints [];
};

layout(std430, binding = 1) buffer lnk
{
    float links [];
};


void main()
{
   // shader is invoked by the number of keypoints detected
   // each shader reads its x,y position from its index in the keypoints buffer
   // TODO read in z from a depthimage
   //
   // For lines, we need to read in sequential links from vetex index

   vec2 pos = vec2(keypoints[int(gl_VertexID) * 3], keypoints[int(gl_VertexID) * 3 + 1]);
   //float depth = imageLoad(depthImage, ivec2(pos)).x;
   float depth = float(textureLod(depthTexture, vec2(pos.x / 255.0f, pos.y / 255.0f), 0.0f).x);

   gl_PointSize = 10.0f;
   gl_Position = vec4(pos.x / 127.0f - 1.0f, -1.0f * (pos.y / 127.0f - 1.0f), 0, 1.0f); 
   // gl_Position = MVP * vec4(joint, 1.0f);
}