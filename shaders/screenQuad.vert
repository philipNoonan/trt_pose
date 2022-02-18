# version 430

in layout(location = 0) vec3 position;
in layout(location = 1) vec2 inTexCoords;


out VS_OUT {
    vec2 outTexCoords;
} vs_out;

void main()
{
    gl_Position = vec4(position, 1.0f);
    vs_out.outTexCoords = vec2(inTexCoords.x, 1.0f - inTexCoords.y);
}