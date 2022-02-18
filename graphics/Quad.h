#pragma once

#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include "Buffer.h"
#include "Texture.h"
#include "VertexArray.h"

namespace gl
{
	class Quad
	{
	private:
		gl::VertexBuffer<GLfloat> vertices;
		gl::IndexBuffer<GLuint> indices;
		gl::VertexArray vao;

	public:
		Quad();
		~Quad();

		void updateVerts(float width, float height);

		void render();

	};
}