#include "Quad.h"

namespace gl
{
	Quad::Quad()
	{
		std::vector<GLfloat> vert =
		{
			//positions        texture coords
			- 1.0, -1.0, 0.0,   0.0, 0.0,
			1.0, -1.0, 0.0,    1.0, 0.0,
			1.0,  1.0, 0.0,    1.0, 1.0,
			-1.0,  1.0, 0.0,   0.0, 1.0
		};
		std::vector<GLshort> index =
		{
			0, 1, 2,
			2, 3, 0
		};

		vertices.create(vert.data(), (int)vert.size(), GL_STATIC_DRAW);
		vao.addVertexAttrib(0, 3, vertices, 20, 0);
		vao.addVertexAttrib(1, 2, vertices, 20, (void *)(12));

		indices.create(index.data(), (int)index.size(), GL_STATIC_DRAW);
		vao.bind();
		indices.bind();
		vao.unbind();
	}

	Quad::~Quad()
	{
	}

	void Quad::updateVerts(float width, float height) {
		std::vector<GLfloat> vert =
		{
			//positions        texture coords
			-1.0f * width, -1.0f * height, 0.0,   0.0, 0.0,
			1.0f * width, -1.0f * height, 0.0,    1.0, 0.0,
			1.0f * width,  1.0f * height, 0.0,    1.0, 1.0,
			-1.0f * width,  1.0f * height, 0.0,   0.0, 1.0
		};
		vertices.update(vert.data(), 0, (int)vert.size());

	}

	void Quad::render()
	{
		vao.bind();
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		vao.unbind();
	}

}