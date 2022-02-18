#pragma once

#include <vector>
#include <map>
#include <GL/glew.h>
#include "Texture.h"
#include "Texture3D.h"

namespace gl
{
	class Framebuffer
	{
	private:
		int width = 0;
		int height = 0;
		int depth = 0;
		GLuint target;

		GLuint id;

		std::map<unsigned int, gl::Texture::Ptr> attachments;
		gl::Texture::Ptr zbuffer;
		gl::Texture3D::Ptr zbuffer3D;


	public:
		Framebuffer();
		~Framebuffer();

		void create(int width, int height, int depth, GLuint target);
		void attach(GLuint texID, int idx,  int level = 0);

		void bind();
		void unbind();

		int getWidth() const;
		int getHeight() const;

		gl::Texture::Ptr getColorAttachment(unsigned int idx = 0) const;
		gl::Texture::Ptr getDepthAttachment() const;

		std::vector<GLenum> getDrawBuffers();

		typedef std::shared_ptr<gl::Framebuffer> Ptr;
	};
}