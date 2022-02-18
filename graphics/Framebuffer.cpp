#include "Framebuffer.h"

namespace gl
{
	Framebuffer::Framebuffer()
	{
		glGenFramebuffers(1, &id);
	}

	Framebuffer::~Framebuffer()
	{
		glDeleteFramebuffers(1, &id);
	}

	void Framebuffer::create(int width, int height, int depth, GLuint target)
	{
		this->width = width;
		this->height = height;
		this->depth = depth;
		this->target = target;


		if (depth == 1) {

			zbuffer = std::make_shared<gl::Texture>();
			zbuffer->create(0, width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT);
			zbuffer->setFiltering(GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST);
			zbuffer->setWarp(GL_CLAMP_TO_EDGE);

			bind();
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, this->target, zbuffer->getID(), 0);
			unbind();

		}	
		else {
			zbuffer3D = std::make_shared<gl::Texture3D>();
			zbuffer3D->create(0, width, height, depth, this->target, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, false);
			zbuffer3D->setFiltering(GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST);
			zbuffer3D->setWarp(GL_CLAMP_TO_EDGE);

			bind();
			glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, zbuffer3D->getID(), 0);
			unbind();
		}


	}

	void Framebuffer::attach(GLuint texID, int idx, int level)
	{
		//attachments[idx] = texID;

		bind();
		if (this->depth == 1) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + idx, this->target, texID, level);
		}
		else {
			glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + idx, texID, level);
		}
		unbind();
	}

	void Framebuffer::bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
	}

	void Framebuffer::unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	int Framebuffer::getWidth() const
	{
		return width;
	}

	int Framebuffer::getHeight() const
	{
		return height;
	}

	gl::Texture::Ptr Framebuffer::getColorAttachment(unsigned int idx) const
	{
		if (attachments.find(idx) != attachments.end())
		{
			return attachments.at(idx);
		}

		return nullptr;
	}

	gl::Texture::Ptr Framebuffer::getDepthAttachment() const
	{
		return zbuffer;
	}

	std::vector<GLenum> Framebuffer::getDrawBuffers()
	{
		std::vector<GLenum> drawBuffers(attachments.size());

		std::vector<GLenum>::iterator itDrawBuffers = drawBuffers.begin();
		std::map<unsigned int, gl::Texture::Ptr>::iterator itAttachments = attachments.begin();
		while (itAttachments != attachments.end())
		{
			*itDrawBuffers = GL_COLOR_ATTACHMENT0 + itAttachments->first;

			++itDrawBuffers;
			++itAttachments;
		}

		return drawBuffers;
	}
}