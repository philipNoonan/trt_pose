#include "Texture.h"
#include <iostream>

namespace gl
{

	Texture::Texture()
	{
		glGenTextures(1, &id);
		std::cout << "id : " << id << std::endl;
	}


	Texture::~Texture()
	{
		glDeleteTextures(1, &id);
	}

	void Texture::create(const void* data, int width, int height, GLuint format, GLuint internalFormat, GLuint type)
	{
		this->width = width;
		this->height = height;

		this->internalFormat = internalFormat;

		this->format = format;
		this->dataType = type;

		bind();
		//glTexStorage2D(GL_TEXTURE_2D, levels, internalformat, w, h); // use mipmapping!
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, dataType, data);
		unbind();
	}

	void Texture::createStorage(int levels, int w, int h, GLuint format, GLuint internalFormat, GLuint type, bool normalized)
	{

		this->width = w;
		this->height = h;

		this->internalFormat = internalFormat;

		this->format = format;
		this->dataType = type;

		bind();
		glTexStorage2D(GL_TEXTURE_2D, levels, internalFormat, w, h);
		unbind();

	}

	void Texture::update(const void* data)
	{


		glBindTexture(GL_TEXTURE_2D, id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, dataType, data);
		glBindTexture(GL_TEXTURE_2D, 0);


	}

	void Texture::mipmap()
	{
		glBindTexture(GL_TEXTURE_2D, id);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void Texture::setFiltering(GLenum maxFilter, GLenum minFilter)
	{


		bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, maxFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
		unbind();
	}

	void Texture::setWarp(GLenum warp)
	{


		bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, warp);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, warp);
		unbind();
	}
	
	void Texture::read(const void* data)
	{
		bind();
		glGetTexImage(GL_TEXTURE_2D, 0, format, dataType, (GLvoid*)data);
		unbind();
	}

	void Texture::read(const void* data, GLenum format, GLenum dataType)
	{
		bind();
		glGetTexImage(GL_TEXTURE_2D, 0, format, dataType, (GLvoid*)data);
		unbind();
	}

	void Texture::bind()
	{
		glBindTexture(GL_TEXTURE_2D, id);
	}

	void Texture::unbind()
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void Texture::bindImage(int idx, int level, GLenum access)
	{
		glBindImageTexture(idx, id, level, GL_FALSE, 0, access, internalFormat);
	}

	void Texture::bindImage(int idx, int level, GLenum access, GLenum internalFormat)
	{
		glBindImageTexture(idx, id, level, GL_FALSE, 0, access, internalFormat);
	}

	void Texture::use(int idx)
	{
		glActiveTexture(GL_TEXTURE0 + idx);
		bind();
	}

	int Texture::getWidth() const
	{
		return width;
	}

	int Texture::getHeight() const
	{
		return height;
	}
	
	int Texture::getID() const
	{
		return id;
	}

}