#pragma once

#include <memory>
#include <GL/glew.h>

namespace gl
{


	enum class TextureFilter
	{
		NEAREST,
		LINEAR,
		NEAREST_LINEAR,
		NEAREST_NEAREST,
		LINEAR_NEAREST,
		LINEAR_LINEAR
	};

	enum class TextureWarp
	{
		REPEAT,
		MIRRORED_REPEAT,
		CLAMP_TO_EDGE,
		CLAM_TO_BORDER
	};

	class Texture
	{
	private:
		int width;
		int height;

		GLuint id;

		GLint internalFormat;
		GLenum format;
		GLenum dataType;

		
	public:
		Texture();
		~Texture();

		void create(const void* data, int width, int height, GLuint format, GLuint internalFormat, GLuint type);
		void createStorage(int levels, int w, int h, GLuint format, GLuint internalFormat, GLuint type, bool normalized);

		void update(const void* data);
		void mipmap();
		void setFiltering(GLenum maxfilter, GLenum minfilter);
		void setWarp(GLenum warp);

		void read(const void* data);
		void read(const void* data, GLenum format, GLenum dataType);

		void bind();
		void unbind();
		void bindImage(int idx, int layer, GLenum access);
		void bindImage(int idx, int layer, GLenum access, GLenum internalFormat);

		void use(int idx = 0);
		
		int getWidth() const;
		int getHeight() const;
		int getID() const;

		typedef std::shared_ptr<Texture> Ptr;
	};
}