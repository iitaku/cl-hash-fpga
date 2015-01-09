#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::string format(const char *fmt, ...)
{
  int length;
  {
    va_list ap;
    va_start(ap, fmt);
#if defined(_MSC_VER)
	length = _vscprintf(fmt, ap);
#else
    length = vsnprintf(NULL, 0, fmt, ap);
#endif
    va_end(ap);
  }

  std::vector<char> buf(length, 0);
  {
    va_list ap;
    va_start(ap, fmt);
#if defined(_MSC_VER)
    vsnprintf_s(&buf[0], length, _TRUNCATE, fmt, ap);
#else
    vsprintf(&buf[0], fmt, ap);
#endif
    va_end(ap);
  }

  std::string s(buf.begin(), buf.end());
  return s;
}

std::vector<char> load(const char *path, std::ios_base::openmode mode=std::ios_base::in)
{
  std::ifstream ifs(path, mode);

  if (!ifs.is_open()) {
    throw std::runtime_error(format("cannot open %s", path));
  }

  // size
  ifs.seekg(0, std::ifstream::end);
  std::ifstream::pos_type end = ifs.tellg();

  ifs.seekg(0, std::ifstream::beg);
  std::ifstream::pos_type beg = ifs.tellg();

  std::size_t buf_size = end-beg;

  // read
  std::vector<char> buf(buf_size, 0);

  ifs.read(&buf[0], buf_size);

  return buf;
}

int main(int argc, char *argv[])
{
  try {

    //const std::size_t size = 1024*1024;
    const std::size_t size = 32;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
      throw std::runtime_error("no OpenCL platforms");
    }

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
      0
    };
    cl::Context context(CL_DEVICE_TYPE_ALL, properties);

    std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());
    cl::CommandQueue queue(context, devices[0], 0);

    std::vector<char> source_buf(load("../sha.cl"));
    cl::Program::Sources source(1, std::make_pair(&source_buf[0], source_buf.size()));
    cl::Program program(context, source);
    try {
      program.build(devices);
    } catch (cl::Error& e) {
      std::string log(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));
      std::cerr << log << std::endl;
      throw;
    }

    cl::Kernel kernel(program, "sha256");

    cl::Buffer h_src(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size);
    cl::Buffer h_dst(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size);

    cl::Buffer d_src(context, CL_MEM_READ_WRITE, size);
    cl::Buffer d_dst(context, CL_MEM_READ_WRITE, size);

    cl::Event event;

    cl_uchar *h_src_ptr = reinterpret_cast<cl_uchar*>(queue.enqueueMapBuffer(h_src, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size));
    cl_uchar *h_dst_ptr = reinterpret_cast<cl_uchar*>(queue.enqueueMapBuffer(h_dst, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size));

    for (int i=0; i<size/sizeof(cl_uchar); ++i) {
      h_src_ptr[i] = 0;
      h_dst_ptr[i] = 0;
    }

    queue.enqueueWriteBuffer(d_src, CL_TRUE, 0, size, h_src_ptr);
    queue.enqueueWriteBuffer(d_dst, CL_TRUE, 0, size, h_dst_ptr);

    kernel.setArg(0, d_src);
    kernel.setArg(1, d_dst);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(size / 32),
                               cl::NullRange,
                               NULL,
                               &event);

    event.wait();

    queue.enqueueReadBuffer(d_dst, CL_TRUE, 0, size, h_dst_ptr);

    for (int i=0; i<size/32; ++i) {
      std::stringstream ss;
      for (int j=0; j<32; ++j) {
        ss << format("%02x", h_dst_ptr[i*32+j]);
      }
      ss << "  " << i;
      std::cout << ss.str() << std::endl;
    }

    queue.enqueueUnmapMemObject(h_src, h_src_ptr);
    queue.enqueueUnmapMemObject(h_dst, h_dst_ptr);

  } catch (const cl::Error& e) {
    std::cerr << format("OpenCL error: %s (%d)", e.what(), e.err()) << std::endl;
    return EXIT_FAILURE;
  } catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Unknown error" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
