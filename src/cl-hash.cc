#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

const char * helloStr =
"__kernel void hello(void)"
"{"
"  printf(\"!\");"
"}";

int main(int argc, char *argv[])
{
  cl_int err = CL_SUCCESS;

  try {

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
      throw std::runtime_error("no OpenCL platforms");
    }

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
      0
    };
    cl::Context context(CL_DEVICE_TYPE_CPU, properties);

    std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());

    cl::Program::Sources source(1, std::make_pair(helloStr,strlen(helloStr)));
    cl::Program program(context, source);
    program.build(devices);

    cl::Kernel kernel(program, "hello", &err);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], 0, &err);
    queue.enqueueNDRangeKernel(
                               kernel,
                               cl::NullRange,
                               cl::NDRange(4,4),
                               cl::NullRange,
                               NULL,
                               &event);

    event.wait();

  } catch (const cl::Error& e) {
    std::cerr
      << "OpenCL error: "
      << e.what()
      << "("
      << e.err()
      << ")"
      << std::endl;
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
