#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include "spd_matrix.h"
#include "cholesky.h"
#include <CL/cl.h>

void CL_CALLBACK onOpenCLError(const char *errinfo,  const void *private_info,
                               size_t cb, void *user_data)
{
    printf("Error while creating context or working in this context : %s", errinfo);
}

typedef struct DeviceDesc{
	cl_device_id    deviceId;
	cl_device_type  deviceType;
	char*           deviceTypeString;
	char*           deviceName;
} DeviceDesc;

int main(int argc, char *argv[])
{
    srand( time( NULL ) );
	if (!argv[1])
	{
	    printf("Specify matrix dimension.\n");
	    exit(-1);
	}
	int dimension = atoi(argv[1]);
	
    cl_int result;

    cl_uint             numEntries = 1;
    cl_platform_id*     platforms;
    cl_uint             numPlatforms;
   
    cl_uint             maxDevices = 1;
    cl_device_id*       deviceIDs;
    cl_uint             numDevices;
    
    cl_context_properties*  properties = 0;
    cl_uint                 usedDevices = 1;
    
    cl_command_queue_properties commandQueueProperties = CL_QUEUE_PROFILING_ENABLE;

    
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numEntries);
    result = clGetPlatformIDs(numEntries, platforms, &numPlatforms);
    if(result != CL_SUCCESS) exit(1);

    
    deviceIDs = (cl_device_id*)malloc(maxDevices*sizeof(cl_device_id));
    result = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, maxDevices, deviceIDs, &numDevices);
    if(result != CL_SUCCESS) exit(2);

    
    cl_context context = clCreateContext(properties, usedDevices, deviceIDs, &onOpenCLError, NULL, &result);
    if(result != CL_SUCCESS) exit(3);

    
    cl_command_queue commands = clCreateCommandQueue(context, deviceIDs[0], commandQueueProperties, &result);
    if(result != CL_SUCCESS) exit(4);

    
    DeviceDesc device_desc;
    device_desc.deviceId = deviceIDs[0];
    size_t actualSize;
    result = clGetDeviceInfo(deviceIDs[0], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_desc.deviceType, &actualSize);

    
    switch(device_desc.deviceType)
    {
         case CL_DEVICE_TYPE_CPU:
            device_desc.deviceTypeString = "Processor";
            break;
         case CL_DEVICE_TYPE_GPU:
            device_desc.deviceTypeString = "Graphics card";
            break;
         case CL_DEVICE_TYPE_ACCELERATOR:
            device_desc.deviceTypeString = "Accelerator";
            break;
         default:
            device_desc.deviceTypeString = "NONE";
            break;
    }

    
    size_t deviceNameLength = 4096;
    char* tempDeviceName = (char*)malloc(4096);
    result |= clGetDeviceInfo(deviceIDs[0], CL_DEVICE_NAME, deviceNameLength, tempDeviceName, &actualSize);
    if(result == CL_SUCCESS){
        device_desc.deviceName = (char*)malloc(actualSize);
        memcpy(device_desc.deviceName, tempDeviceName, actualSize);
        free(tempDeviceName);
    }
    if(result != CL_SUCCESS)
    {
        printf("Error while getting device info\n");
        exit(0);
    }

    printf("%s: %s\n", device_desc.deviceTypeString,device_desc.deviceName);
    printf("CL_KERNEL_WORK_GROUP_SIZE: %d\nThis is the largest Matrix I can work with.\n", CL_KERNEL_WORK_GROUP_SIZE);
    if (dimension > CL_KERNEL_WORK_GROUP_SIZE)
    {
        printf("Matrix too large.\n");
        exit(-1);
    }

    
    char *kernels;
    long input_file_size;
    FILE *input_file = fopen("kernels.c", "rb");
    fseek(input_file, 0, SEEK_END);
    input_file_size = ftell(input_file);
    rewind(input_file);
    kernels = malloc(input_file_size * (sizeof(char)));
    fread(kernels, sizeof(char), input_file_size, input_file);
    fclose(input_file);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernels, NULL, &result);
    if(result != CL_SUCCESS) exit(5);

   
    result = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(result != CL_SUCCESS)
    {
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        printf("--- Build log ---\n %s\n", buffer);
        exit(6);
    }

    cl_kernel choldc_gpu;
    
    choldc_gpu = clCreateKernel(program, "choldc_gpu", &result);
    
    if(result != CL_SUCCESS) exit(7);

    
    printf("You chose Dimension: %d\n", dimension);
    float norm1, norm2, norm3;
    float** A = dmatrix(1, dimension, 1, dimension);
    float** A_clone = dmatrix(1, dimension, 1, dimension);
    float** L = dmatrix(1, dimension, 1, dimension);
    float** L_t = dmatrix(1, dimension, 1, dimension);

    
	A = generate_random_matrix(A, dimension);
    A = construct_symetric_matrix(A, dimension);
	A = matrix_positive_definite(A, dimension);
    norm1 = frobenius_norm(A, dimension);

   
    A_clone = clone_matrix(A, dimension);
    L = choldc(A_clone, L, dimension);
   
    L_t = clone_matrix(L, dimension);
    L_t = transpose_matrix(L_t, dimension);

    /*
    A_clone = multiply(L, L_t, A_clone, dimension);
    norm2 = frobenius_norm(A_clone, dimension);
    printf("Error \tCPU: % 20.16lf\n", fabs(norm1 - norm2));
    */
     
    size_t numberOfValues = dimension * dimension;
    size_t sizeOfBuffers = numberOfValues * sizeof(float);
    float* inputDoubles = (float*)malloc(sizeOfBuffers);
    inputDoubles = convert_to_array(A, dimension);

    
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeOfBuffers, NULL, &result);
    if(result != CL_SUCCESS) exit(8);

 
    cl_bool     blockingWrite = CL_TRUE;
    size_t      offset = 0;
    cl_event    dataInputCopyEvent;
    cl_event*   eventsToWait = NULL;
    cl_uint     numEvents = 0;

    result = clEnqueueWriteBuffer(commands, inputBuffer, blockingWrite, offset, sizeOfBuffers, inputDoubles, numEvents, eventsToWait, &dataInputCopyEvent);
    if(result != CL_SUCCESS) exit(9);

   
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeOfBuffers, NULL, &result);
    if(result != CL_SUCCESS) exit(10);

    
    result = 0;
    result |= clSetKernelArg(choldc_gpu, 0, sizeof(cl_mem), &inputBuffer);
    result |= clSetKernelArg(choldc_gpu, 1, sizeof(cl_mem), &outputBuffer);
    result |= clSetKernelArg(choldc_gpu, 2, sizeof(int), &dimension);
    if(result != CL_SUCCESS) exit(11);
    result = 0;

    cl_uint   workDim = 1;
    size_t*   globalWorkOffset = NULL;
    size_t    globalWorkSize =  dimension;
    size_t    localWorkSize = dimension;
    cl_event  kernelExecEvent;
              eventsToWait = NULL;
              numEvents = 0;

  
    result = clGetKernelWorkGroupInfo(choldc_gpu, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(localWorkSize), &localWorkSize, NULL);
    if(result != CL_SUCCESS) exit(12);

    
    result = clEnqueueNDRangeKernel(commands, choldc_gpu, workDim, globalWorkOffset, &globalWorkSize, NULL, numEvents, eventsToWait, &kernelExecEvent);
   
    clWaitForEvents(1, &kernelExecEvent);
    cl_ulong start = 0, end = 0;
    
    clGetEventProfilingInfo(kernelExecEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(kernelExecEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    cl_double g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-09);
    printf("Time \tGPU: % 20.16lf\n", g_NDRangePureExecTimeMs);
    if(result != CL_SUCCESS)
    {
        printf("%d\n", (unsigned int) result);
        exit(13);
    }

   
    cl_bool     blockingRead = CL_TRUE;
                offset = 0;
    float*     resultArray;
    cl_event    readResultsEvent;
                eventsToWait = NULL;
                numEvents = 0;

    resultArray = (float*)malloc(numberOfValues * sizeof(float));

    
    clEnqueueReadBuffer(commands, outputBuffer, blockingRead, offset, sizeOfBuffers, resultArray, numEvents, eventsToWait, &readResultsEvent);
  
    free(platforms);
    free(deviceIDs);
    free(inputDoubles);
    free(resultArray);

return 0;
}
