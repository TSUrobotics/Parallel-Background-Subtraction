#include "device_launch_parameters.h"
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>


#define FRAMES 3	// Number of frames located within the buffer
//#define PIXELS_PER_FRAME 2073600	// Number of pixels per frame
//#define PIXELS_PER_FRAME 307200
#define PIXELS_PER_FRAME 327680

#define BLOCK 512  // Size of blocks, best if it is a power of 2.

using namespace std;
using namespace cv;


// Globals
unsigned char* currentFrame_GPU;
unsigned char* BlockOfFrames_CPU, * BlockOfFrames_GPU;
float* MeanFrame_GPU;
float* BlockOfLogNormalFrames_GPU;
float* MeanLogNormalFrame_GPU;
float* MedianLogNormalFrame_GPU;
float* StdvLogNormalFrame_GPU;

dim3 dimBlock, dimGrid;

void AllocateMemory()
{
	cudaMalloc((void**)&currentFrame_GPU, PIXELS_PER_FRAME * sizeof(unsigned char));
	// This are the set of frames that will be used to generate the log normal frame
	// and the standard deviation frame
	BlockOfFrames_CPU = (unsigned char*)malloc(FRAMES * PIXELS_PER_FRAME * sizeof(unsigned char));
	cudaMalloc((void**)&BlockOfFrames_GPU, FRAMES * PIXELS_PER_FRAME * sizeof(unsigned char));
	cudaMalloc((void**)&BlockOfLogNormalFrames_GPU, FRAMES * PIXELS_PER_FRAME * sizeof(float));

	// Will hold the log normal frame and the standard deviation of the frames minus the log normal
	cudaMalloc((void**)&MeanFrame_GPU, PIXELS_PER_FRAME * sizeof(float));
	cudaMalloc((void**)&MeanLogNormalFrame_GPU, PIXELS_PER_FRAME * sizeof(float));
	cudaMalloc((void**)&MedianLogNormalFrame_GPU, PIXELS_PER_FRAME * sizeof(float));
	cudaMalloc((void**)&StdvLogNormalFrame_GPU, PIXELS_PER_FRAME * sizeof(float));
}

void SetUpCudaDevices()
{
	dimBlock.x = BLOCK;
	dimBlock.y = 1;
	dimBlock.z = 1;

	dimGrid.x = ((PIXELS_PER_FRAME - 1) / BLOCK) + 1;
	dimGrid.y = 1;
	dimGrid.z = 1;
}
//This function creates the log-normal frames for comparison to the newest image
__global__ void creatingBuffer(float* meanFrame, unsigned char* allFrames, float* allFramesLogNormal, int pixelsPerFrame, float* meanlogNormalFrame,
	float* medianlogNormalFrame, float* stdvLogNormalFrame, int frames)
{
	int id;
	//Mean Matrix
	int pixel = threadIdx.x + blockIdx.x * blockDim.x;
	if (pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for (int i = 0; i < frames; i++)
		{
			sum += (int)allFrames[pixel + pixelsPerFrame * i];
		}
		meanFrame[pixel] = sum / (float)frames;
	}
	//Log-Normal Matrix
	if (pixel < pixelsPerFrame)
	{
		for (int i = 0; i < frames; i++)
		{
			//Same screen location (pixel) but moving through frames (i).
			id = pixel + pixelsPerFrame * i;

			allFramesLogNormal[id] = (float)allFrames[id] - meanFrame[pixel];
			allFramesLogNormal[id] = abs(allFramesLogNormal[id]);

			//Can't take log of zero so to be safe check and move it off zero.
			if (allFramesLogNormal[id] == 0.0f)
			{
				allFramesLogNormal[id] = 0.000001f;
			}

			allFramesLogNormal[id] = logf(allFramesLogNormal[id]);

			//allFramesLogNormal[id] = (float)allFrames[id];  // Remove after debugging.
		}
	}
	//Log Mean Matrix
	if (pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for (int i = 0; i < frames; i++)
		{
			sum += allFramesLogNormal[pixel + pixelsPerFrame * i];
		}
		meanlogNormalFrame[pixel] = sum / (float)frames;
	}
	int used[FRAMES], index, count;
	float median = 0.0;
	float small;
	//Log Median Matrix
	if (pixel < pixelsPerFrame)
	{
		for (int i = 0; i < frames; i++)
		{
			used[i] = 0;
		}
		if (frames % 2 == 0)
		{
			int middle2 = frames / 2;
			int middle1 = middle2 - 1;
			index = -1;
			count = 0;
			while (count <= middle2)
			{
				small = 10000000.0f;  //Needs to be a number larger than anything you would get in a log of a pixel.
				for (int i = 0; i < frames; i++)
				{
					if (allFramesLogNormal[pixel + pixelsPerFrame * i] < small && used[i] == 0)
					{
						small = allFramesLogNormal[pixel + pixelsPerFrame * i];
						index = i;
					}
				}
				if (index == -1) printf("\nError no index found\n");
				used[index] = 1;

				if (count == middle1 || count == middle2)
				{
					median += allFramesLogNormal[pixel + pixelsPerFrame * index];
				}

				count++;
			}
			median /= 2.0f;
		}
		else
		{
			int middle = frames / 2;
			index = -1;
			count = 0;
			while (count <= middle)
			{
				small = 10000000.0f;  //Needs to be a number larger than anything you would get in a log of a pixel.
				for (int i = 0; i < frames; i++)
				{
					if (allFramesLogNormal[pixel + pixelsPerFrame * i] < small)
					{
						if (used[i] == 0)
						{
							small = allFramesLogNormal[pixel + pixelsPerFrame * i];
							index = i;
						}
					}
				}
				if (index == -1) printf("\nError no index found\n");
				used[index] = 1;

				if (count == middle)
				{
					median += allFramesLogNormal[pixel + pixelsPerFrame * index];
				}

				count++;
			}
		}
		medianlogNormalFrame[pixel] = median;
	}
	float temp;
	//Log-Normal STD Matrix
	if (pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for (int i = 0; i < frames; i++)
		{
			temp = allFramesLogNormal[pixel + pixelsPerFrame * i] - meanlogNormalFrame[pixel];
			sum += temp * temp;
		}
		stdvLogNormalFrame[pixel] = sqrtf((sum) / (float)(frames - 1));
	}
}


__global__ void CDFfunction(float* median, float* stdvLogNormalFrame, float* MeanLogNormalFrame, unsigned char* currentFrame, int pixelsPerFrame) {
	int pixel = threadIdx.x + blockIdx.x * blockDim.x;
	if (pixel < pixelsPerFrame)
	{
		float newvalue;
		float x = currentFrame[pixel];
		newvalue = -((logf(x) - median[pixel]) - MeanLogNormalFrame[pixel]) / (sqrtf(2) * stdvLogNormalFrame[pixel]);
		float summ = 0.5f + 0.5f * erff(newvalue);

		//Threshold set to 30%
		if (summ >= 0.3) {
			currentFrame[pixel] = (unsigned char)255;
		}
		else {
			currentFrame[pixel] = (unsigned char)0;
		}
	}
}

void errorCheck(const char* message)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("%s", message);
		printf("\n CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(0);
	}
}

void cleanUp()
{
	free(BlockOfFrames_CPU);

	cudaFree(BlockOfFrames_GPU);
	cudaFree(BlockOfLogNormalFrames_GPU);
	cudaFree(MeanFrame_GPU);
	cudaFree(MeanLogNormalFrame_GPU);
	cudaFree(MedianLogNormalFrame_GPU);
	cudaFree(StdvLogNormalFrame_GPU);

}

int main()
{
	AllocateMemory();
	SetUpCudaDevices();

	//This option is set for a recorded video
	VideoCapture cap("video.avi");

	//The next 4 lines are for an attached camera
	//VideoCapture cap;
	//int DeviceID = 0;//0 is set for the first camera instance
	//int apiID = CAP_ANY;
	//cap.open(DeviceID, apiID);

	Mat frame;
	cap.read(frame);
	int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
	int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
	Mat grayimg, Temp;

	if (!cap.isOpened()) {
		std::cout << "Error! Unable to open camera\n";
		cleanUp();
		return -1;
	}
	unsigned char* TempFrame;
	for (int i = 0; i < FRAMES; i++) {
		cap.read(frame);
		frame.convertTo(Temp, CV_8U);
		cvtColor(Temp, grayimg, COLOR_RGB2GRAY);
		TempFrame = grayimg.ptr<unsigned char>(0);
		memcpy(BlockOfFrames_CPU + i * PIXELS_PER_FRAME, TempFrame, sizeof(unsigned char) * PIXELS_PER_FRAME);
	}

	cudaMemcpyAsync(BlockOfFrames_GPU, BlockOfFrames_CPU, PIXELS_PER_FRAME * FRAMES * sizeof(unsigned char), cudaMemcpyHostToDevice);
	errorCheck("copyFramessUp");
	cudaDeviceSynchronize();

	creatingBuffer << <dimGrid, dimBlock >> > (MeanFrame_GPU, BlockOfFrames_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, MeanLogNormalFrame_GPU, MedianLogNormalFrame_GPU, StdvLogNormalFrame_GPU, FRAMES);

	cudaDeviceSynchronize();

	int i = 0;
	Size frame_size(frame_width, frame_height);
	int frames_per_second = 60;

	//Create and initialize the VideoWriter object
	VideoWriter video("E:\\outputVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, Size(frame_width, frame_height));
	while (true) {
		cap.read(frame);
		cvtColor(frame, grayimg, COLOR_RGB2GRAY);
		TempFrame = grayimg.ptr<unsigned char>(0);
		memcpy(BlockOfFrames_CPU + i * PIXELS_PER_FRAME, TempFrame, sizeof(unsigned char) * (PIXELS_PER_FRAME));
		cudaMemcpyAsync(BlockOfFrames_GPU, BlockOfFrames_CPU, PIXELS_PER_FRAME * FRAMES * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (i == 2) {
			i = -1;
		}
		if (i % 1 == 0) {
			creatingBuffer << <dimGrid, dimBlock >> > (MeanFrame_GPU, BlockOfFrames_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, MeanLogNormalFrame_GPU, MedianLogNormalFrame_GPU, StdvLogNormalFrame_GPU, FRAMES);
		}
		cudaMemcpyAsync(currentFrame_GPU, TempFrame, PIXELS_PER_FRAME * sizeof(unsigned char), cudaMemcpyHostToDevice);
		CDFfunction << <dimGrid, dimBlock >> > (MedianLogNormalFrame_GPU, StdvLogNormalFrame_GPU, MeanLogNormalFrame_GPU, currentFrame_GPU, PIXELS_PER_FRAME);
		errorCheck("CDF function");
		cudaMemcpyAsync(TempFrame, currentFrame_GPU, PIXELS_PER_FRAME * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		imshow("image", grayimg);

		//if a key is pressed break.
		if (waitKey(5) >= 0) {
			video.release();
			cleanUp();
			break;
		}
		cvtColor(grayimg, grayimg, COLOR_GRAY2RGB);
		video.write(grayimg);
	}
	cap.release();
	printf("\n DONE \n");
}

