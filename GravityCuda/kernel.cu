
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono> 
#include <stdio.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <iostream>
#include <stdlib.h> 
#include <vector>
#include <windows.h>
using namespace std;
using namespace std::chrono;
#define SCREEN_WIDTH 1600
#define SCREEN_HEIGHT 1000
#define NUMBER_BALL 128

__device__ float G = 6.67430e-11;
__device__ float M = 1e11;
__device__ float elastic = 0.85;
__device__ float R = 10;
__device__ float t = 1;
static float ballsize = 20;
void init(GLfloat arg[])
{
	for (int i = 0; i < NUMBER_BALL * 2; i+=2)
	{arg[i] = rand() % 1300 + 150;
		arg[i+1] = rand() % 700 + 150;
	}
}
void initV(GLfloat arg[])
{

	for (int n = 0; n < NUMBER_BALL * 2; n++)
	{
		arg[n] = rand() % 6-3;
	}
}
__global__
void collisionCudaShared(GLfloat *X, GLfloat*V)
{
	__shared__ GLfloat s[NUMBER_BALL * 4];
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	if (i < NUMBER_BALL * 2)
	{
		s[i] = X[i];
		s[i + 1] = X[i + 1];
		s[i + NUMBER_BALL * 2] = V[i];
		s[i + 1 + NUMBER_BALL * 2] = V[i + 1];

		if (s[i] <= R)
		{
			s[NUMBER_BALL * 2 + i] = -s[NUMBER_BALL * 2 + i] * elastic;
			s[i] = R;
		}
		if (s[i] >= SCREEN_WIDTH - R)
		{
			s[NUMBER_BALL * 2 + i] = -s[NUMBER_BALL * 2 + i] * elastic;
			s[i] = SCREEN_WIDTH - R;
		}
		if (s[i + 1] <= R)
		{
			s[NUMBER_BALL * 2 + i + 1] = -s[NUMBER_BALL * 2 + i + 1] * elastic;
			s[i + 1] = R;
		}
		if (s[i + 1] >= SCREEN_HEIGHT - R)
		{
			s[NUMBER_BALL * 2 + i + 1] = -s[NUMBER_BALL * 2 + i + 1] * elastic;
			s[i + 1] = SCREEN_HEIGHT - R;
		}
		for (int j = i + 2; j < NUMBER_BALL * 2; j += 2)
		{
			GLfloat dx = s[j] - s[i];
			GLfloat dy = s[j + 1] - s[i + 1];
			GLfloat d = sqrt(dx*dx + dy * dy);
			GLfloat dvx = s[NUMBER_BALL * 2 + j] - s[NUMBER_BALL * 2 + i];
			GLfloat dvy = s[NUMBER_BALL * 2 + j + 1] - s[NUMBER_BALL * 2 + i + 1];
			GLfloat Vxj = s[NUMBER_BALL * 2 + j];
			GLfloat Vyj = s[NUMBER_BALL * 2 + j + 1];
			GLfloat Vxi = s[NUMBER_BALL * 2 + i];
			GLfloat Vyi = s[NUMBER_BALL * 2 + i + 1];
			if (d <= 2 * R)
			{
				GLfloat midx = 0.5*(s[j] + s[i]);
				GLfloat midy = 0.5*(s[j + 1] + s[i + 1]);
				s[j] = (s[j] - midx) * 2 * R / d + midx;
				s[j + 1] = (s[j + 1] - midy) * 2 * R / d + midy;
				s[i] = (s[i] - midx) * 2 * R / d + midx;
				s[i + 1] = (s[i + 1] - midy) * 2 * R / d + midy;
				Vxj = s[NUMBER_BALL * 2 + j] - (dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				Vyj = s[NUMBER_BALL * 2 + j + 1] - (dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				Vxi = s[NUMBER_BALL * 2 + i] - (-dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				Vyi = s[NUMBER_BALL * 2 + i + 1] - (-dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				s[NUMBER_BALL * 2 + j] = Vxj * elastic;
				s[NUMBER_BALL * 2 + j + 1] = Vyj * elastic;
				s[NUMBER_BALL * 2 + i] = Vxi * elastic;
				s[NUMBER_BALL * 2 + i + 1] = Vyi * elastic;
			}
		}
		__syncthreads();

		X[i] = s[i];
		X[i + 1] = s[i + 1];
		V[i] = s[i + NUMBER_BALL * 2];
		V[i + 1] = s[i + 1 + NUMBER_BALL * 2];
	}
}
__global__
void collisionCuda(GLfloat *X, GLfloat*V)
{
	int i = (blockIdx.x * blockDim.x+threadIdx.x )* 2;
	for (int j = i + 2; j < NUMBER_BALL * 2; j += 2)
	{
		GLfloat dx = X[j] - X[i];
		GLfloat dy = X[j + 1] - X[i + 1];
		GLfloat d = sqrt(dx*dx + dy * dy);
		GLfloat dvx = V[j] - V[i];
		GLfloat dvy = V[j + 1] - V[i + 1];
		GLfloat Vxj = V[j];
		GLfloat Vyj = V[j + 1];
		GLfloat Vxi = V[i];
		GLfloat Vyi = V[i + 1];
		if (d <= 2 * R)
		{
			GLfloat midx = 0.5*(X[j] + X[i]);
			GLfloat midy = 0.5*(X[j + 1] + X[i + 1]);
			X[j] = (X[j] - midx) * 2 * R / d + midx;
			X[j + 1] = (X[j + 1] - midy) * 2 * R / d + midy;
			X[i] = (X[i] - midx) * 2 * R / d + midx;
			X[i + 1] = (X[i + 1] - midy) * 2 * R / d + midy;
			Vxj = V[j] - (dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
			Vyj = V[j + 1] - (dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
			Vxi = V[i] - (-dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
			Vyi = V[i + 1] - (-dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
			V[j] = Vxj * elastic;
			V[j + 1] = Vyj * elastic;
			V[i] = Vxi * elastic;
			V[i + 1] = Vyi * elastic;
		}
	}
	if (X[i] <= R)
	{
		V[i] = -V[i] * elastic;
		X[i] = R;
	}
	if (X[i] >= SCREEN_WIDTH - R)
	{
		V[i] = -V[i] * elastic;
		X[i] = SCREEN_WIDTH - R;
	}
	if (X[i + 1] <= R)
	{
		V[i + 1] = -V[i + 1] * elastic;
		X[i + 1] = R;
	}
	if (X[i + 1] >= SCREEN_HEIGHT - R)
	{
		V[i + 1] = -V[i + 1] * elastic;
		X[i + 1] = SCREEN_HEIGHT - R;
	}
}
/*void collision(GLfloat X[], GLfloat V[])
{
	for (int i = 0; i < NUMBER_BALL * 2; i += 2)
	{
		for (int j = i + 2; j < NUMBER_BALL * 2; j += 2)
		{
			GLfloat dx = X[j] - X[i];
			GLfloat dy = X[j + 1] - X[i + 1];
			GLfloat d = sqrt(dx*dx + dy * dy);
			GLfloat dvx = V[j] - V[i];
			GLfloat dvy = V[j + 1] - V[i + 1];
			if (d <= 2 * R)
			{
				GLfloat midx = 0.5*(X[j] + X[i]);
				GLfloat midy = 0.5*(X[j + 1] + X[i + 1]);
				X[j] = (X[j] - midx) * 2 * R / d + midx;
				X[j + 1] = (X[j + 1] - midy) * 2 * R / d + midy;
				X[i] = (X[i] - midx) * 2 * R / d + midx;
				X[i + 1] = (X[i + 1] - midy) * 2 * R / d + midy;
				GLfloat Vxj = V[j] - (dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				GLfloat Vyj = V[j + 1] - (dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				GLfloat Vxi = V[i] - (-dx)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				GLfloat Vyi = V[i + 1] - (-dy)*(dvx*dx + dvy * dy) / (dx*dx + dy * dy);
				V[j] = Vxj * elastic;
				V[j + 1] = Vyj * elastic;
				V[i] = Vxi * elastic;
				V[i + 1] = Vyi * elastic;
			}
		}
		if (X[i] <= R)
		{
			V[i] = -V[i] * elastic;
			X[i] = R;
		}
		if (X[i] >= SCREEN_WIDTH - R)
		{
			V[i] = -V[i] * elastic;
			X[i] = SCREEN_WIDTH - R;
		}
		if (X[i + 1] <= R)
		{
			V[i + 1] = -V[i + 1] * elastic;
			X[i + 1] = R;
		}
		if (X[i + 1] >= SCREEN_HEIGHT - R)
		{
			V[i + 1] = -V[i + 1] * elastic;
			X[i + 1] = SCREEN_HEIGHT - R;
		}
	}
}*/
__global__
void gravityCudaShared(GLfloat *X, GLfloat *V)
{
	__shared__ GLfloat s[NUMBER_BALL * 4];
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	if (i < NUMBER_BALL * 2)
	{
		s[i] = X[i];
		s[i + 1] = X[i + 1];
		s[i + NUMBER_BALL * 2] = V[i];
		s[i + 1 + NUMBER_BALL * 2] = V[i + 1];

		GLfloat newXx = 0;
		GLfloat newXy = 0;
		GLfloat dx = 0;
		GLfloat dy = 0;
		GLfloat d = 0;
		GLfloat ax = 0;
		GLfloat ay = 0;
		for (int j = i + 2; j < NUMBER_BALL * 2; j += 2)
		{
			dx = s[j] - s[i];
			dy = s[j + 1] - s[i + 1];
			d = sqrt(dx*dx + dy * dy);
			ax += dx / (d*d*d);
			ay += dy / (d*d*d);
		}
		for (int j = 0; j < i; j += 2)
		{
			dx = s[j] - s[i];
			dy = s[j + 1] - s[i + 1];
			d = sqrt(dx*dx + dy * dy);
			ax += dx / (d*d*d);
			ay += dy / (d*d*d);
		}
		ax *= G * M;
		ay *= G * M;
		newXx = s[i] + s[NUMBER_BALL * 2 + i] + 0.5*ax*t*t;
		newXy = s[i + 1] + s[NUMBER_BALL * 2 + i + 1] + 0.5*ay*t*t;
		V[i] = s[NUMBER_BALL * 2 + i] + ax * t;
		V[i + 1] = s[NUMBER_BALL * 2 + i + 1] + ay * t;

		__syncthreads();
		X[i] = newXx;
		X[i + 1] = newXy;
	}
}
__global__
void gravityCuda(GLfloat *X, GLfloat *V)
{
	GLfloat newXx = 0;
	GLfloat newXy = 0;
	GLfloat dx = 0;
	GLfloat dy = 0;
	GLfloat d = 0;
	GLfloat ax = 0;
	GLfloat ay = 0;
	int index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	for (int j=index+2;j<NUMBER_BALL*2;j+=2)
	{
			dx = X[j] - X[index];
			dy = X[j + 1] - X[index + 1];
			d = sqrt(dx*dx + dy * dy);
			ax += dx / (d*d*d);
			ay += dy / (d*d*d);
	}
	for (int j = 0; j < index; j += 2)
	{
		dx = X[j] - X[index];
		dy = X[j + 1] - X[index + 1];
		d = sqrt(dx*dx + dy * dy);
		ax += dx / (d*d*d);
		ay += dy / (d*d*d);
	}
	ax *= G * M;
	ay *= G * M;
	newXx=X[index]+V[index]+0.5*ax*t*t;
	newXy = X[index+1]+V[index+1]+0.5*ay*t*t;
	V[index] = V[index] + ax * t;
	V[index + 1] = V[index + 1] + ay * t;

	__syncthreads();
	X[index] = newXx;
	X[index+1] = newXy;

}
/*void gravity(GLfloat X[], GLfloat V[])
{
	GLfloat dx = 0;
	GLfloat dy = 0;
	GLfloat d = 0;
	GLfloat ax = 0;
	GLfloat ay = 0;
	GLfloat newX[NUMBER_BALL * 2] = { 0 };
	for (int i = 0; i < NUMBER_BALL * 2; i += 2)
	{
		ax = 0;
		ay = 0;
		for (int j = 0; j < NUMBER_BALL * 2; j += 2)
		{
			if (j != i)
			{
				dx = X[j] - X[i];
				dy = X[j + 1] - X[i + 1];
				d = sqrt(dx*dx + dy * dy);
				ax += dx / (d*d*d);
				ay += dy / (d*d*d);
			}
		}

		ax *= G * M;
		ay *= G * M;
		newX[i] = X[i] + V[i] + 0.5*ax*t*t;
		newX[i + 1] = X[i + 1] + V[i + 1] + 0.5*ay*t*t;
		V[i] = V[i] + ax * t;
		V[i + 1] = V[i + 1] + ay * t;
	}
	for (int i = 0; i < NUMBER_BALL * 2; i++)
	{
		X[i] = newX[i];
	}
}*/

__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}
int main(void)
{
	GLFWwindow *window;
	// Initialize the library
	if (!glfwInit())
	{
		return -1;
	}

	// Create a windowed mode window and its OpenGL context
	window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Hello World", NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);
	glViewport(0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT); // specifies the part of the window to which OpenGL will draw (in pixels), convert from normalised to pixels
	glMatrixMode(GL_PROJECTION); // projection matrix defines the properties of the camera that views the objects in the world coordinate frame. Here you typically set the zoom factor, aspect ratio and the near and far clipping planes
	glLoadIdentity(); // replace the current matrix with the identity matrix and starts us a fresh because matrix transforms such as glOrpho and glRotate cumulate, basically puts us at (0, 0, 0)
	glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, 0, 1); // essentially set coordinate system
	glMatrixMode(GL_MODELVIEW); // (default matrix mode) modelview matrix defines how your objects are transformed (meaning translation, rotation and scaling) in your world
	glLoadIdentity(); // same as above comment

	float *X, *V;
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&X, 2 * NUMBER_BALL * sizeof(float));
	cudaMallocManaged(&V, 2 * NUMBER_BALL * sizeof(float));
	init(X);
	initV(V);
	int blockSize = (NUMBER_BALL + 31) / 32 * 32;
	int numBlocks = (NUMBER_BALL + blockSize - 1) / blockSize;

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window))
	{
		auto start = high_resolution_clock::now();
		///////global memory 
		//collisionCuda << <numBlocks, blockSize >> > (X, V); 
		//gravityCuda << <numBlocks, blockSize >> > (X, V); 
		///////shared memory
		collisionCudaShared << <numBlocks, blockSize >> > (X, V); 
		gravityCudaShared << <numBlocks, blockSize >> > (X, V); 
		cudaDeviceSynchronize();
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);
		cout << "cost: " << duration.count() << " ms" << endl;

		glClear(GL_COLOR_BUFFER_BIT);
		// render OpenGL here
		glEnable(GL_POINT_SMOOTH);
		glEnableClientState(GL_VERTEX_ARRAY);
		
		glVertexPointer(2, GL_FLOAT, 0, X);
		glPointSize(ballsize);

		glDrawArrays(GL_POINTS, 0, NUMBER_BALL);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_POINT_SMOOTH);
		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();
		//Sleep(20);
	}

	glfwTerminate();
	cudaFree(X);
	cudaFree(V);
	return 0;
}
