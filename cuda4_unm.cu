// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_cuda_img.m_size.y ) return;
	if ( l_x >= t_color_cuda_img.m_size.x ) return;

	uchar3 l_bgr = t_color_cuda_img.at3( l_y, l_x);
	// Store BW point to new image
	//t_bw_cuda_img.m_p_uchar1[ l_y * t_bw_cuda_img.m_size.x + l_x ].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
	t_bw_cuda_img.at1(l_y, l_x).x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 16;
	dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_grayscale<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

__global__ void kernel_Colors( CudaImg t_color_cuda_img, CudaImg t_red_cuda_img,CudaImg t_yellow_cuda_img ,CudaImg t_green_cuda_img  )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    uchar3 l_bgr ;//[ l_y * t_color_cuda_img.m_size.x + l_x ];


    // Store BW point to new image
    l_bgr = t_color_cuda_img.at3( l_y, l_x);

	  t_yellow_cuda_img.at3(l_y, l_x).x = l_bgr.x * 0 + l_bgr.y * 0 + l_bgr.z * 1;
	  t_green_cuda_img.at3(l_y, l_x).y = l_bgr.x * 0 + l_bgr.y * 0 + l_bgr.z * 1;
	  t_red_cuda_img.at3(l_y, l_x).z = l_bgr.x * 0 + l_bgr.y * 0 + l_bgr.z * 1;

}

void cu_run_Colors( CudaImg t_color_cuda_img, CudaImg t_red_cuda_img, CudaImg t_green_cuda_img, CudaImg t_yellow_cuda_img  )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_Colors<<< l_blocks, l_threads >>>( t_color_cuda_img, t_red_cuda_img, t_yellow_cuda_img, t_green_cuda_img);

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

__global__ void kernel_resize(CudaImg bP, CudaImg sP, int ratio)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= sP.m_size.y) return;
	if (l_x >= sP.m_size.x)  return;
	uchar4 vysledek = { 0,0,0,0 };
	if (l_y > 0 && l_x > 0 ) {
		vysledek.x = (sP.at4(l_y * ratio + 1, l_x * ratio - 1).x + sP.at4(l_y * ratio + 1, l_x * ratio).x +
			sP.at4(l_y * ratio + 1, l_x * ratio + 1).x + sP.at4(l_y * ratio, l_x * ratio - 1).x +
			sP.at4(l_y * ratio, l_x * ratio).x + sP.at4(l_y * ratio, l_x * ratio + 1).x +
			sP.at4(l_y * ratio - 1, l_x * ratio - 1).x + sP.at4(l_y * ratio - 1, l_x * ratio).x +
			sP.at4(l_y * ratio - 1, l_x * ratio + 1).x) / 9;
		vysledek.y = (sP.at4(l_y * ratio + 1, l_x * ratio - 1).y + sP.at4(l_y * ratio + 1, l_x * ratio).y +
			sP.at4(l_y * ratio + 1, l_x * ratio + 1).y + sP.at4(l_y * ratio, l_x * ratio - 1).y +
			sP.at4(l_y * ratio, l_x * ratio).y + sP.at4(l_y * ratio, l_x * ratio + 1).y +
			sP.at4(l_y * ratio - 1, l_x * ratio - 1).y + sP.at4(l_y * ratio - 1, l_x * ratio).y +
			sP.at4(l_y * ratio - 1, l_x * ratio + 1).y) / 9;
		vysledek.z = (sP.at4(l_y * ratio + 1, l_x * ratio - 1).z + sP.at4(l_y * ratio + 1, l_x * ratio).z +
			sP.at4(l_y * ratio + 1, l_x * ratio + 1).z + sP.at4(l_y * ratio, l_x * ratio - 1).z +
			sP.at4(l_y * ratio, l_x * ratio).z + sP.at4(l_y * ratio, l_x * ratio + 1).z +
			sP.at4(l_y * ratio - 1, l_x * ratio - 1).z + sP.at4(l_y * ratio - 1, l_x * ratio).z +
			sP.at4(l_y * ratio - 1, l_x * ratio + 1).z) / 9;
		vysledek.w = (sP.at4(l_y * ratio + 1, l_x * ratio - 1).w + sP.at4(l_y * ratio + 1, l_x * ratio).w +
			sP.at4(l_y * ratio + 1, l_x * ratio + 1).w + sP.at4(l_y * ratio, l_x * ratio - 1).w +
			sP.at4(l_y * ratio, l_x * ratio).w + sP.at4(l_y * ratio, l_x * ratio + 1).w +
			sP.at4(l_y * ratio - 1, l_x * ratio - 1).w + sP.at4(l_y * ratio - 1, l_x * ratio).w +
			sP.at4(l_y * ratio - 1, l_x * ratio + 1).w) / 9;
	}
	bP.at4(l_y, l_x) = vysledek;
}
void cu_resize(CudaImg bP, CudaImg sP, int ratio)
{
	cudaError_t l_cerr;
	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks((sP.m_size.x + l_block_size - 1) / l_block_size,
		(sP.m_size.y + l_block_size - 1) / l_block_size);
	dim3 l_threads(l_block_size, l_block_size);
	kernel_resize <<< l_blocks, l_threads >>> (bP, sP, ratio);
	if ((l_cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
	cudaDeviceSynchronize();
}
__global__ void kernel_rotate(CudaImg t_color_pic, CudaImg t_color_pic_rotated)
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	int newX = t_color_pic_rotated.m_size.x - l_y;
	int newY = l_x;

	t_color_pic_rotated.m_p_uchar3[ newY * t_color_pic_rotated.m_size.x +  newX] = t_color_pic.m_p_uchar3[ l_y * t_color_pic.m_size.x + l_x ];
}

void cu_rotate(CudaImg sP, CudaImg t_color_pic_rotated)
{
	cudaError_t l_cerr;
	int l_block_size = 16;
	dim3 l_blocks((sP.m_size.x + l_block_size - 1) / l_block_size,
			(sP.m_size.y + l_block_size - 1) / l_block_size);
		dim3 l_threads(l_block_size, l_block_size);
	kernel_rotate<<<l_blocks, l_threads>>>( sP, t_color_pic_rotated );
	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}
