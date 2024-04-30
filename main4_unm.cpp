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
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
//void cu_run_grayscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
//void cu_run_Colors( CudaImg t_color_cuda_img, CudaImg t_red_cuda_img, CudaImg t_yellow_cuda_img, CudaImg t_green_cuda_img );
void cu_resize(CudaImg bP, CudaImg sP, int ratio);
void cu_rotate(CudaImg sP, CudaImg t_color_pic_rotated);
int main( int t_numarg, char **t_arg )
{
	t_arg[1]="kytka.png";
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	/*if ( t_numarg < 2 )
	{
		printf( "Enter picture filename!\n" );
		return 1;
	}*/

	// Load image
	cv::Mat l_bgr_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );

	cv::Mat l_color_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR );
	cv::Mat l_color2_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR );
	cv::Mat l_color3_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR );

	if ( !l_bgr_cv_img.data )
	{
		printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
		return 1;
	}

	// create empty BW image
	/*cv::Mat l_bw_cv_img( l_bgr_cv_img.size(), CV_8U );
	cv::Mat l_red_cv_img( l_bgr_cv_img.size(), CV_8UC3 );
    cv::Mat l_yellow_cv_img( l_bgr_cv_img.size(), CV_8UC3 );
    cv::Mat l_green_cv_img( l_bgr_cv_img.size(), CV_8UC3 );*/
    cv::Mat l_temp(l_bgr_cv_img.size(),CV_8UC4);
    cv::Mat bgr_img = cv::imread( "kytka.png", cv::IMREAD_COLOR );
    cv::imshow("kytka.png", bgr_img);
	// data for CUDA
	//CudaImg l_bgr_cuda_img, l_bw_cuda_img, l_red_cuda_img,l_color_cuda_img,l_yellow_cuda_img,l_green_cuda_img,l_color2_cuda_img,l_color3_cuda_img;

	/*l_bgr_cuda_img.m_size.x = l_bw_cuda_img.m_size.x = l_bgr_cv_img.size().width;
	l_bgr_cuda_img.m_size.y = l_bw_cuda_img.m_size.y = l_bgr_cv_img.size().height;
	l_bgr_cuda_img.m_p_uchar3 = ( uchar3 * ) l_bgr_cv_img.data;
	//l_bw_cuda_img.m_p_uchar1 = ( uchar1 * ) l_bw_cv_img.data;

	l_color_cuda_img.m_size.x = l_red_cuda_img.m_size.x = l_color_cv_img.size().width;
    l_color_cuda_img.m_size.y = l_red_cuda_img.m_size.y = l_color_cv_img.size().height;
    l_color_cuda_img.m_p_uchar3 = ( uchar3 * ) l_color_cv_img.data;

    l_color3_cuda_img.m_size.x = l_green_cuda_img.m_size.x = l_color2_cv_img.size().width;
    l_color3_cuda_img.m_size.y = l_green_cuda_img.m_size.y = l_color2_cv_img.size().height;
    l_color3_cuda_img.m_p_uchar3 = ( uchar3 * ) l_color3_cv_img.data;

	l_color2_cuda_img.m_size.x = l_yellow_cuda_img.m_size.x = l_color3_cv_img.size().width;
    l_color2_cuda_img.m_size.y = l_yellow_cuda_img.m_size.y = l_color3_cv_img.size().height;
    l_color2_cuda_img.m_p_uchar3 = ( uchar3 * ) l_color2_cv_img.data;
*/
	//l_red_cuda_img.m_p_uchar3 = ( uchar3 * ) l_red_cv_img.data;
    //l_green_cuda_img.m_p_uchar3 = ( uchar3 * ) l_green_cv_img.data;
	//l_yellow_cuda_img.m_p_uchar3 = ( uchar3 * ) l_yellow_cv_img.data;

	// Function calling from .cu file
	//cu_run_grayscale( l_bgr_cuda_img, l_bw_cuda_img );
	//cu_run_Colors( l_color_cuda_img, l_red_cuda_img,l_green_cuda_img,l_yellow_cuda_img );
	// Show the Color and BW image
	//cv::imshow( "ColorFull", l_bgr_cv_img );
	//cv::imshow( "GrayScale", l_bw_cv_img );

	
    //cv::imshow( "RED", l_red_cv_img );
    //cv::imshow( "GREEN", l_green_cv_img );
    //cv::imshow( "BLUE", l_yellow_cv_img );
	CudaImg dvakratVetsi,tempImg;
	tempImg.m_size.x=l_temp.size().width;
	tempImg.m_size.y=l_temp.size().height;
	tempImg.m_p_uchar4=(uchar4*)l_temp.data;
	cv::Mat l_bigger(tempImg.m_size.x,tempImg.m_size.y,CV_8UC4);
	dvakratVetsi.m_size.x=l_bigger.size().width;
	dvakratVetsi.m_size.y=l_bigger.size().height;
	dvakratVetsi.m_p_uchar4=(uchar4*)l_bigger.data;
	//l_color_cuda_img.m_size.x = l_red_cuda_img.m_size.x*2;
	//l_color_cuda_img.m_size.y = l_red_cuda_img.m_size.y*2;
	//l_color_cuda_img.m_p_uchar3 = ( uchar3 * ) l_bigger.data;
	//cu_resize(tempImg,dvakratVetsi,2);
	//cv::imshow("TEST",l_bigger);






	cv::Mat img( 200, 350, CV_8UC3 );
	CudaImg l_pic_sign;
	l_pic_sign.m_size.x = img.cols;
	l_pic_sign.m_size.y = img.rows;
	l_pic_sign.m_p_uchar3 = ( uchar3 * ) img.data;


	cv::Mat img_rotated(350, 200, CV_8UC3 );
	CudaImg l_pic_rotated;
	l_pic_rotated.m_size.x = img_rotated.cols;
	l_pic_rotated.m_size.y = img_rotated.rows;
	l_pic_rotated.m_p_uchar3 = ( uchar3 * ) img_rotated.data;
	cu_rotate(l_pic_sign, l_pic_rotated);
	cv::imshow("rotated", img_rotated);

	cv::waitKey( 0 );
}

