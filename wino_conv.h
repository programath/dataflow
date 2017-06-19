#include "ap_fixed.h"
#include "math.h"
#include "ap_int.h"
//#include "hls_half.h"
//#include "hls_double.h"
#include "math.h"

extern "C" float sqrtf(float);
#define NEW
#define FIXED_NUMBER
#ifdef FIXED_NUMBER
typedef ap_ufixed<16,5> data_tf;
typedef ap_fixed<16,1> data_tw;
typedef ap_fixed<16,5> data_tb;
typedef ap_fixed<22,7> data_tt;//* 8
typedef ap_fixed<20,8> data_tx;
typedef ap_fixed<23,9> data_tp;
typedef ap_fixed<18,8> data_tm;
typedef ap_fixed<18,4> data_ti;
typedef ap_fixed<20,6> data_to;
typedef ap_fixed<16,6> fc_data_tf;
#else
typedef double data_tf;
typedef double data_tb;
typedef double data_tm;
typedef double data_tw;
typedef double data_tt;
typedef double data_ti;
typedef double data_to;
typedef double fc_data_tf;
#endif
typedef unsigned char input_layer;


typedef struct{
    data_tf a,b,c,d;
} quad_tf;


typedef struct{
    data_tw a,b,c,d;
} quad_tw;

#define image_h 320
#define image_w 448
#define image_ch 3
typedef ap_uint<16> layer_1_type;
typedef ap_uint<8> data_r;

typedef struct{
    data_r l,m,r,empty;
} data_ro;
// typedef ap_int<16> data_g;
// typedef ap_int<16> data_i;

// typedef struct
// {
// 	data_r a1, a2, a3;
// } __attribute__ ((packed, aligned(4))) image_data;

typedef struct
{
    data_r a1, a2, a3, a4;
}image_data;
typedef struct
{
    layer_1_type a,b,c,d;
}quad_layer_1_type;
#define in_h  641 // 384  ///////
#define in_w  481  //192 ///////
#define buf_h  (in_h+5)  //386
#define buf_w  (in_w+5)   //194
#define conv_w (in_w-4)
#define in_ch  64
#define out_h  321   //384
#define out_w  241  //192
#define buf_o  (out_w+5)
#define out_ch 64
#define quad_ch (in_ch/4)

//#define cal_block_out_buf_w  in_w

#define corr_h 		80
#define corr_w 		112
#define corr_ch 	64
#define corr_buf_w  200
#define corr_pad	10

#define fc_h 81
#define fc_w 61
#define fc_in_ch 128
#define fc_out_ch 2


#pragma SDS data copy(input[0 : local_in_h * local_in_w * local_in_ch/(nnup_fac*nnup_fac*4)], bias[0 : local_out_w*local_out_h*local_out_ch/4], output[0 : local_out_h * local_out_w * local_out_ch/4], kernel[0 :k_num * 3 * 3 * local_in_ch * local_out_ch/4])///fix
#pragma SDS data access_pattern(input:SEQUENTIAL, bias:SEQUENTIAL, kernel:SEQUENTIAL, output:SEQUENTIAL)
#pragma SDS data sys_port(input:AFI, bias:AFI, output:AFI, kernel:AFI)
#pragma SDS data data_mover(input:AXIDMA_SIMPLE, bias:AXIDMA_SIMPLE, output:AXIDMA_SIMPLE, kernel:AXIDMA_SIMPLE)
#pragma SDS data mem_attribute(input:NON_CACHEABLE, bias:NON_CACHEABLE, output:NON_CACHEABLE, kernel:NON_CACHEABLE)
void wino_conv_wrapper(quad_tf input[in_h*in_w*in_ch/4], quad_tf bias[out_w*out_h*out_ch/4], quad_tf output[out_h*out_w*out_ch/4], quad_tw kernel[3 * 3 * out_ch * in_ch / 4], int local_in_h, int local_in_w, int local_out_h, int local_out_w, int local_in_ch, int local_out_ch, int local_k_sz, int stride, bool R, int nnup_fac, int k_num, int if_4);

//void wino_conv_wrapper(quad_tf * input, quad_tf * bias, quad_tf * output, quad_tw * kernel, int local_in_h, int local_in_w, int local_out_h, int local_out_w, int local_in_ch, int local_out_ch, int local_k_sz, int stride, bool R, int nnup_fac, int k_num, int if_4);


#pragma SDS data access_pattern(input_L:SEQUENTIAL, input_R:SEQUENTIAL, corr_1:SEQUENTIAL, corr_2:SEQUENTIAL)
#pragma SDS data sys_port(input_L:AFI, input_R:AFI, corr_1:AFI, kernel:AFI)
//#pragma SDS data mem_attribute(input_L:NON_CACHEABLE, input_R:NON_CACHEABLE, corr_1:NON_CACHEABLE, corr_2:NON_CACHEABLE)
void corr(quad_tf input_L[corr_h * corr_w * corr_ch / 4], quad_tf input_R[corr_h * corr_w * corr_ch / 4], quad_tf corr_1[corr_h * corr_w * corr_ch / 4], quad_tf corr_2[corr_h * corr_w * 20 / 4]);

#pragma SDS data sys_port(input_L:AFI, input_R:AFI, output_L:AFI, output_R:AFI)
#pragma SDS data access_pattern(input_L:SEQUENTIAL, input_R:SEQUENTIAL, output_L:SEQUENTIAL, output_R:SEQUENTIAL)
void init_layer(image_data input_L[image_h*image_w*3/4], image_data input_R[image_h*image_w*3/4], quad_tf output_L[image_h*image_w], quad_tf output_R[image_h*image_w]);


#pragma SDS data copy(input_1[0 : fc_h*fc_w*16], input_2[0 : fc_h*fc_w*16], kernel_l_1[0:fc_in_ch/4], kernel_l_2[0:fc_in_ch/4], kernel_m_1[0 : fc_in_ch/4], kernel_m_2[0:fc_in_ch/4], kernel_r_1[0 : fc_in_ch/4], kernel_r_2[0 : fc_in_ch/4], output[0:fc_h*fc_w])
//#pragma SDS data sys_port(input_1:AFI, input_2:AFI, kernel_l:AFI, kernel_m:AFI, kernel_r:AFI, output:AFI)
#pragma SDS data mem_attribute(input_1:NON_CACHEABLE, input_2:NON_CACHEABLE, kernel_l_1:NON_CACHEABLE, kernel_l_2:NON_CACHEABLE, kernel_m_1:NON_CACHEABLE,kernel_m_2:NON_CACHEABLE, kernel_r_1:NON_CACHEABLE, kernel_r_2:NON_CACHEABLE, output:NON_CACHEABLE)
#pragma SDS data access_pattern(input_1:SEQUENTIAL, input_2:SEQUENTIAL, kernel_l_1:SEQUENTIAL, kernel_l_2:SEQUENTIAL, kernel_m_1:SEQUENTIAL,kernel_m_2:SEQUENTIAL, kernel_r_1:SEQUENTIAL, kernel_r_2:SEQUENTIAL, output:SEQUENTIAL)
#pragma SDS data data_mover(input_1:AXIDMA_SIMPLE, input_2:AXIDMA_SIMPLE, kernel_l_1:AXIFIFO, kernel_l_2:AXIFIFO, kernel_m_1:AXIFIFO, kernel_m_2:AXIFIFO, kernel_r_1:AXIFIFO, kernel_r_2:AXIFIFO, output:AXIDMA_SIMPLE)
void fc_soft_layer(quad_tf input_1[fc_h*fc_w*16], quad_tf input_2[fc_h*fc_w*16],
                   quad_tw kernel_l_1[fc_in_ch/4], quad_tw kernel_l_2[fc_in_ch/4],
                   quad_tw kernel_m_1[fc_in_ch/4], quad_tw kernel_m_2[fc_in_ch/4],
                   quad_tw kernel_r_1[fc_in_ch/4], quad_tw kernel_r_2[fc_in_ch/4],
                   data_ro output[fc_h*fc_w],
                   data_tw fc_bias0,data_tw fc_bias1,data_tw fc_bias2,data_tw fc_bias3,data_tw fc_bias4,data_tw fc_bias5);
