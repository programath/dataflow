#include "wino_conv.h"
#include <iostream>
using namespace std;
#define Tm	4
#define Tn	4

#define out_tile_sz	4
#define in_tile_sz	6

#ifdef FIXED_NUMBER
ap_fixed<25,1>tmp_64_par = 0.015625;
ap_fixed<25,2>mean_b = 1.64204688;
ap_fixed<25,2>mean_g = 1.82467188;
ap_fixed<25,2>mean_r = 1.9325;
#else
double tmp_64_par = 0.015625;
#endif

//typedef data_tt T;
template<typename T>
T div_2(T in)
{
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = in >> 1;//(in >> 3) + (in >> 5) + (in >> 7) + (in >> 9) + (in >> 11) + (in >> 13) + (in >> 15) + (in >> 17);
#else
    out = in / 2;
#endif
    return out;
}
//
template<typename T>
T div_4(T in)
{
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = in >> 2;//(in >> 3) + (in >> 5) + (in >> 7) + (in >> 9) + (in >> 11) + (in >> 13) + (in >> 15) + (in >> 17);
#else
    out = in / 4;
#endif
    return out;
}
//
template<typename T>
T div_3(T in)
{
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = (in >> 2) + (in >> 4) + (in >> 6) + (in >> 8) + (in >> 10) + (in >> 12) + (in >> 14) + (in >> 16);// + (in >> 18);// + (in >> 20) + (in >> 22);// + (in >> 24);
#else
    out = in / 3;
#endif
    return out;
}

template<typename T>
T div_3_mul_2(T in)
{
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = (in >> 1) + (in >> 3) + (in >> 5) + (in >> 7) + (in >> 9) + (in >> 11) + (in >> 13) + (in >> 15);// + (in >> 18);// + (in >> 20) + (in >> 22);// + (in >> 24);
#else
    out = in / 3;
#endif
    return out;
}
//
template<typename T>
T mul_2(T in){
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = (in << 1);// + (in >> 6) + (in >> 8) + (in >> 10) + (in >> 12) + (in >> 14) + (in >> 16) + (in >> 18);
#else
    out = in * 2;
#endif
    return out;
}

template<typename T>
T mul_4(T in){
#pragma HLS INLINE
    T out;
#ifdef FIXED_NUMBER
    out = (in << 2);// + (in >> 6) + (in >> 8) + (in >> 10) + (in >> 12) + (in >> 14) + (in >> 16) + (in >> 18);
#else
    out = in * 4;
#endif
    return out;
}

void Winograd_Mul(data_tx transform2_input[6][6], data_tw kernel[3][3], data_to output[4][4])
{
#pragma HLS INLINE off
#pragma HLS PIPELINE
    data_tt transform3_input[6][6];
    data_ti transform1_kernel[6][3];
    data_ti transform2_kernel[6][6];
    data_tp transform_output[4][6];

    int i,j,k;

// #pragma HLS array_partition variable=transform1_input complete dim=0
// #pragma HLS array_partition variable=transform2_input complete dim=0
#pragma HLS array_partition variable=transform1_kernel complete dim=0
#pragma HLS array_partition variable=transform2_kernel complete dim=0
#pragma HLS array_partition variable=transform_output complete dim=0

#ifdef NEW

    G_kernel_loop:
    for(i = 0; i < 3; ++i)
    {
        transform1_kernel[0][i] = (kernel[0][i]);
        transform1_kernel[1][i] = (kernel[0][i] + kernel[1][i] + kernel[2][i]);
        transform1_kernel[2][i] = (kernel[0][i] - kernel[1][i] + kernel[2][i]);
        transform1_kernel[3][i] = (div_2(kernel[0][i]) + ((kernel[1][i]) + mul_2(kernel[2][i])));
        transform1_kernel[4][i] = (div_2(kernel[0][i]) - ((kernel[1][i]) - mul_2(kernel[2][i])));
        transform1_kernel[5][i] = kernel[2][i];
    }

    transform1_kernel_G_transposition_loop:
    for(i = 0; i < 6; ++i)
    {
        transform2_kernel[i][0] = (transform1_kernel[i][0]);
        transform2_kernel[i][1] = (transform1_kernel[i][0] + transform1_kernel[i][1] + transform1_kernel[i][2]);
        transform2_kernel[i][2] = (transform1_kernel[i][0] - transform1_kernel[i][1] + transform1_kernel[i][2]);
        transform2_kernel[i][3] = (div_2(transform1_kernel[i][0]) + (transform1_kernel[i][1] + mul_2(transform1_kernel[i][2])));
        transform2_kernel[i][4] = (div_2(transform1_kernel[i][0]) - (transform1_kernel[i][1] - mul_2(transform1_kernel[i][2])));
        transform2_kernel[i][5] = transform1_kernel[i][2];
    }

    GEMM_loop:
    for(i = 0; i < 6; ++i)
        for(j = 0; j < 6; ++j)
        {
            transform3_input[i][j] = transform2_kernel[i][j] * transform2_input[i][j];
        }

    A_transposition_GEMM_loop:
    for(i = 0; i < 6; ++i)
    {
        transform_output[0][i] = transform3_input[0][i] - div_3(mul_2((data_tt)(transform3_input[1][i] + transform3_input[2][i])) - div_2((data_tt)(transform3_input[3][i] + transform3_input[4][i])));
        transform_output[1][i] = div_3(mul_2((data_tt)(-transform3_input[1][i] + transform3_input[2][i])) + (data_tt)(transform3_input[3][i] - transform3_input[4][i]));
        transform_output[2][i] = div_3_mul_2((data_tt)(-transform3_input[1][i] - transform3_input[2][i] + transform3_input[3][i] + transform3_input[4][i]));
        transform_output[3][i] = div_3_mul_2((data_tt)(-transform3_input[1][i] + transform3_input[2][i] + mul_2((data_tt)(transform3_input[3][i] - transform3_input[4][i])))) + mul_4((data_tt)(transform3_input[5][i]));
        // transform_output[0][i] = transform3_input[0][i] - div_3(mul_2((transform3_input[1][i] + transform3_input[2][i])) - div_2((data_tt)(transform3_input[3][i] + transform3_input[4][i])));
        // transform_output[1][i] = div_3(mul_2((-transform3_input[1][i] + transform3_input[2][i])) + (transform3_input[3][i] - transform3_input[4][i]));
        // transform_output[2][i] = div_3(mul_2((-transform3_input[1][i] - transform3_input[2][i] + transform3_input[3][i] + transform3_input[4][i])));
        // transform_output[3][i] = div_3(mul_2((-transform3_input[1][i] + transform3_input[2][i] + mul_2((transform3_input[3][i] - transform3_input[4][i]))))) + mul_4((transform3_input[5][i]));

    }

    A_loop:
    for(i = 0; i < 4; ++i)
    {
//		output[i][0] = transform_output[i][0] - div_3(mul_2((data_tt)(transform_output[i][1] + transform_output[i][2])) -  div_2((data_tt)(transform_output[i][3] + transform_output[i][4])));
//		output[i][1] = div_3(mul_2((data_tt)(-transform_output[i][1] + transform_output[i][2])) + (data_tt)(transform_output[i][3] - transform_output[i][4]));
//		output[i][2] = div_3(mul_2((data_tt)(-transform_output[i][1] - transform_output[i][2] + transform_output[i][3] + transform_output[i][4])));
//		output[i][3] = div_3(mul_2((data_tt)(-transform_output[i][1] + transform_output[i][2] + mul_2((data_tt)(transform_output[i][3] - transform_output[i][4]))))) + mul_4((data_tt)(transform_output[i][5]));
        output[i][0] = transform_output[i][0] - div_3(mul_2((transform_output[i][1] + transform_output[i][2])) -  div_2((transform_output[i][3] + transform_output[i][4])));
        output[i][1] = div_3(mul_2((-transform_output[i][1] + transform_output[i][2])) + (transform_output[i][3] - transform_output[i][4]));
        output[i][2] = div_3_mul_2((-transform_output[i][1] - transform_output[i][2] + transform_output[i][3] + transform_output[i][4]));
        output[i][3] = div_3_mul_2((-transform_output[i][1] + transform_output[i][2] + mul_2((transform_output[i][3] - transform_output[i][4])))) + mul_4((transform_output[i][5]));

    }

#else
    B_transposition_input_loop:
    for(i = 0; i < 6; ++i)
    {
        transform1_input[0][i] = (4)*input[0][i] - mul_5(input[2][i]) + input[4][i];
        transform1_input[1][i] = (-4)*(input[1][i] + input[2][i]) + input[3][i] + input[4][i];
        transform1_input[2][i] = (4)*(input[1][i] - input[2][i]) - input[3][i] + input[4][i];
        transform1_input[3][i] = (-2)*(input[1][i] - input[3][i]) - input[2][i] + input[4][i];
        transform1_input[4][i] = (2)*(input[1][i] - input[3][i]) - input[2][i] + input[4][i];
        transform1_input[5][i] = (4)*input[1][i] - mul_5(input[3][i]) + input[5][i];
    }

    transform1_input_B_loop:
    for(i = 0; i < 6; ++i)
    {
        transform2_input[i][0] = (4)*transform1_input[i][0] - mul_5(transform1_input[i][2]) + transform1_input[i][4];
        transform2_input[i][1] = (-4)*(transform1_input[i][1] + transform1_input[i][2]) + transform1_input[i][3] + transform1_input[i][4];
        transform2_input[i][2] = (4)*(transform1_input[i][1] - transform1_input[i][2]) - transform1_input[i][3] + transform1_input[i][4];
        transform2_input[i][3] = (-2)*(transform1_input[i][1] - transform1_input[i][3]) - transform1_input[i][2] + transform1_input[i][4];
        transform2_input[i][4] = (2)*(transform1_input[i][1] - transform1_input[i][3]) - transform1_input[i][2] + transform1_input[i][4];
        transform2_input[i][5] = (4)*transform1_input[i][1]- mul_5(transform1_input[i][3]) + transform1_input[i][5];
    }

    G_kernel_loop:
    for(i = 0; i < 3; ++i)
    {
        transform1_kernel[0][i] = (kernel[0][i]/4);
        data_tw tmp1 = -kernel[0][i] - kernel[1][i] - kernel[2][i];
        transform1_kernel[1][i] = div_6(tmp1);
        data_tw tmp2 = -kernel[0][i] + kernel[1][i] - kernel[2][i];
        transform1_kernel[2][i] = div_6(tmp2);
        data_tw tmp3 = (kernel[0][i]/4) + (kernel[1][i]/2) + kernel[2][i];
        transform1_kernel[3][i] = div_6(tmp3);
        data_tw tmp4 = (kernel[0][i]/4) - (kernel[1][i]/2) + kernel[2][i];
        transform1_kernel[4][i] = div_6(tmp4);
        transform1_kernel[5][i] = kernel[2][i];
    }

    transform1_kernel_G_transposition_loop:
    for(i = 0; i < 6; ++i)
    {
        transform2_kernel[i][0] = transform1_kernel[i][0]/4;
        data_tw tmp1 = -transform1_kernel[i][0] - transform1_kernel[i][1] - transform1_kernel[i][2];
        transform2_kernel[i][1] = div_6(tmp1);
        data_tw tmp2 = -transform1_kernel[i][0] + transform1_kernel[i][1] - transform1_kernel[i][2];
        transform2_kernel[i][2] = div_6(tmp2);
        data_tw tmp3 = (transform1_kernel[i][0]/4) + (transform1_kernel[i][1]/2) + transform1_kernel[i][2];
        transform2_kernel[i][3] = div_6(tmp3);
        data_tw tmp4 = (transform1_kernel[i][0]/4) - (transform1_kernel[i][1]/2) + transform1_kernel[i][2];
        transform2_kernel[i][4] = div_6(tmp4);
        transform2_kernel[i][5] = transform1_kernel[i][2];
    }
    GEMM_loop:
    for(i = 0; i < 6; ++i)
        for(j = 0; j < 6; ++j)
        {
            transform1_input[i][j] = transform2_kernel[i][j] * transform2_input[i][j];
        }

    A_transposition_GEMM_loop:
    for(i = 0; i < 6; ++i)
    {
        transform_output[0][i] = transform1_input[0][i] + transform1_input[1][i] + transform1_input[2][i] + transform1_input[3][i] + transform1_input[4][i];
        transform_output[1][i] = transform1_input[1][i] - transform1_input[2][i] + (2)*(transform1_input[3][i] - transform1_input[4][i]);
        transform_output[2][i] = transform1_input[1][i] + transform1_input[2][i] + (4)*(transform1_input[3][i] + transform1_input[4][i]);
        transform_output[3][i] = transform1_input[1][i] - transform1_input[2][i] + (8)*(transform1_input[3][i] - transform1_input[4][i]) + transform1_input[5][i];
    }

    A_loop:
    for(i = 0; i < 4; ++i)
    {
        output[i][0] = transform_output[i][0] + transform_output[i][1] + transform_output[i][2] + transform_output[i][3] + transform_output[i][4];
        output[i][1] = transform_output[i][1] - transform_output[i][2] + (2)*(transform_output[i][3] - transform_output[i][4]);
        output[i][2] = transform_output[i][1] + transform_output[i][2] + (4)*(transform_output[i][3] + transform_output[i][4]);
        output[i][3] = transform_output[i][1] - transform_output[i][2] + (8)*(transform_output[i][3] - transform_output[i][4]) + transform_output[i][5];
    }
#endif
}


void tozero(data_to a[Tn][out_tile_sz][out_tile_sz])
{
#pragma HLS INLINE
    for(int t = 0; t < Tn; ++t){
        for(int i = 0; i < out_tile_sz; ++i){
            for(int j = 0; j < out_tile_sz; ++j){
#pragma HLS PIPELINE
                a[t][i][j] = 0;
            }
        }
    }
}

void cal_block(data_tf in_buf_1[buf_w][in_ch], data_tf in_buf_2[buf_w][in_ch], data_tf in_buf_3[buf_w][in_ch], data_tf in_buf_4[buf_w][in_ch], data_tf in_buf_5[buf_w][in_ch], data_tf in_buf_6[buf_w][in_ch], data_tw k_buf[out_ch][in_ch][3][3], data_to out_buf[4][buf_o][out_ch], int o_w, int m_off, int max_n, int in_off, int stride){

#pragma HLS INLINE off

    col_loop:

    data_to wino_rst[out_tile_sz][out_tile_sz];
#pragma HLS array_partition variable=wino_rst complete dim=0
    //

    data_tw kernel_tile[3][3];
#pragma HLS array_partition variable=kernel_tile complete dim=0

    data_to add_rst[Tn][out_tile_sz][out_tile_sz];
#pragma HLS array_partition variable=add_rst complete dim=0

    for(int col = 0; col < o_w ; col += 4){
//#pragma HLS LOOP_TRIPCOUNT min=28 max=28
#pragma HLS LOOP_TRIPCOUNT min=61 max=61
        for(int n_off = 0; n_off < max_n; ++n_off){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE
            tozero(add_rst);
            in_channel_loop:
            for(int tm = 0; tm < Tm; ++tm){
                data_to in_tile[in_tile_sz][in_tile_sz];
#pragma HLS array_partition variable=in_tile complete dim=0
                for(int i = 0; i < 6; ++i){
                    in_tile[0][i] = in_buf_1[col + i][tm + Tm * m_off];
                    in_tile[1][i] = in_buf_2[col + i][tm + Tm * m_off];
                    in_tile[2][i] = in_buf_3[col + i][tm + Tm * m_off];
                    in_tile[3][i] = in_buf_4[col + i][tm + Tm * m_off];
                    in_tile[4][i] = in_buf_5[col + i][tm + Tm * m_off];
                    in_tile[5][i] = in_buf_6[col + i][tm + Tm * m_off];
                }

                data_tm transform1_input[6][6];
                data_tx transform2_input[6][6];
#pragma HLS array_partition variable=transform1_input complete dim=0
#pragma HLS array_partition variable=transform2_input complete dim=0

                B_transposition_input_loop:
                for(int i = 0; i < 6; ++i)
                {
                    transform1_input[0][i] = in_tile[0][i] - in_tile[2][i] - div_4(in_tile[2][i] - in_tile[4][i]);
                    transform1_input[1][i] = -((in_tile[1][i] + in_tile[2][i])) + div_4(in_tile[3][i] + in_tile[4][i]);
                    transform1_input[2][i] = (in_tile[1][i] - in_tile[2][i]) - div_4(in_tile[3][i] - in_tile[4][i]);
                    transform1_input[3][i] = -(in_tile[1][i] - in_tile[3][i]) - div_2(in_tile[2][i] - in_tile[4][i]);
                    transform1_input[4][i] = (in_tile[1][i] - in_tile[3][i]) - div_2(in_tile[2][i] - in_tile[4][i]);
                    transform1_input[5][i] = in_tile[1][i] - in_tile[3][i] - div_4(in_tile[3][i] - in_tile[5][i]);
                }

                transform1_input_B_loop:
                for(int i = 0; i < 6; ++i){

                    transform2_input[i][0] = transform1_input[i][0] - transform1_input[i][2] - div_4(transform1_input[i][2] - transform1_input[i][4]);
                    transform2_input[i][1] = -(transform1_input[i][1] + transform1_input[i][2]) + div_4(transform1_input[i][3] + transform1_input[i][4]);
                    transform2_input[i][2] = (transform1_input[i][1] - transform1_input[i][2]) - div_4(transform1_input[i][3] - transform1_input[i][4]);
                    transform2_input[i][3] = (-transform1_input[i][1] + transform1_input[i][3]) - div_2(transform1_input[i][2] - transform1_input[i][4]);
                    transform2_input[i][4] = (transform1_input[i][1] - transform1_input[i][3]) - div_2(transform1_input[i][2] - transform1_input[i][4]);
                    transform2_input[i][5] = transform1_input[i][1] - transform1_input[i][3] - div_4(transform1_input[i][3] - transform1_input[i][5]);
                }

                out_channel_loop:
                for(int tn = 0; tn < Tn; ++tn){
                    for(int i = 0; i < 3; ++i){
                        for(int j = 0; j < 3; ++j){
                            kernel_tile[i][j] = k_buf[tn + Tn * n_off][tm + Tm * m_off][i][j];
                        }
                    }

                    Winograd_Mul(transform2_input, kernel_tile, wino_rst);
                    for(int tr = 0; tr < out_tile_sz; ++tr){
                        for(int tc = 0; tc < out_tile_sz; ++tc){
                            add_rst[tn][tr][tc] += wino_rst[tr][tc];
                        }
                    }
                }
            }
            // notice that base address colidx=2k might be 4m or 4m+2 which confuses the compiler
            // force base address to be multiples of 4, which means 4*idx
            // if colidx == 4m than keep every thing the same, or (4m+2) results in the indices become (4m+2), (4m+3), (4(m+1)+0), (4(m+1)+1)
            int icolidx = col >> 2;
            int tcolidx = icolidx >> 1;
            for(int tn = 0; tn < Tn; ++tn){
                for(int tr = 0; tr < out_tile_sz; ++tr){
                    int idx1,idx2,idx3,idx4;
                    data_to tmp1, tmp2, tmp3, tmp4;
                    if (stride == 1){
                        tmp1 = add_rst[tn][tr][0];
                        tmp2 = add_rst[tn][tr][1];
                        tmp3 = add_rst[tn][tr][2];
                        tmp4 = add_rst[tn][tr][3];
                        idx1 = icolidx;//1
                        idx2 = icolidx;//2
                        idx3 = icolidx;//3
                        idx4 = icolidx;//4
                    }
                    else {
                        if (icolidx & 0x01){
                            idx1 = tcolidx + 1;//3
                            idx2 = tcolidx + 1;//4
                            idx3 = tcolidx;//1
                            idx4 = tcolidx;//2
                            tmp1 = 0;
                            tmp2 = 0;
                            tmp3 = add_rst[tn][tr][0];
                            tmp4 = add_rst[tn][tr][2];
                        }
                        else {
                            idx1 = tcolidx;//1
                            idx2 = tcolidx;//2
                            idx3 = tcolidx;//3
                            idx4 = tcolidx;//4
                            tmp1 = add_rst[tn][tr][0];
                            tmp2 = add_rst[tn][tr][2];
                            tmp3 = 0;
                            tmp4 = 0;
                        }
                    }
                    out_buf[tr][idx1*4+0][tn + Tn * n_off] += tmp1;
                    out_buf[tr][idx2*4+1][tn + Tn * n_off] += tmp2;
                    out_buf[tr][idx3*4+2][tn + Tn * n_off] += tmp3;
                    out_buf[tr][idx4*4+3][tn + Tn * n_off] += tmp4;
                }
            }
        }
    }
}




void convolver(data_tf in_buf_1[buf_w][in_ch], data_tf in_buf_2[buf_w][in_ch], data_tf in_buf_3[buf_w][in_ch], data_tf in_buf_4[buf_w][in_ch], data_tf in_buf_5[buf_w][in_ch], data_tf in_buf_6[buf_w][in_ch], data_tw k_buf[out_ch][in_ch][3][3], data_to out_buf[4][buf_o][out_ch], int o_w, int max_m, int max_n, int local_k_sz, int stride){
#pragma HLS INLINE off
    for(int m_off = 0; m_off < max_m; ++m_off){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        cal_block(in_buf_1, in_buf_2, in_buf_3, in_buf_4, in_buf_5, in_buf_6, k_buf, out_buf, o_w, m_off, max_n, 0, stride);
    }
}

/*
void load_line(quad_tf *input, data_tf in_buf[buf_w][in_ch], int r_off, int local_i_h, int local_i_w, int local_in_ch, int if_4){
#pragma HLS INLINE off
    int flag = r_off < local_i_h;
    int row_sz = local_i_w * local_in_ch;
    int idx_offset = (r_off * row_sz)>>2;
    int idx = idx_offset;
    if(!if_4){
        for(int i = 0; (i < local_i_w) && (flag); ++i){
//#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
            for(int j = 0; j < local_in_ch; j+=4){
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
#pragma HLS PIPELINE
                //if ((!if_4) && j > 0) break;
                //quad_tf tmp = input[(r_off * row_sz + i * local_in_ch + j) / 4];
            	quad_tf tmp = input[idx];
                in_buf[i][j] = tmp.a;
                in_buf[i][j+1] = tmp.b;
                in_buf[i][j+2] = tmp.c;
                in_buf[i][j+3] = tmp.d;
                idx++;
            }
        }
    }
    else{
        for(int i = 0; (i < local_i_w) && flag; ++i){
//#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
            //for(int j = 0; j < 4; j+=4){
//#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE
            int j = 0;
            //quad_tf tmp = input[(r_off * row_sz + i * local_in_ch + j) / 4];
            quad_tf tmp = input[idx];
            in_buf[i][j] = tmp.a;
            in_buf[i][j+1] = tmp.b;
            in_buf[i][j+2] = tmp.c;
            in_buf[i][j+3] = tmp.d;
            idx++;
        }
    }
}
*/
//void zero_outbuf(data_tf out_buf[4][out_w][out_ch]){
//#pragma HLS INLINE
//    zero_outbuf_loop:
//    for(int col = 0; col < out_w; col += 4){
//        for(int ch = 0; ch < out_ch; ch += 4){
//#pragma HLS PIPELINE
//            for(int i = 0; i < 4; ++i){
//                for(int j = 0; j < 4; ++j){
//                    for(int k = 0; k < 4; ++k){
//                        out_buf[i][j+col][k+ch] = 0;
//                    }
//                }
//            }
//        }
//    }
//}

void zero_outbuf(data_to out_buf[4][buf_o][out_ch]){
#pragma HLS INLINE
    zero_outbuf_loop:
    for(int col = 0; col < buf_o; ++col){
        for(int ch = 0; ch < out_ch; ch += 4){
#pragma HLS PIPELINE
            for(int i = 0; i < 4; ++i){
                ///      for(int j = 0; j < 4; ++j){
                for(int k = 0; k < 4; ++k){
                    out_buf[i][col][k+ch] = 0;
                }
            }
        }
    }
}

void load_lines(quad_tf *input, data_tf in_buf_1[buf_w][in_ch], data_tf in_buf_2[buf_w][in_ch], data_tf in_buf_3[buf_w][in_ch], data_tf in_buf_4[buf_w][in_ch], int r_off, int local_i_h, int local_i_w, int local_in_ch, int if_4){
#pragma HLS INLINE off
//    load_line(input, in_buf_1, r_off + 0, local_i_h, local_i_w, local_in_ch, if_4);
//    load_line(input, in_buf_2, r_off + 1, local_i_h, local_i_w, local_in_ch, if_4);
//    load_line(input, in_buf_3, r_off + 2, local_i_h, local_i_w, local_in_ch, if_4);
//    load_line(input, in_buf_4, r_off + 3, local_i_h, local_i_w, local_in_ch, if_4);
    int flag = r_off < local_i_h;
    int row_sz = local_i_w * local_in_ch;
    int idx_offset = (r_off * row_sz)>>2;
    int idx = idx_offset;
    int num_bound;
    if (r_off == 0)
        num_bound = 2;
    else
        num_bound = 4;
    for (int num = 0; num < num_bound; num++){
        for(int i = 0; (i < local_i_w) && (flag); ++i){
            //#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS LOOP_TRIPCOUNT min=246 max=246
            for(int j = 0; j < local_in_ch; j+=4){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE
                //if ((!if_4) && j > 0) break;
                //quad_tf tmp = input[(r_off * row_sz + i * local_in_ch + j) / 4];
                quad_tf tmp = input[idx];
                switch (num){
                    case 0:
                        in_buf_1[i][j] = tmp.a;
                        in_buf_1[i][j+1] = tmp.b;
                        in_buf_1[i][j+2] = tmp.c;
                        in_buf_1[i][j+3] = tmp.d;
                        break;
                    case 1:
                        in_buf_2[i][j] = tmp.a;
                        in_buf_2[i][j+1] = tmp.b;
                        in_buf_2[i][j+2] = tmp.c;
                        in_buf_2[i][j+3] = tmp.d;
                        break;
                    case 2:
                        in_buf_3[i][j] = tmp.a;
                        in_buf_3[i][j+1] = tmp.b;
                        in_buf_3[i][j+2] = tmp.c;
                        in_buf_3[i][j+3] = tmp.d;
                        break;
                    case 3:
                        in_buf_4[i][j] = tmp.a;
                        in_buf_4[i][j+1] = tmp.b;
                        in_buf_4[i][j+2] = tmp.c;
                        in_buf_4[i][j+3] = tmp.d;
                        break;
                    default:
                        break;
                }
                idx++;
            }
        }
    }
}

void load_kernel(quad_tw *kernel, data_tw k_buf[out_ch][in_ch][3][3], int local_in_ch, int local_out_ch){
#pragma HLS INLINE
    int ch = local_in_ch;
    for(int m = 0; m < ch; ++m){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        for(int n = 0; n < local_out_ch; n+=4){
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            for(int i = 0; i < 3; ++i){
                for(int j = 0; j < 3; ++j){
#pragma HLS PIPELINE
                    int idx = ((m * local_out_ch + n) / 4 * 3 + i) * 3 + j;
                    k_buf[n+0][m][i][j] = kernel[idx].a;
                    k_buf[n+1][m][i][j] = kernel[idx].b;
                    k_buf[n+2][m][i][j] = kernel[idx].c;
                    k_buf[n+3][m][i][j] = kernel[idx].d;

                }
            }
        }
    }
}

void store_output(data_to out_buf[4][buf_o][out_ch], quad_tf *output, quad_tf *bias, int row, int local_out_h, int local_out_w, int local_out_ch, int stride, bool relu){

#pragma HLS INLINE off

    int idx = ((row * local_out_w * local_out_ch) >> 2);
    int i_ceil = 4;
    if(stride == 2)
        i_ceil = 2;
    if (local_out_h - row < i_ceil)
        i_ceil = local_out_h - row;
    //TODO
    for(int i = 0; i < i_ceil; i += 1){
//#pragma HLS LOOP_TRIPCOUNT min=2 max=4
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        for(int col = 0; col < local_out_w; col += 1){
//#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS LOOP_TRIPCOUNT min=241 max=241
            //for(int j = 0; j < local_out_ch; j += 4){
            for(int j = 0; j < out_ch; j += 4){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
//#pragma HLS latency max = 4
#pragma HLS PIPELINE
                if (j < local_out_ch){
                    int istride = i;
                    if (stride == 2){
                        istride = i << 1;
//						colstride = col << 1;
                    }
                    //if(local_out_ch == 32 && j == 32) break;
                    data_to tmp_1 = (data_tb)(bias[idx].a) + out_buf[istride][col][j];
                    if(relu && (tmp_1 < 0)){
                        //tmp_1 = tmp_1/10;
                        tmp_1 = 0;
                    }
                    output[idx].a = tmp_1;
                    out_buf[istride][col][j] = 0;

                    data_to tmp_2 = (data_tb)(bias[idx].b) + out_buf[istride][col][j+1];
                    if(relu && (tmp_2 < 0)){
                        //tmp_2 = tmp_2/10;
                        tmp_2 = 0;
                    }
                    output[idx].b = tmp_2;
                    out_buf[istride][col][j+1] = 0;

                    data_to tmp_3 = (data_tb)(bias[idx].c) + out_buf[istride][col][j+2];
                    if(relu && (tmp_3 < 0)){
                        //tmp_3 = tmp_3/10;
                        tmp_3 = 0;
                    }
                    output[idx].c = tmp_3;
                    out_buf[istride][col][j+2] = 0;

                    data_to tmp_4 = (data_tb)(bias[idx].d) + out_buf[istride][col][j+3];
                    if(relu && (tmp_4 < 0)){
                        //tmp_4 = tmp_4/10;
                        tmp_4 = 0;
                    }
                    output[idx].d = tmp_4;
                    out_buf[istride][col][j+3] = 0;

                    ++idx;
                }
            }
        }
    }
}
#ifndef FIXED_NUMBER
void naive_conv(quad_tf *input, quad_tf *bias, quad_tf *output, quad_tw *kernel){//, int o_h, int o_w ){
    data_tf input_data[34][34][out_ch];
    data_tw k_buf[out_ch][in_ch][3][3];
    data_tf output_data[32][32][out_ch];
    //load_input(input, input_data);
    int idx = 0;
    for (int i = 0; i < 34; i++)
        for(int j = 0; j < 34; j++)
            for (int k = 0; k < 64; k = k + 4){
                if (i == 0 || i == 33)
                    for (int m = 0; m < 4; m++) input_data[i][j][k+m] = 0;
                else if (j == 0 || j == 33)
                    for (int m = 0; m < 4; m++) input_data[i][j][k+m] = 0;
                else{
                    input_data[i][j][k] = input[idx].a;
                    input_data[i][j][k+1] = input[idx].b;
                    input_data[i][j][k+2] = input[idx].c;
                    input_data[i][j][k+3] = input[idx].d;
                    idx++;
                }
            }
    load_kernel(kernel, k_buf, 64, 64);
    for (int o_ch = 0; o_ch < 64; o_ch++){

        for (int i = 0; i < out_h; i++){
            for (int j = 0; j < out_w; j++){
                float sum = 0;
                for (int i_ch = 0; i_ch < 64; i_ch++){
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                            sum += input_data[i+k][j+l][i_ch]*k_buf[o_ch][i_ch][k][l];
                }
                output_data[i][j][o_ch] = sum;
            }
        }
    }
    idx = 0;
    for (int i = 0; i < out_h; i++)
        for (int j = 0; j < out_w; j++)
            for (int o_ch = 0; o_ch < 64; o_ch+=4){
                output[idx].a = output_data[i][j][o_ch];
                output[idx].b= output_data[i][j][o_ch+1];
                output[idx].c = output_data[i][j][o_ch+2];
                output[idx].d = output_data[i][j][o_ch+3];
                idx++;
            }

}
#endif



void wino_conv(quad_tf *input, quad_tf *bias, quad_tw *kernel, quad_tf *output,
               int local_out_h, int local_out_w,
               int local_in_h, int local_in_w,
               int local_in_ch, int local_out_ch,
               int local_k_sz, int stride, bool relu, int if_4){

    data_tf in_buf_1[buf_w][in_ch], in_buf_2[buf_w][in_ch], in_buf_3[buf_w][in_ch],
            in_buf_4[buf_w][in_ch], in_buf_5[buf_w][in_ch], in_buf_6[buf_w][in_ch],
            in_buf_7[buf_w][in_ch], in_buf_8[buf_w][in_ch], in_buf_9[buf_w][in_ch],
            in_buf_10[buf_w][in_ch], in_buf_11[buf_w][in_ch], in_buf_12[buf_w][in_ch];

#pragma HLS array_partition variable=in_buf_1 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_2 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_2 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_3 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_3 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_4 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_4 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_5 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_5 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_6 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_6 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_7 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_7 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_8 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_8 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_9 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_9 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_10 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_10 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_11 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_11 cyclic factor=4 dim=2
#pragma HLS array_partition variable=in_buf_12 cyclic factor=8 dim=1
#pragma HLS array_partition variable=in_buf_12 cyclic factor=4 dim=2

    data_tw k_buf[out_ch][in_ch][3][3];
    //#pragma HLS RESOURCE variable=k_buf core=RAM_2P_LUTRAM
#pragma HLS array_partition variable=k_buf complete dim=3
#pragma HLS array_partition variable=k_buf complete dim=4
#pragma HLS array_partition variable=k_buf cyclic factor=4 dim=1
#pragma HLS array_partition variable=k_buf cyclic factor=2 dim=2

    data_to out_buf_1[4][buf_o][out_ch];
    data_to out_buf_2[4][buf_o][out_ch];
#pragma HLS array_partition variable=out_buf_1 complete dim=1
#pragma HLS array_partition variable=out_buf_1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=out_buf_1 cyclic factor=4 dim=3
#pragma HLS array_partition variable=out_buf_2 complete dim=1
#pragma HLS array_partition variable=out_buf_2 cyclic factor=4 dim=2
#pragma HLS array_partition variable=out_buf_2 cyclic factor=4 dim=3

//    int left = 1;
//    int right = 1 + (-local_out_w) % 4;

    int max_m = (local_in_ch + Tm - 1) / Tm;
    int max_n = (local_out_ch + Tn - 1) / Tn;
    ///local_in_h = padding_in_h
    int padded_local_out_h = local_in_h - 2;//((((local_out_h - 1)>>2)+1)<<2);
    int padded_local_out_w = local_in_w - 2;//((((local_out_w - 1)>>2)+1)<<2);

    load_lines(input, in_buf_1, in_buf_2, in_buf_3, in_buf_4, 0, local_in_h, local_in_w, local_in_ch, if_4);
    load_lines(input, in_buf_3, in_buf_4, in_buf_5, in_buf_6, 2, local_in_h, local_in_w, local_in_ch, if_4);
    load_lines(input, in_buf_7, in_buf_8, in_buf_9, in_buf_10, 6, local_in_h, local_in_w, local_in_ch, if_4);

    load_kernel(kernel, k_buf, local_in_ch, local_out_ch);

    zero_outbuf(out_buf_1);
    zero_outbuf(out_buf_2);
    convolver(in_buf_1, in_buf_2, in_buf_3, in_buf_4, in_buf_5, in_buf_6, k_buf, out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz, stride);
    //load_lines(input, in_buf_7, in_buf_8, in_buf_9, in_buf_10, 6, local_in_h, local_in_w, local_in_ch, if_4);

    int row = 10;
    int out_row = 0;
    int out_row_ii = 4 / stride;
    int flag = 0;



    cal_loop:
    for(;row <= local_in_h; row += 4, out_row += out_row_ii){
//#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS LOOP_TRIPCOUNT min=159 max=159
        flag = (row - 10) / 4;
        if (row < local_in_h){
            switch(flag % 6){
                case 0:
                    load_lines(input, in_buf_11, in_buf_12, in_buf_1, in_buf_2, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_5,in_buf_6,in_buf_7,in_buf_8,in_buf_9,in_buf_10, k_buf, out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);

                    break;

                case 1:
                    load_lines(input, in_buf_3, in_buf_4, in_buf_5, in_buf_6, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_9,in_buf_10,in_buf_11,in_buf_12,in_buf_1,in_buf_2, k_buf, out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);

                    break;

                case 2:
                    load_lines(input, in_buf_7, in_buf_8, in_buf_9, in_buf_10, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_1,in_buf_2,in_buf_3,in_buf_4,in_buf_5,in_buf_6, k_buf, out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);

                    break;

                case 3:
                    load_lines(input, in_buf_11, in_buf_12, in_buf_1, in_buf_2, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_5,in_buf_6,in_buf_7,in_buf_8,in_buf_9,in_buf_10, k_buf, out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);

                    break;

                case 4:
                    load_lines(input, in_buf_3, in_buf_4, in_buf_5, in_buf_6, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_9,in_buf_10,in_buf_11,in_buf_12,in_buf_1,in_buf_2, k_buf, out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);

                    break;

                default:
                    load_lines(input, in_buf_7, in_buf_8, in_buf_9, in_buf_10, row, local_in_h, local_in_w, local_in_ch, if_4);
                    convolver(in_buf_1,in_buf_2,in_buf_3,in_buf_4,in_buf_5,in_buf_6, k_buf, out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;
            }
        }
        else{
            switch(flag % 6){
                case 1:
                    convolver(in_buf_9,in_buf_10,in_buf_11,in_buf_12,in_buf_1,in_buf_2, k_buf,  out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_1, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;

                case 2:
                    convolver(in_buf_1,in_buf_2,in_buf_3, in_buf_4,in_buf_5,in_buf_6, k_buf,  out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_2, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;

                case 3:
                    convolver(in_buf_5,in_buf_6,in_buf_7, in_buf_8,in_buf_9,in_buf_10, k_buf,  out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_1, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;

                case 4:
                    convolver(in_buf_9,in_buf_10,in_buf_11,in_buf_12,in_buf_1,in_buf_2, k_buf,  out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_2, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;

                case 5:
                    convolver(in_buf_1,in_buf_2,in_buf_3,in_buf_4,in_buf_5,in_buf_6, k_buf,  out_buf_1, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_2, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_1, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;

                default:
                    convolver(in_buf_5,in_buf_6,in_buf_7,in_buf_8,in_buf_9,in_buf_10, k_buf,  out_buf_2, padded_local_out_w, max_m, max_n, local_k_sz,stride);
                    store_output(out_buf_1, output, bias, out_row, local_out_h, local_out_w, local_out_ch, stride, relu);
                    store_output(out_buf_2, output, bias, out_row + out_row_ii, local_out_h, local_out_w, local_out_ch, stride, relu);
                    break;
            }
        }
    }
}



void padding(quad_tf* input, int h, int w, int ch, quad_tf* padded_input, int left, int right, int blow, int if_4){

    quad_tf zero;
    zero.a = zero.b = zero.c = zero.d = 0;
    int cnt = 0;
    int row_sz = (w + left + right) * ch;
    int in_row_sz = w * ch;

    // cout<<"in padding"<<endl;

    for( ; cnt < left * row_sz; ++cnt){
#pragma HLS PIPELINE
        padded_input[cnt] = zero;
    }
    //cout<<"after first line"<<endl;

    for(int in_off = 0 ; cnt < (left + h) * row_sz; cnt += row_sz, in_off += in_row_sz ){
//		int pad_w_off = local_pad * local_pad * base_in_ch;
        int local_cnt = 0;
        for( ; local_cnt < left * ch; ++local_cnt)
#pragma HLS PIPELINE
                padded_input[cnt + local_cnt] = zero;
        for(int input_cnt = 0; input_cnt < in_row_sz; ++input_cnt){
#pragma HLS PIPELINE
            quad_tf tmp1;
            if (if_4){
                quad_layer_1_type tmp = ((quad_layer_1_type *)input)[ in_off + input_cnt];
                tmp1.a = tmp.a * tmp_64_par - mean_b;
                tmp1.b = tmp.b * tmp_64_par - mean_g;
                tmp1.c = tmp.c * tmp_64_par - mean_r;
                tmp1.d = tmp.d * tmp_64_par - 0;
            }
            else{
                tmp1 = input[ in_off + input_cnt];
            }
            padded_input[cnt + local_cnt + input_cnt] = tmp1;
        }
        for( ; local_cnt < (left + right) * ch; ++local_cnt)
#pragma HLS PIPELINE
                padded_input[cnt + in_row_sz + local_cnt] = zero;
    }
    //cout<<"after middle"<<endl;
    for(int local_cnt = 0; local_cnt < blow * row_sz; ++local_cnt){
#pragma HLS PIPELINE
        padded_input[cnt + local_cnt] = zero;
    }
    //cout<<"after last line"<<endl;
}





//void wino_conv_wrapper(quad_tf* input, quad_tf* bias, quad_tf* output, quad_tw* kernel, int local_in_h, int local_in_w, int local_out_h, int local_out_w, int local_in_ch, int local_out_ch, int local_k_sz, int stride, bool R, int nnup_fac, int k_num, int if_4){
void wino_conv_wrapper(quad_tf input[in_h*in_w*in_ch/4], quad_tf bias[out_h*out_w*out_ch/4], quad_tf output[out_h*out_w*out_ch/4], quad_tw kernel[3 * 3 * out_ch * in_ch / 4], int local_in_h, int local_in_w, int local_out_h, int local_out_w, int local_in_ch, int local_out_ch, int local_k_sz, int stride, bool R, int nnup_fac, int k_num, int if_4){
//cout << "wrapper" << endl;
#pragma HLS DATA_PACK variable=input
#pragma HLS DATA_PACK variable=bias
#pragma HLS DATA_PACK variable=kernel
#pragma HLS DATA_PACK variable=output

#pragma HLS DATAFLOW

 //   quad_tf padded_input[ buf_h * buf_w * (in_ch >> 2)];
//#pragma HLS STREAM variable=padded_input depth=2048
    quad_tf* padded_input = (quad_tf*)malloc(buf_h*buf_w*in_ch/4*sizeof(quad_tf));
    //if(padded_input==NULL){
    //	cout<<"malloc padded failed"<<endl;
    //}

    //quad_tf padded_input[ buf_h * buf_w * in_ch / 4];
    int stride_h = local_out_h * stride;
    int stride_w = local_out_w * stride;

    //int padded_local_in_h = ((stride_h+1)>>2)<<2+2;
    //int padded_local_in_w = ((stride_w+1)>>2)<<2+2;
    int padded_local_out_h = ((((stride_h - 1) >> 2) + 1) << 2);
    int padded_local_out_w = ((((stride_w - 1) >> 2) + 1) << 2);
    int padded_local_in_h = (padded_local_out_h - 1) + local_k_sz;
    int padded_local_in_w = (padded_local_out_w - 1) + local_k_sz;
    //int padded_h_num = padded_local_in_h - local_in_h;
    //int padded_w_num = padded_local_in_w - local_in_w;
    int left = 1;
    int right = padded_local_in_w - local_in_w - left;
    int blow = padded_local_in_h - local_in_h - left;

    padding(input,  local_in_h,  local_in_w, local_in_ch >> 2, padded_input, left, right, blow,if_4);

    wino_conv(padded_input, bias, kernel, output, local_out_h, local_out_w, padded_local_in_h, padded_local_in_w, local_in_ch, local_out_ch, local_k_sz, stride, R, if_4);
    free(padded_input);
}
