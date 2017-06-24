#include "wino_conv.h"
using namespace std;
#define fc_Tn 2
#define fc_Tm 8

void load_fc_line(quad_tf input_1[fc_h*fc_w*16], quad_tf input_2[fc_h*fc_w*16], data_tb buf_line[2][fc_w][fc_in_ch/2], int row){
#pragma HLS INLINE off
// #pragma HLS array_partition variable=buf_line complete dim=1
// #pragma HLS array_partition variable=buf_line cyclic factor=16 dim=3


	int idx = row * fc_w * 16;
	for(int col = 0; col < fc_w; ++col){
		for(int i = 0; i < 64; i += 4){
		#pragma HLS PIPELINE

			buf_line[0][col][i] = input_1[idx].a;
			buf_line[0][col][i+1] = input_1[idx].b;
			buf_line[0][col][i+2] = input_1[idx].c;
			buf_line[0][col][i+3] = input_1[idx].d;

			buf_line[1][col][i] = input_2[idx].a;
			buf_line[1][col][i+1] = input_2[idx].b;
			buf_line[1][col][i+2] = input_2[idx].c;
			buf_line[1][col][i+3] = input_2[idx].d;

			++idx;
		}
	}
}

void load_fc_kernel(quad_tw *kernel_l, quad_tw *kernel_m, quad_tw *kernel_r, data_tw k_buf[3][2][fc_in_ch / 2][fc_out_ch]){
#pragma HLS INLINE off

	for(int in = 0; in < fc_in_ch/2; ++in){
	#pragma HLS PIPELINE
		k_buf[0][0][in][0] = kernel_l[in].a;
		k_buf[0][0][in][1] = kernel_l[in].b;
		k_buf[0][1][in][0] = kernel_l[in].c;
		k_buf[0][1][in][1] = kernel_l[in].d;
	}

	for(int in = 0; in < fc_in_ch/2; ++in){
	#pragma HLS PIPELINE
		k_buf[1][0][in][0] = kernel_m[in].a;
		k_buf[1][0][in][1] = kernel_m[in].b;
		k_buf[1][1][in][0] = kernel_m[in].c;
		k_buf[1][1][in][1] = kernel_m[in].d;
	}

	for(int in = 0; in < fc_in_ch/2; ++in){
	#pragma HLS PIPELINE
		k_buf[2][0][in][0] = kernel_r[in].a;
		k_buf[2][0][in][1] = kernel_r[in].b;
		k_buf[2][1][in][0] = kernel_r[in].c;
		k_buf[2][1][in][1] = kernel_r[in].d;
	}
}

void store_soft_line(data_ro output[fc_h*fc_w], fc_data_tf outbuf[3][fc_w][2], int row){
#pragma HLS INLINE off

	float tmpl1, tmpl2, suml;
	float tmpm1, tmpm2, summ;
	float tmpr1, tmpr2, sumr;
	data_ro tmp;
	int idx = row * fc_w;
	ap_uint<8> coeff = 255;

	for(int col = 0; col < fc_w; ++col){
	#pragma HLS PIPELINE

		tmpl1 = expf(outbuf[0][col][0]);
		tmpl2 = expf(outbuf[0][col][1]);
		suml = tmpl1 + tmpl2;
		//output_l[idx+col].a = tmp1/sum;
		tmp.l = tmpl2/suml * 255;
		tmpm1 = expf(outbuf[1][col][0]);
		tmpm2 = expf(outbuf[1][col][1]);
		summ = tmpm1 + tmpm2;
		//output_m[idx+col].a = tmp1/sum;
		tmp.m = tmpm2/summ * 255;
		tmpr1 = expf(outbuf[2][col][0]);
		tmpr2 = expf(outbuf[2][col][1]);
		sumr = tmpr1 + tmpr2;
		//output_r[idx+col].a = tmp1/sum;
		tmp.r = tmpr2/sumr * 255;
		tmp.empty = 0;
		output[idx+col] = tmp;
	}
}

//void fc_kernel(data_tf in_buf[2][fc_w][fc_in_ch / 2], data_tw k_buf[2][fc_in_ch / 2][fc_out_ch], data_tf out_buf[fc_w][fc_out_ch],
//			   data_tw fc_bias[2]){
void fc_kernel(data_tb in_buf[2][fc_w][fc_in_ch / 2], data_tw k_buf[2][fc_in_ch / 2][fc_out_ch], fc_data_tf out_buf[fc_w][fc_out_ch],
			   data_tw fc_bias[2]){
#pragma HLS INLINE off
// #pragma HLS array_partition variable=in_buf complete dim=1
// #pragma HLS array_partition variable=in_buf cyclic factor=16 dim=3
// #pragma HLS array_partition variable=k_buf complete dim=1
// #pragma HLS array_partition variable=k_buf cyclic factor=16 dim=2
// #pragma HLS array_partition variable=k_buf complete dim=3
// #pragma HLS array_partition variable=out_buf complete dim=2

	for(int col = 0; col < fc_w; ++col){
	#pragma HLS PIPELINE
		out_buf[col][0] = fc_bias[0];
		out_buf[col][1] = fc_bias[1];
	}

	for(int m = 0; m < fc_in_ch/2; m += fc_Tm){
		for(int col = 0; col < fc_w; ++col){
		#pragma HLS PIPELINE

			//data_tf add_rst[fc_Tn];
			fc_data_tf add_rst[fc_Tn];
			#pragma HLS array_partition variable=add_rst complete dim=0
			for(int tn = 0; tn < fc_Tn; ++tn){
				if(!tn){
					for(int i = 0; i < fc_Tn; ++i) add_rst[i] = 0;
				}
				for(int tm = 0; tm < fc_Tm; ++tm){
					data_tb i_tmp_1 = in_buf[0][col][tm+m];
					data_tb i_tmp_2 = in_buf[1][col][tm+m];
					data_tw k_tmp_1 = k_buf[0][tm+m][tn];
					data_tw k_tmp_2 = k_buf[1][tm+m][tn];
					fc_data_tf mul_rst_1 = i_tmp_1 * k_tmp_1;
					fc_data_tf mul_rst_2 = i_tmp_2 * k_tmp_2;
					fc_data_tf mul_rst_t = mul_rst_1 + mul_rst_2;
					add_rst[tn] += mul_rst_t;
				}
			}

			for(int tn = 0; tn < fc_Tn; ++tn){
				out_buf[col][tn] += add_rst[tn];
			}
		}
	}

		 // for(int col = 0; col<fc_w;++col)
		 // {
		//	  cout<<out_buf[col][0]<<" ";
		//  }
	       //  cout<<endl;
}

void fc_convolve(data_tb in_buf[2][fc_w][fc_in_ch / 2],
				 data_tw k_buf[3][2][fc_in_ch / 2][fc_out_ch],// data_tw k_buf_m[2][fc_in_ch / 2][fc_out_ch], data_tw k_buf_r[2][fc_in_ch / 2][fc_out_ch],
				 fc_data_tf out_buf[3][fc_w][fc_out_ch],// fc_data_tf out_buf_m[fc_w][fc_out_ch], fc_data_tf out_buf_r[fc_w][fc_out_ch],
				 data_tw fc_bias[3][2]){//, data_tw fc_bias_m[2], data_tw fc_bias_r[2]){

#pragma HLS INLINE off
// data_tw*** k_buf[3];
// fc_data_tf** out_buf[3];
// data_tw* fc_bias[3];
// k_buf[0] = k_buf_l;
// k_buf[1] = k_buf_m;
// k_buf[2] = k_buf_r;
// out_buf[0] = out_buf_l;
// out_buf[1] = out_buf_m;
// out_buf[2] = out_buf_r;
// fc_bias[0] = fc_bias_l;
// fc_bias[0] = fc_bias_m;
// fc_bias[0] = fc_bias_r;
for(int i=0;i<3;i++)
	fc_kernel(in_buf, k_buf[i], out_buf[i], fc_bias[i]);
	//fc_kernel(in_buf, k_buf_m, out_buf_m, fc_bias_m);
	//fc_kernel(in_buf, k_buf_r, out_buf_r, fc_bias_r);
}

void fc_soft_layer(quad_tf input_1[fc_h*fc_w*16], quad_tf input_2[fc_h*fc_w*16],
				   quad_tw kernel_l[fc_in_ch*fc_out_ch/4], quad_tw kernel_m[fc_in_ch*fc_out_ch/4], quad_tw kernel_r[fc_in_ch*fc_out_ch/4],
				   data_ro output[fc_h*fc_w], data_tw fc_bias0,data_tw fc_bias1,data_tw fc_bias2,data_tw fc_bias3,data_tw fc_bias4,data_tw fc_bias5){
#pragma HLS DATA_PACK variable=input_1
#pragma HLS DATA_PACK variable=input_2
#pragma HLS DATA_PACK variable=kernel_l
#pragma HLS DATA_PACK variable=kernel_m
#pragma HLS DATA_PACK variable=kernel_r
#pragma HLS DATA_PACK variable=output


	data_tb buf_line_1[2][fc_w][fc_in_ch/2];
	data_tb buf_line_2[2][fc_w][fc_in_ch/2];
#pragma HLS array_partition variable=buf_line_1 complete dim=1
#pragma HLS array_partition variable=buf_line_1 cyclic factor=8 dim=3

#pragma HLS array_partition variable=buf_line_2 complete dim=1
#pragma HLS array_partition variable=buf_line_2 cyclic factor=8 dim=3


	fc_data_tf outbuf_1[3][fc_w][fc_out_ch];//, outbuf_m_1[fc_w][fc_out_ch], outbuf_r_1[fc_w][fc_out_ch];
	fc_data_tf outbuf_2[3][fc_w][fc_out_ch];//, outbuf_m_2[fc_w][fc_out_ch], outbuf_r_2[fc_w][fc_out_ch];

#pragma HLS RESOURCE variable=outbuf_1 core=RAM_2P_LUTRAM
#pragma HLS array_partition variable=outbuf_1 complete dim=1
#pragma HLS array_partition variable=outbuf_1 complete dim=3

#pragma HLS RESOURCE variable=outbuf_2 core=RAM_2P_LUTRAM
#pragma HLS array_partition variable=outbuf_2 complete dim=1
#pragma HLS array_partition variable=outbuf_2 complete dim=3

	data_tw k_buf[3][2][fc_in_ch/2][fc_out_ch];//, k_buf_m[2][fc_in_ch/2][fc_out_ch], k_buf_r[2][fc_in_ch/2][fc_out_ch];

//#pragma HLS RESOURCE variable=k_buf_l core=RAM_S2P_LUTRAM
//#pragma HLS RESOURCE variable=k_buf_m core=RAM_S2P_LUTRAM
#pragma HLS RESOURCE variable=k_buf core=RAM_S2P_LUTRAM
//#pragma HLS array_partition variable=k_buf complete dim=1
#pragma HLS array_partition variable=k_buf complete dim=2
#pragma HLS array_partition variable=k_buf cyclic factor=8 dim=3
#pragma HLS array_partition variable=k_buf complete dim=4
// #pragma HLS array_partition variable=k_buf_m complete dim=1
// #pragma HLS array_partition variable=k_buf_m cyclic factor=16 dim=2
// #pragma HLS array_partition variable=k_buf_m complete dim=3
// #pragma HLS array_partition variable=k_buf_r complete dim=1
// #pragma HLS array_partition variable=k_buf_r cyclic factor=16 dim=2
// #pragma HLS array_partition variable=k_buf_r complete dim=3

	data_tw fc_bias[3][2];
//#pragma HLS array_partition variable=k_buf complete dim=1
//#pragma HLS array_partition variable=k_buf complete dim=2
	fc_bias[0][0] = fc_bias0;
	fc_bias[0][1] = fc_bias1;
	fc_bias[1][0] = fc_bias2;
	fc_bias[1][1] = fc_bias3;
	fc_bias[2][0] = fc_bias4;
	fc_bias[2][1] = fc_bias5;

	load_fc_kernel(kernel_l, kernel_m, kernel_r, k_buf);
	load_fc_line(input_1, input_2, buf_line_1, 0);
	if(1){
		fc_convolve(buf_line_1, k_buf, outbuf_1, fc_bias);
		load_fc_line(input_1, input_2, buf_line_2, 1);
	}

	int row = 2;
	int flag = 1;

	for(; row < fc_h; ++row){

		if(flag){
			load_fc_line(input_1, input_2, buf_line_1, row);
			fc_convolve(buf_line_2, k_buf, outbuf_2, fc_bias);
			store_soft_line(output, outbuf_1, row-2);
			flag = !flag;
		}

		else{
			load_fc_line(input_1, input_2, buf_line_2, row);
			fc_convolve(buf_line_1, k_buf, outbuf_1, fc_bias);
			store_soft_line(output, outbuf_2, row-2);
			flag = !flag;
		}
	}

	if(flag){
		fc_convolve(buf_line_2, k_buf, outbuf_2, fc_bias);
		store_soft_line(output, outbuf_1, row-2);
		store_soft_line(output, outbuf_2, row-1);
	}
	else{
		fc_convolve(buf_line_1, k_buf, outbuf_1, fc_bias);
		store_soft_line(output, outbuf_2, row-2);
		store_soft_line(output, outbuf_1, row-1);
	}
}
