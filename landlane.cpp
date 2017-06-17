#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
//#include "sds_lib.h"
#include "wino_conv.h"
//#include "CImg.h"
#include <ctime>

#ifdef __SDSCC__
#include "sds_lib.h"
#define malloc(x) (sds_alloc_non_cacheable(x))
#define free(x) (sds_free(x))
#define clock(x) (sds_clock_counter(x))
#endif

using namespace std;


//#define CHECK_MALLOC(x)  ()

///////////////
int layer_num =0;



//////////////



int _count = 1;
int write_flag=0;
extern unsigned char *input_data;
extern int video_flag;

extern int odd;

inline void CHECK_MALLOC(void* x){
	if(x == NULL) {
		std::cout<<"allocate failed and exit" << _count <<std::endl;
		exit(-1);
	}
	_count ++;
}

// void corr(quad_tf *conv2a, quad_tf *conv2b, quad_tf *corr_1, quad_tf *corr_2){

// }

layer_1_type *input_r; //0.29MB
//layer_1_type *input_r_0; //0.29MB
//layer_1_type *input_r_1; //0.29MB

data_tf *conv1; //6.88MB
data_tw *conv1_weight; //0.027MB
data_tf *conv1_bias; //6.88MB

data_tf *conv2; //2.29MB
data_tw *conv2_weight; //0.11MB
data_tf *conv2_bias; //2.29MB

data_tf *conv3; //2.29MB
data_tw *conv3_weight; //0.11MB
data_tf *conv3_bias; //2.29MB

data_tf *conv4;
data_tw *conv4_weight;
data_tf *conv4_bias;

data_tf *conv5;
data_tw *conv5_weight;
data_tf *conv5_bias;

data_tf *conv6;
data_tw *conv6_weight;
data_tf *conv6_bias;

data_tf *conv7;
data_tw *conv7_weight;
data_tf *conv7_bias;

data_tf *conv8_1; //6.88MB
data_tw *conv8_1_weight; //0.027MB
data_tf *conv8_1_bias; //6.88MB

data_tf *conv8_2; //2.29MB
data_tw *conv8_2_weight; //0.11MB
data_tf *conv8_2_bias; //2.29MB

data_tf *conv9_1; //6.88MB
data_tw *conv9_1_weight; //0.027MB
data_tf *conv9_1_bias; //6.88MB

data_tf *conv9_2; //2.29MB
data_tw *conv9_2_weight; //0.11MB
data_tf *conv9_2_bias; //2.29MB

data_tf *conv10_1[2]; //6.88MB
data_tw *conv10_1_weight[2]; //0.027MB
data_tf *conv10_1_bias[2]; //6.88MB

data_tf *conv10_2[2]; //2.29MB
data_tw *conv10_2_weight[4]; //0.11MB
data_tf *conv10_2_bias[2]; //2.29MB

data_tf *conv11_1[2]; //6.88MB
data_tw *conv11_1_weight[4]; //0.027MB
data_tf *conv11_1_bias[2]; //6.88MB

data_tf *conv11_2[2]; //2.29MB
data_tw *conv11_2_weight[4]; //0.11MB
data_tf *conv11_2_bias[2]; //2.29MB

data_ro *output_image;
data_tw *fc_1x1_l_weight, *fc_1x1_m_weight, *fc_1x1_r_weight;
data_tw *fc_1x1_l_bias, *fc_1x1_m_bias, *fc_1x1_r_bias;
data_tw *fc_1x1_bias;

void data_malloc_init(){

	input_r = (layer_1_type*)malloc(641 * 481  * 4 * sizeof(data_tf));
	CHECK_MALLOC(input_r);
	//input_r_1 = (layer_1_type*)malloc(642 * 482  * sizeof(quad_layer_1_type));
	//CHECK_MALLOC(input_r_1);

	conv1 = (data_tf*)malloc(321 * 241 * 32 * sizeof(data_tf));
	conv1_bias = (data_tf*)malloc(321 * 241 * 32 * sizeof(data_tf));
	CHECK_MALLOC(conv1);
	CHECK_MALLOC(conv1_bias);

	conv2 = (data_tf*)malloc(321 * 241 * 32 * sizeof(data_tf));
	conv2_bias = (data_tf*)malloc(321 * 241 * 32 * sizeof(data_tf));
	CHECK_MALLOC(conv2);
	CHECK_MALLOC(conv2_bias);

	conv3 = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	conv3_bias = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv3);
	CHECK_MALLOC(conv3_bias);

	conv4 = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	conv4_bias = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv4);
	CHECK_MALLOC(conv4_bias);

	conv5 = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	conv5_bias = (data_tf*)malloc(161 * 121 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv5);
	CHECK_MALLOC(conv5_bias);

	conv6 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv6_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv6);
	CHECK_MALLOC(conv6_bias);

	conv7 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv7_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv7);
	CHECK_MALLOC(conv7_bias);

	conv8_1 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv8_1_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv8_1);
	CHECK_MALLOC(conv8_1_bias);

	conv8_2 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv8_2_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv8_2);
	CHECK_MALLOC(conv8_2_bias);

	conv9_1 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv9_1_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv9_1);
	CHECK_MALLOC(conv9_1_bias);

	conv9_2 = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	conv9_2_bias = (data_tf*)malloc(81 * 61 * 64 * sizeof(data_tf));
	CHECK_MALLOC(conv9_2);
	CHECK_MALLOC(conv9_2_bias);

	conv10_1[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	conv10_1_bias[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	CHECK_MALLOC(conv10_1[0]);
	CHECK_MALLOC(conv10_1_bias[0]);
	for(int i = 1; i < 2; ++i) conv10_1[i] = &(conv10_1[0][i * 81 * 61 * 64]);
	for(int i = 1; i < 2; ++i) conv10_1_bias[i] = &(conv10_1_bias[0][i * 81 * 61 * 64]);

	conv10_2[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	conv10_2_bias[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	CHECK_MALLOC(conv10_2[0]);
	CHECK_MALLOC(conv10_2_bias[0]);
	for(int i = 1; i < 2; ++i) conv10_2[i] = &(conv10_2[0][i * 81 * 61 * 64]);
	for(int i = 1; i < 2; ++i) conv10_2_bias[i] = &(conv10_2_bias[0][i * 81 * 61 * 64]);

	conv11_1[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	conv11_1_bias[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	CHECK_MALLOC(conv11_1[0]);
	CHECK_MALLOC(conv11_1_bias[0]);
	for(int i = 1; i < 2; ++i) conv11_1[i] = &(conv11_1[0][i * 81 * 61 * 64]);
	for(int i = 1; i < 2; ++i) conv11_1_bias[i] = &(conv11_1_bias[0][i * 81 * 61 * 64]);

	conv11_2[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	conv11_2_bias[0] = (data_tf*)malloc(81 * 61 * 128 * sizeof(data_tf));
	CHECK_MALLOC(conv11_2[0]);
	CHECK_MALLOC(conv11_2_bias[0]);
	for(int i = 1; i < 2; ++i) conv11_2[i] = &(conv11_2[0][i * 81 * 61 * 64]);
	for(int i = 1; i < 2; ++i) conv11_2_bias[i] = &(conv11_2_bias[0][i * 81 * 61 * 64]);

	output_image = (data_ro*)malloc(81 * 61 * sizeof(data_ro));
	CHECK_MALLOC(output_image);

	fc_1x1_l_bias = (data_tw*)malloc(2 * sizeof(data_tw));
	fc_1x1_m_bias = (data_tw*)malloc(2 * sizeof(data_tw));
	fc_1x1_r_bias = (data_tw*)malloc(2 * sizeof(data_tw));
	fc_1x1_bias = (data_tw*)malloc(6 * sizeof(data_tw));
	CHECK_MALLOC(fc_1x1_l_bias);
	CHECK_MALLOC(fc_1x1_m_bias);
	CHECK_MALLOC(fc_1x1_r_bias);
	CHECK_MALLOC(fc_1x1_bias);

}

void weight_malloc_init(){

	conv1_weight = (data_tw*)malloc(4 * 32 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv1_weight);

	conv2_weight = (data_tw*)malloc(32 * 32 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv2_weight);

	conv3_weight = (data_tw*)malloc(32 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv3_weight);

	conv4_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv4_weight);

	conv5_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv5_weight);

	conv6_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv6_weight);

	conv7_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv7_weight);

	conv8_1_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv8_1_weight);

	conv8_2_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv8_2_weight);

	conv9_1_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv9_1_weight);

	conv9_2_weight = (data_tw*)malloc(64 * 64 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv9_2_weight);

	conv10_1_weight[0] = (data_tw*)malloc(64 * 128 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv10_1_weight[0]);
	for(int i = 1; i < 2; ++i) conv10_1_weight[i] = &(conv10_1_weight[0][i * 64 * 64 * 3 * 3]);

	conv10_2_weight[0] = (data_tw*)malloc(128 * 128 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv10_2_weight[0]);
	for(int i = 1; i < 4; ++i) conv10_2_weight[i] = &(conv10_2_weight[0][i * 64 * 64 * 3 * 3]);

	conv11_1_weight[0] = (data_tw*)malloc(128 * 128 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv11_1_weight[0]);
	for(int i = 1; i < 4; ++i) conv11_1_weight[i] = &(conv11_1_weight[0][i * 64 * 64 * 3 * 3]);

	conv11_2_weight[0] = (data_tw*)malloc(128 * 128 * 3 * 3 * sizeof(data_tw));
	CHECK_MALLOC(conv11_2_weight[0]);
	for(int i = 1; i < 4; ++i) conv11_2_weight[i] = &(conv11_2_weight[0][i * 64 * 64 * 3 * 3]);

	fc_1x1_l_weight = (data_tw*)malloc(128 * 2 * sizeof(data_tw));
	CHECK_MALLOC(fc_1x1_l_weight);

	fc_1x1_m_weight = (data_tw*)malloc(128 * 2 * sizeof(data_tw));
	CHECK_MALLOC(fc_1x1_m_weight);

	fc_1x1_r_weight = (data_tw*)malloc(128 * 2 * sizeof(data_tw));
	CHECK_MALLOC(fc_1x1_r_weight);


}

void load_weight(data_tw *_weight, char *file_name,  int weight_sz){

	ifstream weight_file;
	weight_file.open(file_name);
	if(!weight_file){
		cout << "cannot open " << file_name << endl;
		exit(-1);
	}
	int i = 0;
	while(weight_file >> _weight[i]){
		++i;
	}
	cout << file_name << endl;
	assert(i == weight_sz);
	weight_file.close();
}

void weight_assign(){
/////in1_out1,in2_out1，in1_out2,in2_out2;
	/////weight��?要在文件里已经排��?


	//load_weight(conv1_weight, "./weight/conv1_weight.txt", 4 * 32 * 3 * 3);
	//load_weight(conv2_weight, "./weight/conv2_weight.txt", 32 * 32 * 3 * 3);

	//load_weight(conv3_weight, "./weight/conv3_weight.txt", 32 * 64 * 3 * 3);

	//load_weight(conv4_weight, "weight/conv4_weight.txt", 64 * 64 * 3 * 3);

//	load_weight(conv5_weight, "weight/conv5_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv6_weight, "weight/conv6_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv7_weight, "weight/conv7_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv8_1_weight, "weight/conv8_1_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv8_2_weight, "weight/conv8_2_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv9_1_weight, "weight/conv9_1_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv9_2_weight, "weight/conv9_2_weight.txt", 64 * 64 * 3 * 3);
//
//	load_weight(conv10_1_weight[0], "weight/conv10_1_weight.txt", 64 * 128 * 3 * 3);
//
//	load_weight(conv10_2_weight[0], "weight/conv10_2_weight.txt", 128 * 128 * 3 * 3);
//
//	load_weight(conv11_1_weight[0], "weight/conv11_1_weight.txt", 128 * 128 * 3 * 3);
//
//	load_weight(conv11_2_weight[0], "weight/conv11_2_weight.txt", 128 * 128 * 3 * 3);
//
//	load_weight(fc_1x1_l_weight, "weight/fc_1x1_left_weight.txt", 2 * 128);
//
//	load_weight(fc_1x1_m_weight, "weight/fc_1x1_guide_weight.txt", 2 * 128);
//
//	load_weight(fc_1x1_r_weight, "weight/fc_1x1_right_weight.txt", 2 * 128);
//
//	load_weight(fc_1x1_l_bias, "weight/fc_1x1_left_bias.txt", 2);
//
//	load_weight(fc_1x1_m_bias, "weight/fc_1x1_guide_bias.txt", 2);
//
//	load_weight(fc_1x1_r_bias, "weight/fc_1x1_right_bias.txt", 2);


	load_weight(conv1_weight, "./weight/reshape_conv1.txt", 4 * 32 * 3 * 3);
	load_weight(conv2_weight, "./weight/reshape_conv2.txt", 32 * 32 * 3 * 3);

	load_weight(conv3_weight, "./weight/reshape_conv3.txt", 32 * 64 * 3 * 3);

	load_weight(conv4_weight, "weight/reshape_conv4.txt", 64 * 64 * 3 * 3);

	load_weight(conv5_weight, "weight/reshape_conv5.txt", 64 * 64 * 3 * 3);

	load_weight(conv6_weight, "weight/reshape_conv6.txt", 64 * 64 * 3 * 3);

	load_weight(conv7_weight, "weight/reshape_conv7.txt", 64 * 64 * 3 * 3);

	load_weight(conv8_1_weight, "weight/reshape_conv8_1.txt", 64 * 64 * 3 * 3);

	load_weight(conv8_2_weight, "weight/reshape_conv8_2.txt", 64 * 64 * 3 * 3);

	load_weight(conv9_1_weight, "weight/reshape_conv9_1.txt", 64 * 64 * 3 * 3);

	load_weight(conv9_2_weight, "weight/reshape_conv9_2.txt", 64 * 64 * 3 * 3);

	load_weight(conv10_1_weight[0], "weight/reshape_conv10_1.txt", 64 * 128 * 3 * 3);

	load_weight(conv10_2_weight[0], "weight/reshape_conv10_2.txt", 128 * 128 * 3 * 3);

	load_weight(conv11_1_weight[0], "weight/reshape_conv11_1.txt", 128 * 128 * 3 * 3);

	load_weight(conv11_2_weight[0], "weight/reshape_conv11_2.txt", 128 * 128 * 3 * 3);

	load_weight(fc_1x1_l_weight, "weight/fc_1x1_left_weight.txt", 2 * 128);

	load_weight(fc_1x1_m_weight, "weight/fc_1x1_guide_weight.txt", 2 * 128);

	load_weight(fc_1x1_r_weight, "weight/fc_1x1_right_weight.txt", 2 * 128);

	load_weight(fc_1x1_l_bias, "weight/fc_1x1_left_bias.txt", 2);

	load_weight(fc_1x1_m_bias, "weight/fc_1x1_guide_bias.txt", 2);

	load_weight(fc_1x1_r_bias, "weight/fc_1x1_right_bias.txt", 2);

	fc_1x1_bias[0] = fc_1x1_l_bias[0];
	fc_1x1_bias[1] = fc_1x1_l_bias[1];
	fc_1x1_bias[2] = fc_1x1_m_bias[0];
	fc_1x1_bias[3] = fc_1x1_m_bias[1];
	fc_1x1_bias[4] = fc_1x1_r_bias[0];
	fc_1x1_bias[5] = fc_1x1_r_bias[1];

}

void load_bias(data_tf *_bias, char *file_name, int _h, int _w, int bias_sz){
///////��?后load前半部分和后半部分各64channel
	ifstream bias_file;
	bias_file.open(file_name);
	if(!bias_file){
		cout << "cannot open " << file_name << endl;
		exit(-1);
	}
	data_tf *bias_tmp;
	bias_tmp = (data_tf*)malloc(bias_sz * sizeof(data_tf));
	int t = 0;
	while(bias_file >> bias_tmp[t]){
		++t;
	}
	assert(t == bias_sz);
	bias_file.close();

	if(bias_sz <= 64){
		for(int i = 0; i < _h; ++i)
		for(int j = 0; j < _w; ++j)
		for(int ch = 0; ch < bias_sz; ++ch){
			_bias[i * _w * bias_sz + j * bias_sz + ch] = bias_tmp[ch];
		}
	}
	else{
		for(int i = 0; i < _h; ++i)
		for(int j = 0; j < _w; ++j)
		for(int ch = 0; ch < 64; ++ch){
			_bias[i * _w * 64 + j * 64 + ch] = bias_tmp[ch];
		}

		int off = _h * _w * 64;
		for(int i = 0; i < _h; ++i)
		for(int j = 0; j < _w; ++j)
		for(int ch = 0; ch < 64; ++ch){
			_bias[off + i * _w *(bias_sz-64) + j * (bias_sz-64) + ch] = bias_tmp[64+ch];
		}
	}

	free(bias_tmp);
}


void bias_assign(){

	load_bias(conv1_bias, "bias/conv1_bias.txt", 321, 241, 32);

	load_bias(conv2_bias, "bias/conv2_bias.txt", 321, 241, 32);

	load_bias(conv3_bias, "bias/conv3_bias.txt", 161, 121, 64);

	load_bias(conv4_bias, "bias/conv4_bias.txt", 161, 121, 64);

	load_bias(conv5_bias, "bias/conv5_bias.txt", 161, 121, 64);

	load_bias(conv6_bias, "bias/conv6_bias.txt", 81, 61, 64);

	load_bias(conv7_bias, "bias/conv7_bias.txt", 81, 61, 64);

	load_bias(conv8_1_bias, "bias/conv8_1_bias.txt", 81, 61, 64);

	load_bias(conv8_2_bias, "bias/conv8_2_bias.txt", 81, 61, 64);

	load_bias(conv9_1_bias, "bias/conv9_1_bias.txt", 81, 61, 64);

	load_bias(conv9_2_bias, "bias/conv9_2_bias.txt", 81, 61, 64);

	load_bias(conv10_1_bias[0], "bias/conv10_1_bias.txt", 81, 61, 128);

	load_bias(conv10_2_bias[0], "bias/conv10_2_bias.txt", 81, 61, 128);

	load_bias(conv11_1_bias[0], "bias/conv11_1_bias.txt", 81, 61, 128);

	load_bias(conv11_2_bias[0], "bias/conv11_2_bias.txt", 81, 61, 128);

}

void forward(){

	layer_num=1;
	cout << "1" << endl;
	wino_conv_wrapper((quad_tf*)input_r, (quad_tf*)conv1_bias, (quad_tf*)conv1, (quad_tw*)conv1_weight, 641, 481, 321, 241, 4, 32, 3, 2, 1, 1, 1, 1);
	for (int ch = 0; ch < 1; ch++){
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv1[(i * 241 + j)*32 + ch] << " ";
		}
		cout << endl;
	}
		cout << "*************channel" << ch <<"********************" << endl;
	}
	++layer_num;
	cout << "2" << endl;
	wino_conv_wrapper((quad_tf*)conv1, (quad_tf*)conv2_bias, (quad_tf*)conv2, (quad_tw*)conv2_weight, 321, 241, 321, 241, 32, 32, 3, 1, 1, 1, 1, 0);
	for (int ch = 0; ch < 1; ch++){
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv2[(i * 241 + j)*32 + ch] << " ";
		}
		cout << endl;
	}
	cout << "**************channel"<<ch<<"*******************" << endl;
	}

	++layer_num;
	cout << "3" << endl;
	wino_conv_wrapper((quad_tf*)conv2, (quad_tf*)conv3_bias, (quad_tf*)conv3, (quad_tw*)conv3_weight, 321, 241, 161, 121, 32, 64, 3, 2, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv3[(i * 121 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;


	++layer_num;
	cout << "4" << endl;
	wino_conv_wrapper((quad_tf*)conv3, (quad_tf*)conv4_bias, (quad_tf*)conv4, (quad_tw*)conv4_weight, 161, 121, 161, 121, 64, 64, 3, 1, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv4[(i * 121 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "5" << endl;
	wino_conv_wrapper((quad_tf*)conv4, (quad_tf*)conv5_bias, (quad_tf*)conv5, (quad_tw*)conv5_weight, 161, 121, 161, 121, 64, 64, 3, 1, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv5[(i * 121 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "6" << endl;
	wino_conv_wrapper((quad_tf*)conv5, (quad_tf*)conv6_bias, (quad_tf*)conv6, (quad_tw*)conv6_weight, 161, 121, 81, 61, 64, 64, 3, 2, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv6[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "7" << endl;
	wino_conv_wrapper((quad_tf*)conv6, (quad_tf*)conv7_bias, (quad_tf*)conv7, (quad_tw*)conv7_weight, 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
		for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv7[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "8_1" << endl;
	wino_conv_wrapper((quad_tf*)conv7, (quad_tf*)conv8_1_bias, (quad_tf*)conv8_1, (quad_tw*)conv8_1_weight, 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv8_1[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "8_2" << endl;
	wino_conv_wrapper((quad_tf*)conv8_1, (quad_tf*)conv8_2_bias, (quad_tf*)conv8_2, (quad_tw*)conv8_2_weight, 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv8_2[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "9_1" << endl;
	wino_conv_wrapper((quad_tf*)conv8_2, (quad_tf*)conv9_1_bias, (quad_tf*)conv9_1, (quad_tw*)conv9_1_weight, 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv9_1[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "9_2" << endl;
	wino_conv_wrapper((quad_tf*)conv9_1, (quad_tf*)conv9_2_bias, (quad_tf*)conv9_2, (quad_tw*)conv9_2_weight, 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv9_2[(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "10_1" << endl;
	wino_conv_wrapper((quad_tf*)conv9_2, (quad_tf*)conv10_1_bias[0], (quad_tf*)conv10_1[0], (quad_tw*)conv10_1_weight[0], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv10_1[0][(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	wino_conv_wrapper((quad_tf*)conv9_2, (quad_tf*)conv10_1_bias[1], (quad_tf*)conv10_1[1], (quad_tw*)conv10_1_weight[1], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv10_1[1][(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "10_2" << endl;
	wino_conv_wrapper((quad_tf*)conv10_1[0], (quad_tf*)conv10_2_bias[0], (quad_tf*)conv10_2[0], (quad_tw*)conv10_2_weight[0], 81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv10_1[1], (quad_tf*)conv10_2[0], (quad_tf*)conv10_2[0], (quad_tw*)conv10_2_weight[1], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv10_2[0][(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	wino_conv_wrapper((quad_tf*)conv10_1[0], (quad_tf*)conv10_2_bias[1], (quad_tf*)conv10_2[1], (quad_tw*)conv10_2_weight[2], 81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv10_1[1], (quad_tf*)conv10_2[1], (quad_tf*)conv10_2[1], (quad_tw*)conv10_2_weight[3], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 15; j++){
			cout << conv10_2[1][(i *61 + j)*64] << " ";
		}
		cout << endl;
	}
	cout << "*********************************" << endl;
	++layer_num;
	cout << "11_1" << endl;
	wino_conv_wrapper((quad_tf*)conv10_2[0], (quad_tf*)conv11_1_bias[0], (quad_tf*)conv11_1[0], (quad_tw*)conv11_1_weight[0],  81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv10_2[1], (quad_tf*)conv11_1[0], (quad_tf*)conv11_1[0], (quad_tw*)conv11_1_weight[1],  81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
				for (int j = 0; j < 15; j++){
					cout << conv11_1[0][(i *61 + j)*64] << " ";
				}
				cout << endl;
			}
			cout << "*********************************" << endl;
	wino_conv_wrapper((quad_tf*)conv10_2[0], (quad_tf*)conv11_1_bias[1], (quad_tf*)conv11_1[1], (quad_tw*)conv11_1_weight[2], 81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv10_2[1], (quad_tf*)conv11_1[1], (quad_tf*)conv11_1[1], (quad_tw*)conv11_1_weight[3], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
			for (int j = 0; j < 15; j++){
				cout << conv11_1[1][(i *61 + j)*64] << " ";
			}
			cout << endl;
		}
	cout << "*********************************" << endl;

	++layer_num;
	cout << "11_2" << endl;
	wino_conv_wrapper((quad_tf*)conv11_1[0], (quad_tf*)conv11_2_bias[0], (quad_tf*)conv11_2[0], (quad_tw*)conv11_2_weight[0], 81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv11_1[1], (quad_tf*)conv11_2[0], (quad_tf*)conv11_2[0], (quad_tw*)conv11_2_weight[1], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int i = 0; i < 15; i++){
					for (int j = 0; j < 15; j++){
						cout << conv11_2[0][(i *61 + j)*64] << " ";
					}
					cout << endl;
				}
	cout << "*********************************" << endl;
	wino_conv_wrapper((quad_tf*)conv11_1[0], (quad_tf*)conv11_2_bias[1], (quad_tf*)conv11_2[1], (quad_tw*)conv11_2_weight[2], 81, 61, 81, 61, 64, 64, 3, 1, 0, 1, 1, 0);
	wino_conv_wrapper((quad_tf*)conv11_1[1], (quad_tf*)conv11_2[1], (quad_tf*)conv11_2[1], (quad_tw*)conv11_2_weight[3], 81, 61, 81, 61, 64, 64, 3, 1, 1, 1, 1, 0);
	for (int ch = 0; ch < 64; ch++){
		for (int i = 0; i < 15; i++){
			for (int j = 0; j < 15; j++){
				cout << conv11_2[1][(i *61 + j)*64+ch] << " ";
			}
			cout << endl;
		}

		cout << "************"<<ch<<"*********************" << endl;
	}
	cout << "fc" << endl;
	fc_soft_layer((quad_tf*)conv11_2[0], (quad_tf*)conv11_2[1], (quad_tw*)fc_1x1_l_weight, (quad_tw*)fc_1x1_m_weight, (quad_tw*)fc_1x1_r_weight, (data_ro *)output_image, fc_1x1_bias[0],fc_1x1_bias[1],fc_1x1_bias[2],fc_1x1_bias[3],fc_1x1_bias[4],fc_1x1_bias[5]);

	cout<<"finish layer"<<endl;
	for(int i = 0; i < 81; ++i){
    	for(int j = 0; j < 61; ++j){
            cout << output_image[i*fc_w+j].l << " ";
        }
        cout << endl;
    }
 	cout<<endl;
}

void* landlane_init(void *s){

	uint64_t start,end;
	double run_time;

	cout << "start data_malloc_init!" << endl;
	//start = sds_clock_counter();
	data_malloc_init();
	//end = sds_clock_counter();
	run_time = (end - start)/800000.0;
	cout << "data_malloc_init finished!" << "running time =  " << run_time << " ms" << endl;

	cout << "start weight_malloc_init!" << endl;
	//start = sds_clock_counter();
	weight_malloc_init();
	//end = sds_clock_counter();
	run_time = (end - start)/800000.0;
	cout << "weight_malloc_init finished!" << "running time =  " << run_time << " ms" << endl;

	cout << "start loading weight!" << endl;
	//start = sds_clock_counter();
	weight_assign();
	//end = sds_clock_counter();
	run_time = (end - start)/800000.0;
	cout << "weight_assign finished!" << "running time =  " << run_time << " ms" << endl;

	cout << "start loading bias!" << endl;
	//start = sds_clock_counter();
	bias_assign();
	//end = sds_clock_counter();
	run_time = (end - start)/800000.0;
	cout << "bias_assign finished!" << "running time =  " << run_time << " ms" << endl;

	cout <<"start loading input_data"<<endl;
	//start = sds_clock_counter();

	int idx = 0;
	ifstream infile;
	infile.open("conv1_input.txt");
	if (!infile)
		cout << "open error" << endl;
	for(int i = 0; i <in_h ; ++i){//641
		for(int j = 0; j < in_w; ++j){//481
			for(int k = 0; k < 4; ++k){
					infile >> input_r[idx];
					++idx;
			}
		}
	}
	//end = sds_clock_counter();
	run_time = (end - start)/800000.0;
	cout << "loading input_data finished!" << "running time =  " << run_time << " ms" << endl;

	return (void *)0;
	}

//void* cpy(void *s)
//{
//	if(odd==0)
//	{
//	memcpy((layer_1_type*)input_r_0, (layer_1_type*)input_data, 641*481*sizeof(quad_layer_1_type));
//	odd=1;
//	}
//	else
//	{
//	memcpy((layer_1_type*)input_r_1, (layer_1_type*)input_data, 641*481*sizeof(quad_layer_1_type));
//	odd=0;
//	}
//	return (void *)0;
//}

int show_11_2()
{
	int out_h_11,out_w_11;
	for( out_h_11=0;out_h_11<81;out_h_11++){
		for( out_w_11=0;out_w_11<61;out_w_11++)
		{
			cout<<conv11_2[0][(out_h_11*61+out_w_11)*64+0]<<' ';
		}
		cout<<endl;
	}

	cout<<"####################################"<<endl;

	for( out_h_11=0;out_h_11<81;out_h_11++){
		for( out_w_11=0;out_w_11<61;out_w_11++)
		{
			cout<<conv11_2[1][(out_h_11*61+out_w_11)*64+0]<<' ';
		}
		cout<<endl;
	}
}

int main()
{

			//printf("in putd data is %x",input_data);
			///memcpy((layer_1_type*)input_r, (layer_1_type*)input_data, 641*481*sizeof(quad_layer_1_type));
			//video_flag = 0;
			landlane_init((void *)0);

			for(int i = 0;i<10;i++)
				cout<<input_r[i]<<" " <<endl;

			cout << "start forward!" << endl;
			double start,end,run_time;
			//start = sds_clock_counter();
			forward();
			//end = sds_clock_counter();
			run_time = (end - start)/800000.0;
			cout << "running time =  " << run_time << " ms" << endl;
			//write_flag = 1;
			cout << "forward finished!" << endl;
			//show_11_2();
//			show_output_l();
//			show_output_m();
//			show_output_r();
			return 0;
		    //run_time = (end - start)/800000.0;
			//cout << "running time =  " << run_time << " ms" << endl;
		//    }
//		 }
}
