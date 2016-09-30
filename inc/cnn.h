#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mat.h"
#include "config.h"
#include "weight.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

#define Tanh 0
#define Sigmoid 1
#define Linear 2

// 卷积层
typedef struct {
    // 关于特征模板的权重分布，这里是一个四维数组
    // 其大小为outChannels*inChannels*mapSize*mapSize大小
    INT32U  outChannels;			//map的通道数,0-dim
    INT32U  inChannels; 		    //map的通道数,1-dim
    INT32U  height;					//map的高度,2-dim
    INT32U  widthStep;				//map的宽度,3-dim
//    float *	mapData;				//map首地址
//    float *basicData;   //偏置首地址，偏置的大小，为outChannels
} CovLayer;

// 采样层 pooling
typedef struct {
//    输入是 3D 图像
//    输出是 3D 图像
    INT32U  channels;			    //map的通道数,0-dim
    INT32U  height;					//map的高度,1-dim
    INT32U  widthStep;				//map的宽度,2-dim
    int poolType;     //Pooling的方法
} PoolLayer;

// 输出层 全连接的神经网络
typedef struct {
//    输入是1D矩阵
//    输出是1D矩阵
    INT32U inputNum;   //输入数据的数目
    INT32U outputNum;  //输出数据的数目
//    float **wData; // 权重数据，为一个inputNum*outputNum大小
//    float *basicData;   //偏置，大小为outputNum大小
} Dense;

//一个卷积神经网络
typedef struct {
//    int layerNum;
//    int num_labels;
//    nSize inputSize;
    INT32U inputHeight;   //输入数据的高度
    INT32U inputWidthStep;   //输入数据的宽度
    CovLayer C11;
    PoolLayer S12;
    CovLayer C21;
    PoolLayer S22;
    CovLayer C31;
    PoolLayer S32;
    Dense FC1;
    Dense FC2;
} CNN;

//数字识别混合CNN模型
typedef struct {
//    34分类器
//    标志位,是否初始化过
    BOOL init;
    CNN model_all;

//    2分类 0D
//    CNN* model_binary_0D;
//    2分类 1I
//    CNN* model_binary_1I;
//    2分类 2Z
//    CNN* model_binary_2Z;
//    2分类 56
//    CNN* model_binary_56;
//    2分类 4A
//    CNN* model_binary_4A;
//    2分类 8B
//    CNN* model_binary_8B;
} CharCNNClassifier;


//CNN model init
void CNNModelInit(CharCNNClassifier *model);

CHAR CNNModelPredict(CharCNNClassifier *model, IplImage *pImage);

#endif
