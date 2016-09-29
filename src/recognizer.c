/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-29
* DESCRIPTION : 字符识别器
*               函数接口 ： CHAR RecognizeSCAU(IplImage *pImage, int version, int location);
*
****************************************************************/

#include <recognizer.h>
#include <config.h>
#include <cnn.h>
#include "math.h"


//模型初始化配置
void Model_Init(CharCNNClassifier *model) {
    CNNModelInit(model);
}


//模型预测
CHAR Predict(CharCNNClassifier *model, IplImage *pImage) {
    return CNNModelPredict(model, pImage);
}

CHAR RecognizeSCAU(IplImage *pImage, int version, int location) {

    CHAR y_pred;

    #if DEBUG_LEVEL>5
        printmat(pImage);
    #endif

    CharCNNClassifier model;
    Model_Init(&model);

    y_pred = Predict(&model, pImage);

    return y_pred;
}