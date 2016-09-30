/**************************************************************
* Created by jdwang on 2016-09-15.
* DESCRIPTION : 
****************************************************************/

#ifndef DR_C_RECOGNIZER_H
#define DR_C_RECOGNIZER_H

#include "typedef.h"
#include "config.h"
#include "mat.h"
#include "cnn.h"



CHAR RecognizeSCAU(IplImage *pImage, int version,int location);
//模型初始化
void Model_Init();
//识别，返回预测结果
CHAR Predict(CharCNNClassifier * model,IplImage *pImage);

#endif //DR_C_RECOGNIZER_H
