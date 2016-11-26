/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-29
* DESCRIPTION : 字符识别器
*               函数接口 ： CHAR RecognizeSCAU(IplImage *pImage, int version, int location);
*
****************************************************************/

#include <recognizer.h>

static CharCNNClassifier Char_Model;


//模型初始化配置
void Model_Init() {
    CNNModelInit(&Char_Model);
}


//模型预测
CHAR Predict(CharCNNClassifier *model, IplImage *pImage) {
    return CNNModelPredict(model, pImage);
}

CHAR RecognizeSCAU(IplImage *pImage, int version, int location) {

    CHAR y_pred;
//    struct timeval start;

    #if DEBUG_LEVEL>5
        printmat(pImage);
    #endif

    if (Char_Model.init==0)
        Model_Init();

    y_pred = Predict(&Char_Model, pImage);

    return y_pred;
}