/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-12-19
* DESCRIPTION : 字符识别器
*               函数接口 ： CHAR RecognizeSCAU(IplImage *pImage, int version, int location);
*
****************************************************************/

#include <recognizer.h>

static CharCNNClassifier Char_Model;
static DigitCNNClassifier DigitChar_Model;
static LetterCNNClassifier LetterChar_Model;


//模型初始化配置
void Model_Init() {
    CNNModelInit(&Char_Model);
    DigitCNNModelInit(&DigitChar_Model);
    LetterCNNModelInit(&LetterChar_Model);
}


//模型预测
CHAR Predict(IplImage *pImage, int version, int location) {

    if (location == 0)//字母识别
        return LetterCNNModelPredict(&LetterChar_Model, pImage);
    else if ((location > 0) && (location < 4))//混合识别
        return CNNModelPredict(&Char_Model, pImage);
    else   //数字识别
        return DigitCNNModelPredict(&DigitChar_Model, pImage);


}
// 识别 location ==0 --->字母; 4>location >0 --->混合; location >=4 --->数字
CHAR RecognizeSCAU(IplImage *pImage, int version, int location) {

    CHAR y_pred;
//    struct timeval start;

#if DEBUG_LEVEL > 5
    printmat(pImage);
#endif

    if (Char_Model.init == 0)
        Model_Init();

    y_pred = Predict(pImage,version,location);

    return y_pred;
}