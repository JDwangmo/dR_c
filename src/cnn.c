/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-12-19
* DESCRIPTION :
 *              （1） CNN综合分类器
 *              （2） CNN二分类分类器
 *              （3） 局部灰度值二分类
****************************************************************/

#include <cnn.h>

//CNN model init
void CNNModelInit(CharCNNClassifier *model) {


    //region 34分类
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_all.inputHeight = 15;
    model->model_all.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_all.C11.outChannels = 10;
    model->model_all.C11.inChannels = 1;
    model->model_all.C11.height = 3;
    model->model_all.C11.widthStep = 3;

    //region 该部分注释掉 —— 改为单卷积
//    //    2's layer - convolution Layer 2 : 25*1*5*5
//    model->model_all.C21.outChannels = 25;
//    model->model_all.C21.inChannels = 25;
//    model->model_all.C21.height = 5;
//    model->model_all.C21.widthStep = 5;
//
////    2's layer - convolution layer 3 : 25*1*7*7
//    model->model_all.C31.outChannels = 25;
//    model->model_all.C31.inChannels = 25;
//    model->model_all.C31.height = 7;
//    model->model_all.C31.widthStep = 7;
    //endregion

//    3's layer - subsample layer 1 : 10*2*2
    model->model_all.S12.channels = 10;
    model->model_all.S12.height = 2;
    model->model_all.S12.widthStep = 2;
    model->model_all.S12.poolType = MaxPool;

    //region 该部分注释掉 —— 改为单卷积
    //    3's layer - subsample layer 2 : 25*2*2
//    model->model_all.S22.channels = 25;
//    model->model_all.S22.height = 2;
//    model->model_all.S22.widthStep = 2;
//    model->model_all.S22.poolType = MaxPool;

//    3's layer - subsample layer 3 : 25*2*2
//    model->model_all.S32.channels = 2;
//    model->model_all.S32.height = 2;
//    model->model_all.S32.widthStep = 2;
//    model->model_all.S32.poolType = MaxPool;
    //endregion

//    4's layer - full connected layer : 360*40
    model->model_all.FC1.inputNum = 360;
    model->model_all.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*34
    model->model_all.FC2.inputNum = 40;
    model->model_all.FC2.outputNum = 34;
    //endregion



    //region 二分类器 5-6
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_binary_56.inputHeight = 15;
    model->model_binary_56.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_56.C11.outChannels = 10;
    model->model_binary_56.C11.inChannels = 1;
    model->model_binary_56.C11.height = 3;
    model->model_binary_56.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_56.S12.channels = 10;
    model->model_binary_56.S12.height = 2;
    model->model_binary_56.S12.widthStep = 2;
    model->model_binary_56.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 360*40
    model->model_binary_56.FC1.inputNum = 360;
    model->model_binary_56.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*2
    model->model_binary_56.FC2.inputNum = 40;
    model->model_binary_56.FC2.outputNum = 2;


    //endregion

    //region 二分类器 8-B,局部特征（左半边）
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_binary_8B.inputHeight = 15;
    model->model_binary_8B.inputWidthStep = 8;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_8B.C11.outChannels = 10;
    model->model_binary_8B.C11.inChannels = 1;
    model->model_binary_8B.C11.height = 3;
    model->model_binary_8B.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_8B.S12.channels = 10;
    model->model_binary_8B.S12.height = 2;
    model->model_binary_8B.S12.widthStep = 2;
    model->model_binary_8B.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 360*40
    model->model_binary_8B.FC1.inputNum = 180;
    model->model_binary_8B.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*2
    model->model_binary_8B.FC2.inputNum = 40;
    model->model_binary_8B.FC2.outputNum = 2;


    //endregion


    //region 3分类器 0-D-Q
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_binary_0DQ.inputHeight = 15;
    model->model_binary_0DQ.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_0DQ.C11.outChannels = 10;
    model->model_binary_0DQ.C11.inChannels = 1;
    model->model_binary_0DQ.C11.height = 3;
    model->model_binary_0DQ.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_0DQ.S12.channels = 10;
    model->model_binary_0DQ.S12.height = 2;
    model->model_binary_0DQ.S12.widthStep = 2;
    model->model_binary_0DQ.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 360*40
    model->model_binary_0DQ.FC1.inputNum = 360;
    model->model_binary_0DQ.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*2
    model->model_binary_0DQ.FC2.inputNum = 40;
    model->model_binary_0DQ.FC2.outputNum = 3;


    //endregion
}

//CNN model init
void LetterCNNModelInit(LetterCNNClassifier *model) {


    //region 34分类
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_all.inputHeight = 15;
    model->model_all.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_all.C11.outChannels = 10;
    model->model_all.C11.inChannels = 1;
    model->model_all.C11.height = 3;
    model->model_all.C11.widthStep = 3;

//    3's layer - subsample layer 1 : 10*2*2
    model->model_all.S12.channels = 10;
    model->model_all.S12.height = 2;
    model->model_all.S12.widthStep = 2;
    model->model_all.S12.poolType = MaxPool;


//    4's layer - full connected layer : 360*40
    model->model_all.FC1.inputNum = 360;
    model->model_all.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*34
    model->model_all.FC2.inputNum = 40;
    model->model_all.FC2.outputNum = 34;
    //endregion


    //region 二分类器 0-D
    //    1's layer - input layer : 15*8
    model->init = 1;
    model->model_binary_0D.inputHeight = 15;
    model->model_binary_0D.inputWidthStep = 8;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_0D.C11.outChannels = 10;
    model->model_binary_0D.C11.inChannels = 1;
    model->model_binary_0D.C11.height = 3;
    model->model_binary_0D.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_0D.S12.channels = 10;
    model->model_binary_0D.S12.height = 2;
    model->model_binary_0D.S12.widthStep = 2;
    model->model_binary_0D.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 180*20
    model->model_binary_0D.FC1.inputNum = 180;
    model->model_binary_0D.FC1.outputNum = 20;

//    5's layer - full connected layer : 20*2
    model->model_binary_0D.FC2.inputNum = 20;
    model->model_binary_0D.FC2.outputNum = 2;

    //endregion

    //region 二分类器 0-Q
    //    1's layer - input layer : 7*15
    model->init = 1;
    model->model_binary_0Q.inputHeight = 7;
    model->model_binary_0Q.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_0Q.C11.outChannels = 10;
    model->model_binary_0Q.C11.inChannels = 1;
    model->model_binary_0Q.C11.height = 3;
    model->model_binary_0Q.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_0Q.S12.channels = 10;
    model->model_binary_0Q.S12.height = 2;
    model->model_binary_0Q.S12.widthStep = 2;
    model->model_binary_0Q.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 120*20
    model->model_binary_0Q.FC1.inputNum = 120;
    model->model_binary_0Q.FC1.outputNum = 20;

//    5's layer - full connected layer : 20*2
    model->model_binary_0Q.FC2.inputNum = 20;
    model->model_binary_0Q.FC2.outputNum = 2;

    //endregion

    //region 二分类器 0-G
    //    1's layer - input layer : 15*7
    model->init = 1;
    model->model_binary_0G.inputHeight = 10;
    model->model_binary_0G.inputWidthStep = 10;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_binary_0G.C11.outChannels = 10;
    model->model_binary_0G.C11.inChannels = 1;
    model->model_binary_0G.C11.height = 3;
    model->model_binary_0G.C11.widthStep = 3;

    //    3's layer - subsample layer 1 : 10*2*2
    model->model_binary_0G.S12.channels = 10;
    model->model_binary_0G.S12.height = 2;
    model->model_binary_0G.S12.widthStep = 2;
    model->model_binary_0G.S12.poolType = MaxPool;

    //    4's layer - full connected layer : 120*20
    model->model_binary_0G.FC1.inputNum = 120;
    model->model_binary_0G.FC1.outputNum = 20;

//    5's layer - full connected layer : 20*2
    model->model_binary_0G.FC2.inputNum = 20;
    model->model_binary_0G.FC2.outputNum = 2;

    //endregion


}

//CNN model init --- digit
void DigitCNNModelInit(DigitCNNClassifier *model) {


    //region 34分类
    //    1's layer - input layer : 15*15
    model->init = 1;
    model->model_all.inputHeight = 15;
    model->model_all.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 10*1*3*3
    model->model_all.C11.outChannels = 10;
    model->model_all.C11.inChannels = 1;
    model->model_all.C11.height = 3;
    model->model_all.C11.widthStep = 3;



//    3's layer - subsample layer 1 : 10*2*2
    model->model_all.S12.channels = 10;
    model->model_all.S12.height = 2;
    model->model_all.S12.widthStep = 2;
    model->model_all.S12.poolType = MaxPool;

//    4's layer - full connected layer : 360*40
    model->model_all.FC1.inputNum = 360;
    model->model_all.FC1.outputNum = 40;

//    5's layer - full connected layer : 40*10
    model->model_all.FC2.inputNum = 40;
    model->model_all.FC2.outputNum = 10;
    //endregion



}

int TanhApproximateFunction(int x) {
    /*
     * tanh x=sinh x / cosh x
     * 其中sinh x=(e^(x)-e^(-x))/2 ，cosh x=(e^x+e^(-x))/2
     * 所以tanhx = (e^(x)-e^(-x)) /(e^x+e^(-x))
     * */
//    return (int)(tanh(x));
    if (x > 10)
        return 1;
    else if (x < -10)
        return -1;
    else
        return (int) ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
}

CHAR CNNModelPredict(CharCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_all.C11.outChannels * 13 * 13];
//    int c21_output[25 * 11 * 11], c31_output[25 * 9 * 9];
    int merge_output[model->model_all.FC1.inputNum];
    int fc1_output[model->model_all.FC1.outputNum];
    char index_to_char[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'I', 'J', 'K', 'L',
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z'};
    /*,fc2_output[model->model_all.FC2.outputNum]*/
//    5-->3-->7
//    length - 1925
//    int merge_output[25 * 5 * 5 + 25 * 6 * 6 + 25 * 4 * 4];

    //region 3*3 convolution
    //    output size
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 13;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0;
                for (row = 0; row < model->model_all.C11.height; row++) {
                    for (col = 0; col < model->model_all.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_all.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%ld,%d,%ld\n", C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(
                        sum + C11_B_Weight[k]);
//                printf("%ld\n",sum + C11_B_Weight[k]);
            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 6;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 6 + i * 6 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion

    //region 5*5 convolution
    //    output size
//    5*5
//    15-5+1=11
//    output_row = 11;
//    output_col = 11;
//    for (k = 0; k < model->model_all.C21.outChannels; k++) {
////        for each filters
//        for (i = 0; i < output_row; i++) {
//            //        i represents start point in row
//            for (j = 0; j < output_col; j++) {
//                //        j represents start point in col
//                sum = 0;
//                for (row = 0; row < model->model_all.C21.height; row++) {
//                    for (col = 0; col < model->model_all.C21.widthStep; col++) {
//                        //      index position
//                        map_index_position = k * 5 * 5 + row * model->model_all.C21.widthStep + col;
//                        image_index_position = (i + row) * pImage->widthStep + (j + col);
////                        printf("%d,%d\n", map_index_position, image_index_position);
////                        printf("%f,%d\n", C11_Map_Weight[map_index_position], pImage->imageData[image_index_position]);
//                        sum += C21_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
//                    }
//                }
////                11*11
//                c21_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(sum + C21_B_Weight[k]);
//            }
//        }
//        printmat2(c21_output, output_row, output_row);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//    }
//    output_row = 5;
//    output_col = 5;
//    for (k = 0; k < model->model_all.C21.outChannels; k++) {
////        for each filters
//        for (i = 0; i < output_row; i++) {
//            //        i represents start point in row
//            for (j = 0; j < output_col; j++) {
//                //        j represents start point in col
//                max = -10;
//                for (row = 0; row < 2; row++) {
//                    for (col = 0; col < 2; col++) {
//                        value = c21_output[k * 11 * 11 + (2 * i + row) * 11 + (2 * j + col)];
//                        if (value > max) {
//                            max = value;
//                        }
//
//                    }
//                }
//                merge_output[k * output_row * output_row + i * output_row + j] = max;
//            }
//        }
////        printmat2(merge_output, 6, 6);
////                printf("%f",sum);
////                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
////        assert(NULL);
//    }
//    //endregion
//
//    //region 7*7 convolution
//    //    output size
////    7*7
////    15-7+1=9
//    output_row = 9;
//    output_col = 9;
//    for (k = 0; k < model->model_all.C31.outChannels; k++) {
////        for each filters
//        for (i = 0; i < output_row; i++) {
//            //        i represents start point in row
//            for (j = 0; j < output_col; j++) {
//                //        j represents start point in col
//                sum = 0;
//                for (row = 0; row < model->model_all.C31.height; row++) {
//                    for (col = 0; col < model->model_all.C31.widthStep; col++) {
//                        //      index position
//                        map_index_position = k * 7 * 7 + row * model->model_all.C31.widthStep + col;
//                        image_index_position = (i + row) * pImage->widthStep + (j + col);
////                        printf("%d,%d\n", map_index_position, image_index_position);
////                        printf("%f,%d\n", C11_Map_Weight[map_index_position], pImage->imageData[image_index_position]);
//                        sum += C31_Map_Weight[map_index_position]* pImage->imageData[image_index_position];
//                    }
//                }
////                9*9
//                c31_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(sum + C31_B_Weight[k]);
//            }
//        }
////        printmat2(c31_output, output_row, output_row);
//    }
//    output_row = 4;
//    output_col = 4;
//    for (k = 0; k < model->model_all.C31.outChannels; k++) {
////        for each filters
//        for (i = 0; i < output_row; i++) {
//            //        i represents start point in row
//            for (j = 0; j < output_col; j++) {
//                //        j represents start point in col
//                max = -10;
//                for (row = 0; row < 2; row++) {
//                    for (col = 0; col < 2; col++) {
//                        value = c31_output[k * 9 * 9 + (2 * i + row) * 9 + (2 * j + col)];
//                        if (value > max) {
//                            max = value;
//                        }
//
//                    }
//                }
//                merge_output[25 * 6 * 6 + 25 * 5 * 5 + k * output_row * output_row + i * output_row + j] = max;
//            }
//        }
//        printmat2(merge_output, 4, 4);

//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
//    }
    //endregion

    //region FC1 - 360*40

    for (row = 0; row < model->model_all.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC1.inputNum; col++) {
            sum += merge_output[col] * FC1_Map_Weight[row * model->model_all.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

//    for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//        printf("%f,",fc1_output[col]);
//    }
//    printf("\n");
//
//    for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[7*model->model_all.FC2.inputNum+col]);
//    }
//    printf("\n");
//    for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[33*model->model_all.FC2.inputNum+col]);
//    }
//    printf("\n");
    //region FC2 - 40*34
    max = -1000000;
    k = 0;


    for (row = 0; row < model->model_all.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],FC2_Map_Weight[row * model->model_all.FC2.inputNum + col]);
            sum += fc1_output[col] * FC2_Map_Weight[row * model->model_all.FC2.inputNum + col];
        }
        value = sum + FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion
//    return index_to_char[k];
    if (index_to_char[k] == '5' || index_to_char[k] == '6')
        return index_to_char[CNNModelPredictBinary56(model, pImage)];
    else if (k == 0 || index_to_char[k] == 'D' || index_to_char[k] == 'Q')
        return index_to_char[CNNModelPredictBinary0DQ(model, pImage)];
    else if (k == 8 || index_to_char[k] == 'B')
        return index_to_char[CNNModelPredictBinary8B(model, pImage)];
    //    使用局部灰度值来二分类
    else if (index_to_char[k] == 'E' || index_to_char[k] == 'F')
        return LocalRegionGrayValuePredictEF(pImage);
    else if (index_to_char[k] == 'R' || index_to_char[k] == 'P')
        return LocalRegionGrayValuePredictRP(pImage);
    else if (index_to_char[k] == 'T' || index_to_char[k] == 'I')
        return LocalRegionGrayValuePredictTI(pImage);
    else
        return index_to_char[k];
//    return '1' ;
}

CHAR LetterCNNModelPredict(LetterCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_all.C11.outChannels * 13 * 13];
    int merge_output[model->model_all.FC1.inputNum];
    int fc1_output[model->model_all.FC1.outputNum];
    char index_to_char[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'I', 'J', 'K', 'L', 'M', 'N', '0', 'P', 'Q',
                            'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z'};
    CHAR result;
    //region 3*3 convolution
    //    output size
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 13;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0;
                for (row = 0; row < model->model_all.C11.height; row++) {
                    for (col = 0; col < model->model_all.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_all.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%ld,%d,%ld\n", C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += Letter_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(
                        sum + Letter_C11_B_Weight[k]);
//                printf("%ld\n",sum + C11_B_Weight[k]);
            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 6;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 6 + i * 6 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_all.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC1.inputNum; col++) {
            sum += merge_output[col] * Letter_FC1_Map_Weight[row * model->model_all.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Letter_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

    //region FC2 - 40*34
    max = -1000000;
    k = 0;


    for (row = 0; row < model->model_all.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],FC2_Map_Weight[row * model->model_all.FC2.inputNum + col]);
            sum += fc1_output[col] * Letter_FC2_Map_Weight[row * model->model_all.FC2.inputNum + col];
        }
        value = sum + Letter_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }

    result = index_to_char[k];
    //endregion
    if (result == '0' || result == 'D')
        if (LetterCNNModelPredictBinary0D(model, pImage) == 'D')
            return 'D';
        else
            result = '0';
    if (result == '0' || result == 'Q')
        if (LetterCNNModelPredictBinary0Q(model, pImage) == 'Q')
            return 'Q';
        else
            result = '0';
    if (result == '0' || result == 'G')
        if (LetterCNNModelPredictBinary0G(model, pImage) == 'G')
            return 'G';
        else
            result = '0';
//    使用局部灰度值来二分类
    if (result == 'E' || result == 'F')
        return LocalRegionGrayValuePredictEF(pImage);
    if (result == 'R' || result == 'P')
        return LocalRegionGrayValuePredictRP(pImage);
    if (result == 'T' || result == 'I')
        return LocalRegionGrayValuePredictTI(pImage);
    return result;
}


CHAR LocalRegionGrayValuePredictEF(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0, min_value = 300;
    for (row = 10; row < 14; row++) {
        for (col = 5; col < 14; col++) {
            // get the pixel
            locate_value = (INT32U) (255 - pImage->imageData[row * pImage->widthStep + col]);
            sum_value += locate_value;
            if (locate_value < min_value) {
                min_value = locate_value;
            }
        }
    }
//    总共36个像素点：4*9
//    printf("%d,%d,%d\n",sum_value,min_value,(sum_value-min_value*36));
    if(sum_value-min_value*36>1933)
        return 'E';
    else
        return 'F';
}

CHAR LocalRegionGrayValuePredictRP(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0, min_value = 300;
    for (row = 8; row < 14; row++) {
        for (col = 8; col < 14; col++) {
            // get the pixel
            locate_value = (INT32U) (255 - pImage->imageData[row * pImage->widthStep + col]);
            sum_value += locate_value;
            if (locate_value < min_value) {
                min_value = locate_value;
            }
        }
    }
//    总共36个像素点：6*6=36
//    printf("%d,%d,%d\n",sum_value,min_value,(sum_value-min_value*36));
    if(sum_value-min_value*36>1953)
        return 'R';
    else
        return 'P';
}
CHAR LocalRegionGrayValuePredictTI(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0, min_value = 300;
    for (row = 1; row < 5; row++) {
        for (col = 1; col < 14; col++) {
            // get the pixel
            locate_value = (INT32U) (255 - pImage->imageData[row * pImage->widthStep + col]);
            sum_value += locate_value;
            if (locate_value < min_value) {
                min_value = locate_value;
            }
        }
    }
//    总共36个像素点：4*13=52
//    printf("%d,%d,%d\n",sum_value,min_value,(sum_value-min_value*36));
    if(sum_value-min_value*52>3219)
        return 'T';
    else
        return 'I';
}

CHAR DigitCNNModelPredict(DigitCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_all.C11.outChannels * 13 * 13];
//    int c21_output[25 * 11 * 11], c31_output[25 * 9 * 9];
    int merge_output[model->model_all.FC1.inputNum];
    int fc1_output[model->model_all.FC1.outputNum];
    char index_to_char[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    /*,fc2_output[model->model_all.FC2.outputNum]*/
//    5-->3-->7
//    length - 1925
//    int merge_output[25 * 5 * 5 + 25 * 6 * 6 + 25 * 4 * 4];

    //region 3*3 convolution
    //    output size
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 13;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0;
                for (row = 0; row < model->model_all.C11.height; row++) {
                    for (col = 0; col < model->model_all.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_all.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%ld,%d,%ld\n", C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += Digit_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(
                        sum + Digit_C11_B_Weight[k]);
//                printf("%ld\n",sum + C11_B_Weight[k]);
            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 6;
    for (k = 0; k < model->model_all.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 6 + i * 6 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion

    //region FC1 - 360*40

    for (row = 0; row < model->model_all.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC1.inputNum; col++) {
            sum += merge_output[col] * Digit_FC1_Map_Weight[row * model->model_all.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Digit_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

    //region FC2 - 40*34
    max = -1000000;
    k = 0;
    for (row = 0; row < model->model_all.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_all.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],FC2_Map_Weight[row * model->model_all.FC2.inputNum + col]);
            sum += fc1_output[col] * Digit_FC2_Map_Weight[row * model->model_all.FC2.inputNum + col];
        }
        value = sum + Digit_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion
    return index_to_char[k];

}


CHAR CNNModelPredictBinary0DQ(CharCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_0DQ.C11.outChannels * 13 * 13];
//    int c21_output[25 * 11 * 11], c31_output[25 * 9 * 9];
    int merge_output[model->model_binary_0DQ.FC1.inputNum];
    int fc1_output[model->model_binary_0DQ.FC1.outputNum];
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 13;
    for (k = 0; k < model->model_binary_0DQ.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0;
                for (row = 0; row < model->model_binary_0DQ.C11.height; row++) {
                    for (col = 0; col < model->model_binary_0DQ.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_0DQ.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%ld,%d,%ld\n", C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += Binary0DQ_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(
                        sum + Binary0DQ_C11_B_Weight[k]);
//                printf("%ld\n",sum + C11_B_Weight[k]);
            }
        }
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 6;
    for (k = 0; k < model->model_binary_0DQ.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 6 + i * 6 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_binary_0DQ.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0DQ.FC1.inputNum; col++) {
            sum += merge_output[col] * Binary0DQ_FC1_Map_Weight[row * model->model_binary_0DQ.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Binary0DQ_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

//    for (col = 0; col < model->model_binary_0DQ.FC2.inputNum; col++) {
//        printf("%f,",fc1_output[col]);
//    }
//    printf("\n");
//
//    for (col = 0; col < model->model_binary_0DQ.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[7*model->model_binary_0DQ.FC2.inputNum+col]);
//    }
//    printf("\n");
//    for (col = 0; col < model->model_binary_0DQ.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[33*model->model_binary_0DQ.FC2.inputNum+col]);
//    }
//    printf("\n");
    //region FC2 - 40*34
    max = -10000;
    k = 0;

    for (row = 0; row < model->model_binary_0DQ.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0DQ.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],FC2_Map_Weight[row * model->model_binary_0DQ.FC2.inputNum + col]);
            sum += fc1_output[col] * Binary0DQ_FC2_Map_Weight[row * model->model_binary_0DQ.FC2.inputNum + col];
        }
        value = sum + Binary0DQ_FC2_B_Weight[row];
//        printf("%ld\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return 0;
    else if (k == 1)
        return 13;
    else
        return 25;
//    return '1' ;
}


CHAR CNNModelPredictBinary56(CharCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_56.C11.outChannels * 13 * 13];
//    int c21_output[25 * 11 * 11], c31_output[25 * 9 * 9];
    int merge_output[model->model_binary_56.FC1.inputNum];
    int fc1_output[model->model_binary_56.FC1.outputNum];
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 13;
    for (k = 0; k < model->model_binary_56.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0;
                for (row = 0; row < model->model_binary_56.C11.height; row++) {
                    for (col = 0; col < model->model_binary_56.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_56.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%ld,%d,%ld\n", C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += Binary56_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = TanhApproximateFunction(
                        sum + Binary56_C11_B_Weight[k]);
//                printf("%ld\n",sum + C11_B_Weight[k]);
            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 6;
    for (k = 0; k < model->model_binary_56.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 6 + i * 6 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_binary_56.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_56.FC1.inputNum; col++) {
            sum += merge_output[col] * Binary56_FC1_Map_Weight[row * model->model_binary_56.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Binary56_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

//    for (col = 0; col < model->model_binary_56.FC2.inputNum; col++) {
//        printf("%f,",fc1_output[col]);
//    }
//    printf("\n");
//
//    for (col = 0; col < model->model_binary_56.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[7*model->model_binary_56.FC2.inputNum+col]);
//    }
//    printf("\n");
//    for (col = 0; col < model->model_binary_56.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[33*model->model_binary_56.FC2.inputNum+col]);
//    }
//    printf("\n");
    //region FC2 - 40*34
    max = -10000;
    k = 0;

    for (row = 0; row < model->model_binary_56.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_56.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],FC2_Map_Weight[row * model->model_binary_56.FC2.inputNum + col]);
            sum += fc1_output[col] * Binary56_FC2_Map_Weight[row * model->model_binary_56.FC2.inputNum + col];
        }
        value = sum + Binary56_FC2_B_Weight[row];
//        printf("%ld\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return 5;
    else
        return 6;
//    return '1' ;
}


CHAR CNNModelPredictBinary8B(CharCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_8B.C11.outChannels * 13 * 6];
//    int c21_output[25 * 11 * 11], c31_output[25 * 9 * 9];
    int merge_output[model->model_binary_8B.FC1.inputNum];
    int fc1_output[model->model_binary_8B.FC1.outputNum];
//    printmat(pImage);
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 6;
    for (k = 0; k < model->model_binary_8B.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
//                printf("*************%d,%d,%d\n",k,i,j);
                sum = 0;
                for (row = 0; row < model->model_binary_8B.C11.height; row++) {
                    for (col = 0; col < model->model_binary_8B.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_8B.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%d,%d,%d\n", Binary8B_C11_Map_Weight[map_index_position],
//                               pImage->imageData[image_index_position],
//                               Binary8B_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position]);
                        sum += Binary8B_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_col + i * output_col + j] = TanhApproximateFunction(
                        sum + Binary8B_C11_B_Weight[k]);
//                printf("%d\n",sum + Binary8B_C11_B_Weight[k]);


            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 3;
    for (k = 0; k < model->model_binary_8B.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 6 + (2 * i + row) * 6 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 3 + i * 3 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_binary_8B.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_8B.FC1.inputNum; col++) {
            sum += merge_output[col] * Binary8B_FC1_Map_Weight[row * model->model_binary_8B.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Binary8B_FC1_B_Weight[row]);
//        printf("%d\n",sum + Binary8B_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

//    for (col = 0; col < model->model_binary_8B.FC2.inputNum; col++) {
//        printf("%f,",fc1_output[col]);
//    }
//    printf("\n");
//
//    for (col = 0; col < model->model_binary_8B.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[7*model->model_binary_8B.FC2.inputNum+col]);
//    }
//    printf("\n");
//    for (col = 0; col < model->model_binary_8B.FC2.inputNum; col++) {
//        printf("%f,",FC2_Map_Weight[33*model->model_binary_8B.FC2.inputNum+col]);
//    }
//    printf("\n");
    //region FC2 - 40*34
    max = -1000000;
    k = 0;

    for (row = 0; row < model->model_binary_8B.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_8B.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],Binary8B_FC2_Map_Weight[row * model->model_binary_8B.FC2.inputNum + col]);
            sum += fc1_output[col] * Binary8B_FC2_Map_Weight[row * model->model_binary_8B.FC2.inputNum + col];
        }
        value = sum + Binary8B_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return 8;
    else
        return 11;
//    return '1' ;
}


CHAR LetterCNNModelPredictBinary0D(LetterCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_0D.C11.outChannels * 13 * 6];
    int merge_output[model->model_binary_0D.FC1.inputNum];
    int fc1_output[model->model_binary_0D.FC1.outputNum];
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 6;
    for (k = 0; k < model->model_binary_0D.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
//                printf("*************%d,%d,%d\n",k,i,j);
                sum = 0;
                for (row = 0; row < model->model_binary_0D.C11.height; row++) {
                    for (col = 0; col < model->model_binary_0D.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_0D.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
                        sum += Letter_0D_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_col + i * output_col + j] = TanhApproximateFunction(
                        sum + Letter_0D_C11_B_Weight[k]);
//                printf("%d\n",sum + Binary0D_C11_B_Weight[k]);


            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 3;
    for (k = 0; k < model->model_binary_0D.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 6 + (2 * i + row) * 6 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 3 + i * 3 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_binary_0D.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0D.FC1.inputNum; col++) {
            sum += merge_output[col] * Letter_0D_FC1_Map_Weight[row * model->model_binary_0D.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Letter_0D_FC1_B_Weight[row]);
//        printf("%d\n",sum + Binary0D_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

    //region FC2 - 40*34
    max = -1000000;
    k = 0;

    for (row = 0; row < model->model_binary_0D.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0D.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],Binary0D_FC2_Map_Weight[row * model->model_binary_0D.FC2.inputNum + col]);
            sum += fc1_output[col] * Letter_0D_FC2_Map_Weight[row * model->model_binary_0D.FC2.inputNum + col];
        }
        value = sum + Letter_0D_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return '0';
    else
        return 'D';
//    return '1' ;
}

CHAR LetterCNNModelPredictBinary0Q(LetterCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_0Q.C11.outChannels * 5 * 13];
    int merge_output[model->model_binary_0Q.FC1.inputNum];
    int fc1_output[model->model_binary_0Q.FC1.outputNum];
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 5;
    output_col = 13;
    for (k = 0; k < model->model_binary_0Q.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
//                printf("*************%d,%d,%d\n",k,i,j);
                sum = 0;
                for (row = 0; row < model->model_binary_0Q.C11.height; row++) {
                    for (col = 0; col < model->model_binary_0Q.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_0Q.C11.widthStep + col;
                        image_index_position = (8 + i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d,%d\n",image_index_position,8+i,j);
                        sum += Letter_0Q_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_col + i * output_col + j] = TanhApproximateFunction(
                        sum + Letter_0Q_C11_B_Weight[k]);
//                printf("%d\n",c11_output[k * output_row * output_col + i * output_row + j]);


            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 2;
    output_col = 6;
    for (k = 0; k < model->model_binary_0Q.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -1000000;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 5 * 13 + (2 * i + row) * 13 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 2 * 6 + i * 6 + j] = max;
//                printf("%d,%d,%d\n",k * 2 * 6 , i * 2 ,j);
//                printf("%d\n",max);
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 120*20

    for (row = 0; row < model->model_binary_0Q.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0Q.FC1.inputNum; col++) {
            sum += merge_output[col] * Letter_0Q_FC1_Map_Weight[row * model->model_binary_0Q.FC1.inputNum + col];
//            printf("%d\n",merge_output[col]);
        }
//        assert(NULL);
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Letter_0Q_FC1_B_Weight[row]);
//        printf("%d\n",sum + Letter_0Q_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

    //region FC2 - 20*2
    max = -1000000;
    k = 0;

    for (row = 0; row < model->model_binary_0Q.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0Q.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],Binary0Q_FC2_Map_Weight[row * model->model_binary_0Q.FC2.inputNum + col]);
            sum += fc1_output[col] * Letter_0Q_FC2_Map_Weight[row * model->model_binary_0Q.FC2.inputNum + col];
        }
        value = sum + Letter_0Q_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return '0';
    else
        return 'Q';
//    return '1' ;
}

CHAR LetterCNNModelPredictBinary0G(LetterCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    int sum, max, value;
    int c11_output[model->model_binary_0G.C11.outChannels * 13 * 6];
    int merge_output[model->model_binary_0G.FC1.inputNum];
    int fc1_output[model->model_binary_0G.FC1.outputNum];
// convolution - layer
//    3*3
//    15-3+1=13
    output_row = 13;
    output_col = 5;
    for (k = 0; k < model->model_binary_0G.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
//                printf("*************%d,%d,%d\n",k,i,j);
                sum = 0;
                for (row = 0; row < model->model_binary_0G.C11.height; row++) {
                    for (col = 0; col < model->model_binary_0G.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_binary_0G.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (8 + j + col);
                        sum += Letter_0G_C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_col + i * output_col + j] = TanhApproximateFunction(
                        sum + Letter_0G_C11_B_Weight[k]);
//                printf("%d\n",sum + Binary0G_C11_B_Weight[k]);


            }
        }
//                printmat(pImage);
//        printmat2(c11_output, 13, 13);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }


    output_row = 6;
    output_col = 2;
    for (k = 0; k < model->model_binary_0G.C11.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -100000;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c11_output[k * 13 * 5 + (2 * i + row) * 5 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * 6 * 2 + i * 2 + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion


    //region FC1 - 360*40

    for (row = 0; row < model->model_binary_0G.FC1.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0G.FC1.inputNum; col++) {
            sum += merge_output[col] * Letter_0G_FC1_Map_Weight[row * model->model_binary_0G.FC1.inputNum + col];
        }
//        fc1_output[row] = tanhf(sum + FC1_B_Weight[row]);
        fc1_output[row] = TanhApproximateFunction(sum + Letter_0G_FC1_B_Weight[row]);
//        printf("%d\n",sum + Binary0G_FC1_B_Weight[row]);
//        fc1_output[row] = sum + FC1_B_Weight[row];
    }
    //endregion

    //region FC2 - 40*34
    max = -1000000;
    k = 0;

    for (row = 0; row < model->model_binary_0G.FC2.outputNum; row++) {
        sum = 0;
        for (col = 0; col < model->model_binary_0G.FC2.inputNum; col++) {
//            printf("%d,%d\n",fc1_output[col],Binary0G_FC2_Map_Weight[row * model->model_binary_0G.FC2.inputNum + col]);
            sum += fc1_output[col] * Letter_0G_FC2_Map_Weight[row * model->model_binary_0G.FC2.inputNum + col];
        }
        value = sum + Letter_0G_FC2_B_Weight[row];
//        printf("%d\n",value);
        if (value > max) {
            max = value;
            k = row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    if (k == 0)
        return '0';
    else
        return 'G';
//    return '1' ;
}
