/**************************************************************
 * Created by jdwang on 2016-09-15.
 * Last updated on 2017-01-03
 * DESCRIPTION :
 *              （1） CNN综合分类器
 *              （2） CNN二分类分类器
 *              （3） 局部灰度值二分类
****************************************************************/

#include <cnn.h>


// ***************** CNN model init -- merge ***********************
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

//CNN model init -- letter
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

// **************** CNN model predict --- merge ************************
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

    // region 丢弃
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

    //region CNN 二分类
#if MODEL_MODE > 1
    if (index_to_char[k] == '5' || index_to_char[k] == '6')
        k = CNNModelPredictBinary56(model, pImage);
    if (k == 0 || index_to_char[k] == 'D' || index_to_char[k] == 'Q')
        k = CNNModelPredictBinary0DQ(model, pImage);
    if (k == 8 || index_to_char[k] == 'B')
        k = CNNModelPredictBinary8B(model, pImage);
#endif
    //endregion
    // region 使用局部灰度值来二分类
#if MODEL_MODE > 2
    //  R-P:  对区域 [8:14,8:14] 求灰度和, 临界值：1953
    if (index_to_char[k] == 'R' || index_to_char[k] == 'P')
        if (LocalRegionGrayValuePredictRP(pImage) == 'R')
            k = 26;
        else
            k = 24;
//  I-T: : 对区域 [1:5,1:6]，[1:5,9:14] 求灰度和  I：0-44   T：2676-6481  临界值：1316
    if (index_to_char[k] == 'T' || index_to_char[k] == 'I')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictTI(pImage) == 'T')
            k = 28;
        else
            k = 18;

//  T-J: : 对区域 [7:14,1:14] 进行二值化，取这个小区域的左下角 [5::,0:2]，如果这个区域有黑点，就修正为 J
//    注意，这里只是单向修改，即 修改 T 的预测结果 ，对于预测为 J 的 不修改
    if (index_to_char[k] == 'T')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictTJ(pImage) == 'T')
            k = 28;//T
        else
            k = 19;//J

//    E-F : 对区域 [7:,::] 进行二值化，然后取小区域的  [3:7,6:14]    F:：0-1，E：7-24  临界值：3
    if (index_to_char[k] == 'E' || index_to_char[k] == 'F')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictEF(pImage) == 'E')
            k = 14;
        else
            k = 15;
//    7-Z : 对区域 [7:14,1:14] 进行二值化，然后取这个小区域的左下角（4*4，3::,0:4）和右下角（4*4，3::,9::），选择点数最小的这个区域来计算，7:0-1   Z：3-12
    if (index_to_char[k] == 'Z' || index_to_char[k] == '7')
        if (LocalRegionGrayValuePredictZ7(pImage) == 'Z')
            k = 33;//Z
        else
            k = 7;//7
//    8-3：区域 [1:14,1:6]， 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == '8' || index_to_char[k] == '3')
        if (LocalRegionGrayValuePredict83(pImage) == '8')
            k = 8;//8
        else
            k = 3;//3
//    8-6：区域 [1:8,8:14]， 对区域进行二值化， 并判断是否有环，单向 8 修正为 6
    if (index_to_char[k] == '8')
        if (LocalRegionGrayValuePredict86(pImage) == '8')
            k = 8;//8
        else
            k = 6;//6
//    0-C：区域 [1:14,9:14], 对区域进行二值化， 并判断是否有环 --- 这里是单向的，只对预测成 C 修正，预测成0的不修正
    if (index_to_char[k] == 'C')
        if (LocalRegionGrayValuePredict0C(pImage) == '0')
            k = 0;//0
        else
            k = 12;//C
//    P-F：区域 [1:9,8:14] , 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == 'P' || index_to_char[k] == 'F')
        if (LocalRegionGrayValuePredictPF(pImage) == 'P')
            k = 24;//P
        else
            k = 15;//F
//    6-E：区域 [5:14,9:14], 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == '6' || index_to_char[k] == 'E')
        if (LocalRegionGrayValue6redict6E(pImage) == '6')
            k = 6;//6
        else
            k = 14;//E
//    6-5：区域 [6:14,1:7]， 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == '6' || index_to_char[k] == '5')
        if (LocalRegionGrayValue6redict65(pImage) == '6')
            k = 6;//6
        else
            k = 5;//5

#endif
    // endregion
    return index_to_char[k];
}

//  R-P:  对区域 [8:14,8:14] 求灰度和, 临界值：1953
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
    if (sum_value - min_value * 36 > 1953)
        return 'R';
    else
        return 'P';
}


//  I-T: : 对区域 [1:5,1:6]，[1:5,9:14] 求灰度和  I：0-44   T：2676-6481  临界值：1316
CHAR LocalRegionGrayValuePredictTI(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0, min_value = 300;
//    区域 [1:5,1:6], 4*5
    for (row = 1; row < 5; row++) {
        for (col = 1; col < 6; col++) {
            // get the pixel, 黑白颠倒
            locate_value = (INT32U) (255 - pImage->imageData[row * pImage->widthStep + col]);
            sum_value += locate_value;
            if (locate_value < min_value) {
                min_value = locate_value;
            }
        }
    }
//    区域 [1:5,9:14],4*4
    for (row = 1; row < 5; row++) {
        for (col = 9; col < 14; col++) {
            // get the pixel, 黑白颠倒
            locate_value = (INT32U) (255 - pImage->imageData[row * pImage->widthStep + col]);
            sum_value += locate_value;
            if (locate_value < min_value) {
                min_value = locate_value;
            }
        }
    }
//    总共40个像素点：4*5+4*4=36
//    printf("%d,%d,%d\n",sum_value,min_value,(sum_value-min_value*36));
    if (sum_value - min_value * 36 > 1316)
        return 'T';
    else
        return 'I';
}

//  T-J: : 对区域 [7:14,1:14] 进行二值化，取这个小区域的左下角 [5:,0:2]，如果这个区域有白点，就修正为 J
CHAR LocalRegionGrayValuePredictTJ(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
//    区域 [7:14,1:14], 7*13
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 7; row < 14; row++) {
        for (col = 1; col < 14; col++) {

            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);
    //    区域 [12:14,1:3],2*2
    for (row = 12; row < 14; row++) {
        for (col = 1; col < 3; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }
    }
    if (sum_value > 0)
        return 'J';
    else
        return 'T';
}

// 寻找 最优 二值化 阈值
int get_best_threshold(const UINT *count_of_gray) {
    UINT accumulation_of_pixel[255] = {0};
    UINT accumulation_of_gray[255] = {0};
    int i, w1, w2, u1, u2, v, max_v = 0, best_threshold = 0;
//    初始值
    accumulation_of_pixel[0] = count_of_gray[0];
    accumulation_of_gray[0] = 0;
    for (i = 1; i < 256; i++) {
//        <= i 时
        accumulation_of_pixel[i] = accumulation_of_pixel[i - 1] + count_of_gray[i];
        accumulation_of_gray[i] = accumulation_of_gray[i - 1] + count_of_gray[i] * i;
    }
    for (i = 0; i < 256; i++) {
//        <=i ,有多少像素点
        w1 = accumulation_of_pixel[i];
//        >i ,有多少像素点
        w2 = accumulation_of_pixel[255] - w1;
        if (w1 * w2 == 0)
            continue;
        u1 = accumulation_of_gray[i] / w1;
        u2 = (accumulation_of_gray[255] - accumulation_of_gray[i]) / w2;
        v = w1 * w2 * (u1 - u2) * (u1 - u2);
        if (v > max_v) {
            max_v = v;
            best_threshold = i;
        }
    }

    return best_threshold;
}

//    E-F : 对区域 [7:,::] 进行二值化，然后取小区域的  [3:7,6:14]    F:：0-1，E：7-24  临界值：3
CHAR LocalRegionGrayValuePredictEF(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0, min_value = 300;

    //    区域 [7:,:], 7*13
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 7; row < 15; row++) {
        for (col = 0; col < 15; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    //    区域 [10:14,6:14],4*8
    for (row = 12; row < 14; row++) {
        for (col = 6; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }
    }

    if (sum_value > 3)
        return 'E';
    else
        return 'F';
}

//    7-Z : 对区域 [7:14,1:14] 进行二值化，然后取这个小区域的左下角（4*4，3:,0:4）和右下角（4*4，3:,9:），选择点数最小的区域来计算，7:0-1   Z：3-12
CHAR LocalRegionGrayValuePredictZ7(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value1 = 0, sum_value = 0;

    //    区域 [7:14,1:14]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 7; row < 14; row++) {
        for (col = 1; col < 14; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    //    区域 [10:14,1:5] 和
    for (row = 10; row < 14; row++) {
        for (col = 1; col < 5; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

    }
//    区域 [10:14,10:14]
    for (row = 10; row < 14; row++) {
        for (col = 10; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            //            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value1 += locate_value;
        }
    }

    if (sum_value1 < sum_value)
        sum_value = sum_value1;

    if (sum_value > 2)
        return 'Z';
    else
        return '7';
}

//    8-3：区域 [1:14,1:6]， 对区域进行二值化， 并判断是否有环
CHAR LocalRegionGrayValuePredict83(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [1:14,1:6]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 1; row < 14; row++) {
        for (col = 1; col < 6; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 1; row < 14; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 1; col < 6; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
            return '3';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
//            return '3';
        }


        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return '3';
    else
        return '8';
}

//    8-6：区域 [1:8,8:14]， 对区域进行二值化， 并判断是否有环，单向 8 修正为 6
CHAR LocalRegionGrayValuePredict86(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [1:8,8:14]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 1; row < 8; row++) {
        for (col = 8; col < 14; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 1; row < 8; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 8; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
            //        非环
            return '6';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
        }


        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return '6';
    else
        return '8';
}

//    0-C：区域 [1:14,9:14], 对区域进行二值化， 并判断是否有环
CHAR LocalRegionGrayValuePredict0C(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [1:14,9:14]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 1; row < 14; row++) {
        for (col = 9; col < 14; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 1; row < 14; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 9; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
            //        非环
            return 'C';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
        }


        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return 'C';
    else
        return '0';
}

//    P-F：区域 [1:9,8:14] , 对区域进行二值化， 并判断是否有环
CHAR LocalRegionGrayValuePredictPF(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [1:9,8:14]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 1; row < 9; row++) {
        for (col = 8; col < 14; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 1; row < 9; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 8; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
            //        非环
            return 'F';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
        }


        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return 'F';
    else
        return 'P';
}

//    6-E：区域 [5:14,9:14], 对区域进行二值化， 并判断是否有环
CHAR LocalRegionGrayValue6redict6E(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [5:14,9:14]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 5; row < 14; row++) {
        for (col = 9; col < 14; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 5; row < 14; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 9; col < 14; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
            //        非环
            return 'E';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
//            return '3';
        }


        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return 'E';
    else
        return '6';
}

//    6-5：区域 [6:14,1:7]， 对区域进行二值化， 并判断是否有环
CHAR LocalRegionGrayValue6redict65(IplImage *pImage) {
    INT32U row, col, locate_value, sum_value = 0;
    INT32U is_white = 0, white_to_black = 0;

    //    区域 [6:14,1:7]
    UINT count_of_gray[256] = {0};
    int best_threshold;

    for (row = 6; row < 14; row++) {
        for (col = 1; col < 7; col++) {
            // get the pixel,
            locate_value = pImage->imageData[row * pImage->widthStep + col];
            count_of_gray[locate_value] += 1;

        }
    }
    //    寻找 最优 二值化 阈值
    best_threshold = get_best_threshold(count_of_gray);

    for (row = 6; row < 14; row++) {
//        计算这一行 二值化后 灰度值大小
        sum_value = 0;
        for (col = 1; col < 7; col++) {
            // get the pixel
            locate_value = pImage->imageData[row * pImage->widthStep + col];
//            二值化,背景是黑色（0），前景是白色（1）
            if (locate_value <= best_threshold)
                locate_value = 1;
            else
                locate_value = 0;

            sum_value += locate_value;
        }

        if (white_to_black == 1 && sum_value > 0) {
//            白转黑 又转白
            return '5';
        }

        if (is_white == 1 && sum_value == 0) {
//            白转黑
            white_to_black = 1;
        }

        if (sum_value > 0)
            is_white = 1;

    }

    if (is_white == 0)
//        非环
        return '5';
    else
        return '6';
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

// *********************** CNN model predict --- letter ***************************************
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

    //endregion



    //region CNN 二分类
#if MODEL_MODE > 1
    if (index_to_char[k] == '0' || index_to_char[k] == 'D')
        if (LetterCNNModelPredictBinary0D(model, pImage) == '0')
            k = 14;
        else
            k = 3;
    if (index_to_char[k] == '0' || index_to_char[k] == 'Q')
        if (LetterCNNModelPredictBinary0Q(model, pImage) == '0')
            k = 14;
        else
            k = 16;
    if (index_to_char[k] == '0' || index_to_char[k] == 'G')
        if (LetterCNNModelPredictBinary0G(model, pImage) == '0')
            k = 14;
        else
            k = 6;
#endif
    //endregion

    // region 使用局部灰度值来二分类
#if MODEL_MODE > 2
//  R-P:  对区域 [8:14,8:14] 求灰度和, 临界值：1953
    if (index_to_char[k] == 'R' || index_to_char[k] == 'P')
        if (LocalRegionGrayValuePredictRP(pImage) == 'R')
            k = 17;
        else
            k = 15;
//  I-T: : 对区域 [1:5,1:6]，[1:5,9:14] 求灰度和  I：0-44   T：2676-6481  临界值：1316
    if (index_to_char[k] == 'T' || index_to_char[k] == 'I')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictTI(pImage) == 'T')
            k = 19;
        else
            k = 8;

//  T-J: : 对区域 [7:14,1:14] 进行二值化，取这个小区域的左下角 [5::,0:2]，如果这个区域有黑点，就修正为 J
//    注意，这里只是单向修改，即 修改 T 的预测结果 ，对于预测为 J 的 不修改
    if (index_to_char[k] == 'T')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictTJ(pImage) == 'T')
            k = 19;//T
        else
            k = 9;//J

//    E-F : 对区域 [7:,::] 进行二值化，然后取小区域的  [3:7,6:14]    F:：0-1，E：7-24  临界值：3
    if (index_to_char[k] == 'E' || index_to_char[k] == 'F')
        //    使用局部灰度值来二分类
        if (LocalRegionGrayValuePredictEF(pImage) == 'E')
            k = 4;
        else
            k = 5;

//    0-C：区域 [1:14,9:14], 对区域进行二值化， 并判断是否有环 --- 这里是单向的，只对预测成 C 修正，预测成0的不修正
    if (index_to_char[k] == 'C')
        if (LocalRegionGrayValuePredict0C(pImage) == '0')
            k = 14;//0
        else
            k = 2;//C
//    P-F：区域 [1:9,8:14] , 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == 'P' || index_to_char[k] == 'F')
        if (LocalRegionGrayValuePredictPF(pImage) == 'P')
            k = 15;//P
        else
            k = 5;//F

#endif
    // endregion
    return index_to_char[k];

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

// ************************  CNN model predict --- digit   *************************************
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

    // region 使用局部灰度值来二分类
#if MODEL_MODE > 2
//    8-3：区域 [1:14,1:6]， 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == '8' || index_to_char[k] == '3')
        if (LocalRegionGrayValuePredict83(pImage) == '8')
            k = 8;//8
        else
            k = 3;//3

//    8-6：区域 [1:8,8:14]， 对区域进行二值化， 并判断是否有环，单向 8 修正为 6
    if (index_to_char[k] == '8')
        if (LocalRegionGrayValuePredict86(pImage) == '8')
            k = 8;//8
        else
            k = 6;//6

//    6-5：区域 [6:14,1:7]， 对区域进行二值化， 并判断是否有环
    if (index_to_char[k] == '6' || index_to_char[k] == '5')
        if (LocalRegionGrayValue6redict65(pImage) == '6')
            k = 6;//6
        else
            k = 5;//5
#endif
    // endregion

    return index_to_char[k];
}