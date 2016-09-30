/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-29
* DESCRIPTION : 
****************************************************************/

#include "cnn.h"


//CNN model init
void CNNModelInit(CharCNNClassifier *model){
//    1's layer - input layer : 25*25
    model->model_all.inputHeight = 15;
    model->model_all.inputWidthStep = 15;
//    2's layer - convolution Layer 1 : 25*1*3*3
    model->model_all.C11.outChannels = 25;
    model->model_all.C11.inChannels = 25;
    model->model_all.C11.height = 3;
    model->model_all.C11.widthStep = 3;

//    2's layer - convolution Layer 2 : 25*1*5*5
    model->model_all.C21.outChannels = 25;
    model->model_all.C21.inChannels = 25;
    model->model_all.C21.height = 5;
    model->model_all.C21.widthStep = 5;

//    2's layer - convolution layer 3 : 25*1*7*7
    model->model_all.C31.outChannels = 25;
    model->model_all.C31.inChannels = 25;
    model->model_all.C31.height = 7;
    model->model_all.C31.widthStep = 7;

//    3's layer - subsample layer 1 : 25*2*2
    model->model_all.S12.channels = 25;
    model->model_all.S12.height = 2;
    model->model_all.S12.widthStep = 2;
    model->model_all.S12.poolType = MaxPool;

//    3's layer - subsample layer 2 : 25*2*2
    model->model_all.S22.channels = 25;
    model->model_all.S22.height = 2;
    model->model_all.S22.widthStep = 2;
    model->model_all.S22.poolType = MaxPool;

//    3's layer - subsample layer 3 : 25*2*2
    model->model_all.S32.channels = 2;
    model->model_all.S32.height = 2;
    model->model_all.S32.widthStep = 2;
    model->model_all.S32.poolType = MaxPool;

//    4's layer - full connected layer : 1925*400
    model->model_all.FC1.inputNum = 1925;
    model->model_all.FC1.outputNum = 400;

//    5's layer - full connected layer : 400*34
    model->model_all.FC2.inputNum = 400;
    model->model_all.FC2.outputNum = 34;
}

float TanhApproximateFunction(float x){
    return (exp2f(x)-exp2f(-x))/(exp2f(x)+exp2f(-x));
}

CHAR CNNModelPredict(CharCNNClassifier *model, IplImage *pImage) {
    int row, col, i, j, k;
    INT8U output_row, output_col;
    int map_index_position, image_index_position;
    float sum, max, value;
    float c11_output[25 * 13 * 13], c21_output[25 * 11 * 11], c31_output[25 * 9 * 9],fc1_output[model->model_all.FC1.outputNum];
    char index_to_char[]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L',
                          'M','N','P','Q','R','S','T','U','W','X','Y','Z'};
    /*,fc2_output[model->model_all.FC2.outputNum]*/
//    5-->3-->7
//    length - 1925
    float merge_output[25 * 5 * 5 + 25 * 6 * 6 + 25 * 4 * 4];

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
                sum = 0.0;
                for (row = 0; row < model->model_all.C11.height; row++) {
                    for (col = 0; col < model->model_all.C11.widthStep; col++) {
                        //      index position
                        map_index_position = k * 3 * 3 + row * model->model_all.C11.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%f,%d\n", C11_Map_Weight[map_index_position], pImage->imageData[image_index_position]);
                        sum += C11_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
                c11_output[k * output_row * output_row + i * output_row + j] = tanhf(sum + C11_B_Weight[k]);
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
                merge_output[25 * 5 * 5 + k * 6 * 6 + i * 6 + j] = max;
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
    output_row = 11;
    output_col = 11;
    for (k = 0; k < model->model_all.C21.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0.0;
                for (row = 0; row < model->model_all.C21.height; row++) {
                    for (col = 0; col < model->model_all.C21.widthStep; col++) {
                        //      index position
                        map_index_position = k * 5 * 5 + row * model->model_all.C21.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%f,%d\n", C11_Map_Weight[map_index_position], pImage->imageData[image_index_position]);
                        sum += C21_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
//                11*11
                c21_output[k * output_row * output_row + i * output_row + j] = tanhf(sum + C21_B_Weight[k]);
            }
        }
//        printmat2(c21_output, output_row, output_row);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
    }
    output_row = 5;
    output_col = 5;
    for (k = 0; k < model->model_all.C21.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c21_output[k * 11 * 11 + (2 * i + row) * 11 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[k * output_row * output_row + i * output_row + j] = max;
            }
        }
//        printmat2(merge_output, 6, 6);
//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion

    //region 7*7 convolution
    //    output size
//    7*7
//    15-7+1=9
    output_row = 9;
    output_col = 9;
    for (k = 0; k < model->model_all.C31.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                sum = 0.0;
                for (row = 0; row < model->model_all.C31.height; row++) {
                    for (col = 0; col < model->model_all.C31.widthStep; col++) {
                        //      index position
                        map_index_position = k * 7 * 7 + row * model->model_all.C31.widthStep + col;
                        image_index_position = (i + row) * pImage->widthStep + (j + col);
//                        printf("%d,%d\n", map_index_position, image_index_position);
//                        printf("%f,%d\n", C11_Map_Weight[map_index_position], pImage->imageData[image_index_position]);
                        sum += C31_Map_Weight[map_index_position] * pImage->imageData[image_index_position];
                    }
                }
//                9*9
                c31_output[k * output_row * output_row + i * output_row + j] = tanhf(sum + C31_B_Weight[k]);
            }
        }
//        printmat2(c31_output, output_row, output_row);
    }
    output_row = 4;
    output_col = 4;
    for (k = 0; k < model->model_all.C31.outChannels; k++) {
//        for each filters
        for (i = 0; i < output_row; i++) {
            //        i represents start point in row
            for (j = 0; j < output_col; j++) {
                //        j represents start point in col
                max = -10;
                for (row = 0; row < 2; row++) {
                    for (col = 0; col < 2; col++) {
                        value = c31_output[k * 9 * 9 + (2 * i + row) * 9 + (2 * j + col)];
                        if (value > max) {
                            max = value;
                        }

                    }
                }
                merge_output[25 * 6 * 6 + 25 * 5 * 5 + k * output_row * output_row + i * output_row + j] = max;
            }
        }
//        printmat2(merge_output, 4, 4);

//                printf("%f",sum);
//                printf("%f",c11_output[k * 13 * 13 + i * 13 + j]);
//        assert(NULL);
    }
    //endregion

    //region FC1 - 1925*400
    for (row = 0; row < model->model_all.FC1.outputNum; row++) {
        sum = 0.0;
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
    //region FC2 - 400*34
    max = -10000;
    k=0;
    for (row = 0; row < model->model_all.FC2.outputNum; row++) {
        sum = 0.0;
        for (col = 0; col < model->model_all.FC2.inputNum; col++) {
            sum += fc1_output[col] * FC2_Map_Weight[row * model->model_all.FC2.inputNum + col];
        }
        value = sum + FC2_B_Weight[row];
//        printf("%f,",value);
        if(value>max){
            max =value;
            k=row;
        }
    }
    //endregion

//    printf("%f,%d\n",max,k);
    return index_to_char[k];
}