/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-15.
* DESCRIPTION : 关于数组的操作
****************************************************************/

#include <config.h>
#include "mat.h"

//打印矩阵
void printmat(IplImage *pImage) {
    INT32U row, col, locate;
    printf("--------------------------------------\n");
    for (row = 0; row < pImage->height; row++) {
        for (col = 0; col < pImage->widthStep; col++) {
            // compute the location of the pixel
            locate = row * pImage->widthStep + col;
            printf("%-8.2f  ", (float)pImage->imageData[locate]);
        }
        printf("\n");
    }
    printf("--------------------------------------\n");
}

//打印矩阵
void printmat2(float *image,INT8U height,INT8U widthStep) {
    INT32U row, col, locate;
    printf("--------------------------------------\n");
    for (row = 0; row < height; row++) {
        for (col = 0; col < widthStep; col++) {
            // compute the location of the pixel
            locate = row * widthStep + col;
            printf("%-8.2f  ", image[locate]);
        }
        printf("\n");
    }
    printf("--------------------------------------\n");
}




