/************************************************************************
 * Created by jdwang on 2016-09-15.
 * Last updated on 2016-09-29
 * DESCRIPTION : 主函数, 测试字符识别器，为了测试所有，可以直接调用 recognizer.c
************************************************************************/

#include <typedef.h>
#include <config.h>
#include "imageIO.h"
#include "recognizer.h"
#include "stdio.h"
char Index_To_Char[]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L',
                      'M','N','P','Q','R','S','T','U','W','X','Y','Z'};
int main() {
    char * images_file_name,*labels_file_name;
    INT8U * label_array;
    ImageArray image_array;
    INT32U image_index,correct_count=0;
    CHAR y_pred;

    #if DEBUG_LEVEL>0
        printf("开始测试！");
    #endif
//加载图片数据
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_data1.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_data1.mat";
    image_array = ReadImageFromFile(images_file_name);
    label_array = ReadLabelFromFile(labels_file_name);

    for(image_index=0;image_index<image_array.number_of_image;image_index++){
        y_pred = RecognizeSCAU(&image_array.imageList[image_index],0,0);

        #if DEBUG_LEVEL>0
            printf("The %d's image: %c(Predict),%c(Real),True:%d \n",image_index,
                   y_pred,
                   Index_To_Char[label_array[image_index]],
                   y_pred==Index_To_Char[label_array[image_index]]
            );
        #endif
        correct_count += (y_pred==Index_To_Char[label_array[image_index]]);
    }
    #if DEBUG_LEVEL>0
        printf("测试完成！");
        printf("正确个数：%d,准确率：%f",correct_count,(correct_count*1.0)/image_array.number_of_image);
    #endif
    return 0;
}


