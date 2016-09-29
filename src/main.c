/************************************************************************
 * Created by jdwang on 2016-09-15.
 * Last updated on 2016-09-29
 * DESCRIPTION : 主函数, 测试字符识别器，为了测试所有，可以直接调用 recognizer.c
************************************************************************/

#include <typedef.h>
#include "imageIO.h"
#include "recognizer.h"
#include "stdio.h"

int main() {
    char * file_name;
    INT32U image_index;
    CHAR y_pred;
    #if DEBUG_LEVEL>0
        printf("开始测试！");
    #endif
//加载图片数据
    file_name = "/home/jdwang/ClionProjects/dR_c/data/input_data2.mat";
    ImageArray image_array = ReadImageFromFile(file_name);

    for(image_index=0;image_index<image_array.number_of_image;image_index++){
        y_pred = RecognizeSCAU(&image_array.imageList[image_index],0,0);

        #if DEBUG_LEVEL>0
            printf("The %d's image: %c \n",image_index,y_pred);
        #endif
    }
    #if DEBUG_LEVEL>0
        printf("测试完成！");
    #endif
    return 0;
}


