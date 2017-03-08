/************************************************************************
 * Created by jdwang on 2016-09-15.
 * Last updated on 2017-03-08
 * DESCRIPTION : 主函数, 测试字符识别器，为了测试所有，可以直接调用 recognizer.c
************************************************************************/

#include <typedef.h>
#include <config.h>
#include "imageIO.h"
#include "recognizer.h"
#include "sys/time.h"
// ******** 参数设置 ***********
//0-merge,1-digit,2-letter
#define TEST_TYPE 2

//0-test, 1-other, 2-other-new, 3-test20170103, 4-test20170218data
#define DATA_TYPE 4


char Index_To_Char[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                        'J', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z'};

int main() {
    char *images_file_name, *labels_file_name;
    INT8U *label_array;
    ImageArray image_array;
    INT32U image_index, correct_count = 0;
    CHAR y_pred;
    struct timeval start, end;
    int count = 0;
    int flag = 1;

#if DEBUG_LEVEL > 0
    printf("开始测试！");
#endif
//加载图片数据
#if DATA_TYPE ==0
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_test_data.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_test_data.mat";
#elif DATA_TYPE == 1
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_other_data.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_other_data.mat";
#elif DATA_TYPE == 2
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_other_new_data.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_other_new_data.mat";
#elif DATA_TYPE == 3
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_test20170103_data.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_test20170103_data.mat";
#elif DATA_TYPE == 4
    images_file_name = "/home/jdwang/ClionProjects/dR_c/data/images_test20170218data.mat";
    labels_file_name = "/home/jdwang/ClionProjects/dR_c/data/labels_test20170218data.mat";
#endif
    image_array = ReadImageFromFile(images_file_name);
    label_array = ReadLabelFromFile(labels_file_name);
    gettimeofday(&start, 0);

    for (image_index = 0; image_index < image_array.number_of_image; image_index++) {
//        image_index = 91786;
//        printmat(&image_array.imageList[image_index]);
#if TEST_TYPE == 2
        if (label_array[image_index] < 10 && label_array[image_index] > 0)continue;
        flag=0;
#elif TEST_TYPE == 1
        if (label_array[image_index] > 9)continue;
        flag = 5;
#endif
        count++;
        y_pred = RecognizeSCAU(&image_array.imageList[image_index], 0, flag);

#if DEBUG_LEVEL > 0
//        if(y_pred != Index_To_Char[label_array[image_index]])
        printf("The %d's image: %c(Predict),%c(Real),True:%d \n", image_index,
               y_pred,
               Index_To_Char[label_array[image_index]],
               y_pred == Index_To_Char[label_array[image_index]]
        );
#endif
        if ((y_pred == '1' || y_pred == 'I') &&
            (Index_To_Char[label_array[image_index]] == '1' || Index_To_Char[label_array[image_index]] == 'I')) {
//            printf("%c,%c",y_pred,
//                   Index_To_Char[label_array[image_index]]);
            y_pred = Index_To_Char[label_array[image_index]];
        }
        correct_count += (y_pred == Index_To_Char[label_array[image_index]]);
//        assert(NULL);
    }
#if DEBUG_LEVEL > 0
    printf("测试完成！");
    gettimeofday(&end, 0);
    printf("用时：%dms\n", (int) ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000));
    printf("正确个数：%d(共：%d),准确率：%f", correct_count, count, (correct_count * 1.0) / count);
#endif
    return 0;
}


