/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-29
* DESCRIPTION : 
****************************************************************/

#include <typedef.h>
#include "imageIO.h"
#include "config.h"

INT8U * ReadLabelFromFile(const char *file_name){
    FILE * fp;
    INT32U number_of_image;
    INT32U image_index,pixel_index;
    INT8U * labelList;

    #if DEBUG_LEVEL>0
        printf("----------------\n");
        printf("打开文件（label）：%s\n",file_name);
    #endif
    fp = fopen(file_name,"rb");
    if(fp==NULL){
        printf("open image failed!(%s)\n",file_name);
    }
    //    图片的数量
    fread(&number_of_image,sizeof(INT32U),1,fp);
    #if DEBUG_LEVEL>0
        printf("----------------\n");
        printf("number of labels : %d\n",number_of_image);
    #endif

    labelList = malloc(sizeof(INT8U)*number_of_image);
//    for(image_index=0;image_index<number_of_image;image_index++){
    fread(labelList,sizeof(INT8U),number_of_image,fp);
//    assert(NULL);
    return labelList;
}

ImageArray ReadImageFromFile(const char *file_name){

    FILE * fp;
    INT32U number_of_image;
    ImageArray image_array;
    INT32U image_index,pixel_index;
    INT32U number_of_image_pixel;

    #if DEBUG_LEVEL>0
        printf("----------------\n");
        printf("打开文件（image）：%s\n",file_name);
    #endif

    fp = fopen(file_name,"rb");
    if(fp==NULL){
        printf("open image failed!(%s)\n",file_name);
    }

    //    图片的数量
    fread(&number_of_image,sizeof(INT32U),1,fp);

    image_array.number_of_image=number_of_image;
    // 分配数组空间
    image_array.imageList = malloc(sizeof(IplImage)*number_of_image);

    // 设置图像数据
    for(image_index=0;image_index<number_of_image;image_index++){
        image_array.imageList[image_index].height=15;
        image_array.imageList[image_index].widthStep=15;
        // 一张2D图像的所有像素个数
        number_of_image_pixel = image_array.imageList[image_index].height * image_array.imageList[image_index].widthStep;
        image_array.imageList[image_index].imageData = malloc(sizeof(INT8U)*number_of_image_pixel);

        for(pixel_index=0;pixel_index<number_of_image_pixel;pixel_index++){
//            INT8U* in=(INT8U*)malloc(sizeof(INT8U));
//            fread(in,sizeof(INT8U),1,fp);
//            INT32S a;
//            fread(&a,sizeof(INT32S),1,fp);
//            printf("%c,",*in);
            fread(&(image_array.imageList[image_index].imageData[pixel_index]),sizeof(INT8U),1,fp);
        }
    }

    fclose(fp);

    #if DEBUG_LEVEL>0
        printf("the number of images : %d\n",number_of_image);
        printf("Reading image finished！\n");
        printf("----------------\n");
    #endif

    return image_array;

}
