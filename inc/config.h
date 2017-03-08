/**************************************************************
 * Created by jdwang on 2016-09-15.
 * Last updated on 2017-01-03
 * DESCRIPTION : program的全局配置文件
****************************************************************/

#ifndef DR_C_CONFIG_H
#define DR_C_CONFIG_H

// data type define
#include <stdio.h>
#include "assert.h"
#include "typedef.h"

// ******** 参数设置 ***********

// 设置debug的level，数值越大，详情越多，
// 设为0时，关闭debug功能
#define DEBUG_LEVEL 1

//1 - CNN总体，2 - CNN总体+CNN二分类，3 - CNN总体+CNN二分类 + 局部灰度值二分类
#define MODEL_MODE 1



#endif //DR_C_CONFIG_H
