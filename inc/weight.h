/**************************************************************
* Created by jdwang on 2016-09-16.
* DESCRIPTION : 
****************************************************************/

#ifndef DR_C_WEIGHT_H
#define DR_C_WEIGHT_H
//34分类权重
#include "conv_weight33.h"
#include "conv_weight55.h"
#include "conv_weight77.h"
#include "conv_b33.h"
#include "conv_b55.h"
#include "conv_b77.h"
#include "fc1_weight.h"
#include "fc1_b.h"
#include "fc2_weight.h"
#include "fc2_b.h"

//二分类5-6权重
#include "binary56_conv_weight33.h"
#include "binary56_conv_b33.h"
#include "binary56_fc1_weight.h"
#include "binary56_fc1_b.h"
#include "binary56_fc2_weight.h"
#include "binary56_fc2_b.h"

//二分类8-B权重
#include "binary8B_conv_weight33.h"
#include "binary8B_conv_b33.h"
#include "binary8B_fc1_weight.h"
#include "binary8B_fc1_b.h"
#include "binary8B_fc2_weight.h"
#include "binary8B_fc2_b.h"

//3分类0-D-Q权重
#include "binary0DQ_conv_weight33.h"
#include "binary0DQ_conv_b33.h"
#include "binary0DQ_fc1_weight.h"
#include "binary0DQ_fc1_b.h"
#include "binary0DQ_fc2_weight.h"
#include "binary0DQ_fc2_b.h"



//10数字分类权重
#include "digit10_int_weight0.h"
#include "digit10_int_weight1.h"
#include "digit10_int_weight2.h"
#include "digit10_int_weight3.h"
#include "digit10_int_weight4.h"
#include "digit10_int_weight5.h"

//25字母分类权重
#include "letter25_int_weight0.h"
#include "letter25_int_weight1.h"
#include "letter25_int_weight2.h"
#include "letter25_int_weight3.h"
#include "letter25_int_weight4.h"
#include "letter25_int_weight5.h"
//25字母分类权重 -- 0D
#include "letter0D_int_weight0.h"
#include "letter0D_int_weight1.h"
#include "letter0D_int_weight2.h"
#include "letter0D_int_weight3.h"
#include "letter0D_int_weight4.h"
#include "letter0D_int_weight5.h"

//25字母分类权重 -- 0Q
#include "letter0Q_int_weight0.h"
#include "letter0Q_int_weight1.h"
#include "letter0Q_int_weight2.h"
#include "letter0Q_int_weight3.h"
#include "letter0Q_int_weight4.h"
#include "letter0Q_int_weight5.h"

//25字母分类权重 -- 0G
#include "letter0G_int_weight0.h"
#include "letter0G_int_weight1.h"
#include "letter0G_int_weight2.h"
#include "letter0G_int_weight3.h"
#include "letter0G_int_weight4.h"
#include "letter0G_int_weight5.h"

#endif //DR_C_WEIGHT_H
