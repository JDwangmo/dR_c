/**************************************************************
* Created by jdwang on 2016-09-15.
* Last updated on 2016-09-29
* DESCRIPTION : Image IO operation
****************************************************************/

#ifndef DR_C_IMAGEIO_H
#define DR_C_IMAGEIO_H

#include "typedef.h"

ImageArray ReadImageFromFile(const char *file_name);

INT8U * ReadLabelFromFile(const char *file_name);


#endif //DR_C_IMAGEIO_H
