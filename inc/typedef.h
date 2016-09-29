/************************************************************************
* Header File PreProcessor Directive
* Last updated on 2016-09-29
************************************************************************/
#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <stdio.h>				//系统库头文件
#include <string.h>
#include <stdlib.h>


/*********************************************************************************************************
*                                              DATA TYPES
*                                         (Compiler Specific)
**********************************************************************************************************/

typedef unsigned char 		BOOL;
typedef unsigned char		INT8U;	/* Unsigned  8 bit quantity */
typedef signed   char		INT8S;	/* Signed    8 bit quantity */
typedef unsigned short int	INT16U;	/* Unsigned 16 bit quantity */
typedef signed   short int	INT16S;	/* Signed   16 bit quantity */
typedef unsigned int		INT32U;	/* Unsigned 32 bit quantity */
typedef signed   int		INT32S;	/* Signed   32 bit quantity */

typedef volatile unsigned char		VINT8U;		/* volatile Unsigned  8 bit quantity */
typedef volatile signed   char		VINT8S;		/* volatile Signed    8 bit quantity */
typedef volatile unsigned short int	VINT16U;	/* volatile Unsigned 16 bit quantity */
typedef volatile signed   short int	VINT16S;	/* volatile Signed   16 bit quantity */
typedef volatile unsigned int		VINT32U;	/* volatile Unsigned 32 bit quantity */
typedef volatile signed   int		VINT32S;	/* volatile Signed   32 bit quantity */



typedef unsigned long		INT64U;
typedef signed 	 long		INT64S;

//typedef 		 double 	INT64U;	/* Unsigned 32 bit quantity */
//typedef    	 double		INT64S;	/* Signed   32 bit quantity */

//*************************************************************************//
//文件系统输出格式
typedef signed int		INT;
typedef unsigned int	UINT;

/* These types must be 8-bit integer */
typedef char			CHAR;
typedef unsigned char	UCHAR;
typedef unsigned char	BYTE;
//typedef unsigned int	BYTE;

/* These types must be 16-bit integer */
typedef short			SHORT;
typedef unsigned short	USHORT;
typedef unsigned short	WORD;
typedef unsigned short	WCHAR;

/* These types must be 32-bit integer */
typedef long			LONG;
typedef unsigned long	ULONG;
//typedef unsigned long	DWORD;
#define DWORD 			INT32U

//*************************************************************************//

/**********************************************************************************************************/

#define SKIP_PRINTF //有此宏定义则不输出调试信息（printf无效）
//*************************************************************************//
#define RW32F(x)	(*(volatile float*)(x))			//读写SDRAM空间的宏命令
#define RW32U(x)	(*(volatile unsigned int*)(x))	//读写SDRAM空间的宏命令
#define RW16U(x)	(*(volatile unsigned short*)(x))//读写SDRAM空间的宏命令
#define RW8U(x)		(*(volatile unsigned char*)(x))	//读写SDRAM空间字节的宏命令
#define listof(dat)	(sizeof(dat)/sizeof(dat[0]))	//数组长度(常量)

/*********************************************************************************************************
*                                              MAT TYPES
**********************************************************************************************************/

//2D-图像结构-int
typedef struct {
	INT32U  height;					//图像的高度,0-dim
    INT32U  widthStep;				//图像的宽度,1-dim
	INT8U *	imageData;				//图像首地址
}IplImage;


////3D-图像结构
//typedef struct {
//    INT32U  channel;				//图像的通道数,0-dim
//    INT32U  height;					//图像的高度,1-dim
//    INT32U  widthStep;				//图像的宽度,2-dim
//    INT8U *	imageData;				//图像首地址
//}IplImage3D;
//
////4D-图像结构-float
//typedef struct {
//    INT32U  number; 				//图像的通道数,0-dim
//    INT32U  channel;				//图像的通道数,1-dim
//    INT32U  height;					//图像的高度,2-dim
//    INT32U  widthStep;				//图像的宽度,3-dim
//    INT8U *	imageData;				//图像首地址
//}IplImage4D;

//2D-图像数组结构
typedef struct {
    INT32U number_of_image;          //图像的数量
    IplImage * imageList;            //图像数组首地址
	INT8U * labelList;               //图像的标签首地址
}ImageArray;


//------------------------------------------------------------------------

#endif
