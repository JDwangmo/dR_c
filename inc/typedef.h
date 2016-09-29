/************************************************************************
* Header File PreProcessor Directive
* Last updated on 2016-09-29
************************************************************************/
#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <stdio.h>				//ϵͳ��ͷ�ļ�
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
//�ļ�ϵͳ�����ʽ
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

#define SKIP_PRINTF //�д˺궨�������������Ϣ��printf��Ч��
//*************************************************************************//
#define RW32F(x)	(*(volatile float*)(x))			//��дSDRAM�ռ�ĺ�����
#define RW32U(x)	(*(volatile unsigned int*)(x))	//��дSDRAM�ռ�ĺ�����
#define RW16U(x)	(*(volatile unsigned short*)(x))//��дSDRAM�ռ�ĺ�����
#define RW8U(x)		(*(volatile unsigned char*)(x))	//��дSDRAM�ռ��ֽڵĺ�����
#define listof(dat)	(sizeof(dat)/sizeof(dat[0]))	//���鳤��(����)

/*********************************************************************************************************
*                                              MAT TYPES
**********************************************************************************************************/

//2D-ͼ��ṹ-int
typedef struct {
	INT32U  height;					//ͼ��ĸ߶�,0-dim
    INT32U  widthStep;				//ͼ��Ŀ��,1-dim
	INT8U *	imageData;				//ͼ���׵�ַ
}IplImage;


////3D-ͼ��ṹ
//typedef struct {
//    INT32U  channel;				//ͼ���ͨ����,0-dim
//    INT32U  height;					//ͼ��ĸ߶�,1-dim
//    INT32U  widthStep;				//ͼ��Ŀ��,2-dim
//    INT8U *	imageData;				//ͼ���׵�ַ
//}IplImage3D;
//
////4D-ͼ��ṹ-float
//typedef struct {
//    INT32U  number; 				//ͼ���ͨ����,0-dim
//    INT32U  channel;				//ͼ���ͨ����,1-dim
//    INT32U  height;					//ͼ��ĸ߶�,2-dim
//    INT32U  widthStep;				//ͼ��Ŀ��,3-dim
//    INT8U *	imageData;				//ͼ���׵�ַ
//}IplImage4D;

//2D-ͼ������ṹ
typedef struct {
    INT32U number_of_image;          //ͼ�������
    IplImage * imageList;            //ͼ�������׵�ַ
	INT8U * labelList;               //ͼ��ı�ǩ�׵�ַ
}ImageArray;


//------------------------------------------------------------------------

#endif
