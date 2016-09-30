已实现重构34分类 CNN模型的C代码
接口在 src/recognizer.c ---- CHAR RecognizeSCAU(IplImage *pImage, int version, int location);

------- data
测试完成！正确个数：41072,准确率：0.979491

去除 FC1 tanh
测试完成！正确个数：40940,准确率：0.976343

------- data1
完整
正确个数：3398,准确率：0.999412

--- FC1 tanh 替换成 近似解
正确个数：3399,准确率：0.999706
--- 去掉FC1 tanh
正确个数：3396,准确率：0.998824
10个字符：86ms
100个字符：800ms
1000个字符：7528ms
