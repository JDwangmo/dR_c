








******* 20161127 *******
- 训练/测试：20160712的各一半；应用：20161026
- 3个进一步分类：5-6，0-D-Q，8-B
- 训练数据（数量:85041），测试数据（数量:64381），应用数据（数量:243391）
- 34分类：测试（正确率:0.9997204144、错分个数:18），应用（正确率:0.9996343332、错分个数:89）
- 34分类+3个进一步分类：测试（正确率:0.9998446747、错分个数:10），应用（正确率:0.9997370486、错分个数:64）





******* 20160915 *******
已实现重构34分类 整型CNN模型的C代码
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
