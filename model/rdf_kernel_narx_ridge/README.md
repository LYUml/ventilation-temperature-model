# RDF-Kernel NARX-Ridge

Kernel慢趋势加RDF空间输入的NARX-Ridge残差递推模型。

- RDF共享构件生成空间邻接和楼层拓扑；
- RDF界面UA生成实测邻室边界权重；
- Kernel只作为方程内部的慢状态，不是独立输出ensemble；
- 图约束VARX递推慢状态偏差；
- 不使用固定热容量、假定窗朝向或手写房间映射。

锁定测试平均RMSE约0.319 °C，相对Kernel提升约7.3%。

```powershell
python -m model.rdf_kernel_narx_ridge.run
```
