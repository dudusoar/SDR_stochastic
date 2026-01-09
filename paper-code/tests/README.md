# Paper-Code Test Suite

这个目录包含了 `paper-code` 的完整测试套件，用于验证重构后的代码是否能正常运行。

## 📁 文件结构

```
tests/
├── README.md                    # 本文件 - 测试说明
├── TEST_SUMMARY.md              # 测试结果总结
├── run_all_tests.py             # 主测试脚本 - 运行所有测试
├── test_01_data_layer.py        # 测试数据层 (RealMap + DemandGenerator)
├── test_02_order_generation.py  # 测试订单生成 (OrderGenerator)
├── test_03_instance_solution.py # 测试实例和解 (PDPTWInstance + PDPTWSolution)
└── test_04_solver.py            # 测试求解器 (初始解 + ALNS短跑)
```

## 🚀 快速开始

### 运行所有测试

```bash
cd paper-code/tests
python run_all_tests.py
```

### 运行单个测试

```bash
cd paper-code/tests
python test_01_data_layer.py
python test_02_order_generation.py
python test_03_instance_solution.py
python test_04_solver.py
```

## 📊 测试内容

### Test 1: Data Layer (数据层)
- **文件**: `test_01_data_layer.py`
- **测试内容**:
  - RealMap 地图生成
  - DemandGenerator 需求生成
  - 距离矩阵和坐标验证
- **预期时间**: ~1-2秒

### Test 2: Order Generation (订单生成)
- **文件**: `test_02_order_generation.py`
- **测试内容**:
  - OrderGenerator 订单生成
  - 订单表结构验证
  - 时间矩阵生成
- **预期时间**: ~1-2秒

### Test 3: Instance & Solution (实例和解)
- **文件**: `test_03_instance_solution.py`
- **测试内容**:
  - PDPTWInstance 实例创建
  - PDPTWSolution 解的表示
  - 目标函数计算
  - 可行性检查
- **预期时间**: ~1-2秒

### Test 4: Solver (求解器)
- **文件**: `test_04_solver.py`
- **测试内容**:
  - 贪心插入初始解生成
  - **ALNS 算法短跑测试** (2个段, 每段5次迭代)
  - 目标值改进验证
- **预期时间**: ~1-2秒
- **注意**: 这是短期测试，只运行10次迭代。完整优化需要更多迭代。

## ✅ 预期结果

所有测试通过后，你应该看到：

```
================================================================================
Total: 4/4 tests passed
Total time: ~5-10s
================================================================================

*** ALL TESTS PASSED! Paper-code is working correctly. ***
```

## 🔧 测试设计说明

### 为什么使用短跑测试？
- **目的**: 验证代码能正常运行，而不是完整优化
- **参数**:
  - 只运行 2 个 segments
  - 每个 segment 只有 5 次迭代
  - 总共 10 次迭代
- **好处**: 快速验证（~2秒）而不是完整运行（可能需要几分钟）

### 测试数据规模
- **小规模实例**: 2个餐厅, 3-4个客户
- **少量订单**: 5-7个订单
- **快速验证**: 每个测试 1-2秒完成

### 编码兼容性
- 所有输出使用 ASCII 字符（`[OK]`, `[ERROR]`）
- 兼容 Windows GBK 编码环境
- 避免 Unicode 特殊字符导致的显示问题

## 📝 如何添加新测试

1. **创建新测试文件**: `test_XX_feature_name.py`
2. **添加导入设置**:
   ```python
   import sys
   import os
   # Add parent directory to path for imports
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
3. **导入需要的模块**:
   ```python
   from real_map import RealMap
   from demands import DemandGenerator
   # ... 等等
   ```
4. **添加到 run_all_tests.py**:
   ```python
   tests = [
       # ... 现有测试 ...
       ("test_XX_feature_name.py", "TEST X: Feature Name"),
   ]
   ```

## 🐛 常见问题

### ImportError: No module named 'real_map'
**原因**: 没有正确设置 sys.path
**解决**: 确保测试文件包含路径设置代码（见上方）

### UnicodeEncodeError
**原因**: 使用了 Unicode 特殊字符
**解决**: 使用 ASCII 字符（`[OK]`, `[ERROR]`, `[PASS]`, `[FAIL]`）

### 测试超时
**原因**: ALNS 迭代次数过多
**解决**: 减少 `num_segments` 和 `segment_length` 参数

## 📚 相关文档

- **测试总结**: `TEST_SUMMARY.md` - 详细的测试结果和发现
- **项目文档**: `../docs/` - Paper-code 的完整文档
- **主代码**: `../` - 被测试的源代码

## 🎯 下一步

测试通过后，可以：
1. 运行完整的 ALNS 优化（增加 segments 数量）
2. 在真实数据集上验证
3. 与论文结果对比
4. 继续迁移到 vrp-toolkit 架构

---

**最后更新**: 2026-01-09
**测试状态**: ✅ 4/4 测试通过
**总耗时**: ~5-10秒
