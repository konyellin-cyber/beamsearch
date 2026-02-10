# GPU Beamsearch + 打散规则实现指南

## 快速导航

- [1. 架构概览](#1-架构概览)
- [2. 你的业务场景分析](#2-你的业务场景分析)
- [3. 核心设计决策](#3-核心设计决策)
- [4. 实现路线图](#4-实现路线图)
- [5. 性能优化建议](#5-性能优化建议)

---

## 1. 架构概览

### 当前状态

你的推荐系统流程：
```
[精排层] → [混排层：Beamsearch + 打散规则] → [用户展示]
           ↑
           候选集：2000 items
```

### GPU 加速方案

```
输入：2000 个候选 + 打散规则

┌──────────────────────────────────────────┐
│  位置 0: 并行检查 2000 个候选             │
│  GPU 规则检查 → 有效候选 ~1500 个         │
│  选出最高分 → item_A                     │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  位置 1: 已选 [item_A]，并行检查 1999 个  │
│  GPU 规则检查 → 有效候选 ~1400 个         │
│  选出最高分 → item_B                     │
└──────────────────────────────────────────┘
                    ↓
            ... 继续迭代 ...
                    ↓
┌──────────────────────────────────────────┐
│  位置 99: 返回最终 100 个 items           │
└──────────────────────────────────────────┘
```

**关键特点**：
- ✅ 串行推进位置（因为规则存在依赖性）
- ✅ GPU 并行检查每个位置的所有候选（2000 → 100-200 次）
- ✅ 预期性能：1-2ms vs CPU 的 20-30ms

---

## 2. 你的业务场景分析

### 打散规则类型

基于你的描述，整理出三类规则：

#### 类型 1️⃣：坑位过滤/强插

**特点**：给定位置的强约束

**示例规则**：
```
- 首坑（位置 0）不出双列样式
- 位置 0-2 必须有广告（强插）
- 位置 5 必须是视频号
```

**GPU 实现**：
```python
def check_position_rule(candidate_idx, position, rule):
    if position == rule.position:
        # 检查该候选是否满足约束
        return check_forbidden_property(candidate, rule)
    return True  # 其他位置无约束
```

#### 类型 2️⃣：窗口 M 出 N

**特点**：基于已选序列的维度打散

**维度**：
- `itemshowtype`（样式）
- `category_id`（一级类目）
- `tag`（标签）
- `bizuin`（商家/账号）
- `is_beauty`（美女）
- `is_experience`（体验）
- 等等

**示例规则**：
```
- window_size=5, category_id, max=2
  → 任意 5 个连续 item 中，同一类目不超过 2 个

- window_size=3, bizuin, max=1
  → 任意 3 个连续 item 中，同一商家最多 1 个
```

**GPU 实现逻辑**：
```python
def check_window_rule(current_result, candidate, rule):
    # 获取窗口范围
    window_start = max(0, len(current_result) - (rule.window_size - 1))
    
    # 统计该维度在窗口中的出现次数
    count = 0
    for item in current_result[window_start:]:
        if getattr(item, rule.dimension) == getattr(candidate, rule.dimension):
            count += 1
    
    # 检查是否违反规则
    return count < rule.max_count
```

#### 类型 3️⃣：定坑折损

**特点**：检查热内容的占比

**示例规则**：
```
- window=[0, 20], max_heat_ratio=0.3
  → 前 20 个位置中，加热内容最多占 30%

- window=[20, 50], max_heat_ratio=0.2
  → 位置 20-50 中，加热内容最多占 20%
```

**GPU 实现逻辑**：
```python
def check_heat_rule(current_result, candidate, rule, position):
    if not candidate.is_heat:
        return True  # 非热内容不受限制
    
    # 计算窗口内已选的热内容占比
    heat_count = sum(1 for item in current_result if item.is_heat)
    window_size = position - rule.window_start
    
    heat_ratio = (heat_count + 1) / window_size  # +1 是加入这个候选
    return heat_ratio <= rule.max_heat_ratio
```

---

## 3. 核心设计决策

### 决策 1：CPU vs GPU 规则检查

**你的观察是对的**：
- ✅ 规则检查必须是串行的（因为依赖已选序列）
- ✅ 但每个位置内的候选检查可以并行

**CPU 版本**（简单快速）：
```python
# 每个位置逐个检查候选
for candidate in candidates:
    for rule in rules:
        if not rule.check(current_result, candidate):
            valid = False
            break
```
- 优点：实现简单，调试容易
- 缺点：2000 候选 × 10 规则 = 20K 次检查，慢
- **预期耗时**：20-30ms

**GPU 版本**（高性能）：
```python
# 并行检查所有候选
for rule in rules:
    valid_mask = gpu_check_rule(current_result, candidates, rule)
    overall_valid &= valid_mask
```
- 优点：2000 个候选同时检查
- 缺点：需要 CUDA 编程，调试复杂
- **预期耗时**：1-2ms

### 决策 2：实现技术栈

| 阶段 | 技术 | 理由 |
|------|------|------|
| **原型** | Python + CuPy | 快速验证方案，易于迭代 |
| **性能测试** | Python + Numba/Triton | 找瓶颈，优化算法 |
| **生产** | C++ + CUDA | 最优性能，集成到推荐系统 |

### 决策 3：内存架构

**GPU 内存使用**（超级轻量）：
```
候选属性：2000 × 40B ≈ 80 KB
已选序列：100 × 40B ≈ 4 KB
用户特征：64 × 4B ≈ 256 B
规则参数：~10 KB

总计：≈ 100 KB
```

**结论**：即使是消费级 GPU 也轻松胜任，没有内存压力

---

## 4. 实现路线图

### Phase 1：快速验证（1-2 周）

**目标**：验证方案可行性

**任务**：
1. ✅ 基础 CPU 实现
   - 实现三类规则的检查逻辑
   - 验证规则逻辑正确性
   
2. ✅ 生成测试数据
   - 模拟 2000 候选 items
   - 构造真实规则集合
   
3. ✅ 性能基准
   - CPU 版本：20-30ms
   - 目标：GPU 版本 < 5ms

**代码位置**：`/root/beamsearch_implementation_framework.py`

### Phase 2：GPU 原型（2-3 周）

**目标**：GPU 加速实现

**任务**：
1. CuPy 实现
   - 移植规则检查到 GPU
   - 测试每条规则的 GPU 版本
   
2. 性能优化
   - 内存优化：减少 CPU↔GPU 传输
   - 内核优化：并发执行多个规则
   
3. 精度验证
   - GPU 结果 vs CPU 结果
   - 确保逻辑正确性

**关键代码**：
```python
# GPU 规则检查
@cp.fuse()
def gpu_check_window_rule(result_dims, candidate_dims, window_size, max_count):
    count = 0
    for i in range(len(result_dims)):
        if result_dims[i] == candidate_dims[0]:
            count += 1
    return count < max_count
```

### Phase 3：生产部署（2-4 周）

**目标**：集成到推荐系统

**任务**：
1. C++ CUDA 实现
   - 高性能内核
   - 与 C++ 推荐系统集成
   
2. 服务化
   - gRPC 或 HTTP 接口
   - 负载均衡
   
3. 监控 & 告警
   - 实时性能监控
   - 规则违反率统计
   - A/B 测试

**技术栈**：TensorRT / NVIDIA RAPIDS

---

## 5. 性能优化建议

### 优化 1️⃣：减少 CPU↔GPU 传输

**现状**：
```
GPU 规则检查 → valid_mask (2000 bool) → CPU
```

**优化**：
```
GPU 规则检查 → GPU argmax → best_idx (1 int) → CPU
```

**代码**：
```python
# 优化前：传输 2000 bool
valid_mask_cpu = cp.asnumpy(valid_mask_gpu)

# 优化后：只传输 1 个索引
best_idx = cp.argmax(scores_gpu)
best_idx_cpu = cp.asnumpy(best_idx)
```

### 优化 2️⃣：内存预分配

**现状**：每次迭代都分配内存
```python
for pos in range(target_length):
    valid_mask_gpu = cp.ones(2000, dtype=cp.bool_)  # 重复分配
```

**优化**：提前分配，重复使用
```python
# 预分配所有内存
valid_mask_gpu = cp.ones(2000, dtype=cp.bool_)
scores_gpu = cp.zeros(2000, dtype=cp.float32)

for pos in range(target_length):
    valid_mask_gpu.fill(True)  # 重置而不是重新分配
    # ...
```

### 优化 3️⃣：规则预编译

**现状**：每次迭代都解析规则
```python
for pos in range(target_length):
    for rule in rules:
        check_rule(rule)  # 规则每次都被解析
```

**优化**：预编译规则参数
```python
class CompiledRule:
    def __init__(self, rule):
        self.dimension_idx = self._get_dimension_index(rule.dimension)
        self.max_count = rule.max_count
        self.window_size = rule.window_size

compiled_rules = [CompiledRule(r) for r in rules]

for pos in range(target_length):
    for compiled_rule in compiled_rules:
        # 直接使用编译后的参数，无需解析
        check_compiled_rule(compiled_rule)
```

### 优化 4️⃣：并发执行多条规则

**现状**：规则串行检查
```python
valid_mask = [True] * 2000

for rule in rules:
    rule_valid = check_rule(rule)
    valid_mask &= rule_valid  # 逐个更新
```

**优化**：GPU 内并发执行
```python
# CUDA 内核：同时检查多条规则
__global__ void check_all_rules(
    const Item* candidates,
    const Rule* rules,
    int num_candidates,
    int num_rules,
    bool* valid_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int r = 0; r < num_rules; r++) {
        bool valid = check_single_rule(candidates[idx], rules[r]);
        valid_mask[idx * num_rules + r] = valid;
    }
}
```

### 优化 5️⃣：使用 Tensor Core（如果使用 TensorRT）

**矩阵化表示**：
- 将 2000 个候选的维度属性编码为矩阵
- 使用矩阵操作加速维度匹配
- 预期加速：2-5 倍

---

## 常见问题 FAQ

### Q1: 为什么不能完全并行化所有位置？

**A**: 因为窗口规则的依赖性。
```
位置 0: 可以选任何候选
位置 1: 依赖位置 0 的结果（知道了类目、bizuin 等）
位置 2: 依赖位置 0-1 的结果
...
```

这就是为什么需要**串行推进位置**。

### Q2: GPU 上处理 2000 候选会不会太小？

**A**: 不会。虽然 2000 相对较小，但：
1. 规则检查涉及复杂的条件判断（分支多）
2. 每个位置需要重复多次（100 个位置）
3. 总计：2000 × 100 = 20 万次规则检查
4. GPU 的优势：并行处理这 20 万次检查

**类比**：如果用 GPU 处理单个 2000 的数据，确实不划算。但这里是**100 次迭代 × 2000 候选 × 10 规则 = 200 万次**判断，GPU 值得。

### Q3: 如何处理新增/修改规则？

**A**: 规则应该从配置文件加载，不硬编码。

```python
# rules.yaml
rules:
  - type: window
    rule_id: category_diversity
    window_size: 5
    dimension: category_id
    max_count: 2
  
  - type: position
    rule_id: first_no_double
    position: 0
    forbidden_itemshowtype: 3

# Python 代码
rules = BeamsearchRules.from_yaml('rules.yaml')
result = beamsearch.rank(candidates, rules)
```

### Q4: 如何测试规则的正确性？

**A**: 单元测试 + 集成测试 + 监控

```python
# 单元测试
def test_window_rule():
    result = [Item(category=1), Item(category=2)]
    candidate = Item(category=1)
    rule = WindowRule(window_size=3, dimension='category', max_count=1)
    assert not rule.check(result, candidate)  # 应该被过滤

# 集成测试
def test_full_ranking():
    result = beamsearch.rank(candidates, rules, target_length=100)
    assert all_rules_satisfied(result, rules)

# 监控
def monitor_rule_violations():
    violations = count_violations(result, rules)
    log_to_prometheus(violations)
```

### Q5: 生产部署时的关键指标是什么？

**A**：
1. **性能**：P99 延时 < 5ms
2. **准确性**：规则违反率 = 0
3. **稳定性**：GPU 显存使用 < 1GB，无 OOM
4. **兼容性**：与现有评分模型的兼容性

---

## 总结与下一步

### 关键设计点

1. ✅ **串行位置 + GPU 并行规则**：符合你的业务逻辑
2. ✅ **三类规则完整支持**：坑位、窗口、折损
3. ✅ **轻量化 GPU 使用**：内存 < 100KB，适合嵌入式
4. ✅ **性能目标**：1-2ms vs CPU 的 20-30ms

### 建议的下一步行动

**立即**：
- [ ] 整理你的完整打散规则列表
- [ ] 准备 2000 候选的真实或模拟数据
- [ ] 评估当前 CPU 版本的延时基准

**第 1 周**：
- [ ] 实现 CPU 版本（参考提供的代码框架）
- [ ] 验证规则逻辑的正确性
- [ ] 建立性能基准

**第 2-3 周**：
- [ ] CuPy GPU 实现
- [ ] 性能优化和对标
- [ ] 精度验证

**第 4-6 周**：
- [ ] C++ CUDA 优化版本
- [ ] 集成到推荐系统
- [ ] 上线 A/B 测试

---

**准备好开始了吗？请补充你的具体规则列表！**
