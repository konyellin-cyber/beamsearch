# TensorFlow GPU Beamsearch - 最小可行性分析

## 1. 问题

### 场景
- 推荐系统混排层 beamsearch
- 候选集：2000 items
- 输出：100-200 items
- 规则：位置相关的打散规则

### 目标
用 TensorFlow GPU 加速，从 **20-30ms 降到 < 5ms**（5-10 倍提升）

### 三类打散规则
1. **坑位过滤**：给定位置 + 上文 → 过滤候选
2. **窗口 M 出 N**：窗口内某维度最多出现 N 次
3. **定坑折损**：窗口内特定类型不超过 X%

---

## 2. TensorFlow 方案核心

### 为什么 TensorFlow？

**优势**：
- 自动图优化（操作融合、内存优化）
- 高效的向量操作（广播、reduce）
- 与推荐系统易集成（如已用 TF）
- 生产级支持（SavedModel、Serving）

**性能预期**：
- CPU 版本：20-30ms
- TensorFlow GPU：3-5ms
- 提升：5-10 倍

### 核心设计

**不能完全并行化**：
```
规则依赖性：位置 i 依赖位置 0..i-1 的结果
→ 必须串行推进位置
```

**但可以在每个位置并行**：
```
位置 0:
  ├─ GPU 并行：检查 2000 个候选
  ├─ GPU 并行：计算规则有效性
  └─ CPU：选择最高分 → 位置 1
  
位置 1:
  ├─ GPU 并行：检查 1999 个候选
  ├─ GPU 并行：计算规则有效性
  └─ CPU：选择最高分 → 位置 2
  
...
```

**GPU 特别快的操作**：
1. **广播比较**（20x）- 5 个已选 vs 2000 个候选
2. **向量求和**（10x）- 统计匹配结果
3. **条件判断**（7.5x）- 2000 个候选的规则检查

---

## 3. GPU 计算流程图

### 整体流程（GPU vs CPU）

```mermaid
graph TD
    A["输入：2000 候选 + 已选序列"] -->|CPU| B["初始化"]
    B -->|GPU| C["位置 0 的 GPU 计算"]
    C -->|CPU| D["选择最高分"]
    D -->|CPU| E["更新已选序列"]
    E -->|GPU| F["位置 1 的 GPU 计算"]
    F -->|CPU| G["选择最高分"]
    G -->|CPU| H["更新已选序列"]
    H -->|...| I["位置 99"]
    I -->|CPU| J["输出 100 items"]
    
    style C fill:#90EE90
    style F fill:#90EE90
    style I fill:#90EE90
    style D fill:#FFB6C1
    style G fill:#FFB6C1
    style J fill:#FFB6C1
```

### 单个位置的详细流程（GPU 并行）

```mermaid
graph TD
    A["位置 pos：已有 pos 个 items<br/>剩余 2000-pos 个候选"] --> B["GPU 内存准备"]
    
    B --> C["Phase 1: GPU 规则检查（并行）"]
    
    C --> C1["坑位规则<br/>candidate[:, feature] != forbidden"]
    C --> C2["窗口规则<br/>广播比较 + 求和"]
    C --> C3["折损规则<br/>heat_count 计算"]
    
    C1 --> C4["融合为一个<br/>valid_mask"]
    C2 --> C4
    C3 --> C4
    
    C4 --> D["Phase 2: GPU 评分计算"]
    D --> D1["candidate_features @ user_features<br/>= scores"]
    
    D1 --> E["Phase 3: GPU 应用掩码"]
    E --> E1["masked_scores = where<br/>valid_mask, scores, -inf"]
    
    E1 --> F["Phase 4: GPU 选择"]
    F --> F1["best_idx = argmax<br/>masked_scores"]
    
    F1 --> G["Phase 5: CPU 同步"]
    G --> G1["转回 CPU<br/>best_idx 仅 1 个 int"]
    
    G1 --> H["CPU 更新"]
    H --> H1["result.append<br/>candidates[best_idx]"]
    
    H1 --> I["位置推进"]
    
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
    style F fill:#90EE90
    style G4 fill:#FFB6C1
    style H fill:#FFB6C1
```

### 窗口规则的 GPU 计算（最关键）

```mermaid
graph LR
    A["已选序列<br/>5 items<br/>category_ids<br/>shape: 5"] -->|reshape| B["(5,1)"]
    C["所有候选<br/>2000 items<br/>category_ids<br/>shape: 2000"] -->|reshape| D["(1,2000)"]
    
    B -->|广播比较| E["matches<br/>(5, 2000)"]
    D -->|广播比较| E
    
    E -->|cast to int32| F["matches_int<br/>(5, 2000)"]
    F -->|reduce_sum<br/>axis=0| G["match_counts<br/>shape: 2000<br/>每个候选的匹配次数"]
    
    G -->|<= max_count| H["valid_mask<br/>shape: 2000<br/>bool 数组"]
    
    style A fill:#87CEEB
    style C fill:#87CEEB
    style E fill:#FFD700
    style G fill:#FFD700
    style H fill:#90EE90
```

### 数据流转与同步

```mermaid
graph TD
    A["CPU 内存"] -->|首次：80KB| B["GPU 内存<br/>候选特征"]
    C["CPU 内存<br/>已选序列"] -->|每次位置：4KB| D["GPU 内存<br/>已选维度"]
    
    B -->|计算| E["GPU 计算<br/>3-5ms"]
    D -->|计算| E
    
    E -->|结果：2KB<br/>valid_mask| F["CPU 内存<br/>有效掩码"]
    
    F -->|CPU 逻辑| G["选择最高分"]
    G -->|1 个 int| H["CPU 内存<br/>best_idx"]
    
    H -->|更新| C
    
    style A fill:#FFE4E1
    style B fill:#87CEEB
    style D fill:#87CEEB
    style E fill:#90EE90
    style F fill:#FFE4E1
    style G fill:#FFE4E1
    style H fill:#FFE4E1
```

### 三类规则的 GPU 计算

```mermaid
graph TD
    A["候选集：2000 items"] --> B["规则检查"]
    
    B --> B1["坑位规则<br/>if position == target<br/>  candidate[feature] != forbidden"]
    B --> B2["窗口规则<br/>广播比较<br/>match_counts = reduce_sum<br/>match_counts < max_count"]
    B --> B3["折损规则<br/>heat_count = sum is_heat<br/>ratio = heat_count/window<br/>ratio <= threshold"]
    
    B1 --> C["GPU 并行<br/>2000 个线程"]
    B2 --> C
    B3 --> C
    
    C --> D["all_valid<br/>= valid1 & valid2 & valid3"]
    D --> E["输出：bool[2000]<br/>标记有效候选"]
    
    style C fill:#90EE90
    style E fill:#FFD700
```

### 性能瓶颈分析

```mermaid
graph LR
    A["CPU 20-30ms"] --> A1["规则检查：15ms<br/>条件判断多，分支复杂"]
    A --> A2["评分计算：5-10ms<br/>2000 个候选的矩阵操作"]
    A --> A3["同步开销：1-2ms<br/>内存传输"]
    
    B["GPU 3-5ms"] --> B1["规则检查：2-3ms<br/>并行条件判断"]
    B --> B2["评分计算：0.5-1ms<br/>并行矩阵操作"]
    B --> B3["同步开销：1-2ms<br/>内存传输"]
    
    A1 -->|7.5x| B1
    A2 -->|10x| B2
    A3 -->|不变| B3
    
    style A fill:#FFB6C1
    style B fill:#90EE90
    style B1 fill:#FFD700
    style B2 fill:#FFD700
```

---

## 4. TensorFlow 实现概述

### 伪代码

```python
import tensorflow as tf

@tf.function  # JIT 编译，自动优化
def beamsearch_step(result, candidates, position):
    # 已选序列（GPU 张量）
    result_dims = tf.constant(...)  # shape: (pos, num_features)
    
    # 所有候选（GPU 张量）
    candidate_dims = tf.constant(...)  # shape: (2000, num_features)
    
    # Phase 1: 规则检查（GPU）
    valid_mask = check_all_rules(result_dims, candidate_dims, position)
    
    # Phase 2: 评分计算（GPU）
    scores = compute_scores(candidate_dims)
    
    # Phase 3: 应用掩码
    masked_scores = tf.where(valid_mask, scores, -1e10)
    
    # Phase 4: 选择最高分
    best_idx = tf.argmax(masked_scores)
    
    return best_idx, scores, valid_mask

def check_all_rules(result_dims, candidate_dims, position):
    """GPU 上并行检查所有规则"""
    num_candidates = tf.shape(candidate_dims)[0]
    valid_mask = tf.ones(num_candidates, dtype=tf.bool)
    
    # 坑位规则
    if position == rule.position:
        valid_mask &= candidate_dims[:, 1] != forbidden_type
    
    # 窗口规则：广播比较 + 求和
    window_start = tf.maximum(0, tf.shape(result_dims)[0] - window_size + 1)
    result_window = result_dims[window_start:, dim]  # shape: (w,)
    
    # 广播比较：(w, 1) vs (1, 2000) → (w, 2000)
    matches = tf.equal(result_window[:, None], candidate_dims[None:, dim])
    
    # 沿 axis=0 求和
    match_counts = tf.reduce_sum(tf.cast(matches, tf.int32), axis=0)
    valid_mask &= match_counts < max_count
    
    # 折损规则
    if candidate.is_heat:
        heat_count = tf.reduce_sum(tf.cast(result_dims[:, 4], tf.bool))
        heat_ratio = (heat_count + 1) / tf.shape(result_dims)[0]
        valid_mask &= heat_ratio <= max_heat_ratio
    
    return valid_mask
```

### 关键优化

1. **@tf.function 编译**
   - 转换为静态计算图
   - 自动融合相邻操作
   - 性能提升 2-3 倍

2. **广播操作**
   - CPU 嵌套循环：O(w × 2000)，~10ms
   - TensorFlow 广播：O(1)，~0.5ms
   - 提升：20 倍

3. **最小化 CPU-GPU 同步**
   - 每个位置同步一次
   - 传输 2KB bool 掩码
   - 同步开销 < 1-2ms

4. **混合精度（可选）**
   - 条件判断用 int32/bool
   - 评分用 float16（可选）
   - 吞吐量提升 2-3 倍

---

## 5. 信息需求

### 必须提供

- [ ] **打散规则完整列表**
  ```
  - 规则 ID（如 "first_no_double"）
  - 规则类型（坑位/窗口/折损）
  - 具体参数
  - 优先级（如果有冲突）
  ```

- [ ] **候选 item 属性**
  ```
  已知的：
    - score, itemshowtype, category_id, bizuin, is_heat
  
  需要确认：
    - 还有其他维度吗？
    - 哪些是维度值（用于窗口规则）？
    - 哪些是标志位（用于条件判断）？
    - 总共多少个特征维度？
  ```

- [ ] **TensorFlow 环境**
  ```
  - TensorFlow 版本要求
  - GPU 类型和显存
  - 现有系统中 GPU 的使用情况
  ```

- [ ] **性能目标**
  ```
  - 目标延时：3-5ms 还是 < 2ms？
  - P99 要求：多少？
  - 吞吐量：QPS？
  ```

### 架构确认

- [ ] 现有推荐系统是否已用 TensorFlow？
- [ ] 是否有现成的 GPU 推理流程？
- [ ] 推荐模型的输出维度是多少？
- [ ] 是否需要支持 TensorFlow Serving 部署？

---

## 6. 预期与下一步

### 预期

| 指标 | 值 |
|------|-----|
| 性能提升 | 5-10 倍 |
| 延时 | 3-5ms |
| 开发周期 | 2-3 周 |
| 代码行数 | 300-500 行 |

### 下一步流程

1. **你提供信息** ← 现在
2. **我们设计 TensorFlow 方案**
3. **实现原型**
4. **性能基准测试**
5. **集成和部署**

---

## 问题清单（待回答）

将你的回答填入 markdown 复选框中。

### 打散规则
- [ ] 有几条规则？
- [ ] 每条规则的具体定义？
- [ ] 规则之间有优先级吗？
- [ ] 规则是否动态变化？

### 候选 Item
- [ ] 总共多少个维度？
- [ ] 哪些维度用于窗口规则？
- [ ] 是否需要特殊的内存对齐？
- [ ] 特征向量是否预计算好了？

### 系统架构
- [ ] 现有系统是否用 TensorFlow？
- [ ] 推荐模型输出格式？
- [ ] GPU 显存约束？
- [ ] 是否需要 TensorFlow Serving？

### 性能要求
- [ ] 目标延时是多少？
- [ ] P99 要求是多少？
- [ ] 是否需要支持多 GPU？
- [ ] 是否需要支持批处理？

---

**准备好了吗？提供上面的信息，我们开始设计和实现！** 🚀
