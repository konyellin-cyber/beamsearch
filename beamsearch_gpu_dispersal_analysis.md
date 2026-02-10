# GPU 加速 Beamsearch + 打散规则集成方案

## 项目背景

**场景**：精排后的混排层 beamsearch
- **候选集规模**：2000 items
- **输出规模**：通常 100-200 items
- **关键特点**：打散规则存在位置依赖性（if 位置A then 位置B）

**规则类型**：
1. 指定坑位过滤/强插（位置相关强约束）
2. 窗口 M 出 N（维度打散：itemshowtype、类目、标签、bizuin 等）
3. 定坑折损（加热内容窗口折损检查）

---

## 第一部分：规则建模

### 1.1 规则类型详细分析

#### 规则类型 1：坑位过滤/强插

**定义**：给定当前上文 + 坑位位置，对候选 item 进行过滤或强制插入

**示例**：
- 首坑（位置 0）不出现双列样式
- 视频号左右卡在特定位置强插
- 置顶广告必须在位置 0-2

**GPU 实现思路**：
```
对于每个候选 item，并行计算：
  - 能否放在位置 pos？
  - 输出：bool[num_candidates] 表示可行性
```

**数据结构**：
```python
@dataclass
class PositionFilter:
    position: int                    # 约束作用的位置
    rule_id: str                     # 规则唯一标识
    filter_func: Callable            # GPU 或 CPU 执行的过滤函数
    # 示例：itemshowtype != DOUBLE_COLUMN if position == 0
```

#### 规则类型 2：窗口 M 出 N

**定义**：在大小为 M 的滑动窗口内，某个维度最多/最少出现 N 次

**维度列表**：
- `itemshowtype`（样式类型）
- `category_id`（类目 ID）
- `tag`（标签）
- `bizuin`（商家/账号）
- `is_beauty`（美女类）
- `is_experience`（体验类）
- `same_event`（同事件）
- etc.

**示例规则**：
```
- window_size=5, dimension=category_id, max_count=2 
  → 任意 5 个连续 item 中，同一类目不超过 2 个

- window_size=3, dimension=bizuin, max_count=1 
  → 任意 3 个连续 item 中，同一商家最多 1 个
```

**关键约束**：这个规则依赖于**已选序列的状态**
- 需要知道 `[result[0], ..., result[pos-1]]` 才能判断当前位置
- 因此需要**串行推进**，但在每个位置**并行计算**所有候选的可行性

**GPU 实现思路**：
```
输入：
  - current_result: [pos] items (已选项)
  - candidates: [2000] 未选项
  - window_rules: List[WindowRule]

对每条窗口规则，并行计算：
  for each candidate in candidates:
    # 计算如果选择这个 candidate，是否违反规则
    
    # 统计最后 M-1 个已选项中该维度的出现次数
    count_in_window = count_dimension_in_last_window(
        current_result, candidate, dimension, window_size
    )
    
    # 判断是否违反规则
    is_valid[candidate_idx] = count_in_window < max_count

输出：is_valid[2000] bool 数组，标记每个候选是否可选
```

**数据结构**：
```python
@dataclass
class WindowRule:
    window_size: int                # M
    dimension: str                  # 维度（category_id, bizuin 等）
    max_count: int                  # N（最多出现次数）
    min_count: int = 0              # 最少出现次数（可选）
    rule_id: str = ""               # 规则标识
    
    # GPU 计算需要的映射表
    dimension_hash: Dict[Any, int]  # 维度值 → 哈希 ID（加速查找）
```

#### 规则类型 3：定坑折损

**定义**：对加热内容（热商品），计算窗口内折损（与特定类型商品的混合比例）

**示例**：
```
- 如果位置 [0-19] 中选择了超过 30% 的加热内容，则不能再出现加热商品
- 折损 = 非加热占比
- 当折损不足时，加热内容不出（不能放）
```

**GPU 实现思路**：
```
输入：
  - current_result: [pos] items
  - candidate: 当前候选
  - heat_window_rules: List[HeatWindowRule]

对每条规则，并行计算：
  if candidate.is_heat:
    heat_count = count_heat_in_window(current_result, window_start, window_end)
    total_count = end - start
    heat_ratio = heat_count / total_count
    
    if heat_ratio >= max_heat_ratio:
      is_valid[candidate_idx] = False
```

**数据结构**：
```python
@dataclass
class HeatWindowRule:
    window_start: int               # 窗口起始位置（相对或绝对）
    window_end: int                 # 窗口结束位置
    max_heat_ratio: float           # 最大热内容占比
    rule_id: str = ""
```

---

## 第二部分：GPU Beamsearch 架构设计

### 2.1 算法流程

```
输入：candidates[2000], user_features, rules

初始化：
  result = []  # 已选序列
  candidates_mask = [True] * 2000  # 未选标记
  
Main Loop（串行推进）：
  for pos in range(target_length):  # 目标输出长度
    
    # Phase 1: GPU 规则检查（并行）
    valid_mask = gpu_check_all_rules(
        current_result=result,
        all_candidates=candidates,
        candidates_mask=candidates_mask,
        rules=all_rules,
        current_position=pos
    )
    
    # Phase 2: GPU 评分计算（并行，仅对有效候选）
    valid_indices = where(valid_mask & candidates_mask)
    scores = gpu_batch_score(
        candidates[valid_indices],
        user_features
    )
    
    # Phase 3: CPU 选择（选最高分）
    best_idx = argmax(scores)
    best_candidate = candidates[valid_indices[best_idx]]
    
    # Phase 4: 更新状态（CPU）
    result.append(best_candidate)
    candidates_mask[best_idx] = False
    
输出：result[target_length]
```

### 2.2 关键设计：并行粒度

**重点**：由于规则的位置依赖性，你的观察是正确的：

```
┌─ 串行推进（位置）
│  ├─ 位置 0
│  │  └─ GPU 并行：检查所有 2000 个候选
│  ├─ 位置 1
│  │  └─ GPU 并行：检查所有剩余 ~1999 个候选
│  ├─ 位置 2
│  │  └─ ...
│  └─ ...
└─ 每个位置内：充分利用 GPU 并行计算
```

**为什么不能完全并行化**？
- 窗口规则需要知道已选序列的信息
- 某些规则有累积效应（如折损率）
- 贪心+规则的组合本质是串行的

**GPU 的价值在哪**？
- 每个位置需要检查 ~2000 个候选 × 多条规则的组合
- GPU 可以高效地并行这些检查，而不是逐个 CPU 判断

---

## 第三部分：GPU 实现细节

### 3.1 核心 GPU 内核设计

#### 内核 1：窗口 M 出 N 规则检查

```cuda
/**
 * 针对窗口规则的并行检查
 * 
 * 输入：
 *   - result_sequence: [pos] int32，已选 item 的维度值（提前映射）
 *   - candidates_dimension: [2000] int32，候选 item 的维度值
 *   - window_size: M
 *   - max_count: N
 *   - pos: 当前位置
 * 
 * 输出：
 *   - valid_mask: [2000] bool，True 表示该候选不违反规则
 */
__global__ void check_window_rule(
    const int* result_sequence,        // [pos]
    const int* candidates_dimension,   // [2000]
    int pos,
    int window_size,
    int max_count,
    int candidate_count,               // 2000
    bool* valid_mask                   // [2000] output
) {
    int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate_idx >= candidate_count) return;
    
    // 统计最后 window_size-1 个已选项中该维度的出现次数
    int window_start = max(0, pos - (window_size - 1));
    int count_in_window = 0;
    int candidate_dim_value = candidates_dimension[candidate_idx];
    
    for (int i = window_start; i < pos; i++) {
        if (result_sequence[i] == candidate_dim_value) {
            count_in_window++;
        }
    }
    
    // 检查是否满足条件
    valid_mask[candidate_idx] = (count_in_window < max_count);
}
```

#### 内核 2：坑位过滤

```cuda
/**
 * 坑位过滤规则检查
 * 
 * 输入：
 *   - candidates_showtype: [2000] int32，候选的样式类型
 *   - pos: 当前位置
 *   - position_rule: 坑位规则
 *   - result_showtypes: [pos] int32，已选 item 的样式类型
 * 
 * 输出：
 *   - valid_mask: [2000] bool
 */
__global__ void check_position_filter(
    const int* candidates_showtype,
    int pos,
    int position,                      // 规则作用的位置
    int forbidden_showtype,            // 不允许的样式类型
    int candidate_count,
    bool* valid_mask
) {
    int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate_idx >= candidate_count) return;
    
    // 只有在指定位置时才应用规则
    if (pos == position) {
        valid_mask[candidate_idx] = 
            (candidates_showtype[candidate_idx] != forbidden_showtype);
    } else {
        valid_mask[candidate_idx] = true;  // 其他位置无约束
    }
}
```

#### 内核 3：定坑折损检查

```cuda
/**
 * 热内容折损规则检查
 * 
 * 输入：
 *   - candidates_is_heat: [2000] bool
 *   - result_is_heat: [pos] bool，已选 item 的热标记
 *   - window_start, window_end: 窗口范围
 *   - max_heat_ratio: 最大热占比
 * 
 * 输出：
 *   - valid_mask: [2000] bool
 */
__global__ void check_heat_dispersal(
    const bool* candidates_is_heat,
    const bool* result_is_heat,
    int pos,
    int window_start,
    int window_end,
    float max_heat_ratio,
    int candidate_count,
    bool* valid_mask
) {
    int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate_idx >= candidate_count) return;
    
    // 只对热内容候选检查
    if (!candidates_is_heat[candidate_idx]) {
        valid_mask[candidate_idx] = true;
        return;
    }
    
    // 统计当前窗口中已选的热内容数量
    int heat_count = 0;
    int window_size = 0;
    
    for (int i = max(0, window_start); i < pos && i < window_end; i++) {
        window_size++;
        if (result_is_heat[i]) {
            heat_count++;
        }
    }
    
    // 检查加入这个热内容后的折损率
    if (window_size > 0) {
        float current_heat_ratio = (float)heat_count / window_size;
        valid_mask[candidate_idx] = (current_heat_ratio < max_heat_ratio);
    } else {
        valid_mask[candidate_idx] = true;  // 窗口为空时允许
    }
}
```

### 3.2 主程序流程（Python + CUDA）

```python
import cupy as cp
import numpy as np
from typing import List, Dict, Tuple

class GPUBeamsearchWithDispersal:
    def __init__(self, beam_size=100, gpu_id=0):
        self.beam_size = beam_size
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        
    def rank(self, 
             candidates: List[Dict],
             user_features: np.ndarray,
             rules: Dict,
             target_length: int = 100) -> List[Dict]:
        """
        主排序函数
        
        Args:
            candidates: 候选 item 列表，每个 item 包含：
                - id, score, itemshowtype, category_id, bizuin, is_heat, tags 等
            user_features: 用户特征向量 [feature_dim]
            rules: 包含所有打散规则
            target_length: 目标输出长度
            
        Returns:
            排序后的 item 列表 [target_length]
        """
        
        num_candidates = len(candidates)
        result = []
        candidates_mask = np.ones(num_candidates, dtype=bool)  # False = 已选
        
        # 准备 GPU 内存
        # 1. 候选 item 的属性转为 GPU 数组
        candidates_gpu = self._prepare_candidates_gpu(candidates)
        
        # 2. 已选序列的属性（初始为空）
        result_sequence_gpu = cp.zeros((target_length, 10), dtype=cp.int32)
        result_pos = 0
        
        # 3. 用户特征
        user_features_gpu = cp.asarray(user_features, dtype=cp.float32)
        
        # 主循环
        for pos in range(target_length):
            # Phase 1: GPU 规则检查
            valid_mask_gpu = self._gpu_check_all_rules(
                result_sequence_gpu,
                result_pos,
                candidates_gpu,
                rules
            )
            
            # Phase 2: 转回 CPU 并与已选掩码结合
            valid_mask_cpu = cp.asnumpy(valid_mask_gpu)
            valid_mask_combined = valid_mask_cpu & candidates_mask
            
            # Phase 3: GPU 评分计算
            valid_indices = np.where(valid_mask_combined)[0]
            if len(valid_indices) == 0:
                break  # 没有更多可选候选
            
            scores = self._gpu_batch_score(
                candidates_gpu,
                valid_indices,
                user_features_gpu
            )
            
            # Phase 4: CPU 选择最高分
            scores_cpu = cp.asnumpy(scores)
            best_local_idx = np.argmax(scores_cpu)
            best_global_idx = valid_indices[best_local_idx]
            
            # Phase 5: 更新状态
            selected_item = candidates[best_global_idx]
            result.append(selected_item)
            candidates_mask[best_global_idx] = False
            
            # 更新已选序列（GPU）
            self._update_result_sequence_gpu(
                result_sequence_gpu,
                result_pos,
                selected_item
            )
            result_pos += 1
        
        return result
    
    def _prepare_candidates_gpu(self, candidates: List[Dict]):
        """准备候选 item 的 GPU 数组"""
        num_candidates = len(candidates)
        
        # 构造结构化数组，便于 GPU 访问
        candidates_struct = np.zeros(
            num_candidates,
            dtype=[
                ('id', np.int32),
                ('score', np.float32),
                ('itemshowtype', np.int32),
                ('category_id', np.int32),
                ('bizuin', np.int32),
                ('is_heat', np.bool_),
                # ... 其他维度
            ]
        )
        
        for i, candidate in enumerate(candidates):
            candidates_struct[i]['id'] = candidate['id']
            candidates_struct[i]['score'] = candidate['score']
            candidates_struct[i]['itemshowtype'] = candidate.get('itemshowtype', 0)
            candidates_struct[i]['category_id'] = candidate.get('category_id', 0)
            candidates_struct[i]['bizuin'] = candidate.get('bizuin', 0)
            candidates_struct[i]['is_heat'] = candidate.get('is_heat', False)
        
        return cp.asarray(candidates_struct)
    
    def _gpu_check_all_rules(self,
                            result_sequence_gpu,
                            result_pos,
                            candidates_gpu,
                            rules: Dict):
        """并行检查所有规则"""
        num_candidates = candidates_gpu.size
        valid_mask = cp.ones(num_candidates, dtype=cp.bool_)
        
        # 规则 1: 窗口规则
        for window_rule in rules.get('window_rules', []):
            rule_valid = self._check_window_rule_gpu(
                result_sequence_gpu,
                result_pos,
                candidates_gpu,
                window_rule
            )
            valid_mask &= rule_valid
        
        # 规则 2: 坑位规则
        for position_rule in rules.get('position_rules', []):
            rule_valid = self._check_position_rule_gpu(
                result_sequence_gpu,
                result_pos,
                candidates_gpu,
                position_rule
            )
            valid_mask &= rule_valid
        
        # 规则 3: 折损规则
        for heat_rule in rules.get('heat_rules', []):
            rule_valid = self._check_heat_rule_gpu(
                result_sequence_gpu,
                result_pos,
                candidates_gpu,
                heat_rule
            )
            valid_mask &= rule_valid
        
        return valid_mask
    
    def _check_window_rule_gpu(self, result_seq_gpu, result_pos, candidates_gpu, rule):
        """GPU 检查单条窗口规则"""
        # 这里调用 CUDA 内核
        # 实际实现需要用 pycuda 或 cupy 的原始操作
        num_candidates = candidates_gpu.size
        valid_mask = cp.ones(num_candidates, dtype=cp.bool_)
        
        # 简化实现（实际需要 CUDA 内核）
        for i in range(num_candidates):
            # 计算该候选是否违反规则
            violate = False
            
            # 统计最后 window_size-1 个已选 item 的维度值
            dim_value = candidates_gpu[i][rule['dimension']]
            count_in_window = 0
            
            window_start = max(0, result_pos - (rule['window_size'] - 1))
            for j in range(window_start, result_pos):
                if result_seq_gpu[j][rule['dimension']] == dim_value:
                    count_in_window += 1
            
            if count_in_window >= rule['max_count']:
                violate = True
            
            valid_mask[i] = not violate
        
        return valid_mask
    
    def _gpu_batch_score(self, candidates_gpu, valid_indices, user_features_gpu):
        """GPU 批量评分"""
        # 使用预计算的候选评分（如果已有）
        # 或调用评分函数
        scores = candidates_gpu[valid_indices]['score']
        return cp.asarray(scores, dtype=cp.float32)
    
    def _update_result_sequence_gpu(self, result_seq_gpu, pos, item):
        """更新已选序列"""
        # 将 item 的属性信息转为 GPU 格式
        # 这是简化实现
        pass
```

---

## 第四部分：性能分析与优化建议

### 4.1 计算复杂度分析

| 步骤 | 复杂度 | 说明 |
|-----|-------|------|
| GPU 规则检查 | O(K × M × R) | K=2000 候选, M=窗口大小(avg 5), R=规则数 |
| GPU 评分 | O(K × D) | K=2000, D=特征维度 |
| CPU 选择 | O(K log K) 或 O(K) | argmax 操作 |
| 主循环 | O(L × (K + K + K log K)) | L=目标长度(100-200) |

**总体复杂度**：O(L × K × (M × R + D))
- 假设 L=100, K=2000, M=5, R=10, D=64
- 每次迭代：100 × 2000 × (5×10 + 64) = 100 × 2000 × 114 = **22.8M 操作**
- 全程：100 × 22.8M = **2.28B 操作**

**GPU vs CPU 性能**：
- GPU（Tesla A100）：2.2 TFLOPS = ~2000+ 操作/纳秒
- CPU（单核）：~100-200 GFLOPS = ~0.1-0.2 操作/纳秒
- **GPU 优势**：20-30 倍

**预期延时**（仅计算部分）：
- GPU：2.28B / 2000G ≈ **1-2ms**（主要瓶颈是内存）
- CPU：2.28B / 100G ≈ **20-30ms**
- **提升**：10-20 倍

### 4.2 内存分析

**GPU 内存占用**：
```
候选 item 属性：
  - 2000 × (4 int32 + 1 float32 + 8 bool) ≈ 40 KB

已选序列：
  - 100 × (4 int32 + 1 float32 + 8 bool) ≈ 2 KB

用户特征：
  - 64 × float32 ≈ 256 B

规则信息：
  - 各种规则参数 ≈ ~10 KB

总计：< 100 KB（极其轻量，即使是嵌入式 GPU 也不成问题）
```

**带宽分析**：
- 每次迭代需要读取候选属性：2000 × 40B = 80 KB
- GPU HBM 带宽：~2 TB/s
- 传输时间：80 KB / 2 TB/s ≈ **40 纳秒**（可忽略）

### 4.3 优化策略

#### 优化 1：减少 CPU↔GPU 数据传输

**现状**：
```
GPU 检查规则 → valid_mask → CPU 转换 → CPU 选择 → GPU 内存更新
```

**优化**：
```
GPU 完整流程：
  1. GPU 检查规则 → valid_mask
  2. GPU argmax（选择最大评分）→ best_idx
  3. 只传输 best_idx（单个 int）回 CPU
  4. CPU 更新 candidates_mask
  5. GPU 更新 result_sequence
```

**代码示例**：
```python
def _gpu_select_best(self, valid_mask_gpu, scores_gpu):
    """GPU 内选择最大值"""
    masked_scores = cp.where(valid_mask_gpu, scores_gpu, -1e10)
    best_idx = cp.argmax(masked_scores)
    return best_idx
```

#### 优化 2：批处理多个位置

虽然规则有位置依赖性，但可以考虑**分批处理**：
- 处理位置 [0-9]，然后处理 [10-19]，等等
- 每批内部的规则检查相对独立
- 减少 GPU 内核启动的开销

#### 优化 3：规则预编译

```python
class CompiledRule:
    """规则预编译，避免每次迭代解析"""
    def __init__(self, rule_config):
        self.dimension_hash = self._build_hash_map()
        self.cuda_kernel = self._compile_kernel()
        self.block_size = 256
        self.grid_size = (2000 + 255) // 256
```

#### 优化 4：混合精度计算

```python
# 评分计算用 float16（加快 2 倍）
scores_fp16 = cp.asarray(scores, dtype=cp.float16)
# 规则检查用 int8/bool（节省带宽）
```

---

## 第五部分：实现建议和代码框架

### 5.1 系统架构

```
┌───────────────────────────────────────────┐
│         推荐系统混排层                     │
└───────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  候选集准备（CPU）        │
        │  - 精排后的 2000 items    │
        │  - 准备特征向量          │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  数据上传 GPU            │
        │  - candidates → GPU      │
        │  - user_features → GPU   │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  GPU Beamsearch 内核     │
        │  - 规则并行检查          │
        │  - 评分计算              │
        │  - 结果积累              │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  结果转出 CPU            │
        │  - 最终排序结果          │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │      返回给前端           │
        └──────────────────────────┘
```

### 5.2 开发选择

| 工具 | 优点 | 缺点 | 适用场景 |
|-----|------|------|--------|
| **CUDA C++** | 性能最优，控制细致 | 开发周期长，难度大 | 生产部署 |
| **CuPy** | Python 友好，易于原型 | 性能略低于原生 CUDA | 快速迭代、研究 |
| **Triton** | 易于编写高效内核，Python | 还在发展中，文档不足 | 中等复杂度 |
| **Numba** | Python JIT，易上手 | 不如 CUDA 灵活 | 简单内核 |

**建议**：
- **原型阶段**：CuPy + Python，快速验证方案
- **性能优化阶段**：CUDA C++ 或 Triton，精细控制
- **生产部署**：CUDA C++，集成到 C++ 推荐系统中

### 5.3 关键代码组件

```python
# 1. 规则定义
@dataclass
class BeamsearchRules:
    window_rules: List[WindowRule]
    position_rules: List[PositionRule]
    heat_rules: List[HeatRule]
    
    @classmethod
    def from_config(cls, config_dict):
        """从配置文件加载规则"""
        pass

# 2. 候选 item 定义
@dataclass
class CandidateItem:
    id: int
    score: float
    itemshowtype: int
    category_id: int
    bizuin: int
    is_heat: bool
    tags: List[int]
    
    def to_gpu_struct(self):
        """转为 GPU 结构"""
        pass

# 3. GPU Beamsearch 类
class GPUBeamsearch:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
    
    def rank(self, candidates, rules, target_length=100):
        """主排序接口"""
        pass

# 4. 性能监测
class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
    
    def log_timing(self, stage_name, elapsed_ms):
        """记录各阶段耗时"""
        pass
    
    def report(self):
        """生成性能报告"""
        pass
```

---

## 第六部分：具体示例

### 示例：3 条打散规则

**规则 1：窗口规则 - 类目打散**
```python
rule1 = WindowRule(
    window_size=5,
    dimension='category_id',
    max_count=2,
    rule_id='category_diversity_5_2'
)
# 含义：任意 5 个连续 item 中，同一类目不超过 2 个
```

**规则 2：坑位规则 - 首坑不出双列**
```python
rule2 = PositionRule(
    position=0,
    rule_id='first_slot_no_double_column',
    forbidden_property=('itemshowtype', 3)  # 3 = 双列
)
# 含义：位置 0 不能出现样式类型 3（双列）
```

**规则 3：折损规则 - 加热内容折损**
```python
rule3 = HeatWindowRule(
    window_start=0,
    window_end=20,
    max_heat_ratio=0.3,
    rule_id='heat_content_max_30'
)
# 含义：前 20 个位置中，加热内容不超过 30%
```

**执行流程**：
```
位置 0：
  - 检查 rule1：都不违反（没有前置项）
  - 检查 rule2：候选中 itemshowtype=3 的被过滤掉
  - 检查 rule3：候选中 is_heat=true 的被过滤掉（不能放）
  - GPU 并行检查：2000 个候选 → ~1500 个有效
  - 评分选最高分：选出候选 A

位置 1：
  - 检查 rule1：统计类目 ID，候选 B 的类目 ID 已在位置 0 了
    - 如果 A 的类目是 10，则类目 10 的候选被过滤
  - 检查 rule2：位置 1 无约束
  - 检查 rule3：结合 A 是否 heat，判断候选是否 heat
  - GPU 并行检查：~1999 个候选（除去已选的 A）→ ~1400 个有效
  - 评分选最高分：选出候选 B
  
... 继续迭代直到位置 99
```

---

## 第七部分：验证 & 测试建议

### 7.1 单元测试

```python
def test_window_rule():
    """测试窗口规则"""
    rule = WindowRule(window_size=3, dimension='category_id', max_count=1)
    
    result = [
        Item(category_id=1),
        Item(category_id=2),
    ]
    candidate = Item(category_id=1)  # 类目已经出现过
    
    is_valid = check_window_rule(result, candidate, rule)
    assert not is_valid  # 应该被过滤
```

### 7.2 端到端测试

```python
def test_full_beamsearch():
    """测试完整流程"""
    candidates = generate_mock_candidates(2000)
    rules = BeamsearchRules.from_config(load_config('rules.yaml'))
    
    gpu_bs = GPUBeamsearch()
    result = gpu_bs.rank(candidates, rules, target_length=100)
    
    # 验证约束
    assert len(result) == 100
    assert all_rules_satisfied(result, rules)
    assert result is sorted by score
```

### 7.3 性能测试

```python
def benchmark():
    """性能基准测试"""
    for num_candidates in [1000, 2000, 5000]:
        for target_length in [50, 100, 200]:
            result, elapsed_ms = gpu_bs.rank(
                candidates[:num_candidates],
                rules,
                target_length
            )
            print(f"C={num_candidates}, L={target_length}: {elapsed_ms}ms")
```

---

## 总结

### 关键设计点

1. **串行位置推进 + GPU 并行规则检查**：符合你的业务逻辑
2. **三类规则的 GPU 内核**：覆盖坑位、窗口、折损
3. **内存轻量化**：GPU 内存使用 < 100 KB
4. **性能期望**：1-2ms 计算时间（vs CPU 的 20-30ms）

### 推荐的下一步

1. ✅ **确认规则详情**：提供具体的规则列表和优先级
2. ✅ **准备测试数据**：真实或模拟的 2000 候选 item + 规则
3. ✅ **选择实现方案**：CuPy（原型）vs CUDA（生产）
4. ✅ **开发第一个原型**：从简单的窗口规则开始
5. ✅ **性能基准测试**：与 CPU 版本对比

### 文件结构建议

```
beamsearch/
├── src/
│   ├── gpu_beamsearch.py         # 主类
│   ├── rules.py                  # 规则定义
│   ├── kernels.cu                # CUDA 内核
│   └── utils.py                  # 工具函数
├── tests/
│   ├── test_rules.py
│   ├── test_gpu_beamsearch.py
│   └── test_performance.py
├── config/
│   └── rules.yaml               # 规则配置
├── requirements.txt
└── README.md
```

---

**有具体的规则列表吗？我可以为你生成完整的可运行代码实现。**
