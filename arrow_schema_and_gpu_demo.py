import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
import json

# 1. Arrow Schema 定义
def get_arrow_schema():
    schema = pa.schema([
        pa.field("age", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32(), 128)),  # 定长向量
        pa.field("clicked_items", pa.list_(pa.int64())),     # 变长序列
        pa.field("city_id", pa.int32()),
        pa.field("weights", pa.list_(pa.float32())),         # 变长向量
    ])
    return schema

# 2. 保存 Arrow Schema 为 JSON 文件
def save_arrow_schema_to_json(schema, filename="schema.json"):
    # 用 pyarrow 的 serialize 方法保存 schema
    import pyarrow as pa
    buf = schema.serialize()
    with open(filename, 'wb') as f:
        f.write(buf.to_pybytes())
    print(f"Arrow schema serialized and saved to '{filename}'")

# 3. 从文件读取 Arrow Schema
def load_arrow_schema_from_json(filename="schema.json"):
    import pyarrow as pa
    with open(filename, 'rb') as f:
        buf = f.read()
    schema = pa.ipc.read_schema(pa.py_buffer(buf))
    return schema

# 4. 示例数据生成与写入 Parquet
def generate_and_save_parquet(filename="features.parquet", n_samples=100000):
    """
    生成大规模模拟数据并写入 Parquet 文件。
    """
    np.random.seed(42)
    data = {
        "age": np.random.uniform(18, 70, size=n_samples).astype(np.float32),
        "embedding": [np.random.randn(128).astype(np.float32).tolist() for _ in range(n_samples)],
        "clicked_items": [np.random.randint(100, 999, size=np.random.randint(1, 10)).tolist() for _ in range(n_samples)],
        "city_id": np.random.randint(1, 100, size=n_samples).astype(np.int32),
        "weights": [np.random.rand(np.random.randint(1, 8)).astype(np.float32).tolist() for _ in range(n_samples)]
    }
    schema = get_arrow_schema()
    save_arrow_schema_to_json(schema)
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, filename)
    print(f"Parquet file '{filename}' written with {n_samples} samples.")

# CUDA stream 并行读取和处理 Parquet 文件
# 假设数据量大，分块读取，每块用独立 stream 并行 pad/拼接/转 tensor

def cuda_stream_parallel_read(filename="features.parquet", schema_file="features.schema.json", chunk_size=20000):
    import cudf
    import cupy as cp
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import concurrent.futures

    # 读取 schema
    schema = load_arrow_schema_from_json(schema_file)
    # 用 pyarrow 读取行数
    n_rows = pq.read_metadata(filename).num_rows
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    results = [None] * n_chunks

    def process_chunk(chunk_idx):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_rows)
            table = pq.read_table(filename, schema=schema, skip_rows=start, num_rows=end-start)
            gdf = cudf.DataFrame.from_arrow(table)
            # 标量字段
            age_tensor = torch.as_tensor(gdf["age"].values).cuda(non_blocking=True)
            city_id_tensor = torch.as_tensor(gdf["city_id"].values).cuda(non_blocking=True)
            # embedding
            embedding_flat = gdf["embedding"].list.leaves.values
            embedding_tensor = torch.as_tensor(embedding_flat).reshape((-1, 128)).cuda(non_blocking=True)
            # 变长字段并行 pad
            def pad_list_column(col, dtype, pad_value=0):
                # 完全用 cupy kernel 并行，无 for 循环
                offsets = cp.asarray(col._column.offsets.values)
                values = col.list.leaves.values
                batch_size = len(offsets) - 1
                lens = offsets[1:] - offsets[:-1]
                max_len = int(lens.max())
                # 生成 batch_idx 和 pos_idx
                batch_idx = cp.repeat(cp.arange(batch_size), lens)
                pos_idx = cp.concatenate([cp.arange(l) for l in lens.tolist()])
                padded = cp.full((batch_size, max_len), pad_value, dtype=dtype)
                if len(values) > 0:
                    padded[batch_idx, pos_idx] = values
                return torch.as_tensor(padded).cuda(non_blocking=True)
            clicked_tensor = pad_list_column(gdf["clicked_items"], dtype=cp.int64, pad_value=-1)
            weights_tensor = pad_list_column(gdf["weights"], dtype=cp.float32, pad_value=0.0)
            stream.synchronize()
            return {
                "age": age_tensor,
                "city_id": city_id_tensor,
                "embedding": embedding_tensor,
                "clicked_items": clicked_tensor,
                "weights": weights_tensor
            }

    # 多线程调度 stream 并行（每个线程独立 stream）
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(process_chunk, i): i for i in range(n_chunks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
def cuda_stream_parallel_read_overlap(filename="features.parquet", schema_file="features.schema.json", chunk_size=20000):
    import cudf
    import cupy as cp
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import concurrent.futures

    # 读取 schema
    schema = load_arrow_schema_from_json(schema_file)
    # 用 pyarrow 读取行数
    n_rows = pq.read_metadata(filename).num_rows
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    results = [None] * n_chunks

    def process_chunk(chunk_idx):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_rows)
            # 异步预取下一个chunk的数据到GPU
            if chunk_idx < n_chunks - 1:
                next_start = (chunk_idx + 1) * chunk_size
                next_end = min((chunk_idx + 2) * chunk_size, n_rows)
                next_table = pq.read_table(filename, schema=schema, skip_rows=next_start, num_rows=next_end-next_start)
                next_gdf = cudf.DataFrame.from_arrow(next_table)
                next_gdf = next_gdf.cuda(non_blocking=True)
            # 当前stream上处理本chunk
            table = pq.read_table(filename, schema=schema, skip_rows=start, num_rows=end-start)
            gdf = cudf.DataFrame.from_arrow(table)
            # 标量字段
            age_tensor = torch.as_tensor(gdf["age"].values).cuda(non_blocking=True)
            city_id_tensor = torch.as_tensor(gdf["city_id"].values).cuda(non_blocking=True)
            # embedding
            embedding_flat = gdf["embedding"].list.leaves.values
            embedding_tensor = torch.as_tensor(embedding_flat).reshape((-1, 128)).cuda(non_blocking=True)
            # 变长字段并行 pad
            def pad_list_column(col, dtype, pad_value=0):
                # 完全用 cupy kernel 并行，无 for 循环
                offsets = cp.asarray(col._column.offsets.values)
                values = col.list.leaves.values
                batch_size = len(offsets) - 1
                lens = offsets[1:] - offsets[:-1]
                max_len = int(lens.max())
                # 生成 batch_idx 和 pos_idx
                batch_idx = cp.repeat(cp.arange(batch_size), lens)
                pos_idx = cp.concatenate([cp.arange(l) for l in lens.tolist()])
                padded = cp.full((batch_size, max_len), pad_value, dtype=dtype)
                if len(values) > 0:
                    padded[batch_idx, pos_idx] = values
                return torch.as_tensor(padded).cuda(non_blocking=True)
            clicked_tensor = pad_list_column(gdf["clicked_items"], dtype=cp.int64, pad_value=-1)
            weights_tensor = pad_list_column(gdf["weights"], dtype=cp.float32, pad_value=0.0)
            stream.synchronize()
            return {
                "age": age_tensor,
                "city_id": city_id_tensor,
                "embedding": embedding_tensor,
                "clicked_items": clicked_tensor,
                "weights": weights_tensor
            }

    # 多线程调度 stream 并行（每个线程独立 stream）
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(process_chunk, i): i for i in range(n_chunks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            print(f"Chunk {idx+1}/{n_chunks} processed on GPU.")
    print("All chunks processed with CUDA stream parallelism and overlap.")
    return results

# 5. GPU 解析代码（cuDF + PyTorch）
def load_and_parse_gpu(filename="features.parquet"):
    import cudf
    import cupy as cp
    import torch

    # 读取 Arrow schema
    schema = load_arrow_schema_from_json()

    # 读取到 GPU DataFrame
    gdf = cudf.read_parquet(filename, schema=schema)
    print("cuDF DataFrame loaded:")
    print(gdf)

    # 标量字段（全程在GPU）
    age_gpu = gdf["age"].values  # cupy array
    city_id_gpu = gdf["city_id"].values  # cupy array
    # 转为 PyTorch GPU tensor（zero-copy）
    age_tensor = torch.as_tensor(age_gpu)
    city_id_tensor = torch.as_tensor(city_id_gpu)

    # embedding 字段（定长向量，list<fp32[128]>），cuDF 23.x+ 支持 list column
    # 直接用 cuDF 的 list column 转成 cupy，然后转 torch tensor
    embedding_col = gdf["embedding"]
    # flatten 成一维 cupy array
    embedding_flat = embedding_col.list.leaves.values
    # reshape 为 [batch, 128]
    embedding_tensor = torch.as_tensor(embedding_flat).reshape((-1, 128))

    # 变长 list 字段 clicked_items（list<int64>）和 weights（list<float32>）
    # 用 cuDF 的 list column + cupy 并行 pad 拼接
    def pad_list_column(col, dtype, pad_value=0):
        # 获取 offsets
        offsets = col._column.offsets.values.get()
        # 获取所有元素
        values = col.list.leaves.values
        batch_size = len(offsets) - 1
        lens = offsets[1:] - offsets[:-1]
        max_len = lens.max()
        # 用 cupy 并行 pad
        padded = cp.full((batch_size, max_len), pad_value, dtype=dtype)
        for i in range(batch_size):
            l = lens[i]
            if l > 0:
                padded[i, :l] = values[offsets[i]:offsets[i+1]]
        return torch.as_tensor(padded)

    clicked_tensor = pad_list_column(gdf["clicked_items"], dtype=cp.int64, pad_value=-1)
    weights_tensor = pad_list_column(gdf["weights"], dtype=cp.float32, pad_value=0.0)

    # 确保所有 tensor 都在 cuda 上
    embedding_tensor = embedding_tensor.cuda()
    clicked_tensor = clicked_tensor.cuda()
    weights_tensor = weights_tensor.cuda()
    age_tensor = age_tensor.cuda()
    city_id_tensor = city_id_tensor.cuda()

    print("embedding_tensor shape:", embedding_tensor.shape)
    print("clicked_tensor shape:", clicked_tensor.shape)
    print("weights_tensor shape:", weights_tensor.shape)
    print("age_tensor shape:", age_tensor.shape)
    print("city_id_tensor shape:", city_id_tensor.shape)
    return {
        "age": age_tensor,
        "city_id": city_id_tensor,
        "embedding": embedding_tensor,
        "clicked_items": clicked_tensor,
        "weights": weights_tensor
    }

from torch.utils.data import Dataset, DataLoader

class ParquetGPUTorchDataset(Dataset):
    def __init__(self, filename="features.parquet", schema_file="features.schema.json", chunk_size=20000):
        self.filename = filename
        self.schema_file = schema_file
        import pyarrow.parquet as pq
        self.n_rows = pq.read_metadata(filename).num_rows
        self.chunk_size = chunk_size
        self.n_chunks = (self.n_rows + chunk_size - 1) // chunk_size
        self._cache = {}

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        inner_idx = idx % self.chunk_size
        if chunk_idx not in self._cache:
            # 只缓存当前chunk，节省显存
            results = cuda_stream_parallel_read(self.filename, self.schema_file, self.chunk_size)
            self._cache = {chunk_idx: results[chunk_idx]}
        batch = self._cache[chunk_idx]
        # 取出单条数据（所有特征）
        item = {k: v[inner_idx] for k, v in batch.items()}
        return item

if __name__ == "__main__":
    generate_and_save_parquet("features.parquet")
    # 用DataLoader加载GPU数据集
    dataset = ParquetGPUTorchDataset("features.parquet", "features.schema.json", chunk_size=2048)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    for batch in dataloader:
        print("Batch loaded:", {k: v.shape for k, v in batch.items()})
        break  # 只演示一批
