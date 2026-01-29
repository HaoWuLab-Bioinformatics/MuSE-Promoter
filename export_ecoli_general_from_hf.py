import os
import polars as pl

# 配置路径
PARQUET_PATH = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/train-00000-of-00001.parquet"
OUT_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/data/E.coli"


def write_fasta(df, out_path, prefix):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = 0
    with open(out_path, "w") as f:
        # 假设列名为 segment 和 y
        for i, row in enumerate(df.select(["segment", "y"]).iter_rows()):
            seq, y = row[0], row[1]
            if seq:
                f.write(f">{prefix}_{i}|{y}\n{seq}\n")
                count += 1
    print(f"[OK] 写入 {count} 条序列 -> {out_path}")


def main():
    print(f"[Load] 正在读取数据...")
    df = pl.read_parquet(PARQUET_PATH)

    # 1. 提取 E.coli 的正样本 (y=1)
    # 匹配物种名包含 escherichia coli 的正样本
    pos_df = df.filter(
        (pl.col("ppd_original_SpeciesName").str.to_lowercase().str.contains("escherichia coli")) &
        (pl.col("y") == 1)
    )

    num_pos = pos_df.height
    print(f"[Stat] 找到 E.coli 正样本: {num_pos} 条")

    # 2. 提取负样本 (y=0)
    # 因为 E.coli 下没有 y=0，我们从全局 y=0 中随机抽取等量的样本
    neg_pool = df.filter(pl.col("y") == 0)

    if neg_pool.height >= num_pos:
        neg_df = neg_pool.sample(n=num_pos, seed=42)  # 使用随机种子保证可重复
        print(f"[Stat] 从全局负样本池中随机抽取了 {num_pos} 条作为负样本")
    else:
        neg_df = neg_pool
        print(f"[Warn] 负样本不足，全部提取: {neg_df.height} 条")

    # 3. 写入文件
    write_fasta(pos_df, os.path.join(OUT_DIR, "Ecoli_prom.fa"), "Ecoli_pos")
    write_fasta(neg_df, os.path.join(OUT_DIR, "Ecoli_non_prom.fa"), "Ecoli_neg")

    print("\n=== 任务完成 ===")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()