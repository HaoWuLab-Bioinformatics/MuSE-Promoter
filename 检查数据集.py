import os, hashlib
def read_fasta_with_labels(fp):
    hs, ss, ys = [], [], []
    h=None; s=[]
    with open(fp,"r",errors="ignore") as f:
        for ln in f:
            ln=ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                if h is not None and s:
                    seq="".join(s).upper()
                    # header 形如 xxx|1|training
                    y = int(h.split("|")[1]) if "|" in h else None
                    hs.append(h); ss.append(seq); ys.append(y)
                h=ln[1:]; s=[]
            else:
                s.append(ln)
        if h is not None and s:
            seq="".join(s).upper()
            y = int(h.split("|")[1]) if "|" in h else None
            hs.append(h); ss.append(seq); ys.append(y)
    return ss, ys

def stats(seqs, ys, yval):
    xs=[s for s,y in zip(seqs,ys) if y==yval]
    lens=[len(s) for s in xs]
    gc=[(s.count("G")+s.count("C"))/max(1,len(s)) for s in xs]
    nrat=[s.count("N")/max(1,len(s)) for s in xs]
    import numpy as np
    return {
        "n": len(xs),
        "len_mean": float(np.mean(lens)),
        "gc_mean": float(np.mean(gc)),
        "n_mean": float(np.mean(nrat)),
    }

base="/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/B.subtilis"
seqs, ys = read_fasta_with_labels(f"{base}/train/data.fasta")
print("train y=1:", stats(seqs, ys, 1))
print("train y=0:", stats(seqs, ys, 0))

def md5(fp):
    h = hashlib.md5()
    with open(fp, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def check_pair(a, b):
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    print("A:", ra, "md5:", md5(ra))
    print("B:", rb, "md5:", md5(rb))
    print("same_realpath?", ra == rb, "same_md5?", md5(ra)==md5(rb))
    print("-"*60)

base = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/B.subtilis"
check_pair(f"{base}/train/y.txt", f"{base}/test/y.txt")
check_pair(f"{base}/train/data.fasta", f"{base}/test/data.fasta")
check_pair(f"{base}/train/cksnap.txt", f"{base}/test/cksnap.txt")
check_pair(f"{base}/train/mismatch.txt", f"{base}/test/mismatch.txt")
check_pair(f"{base}/train/word2vec.txt", f"{base}/test/word2vec.txt")
def read_fasta_seqs(fp):
    seqs = []
    s = []
    with open(fp, "r", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if s:
                    seqs.append("".join(s))
                    s=[]
            else:
                s.append(line.upper())
        if s:
            seqs.append("".join(s))
    return seqs

base = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/B.subtilis"
tr = read_fasta_seqs(f"{base}/train/data.fasta")
te = read_fasta_seqs(f"{base}/test/data.fasta")

set_tr = set(tr)
set_te = set(te)
overlap = set_tr & set_te

print("train seqs:", len(tr), "unique:", len(set_tr))
print("test  seqs:", len(te), "unique:", len(set_te))
print("overlap unique seqs:", len(overlap))
print("overlap ratio (test):", len(overlap)/max(1,len(set_te)))
import hashlib

def line_hashes(fp, round_digits=6):
    hs=set()
    with open(fp,"r",errors="ignore") as f:
        for ln in f:
            ln=ln.strip()
            if not ln:
                continue
            # 归一化：避免浮点写法差异
            parts = ln.split()
            try:
                parts = [f"{float(x):.{round_digits}f}" for x in parts]
                norm = " ".join(parts)
            except:
                norm = ln
            hs.add(hashlib.md5(norm.encode("utf-8")).hexdigest())
    return hs

base = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/B.subtilis"
for fn in ["mismatch.txt", "cksnap.txt", "tpcp.txt", "psetnc.txt", "rckmer.txt", "word2vec.txt"]:
    tr_h = line_hashes(f"{base}/train/{fn}")
    te_h = line_hashes(f"{base}/test/{fn}")
    inter = len(tr_h & te_h)
    print(fn, "train_unique:", len(tr_h), "test_unique:", len(te_h), "overlap:", inter, "overlap_ratio(test):", inter/max(1,len(te_h)))
import numpy as np

def load_y(fp):
    ys=[]
    with open(fp,"r") as f:
        for ln in f:
            ln=ln.strip()
            if ln!="":
                ys.append(int(float(ln)))
    return np.array(ys, dtype=np.float32)

def load_matrix(fp):
    rows=[]
    with open(fp,"r",errors="ignore") as f:
        for ln in f:
            ln=ln.strip()
            if not ln:
                continue
            try:
                rows.append([float(x) for x in ln.split()])
            except:
                pass
    return np.array(rows, dtype=np.float32)

base = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/B.subtilis/test"
y = load_y(f"{base}/y.txt")

for fn in ["cksnap.txt","tpcp.txt","psetnc.txt","rckmer.txt","mismatch.txt"]:
    X = load_matrix(f"{base}/{fn}")
    if X.shape[0] != y.shape[0]:
        print(fn, "skip shape mismatch", X.shape, y.shape)
        continue
    # 计算每列与 y 的相关性（粗查）
    corrs=[]
    for j in range(X.shape[1]):
        x = X[:,j]
        if np.std(x) < 1e-12:
            corrs.append(0.0)
        else:
            c = np.corrcoef(x, y)[0,1]
            corrs.append(abs(c))
    print(fn, "max |corr(feature,y)| =", max(corrs))
