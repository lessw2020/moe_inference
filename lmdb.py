# Example of LMDB's read performance
import lmdb
import time

env = lmdb.open("./test.db", map_size=1024*1024*1024)  # 1GB map size

# Write some test data
with env.begin(write=True) as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")

# Read performance test
with env.begin() as txn:
    start = time.time()
    for _ in range(1000000):
        value = txn.get(b"key1")  # Very fast memory-mapped read
    end = time.time()
    print(f"Million reads/sec: {1000000/(end-start):.2f}")
