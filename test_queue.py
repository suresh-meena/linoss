import multiprocessing as mp
import time

def worker(q, gpu_id):
    q.put({"type": "total", "gpu_id": gpu_id, "value": 10})
    for i in range(10):
        time.sleep(0.1)
        q.put({"type": "update", "gpu_id": gpu_id})

if __name__ == "__main__":
    mp.set_start_method("spawn")
    q = mp.Queue()
    p1 = mp.Process(target=worker, args=(q, 0))
    p2 = mp.Process(target=worker, args=(q, 1))
    p1.start()
    p2.start()
    
    totals = {0: 0, 1: 0}
    progress = {0: 0, 1: 0}
    
    totals_received = 0
    while totals_received < 2:
        msg = q.get()
        if msg["type"] == "total":
            totals[msg["gpu_id"]] = msg["value"]
            totals_received += 1
            
    while p1.is_alive() or p2.is_alive() or not q.empty():
        while not q.empty():
            msg = q.get()
            if msg["type"] == "update":
                progress[msg["gpu_id"]] += 1
                
        # Simple text-based progress output
        print(f"\rGPU 0: [{progress[0]}/{totals[0]}] | GPU 1: [{progress[1]}/{totals[1]}]", end="")
        time.sleep(0.1)
        
    print() # Newline at the end
    p1.join()
    p2.join()
