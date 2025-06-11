import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
import time
import numpy as np

def producer(queue, producer_done, consumer_ready):
    """生产者进程：创建张量并放入队列，等待消费者完成"""
    print("\nProducer: Waiting for consumer to be ready...")
    consumer_ready.wait()  # 等待消费者准备好
    
    print("\nProducer: Creating tensors...")
    
    # 创建不同类型和形状的张量
    tensors = {
        'int_tensor': torch.randint(0, 100, (4096, 4096), device='cuda', dtype=torch.int32),
        'float_tensor': torch.randn(5, 2, device='cuda', dtype=torch.float32),
    }
    
    # 添加元数据
    metadata = {
        'created_at': time.time(),
        'description': 'Sample tensors with various dtypes and shapes'
    }
    
    # 打印生产者的张量信息
    print("\nProducer: Created tensors:")
    for key, tensor in tensors.items():
        print(f"  {key}: {tensor}")
    
    # 将张量和元数据放入队列
    print("\nProducer: Sending tensors to queue...")
    queue.put((tensors, metadata))
    
    # 等待消费者处理完成
    print("Producer: Waiting for consumer to finish processing...")
    producer_done.wait()
    print("Producer: Received completion signal. Exiting.")

def consumer(queue, producer_done, consumer_ready):
    """消费者进程：处理张量后通知生产者"""
    print("\nConsumer: Ready. Signaling producer to start...")
    consumer_ready.set()  # 通知生产者可以开始了
    
    start = time.time()
    print("\nConsumer: Waiting for tensors from queue...")
    
    # 从队列获取张量
    tensors, metadata = queue.get()
    
    # 打印元数据
    print(f"\nConsumer: Received metadata - {metadata['description']}")
    print(f"Consumer: Created at: {time.ctime(metadata['created_at'])}")
    
    # 处理接收到的张量
    print("\nConsumer: Processing tensors:")
    
    # 1. 整数张量
    int_tensor = tensors['int_tensor']
    
    # 2. 浮点张量
    float_tensor = tensors['float_tensor']
    
    end = time.time()
    print(f"restore tensor used {1000*(end-start)}:.2f milli seconds")
    
    
    # 通知生产者可以安全退出了
    print("\nConsumer: Finished processing. Signaling producer to exit...")
    producer_done.set()
    print("Consumer: Exiting.")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    print("Main: Starting multiprocessing demo with tensor sharing")
    
    # 创建队列和事件
    queue = Queue(maxsize=1)  # 限制队列大小为1
    producer_done = Event()   # 生产者完成事件
    consumer_ready = Event()  # 消费者就绪事件
    
    # 创建进程
    p_producer = Process(target=producer, args=(queue, producer_done, consumer_ready))
    p_consumer = Process(target=consumer, args=(queue, producer_done, consumer_ready))
    
    # 启动进程
    p_producer.start()
    p_consumer.start()
    
    # 等待进程完成
    p_producer.join()
    p_consumer.join()
    
    print("\nMain: All processes completed successfully")