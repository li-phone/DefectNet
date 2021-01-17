from concurrent.futures import ThreadPoolExecutor
import threading
import time

# 定义一个准备作为线程任务的函数
count = 100


def action(max):
    global count
    while count > 0:
        print(threading.current_thread().name + ',' + str(count), '\n')
        count -= 1
    return count


# 创建一个包含4条线程的线程池
with ThreadPoolExecutor(max_workers=4) as pool:
    # 使用线程执行map计算
    # 后面元组有3个元素，因此程序启动3条线程来执行action函数
    results = pool.map( , (50, 100, 150))
    print('--------------')
    for r in results:
        print(r)
