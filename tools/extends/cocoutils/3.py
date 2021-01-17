import threading

balance = 0
main_thread_lock = threading.Lock()  # 这里是在主线程中设置的一把线程锁


# 获得锁 main_thread_lock.acquire() 这个方法或阻塞当前线程，直到获得一把锁。但是不会阻塞主线程
# 释放锁 main_thread_lock.release() 此时其他的方法可以去申请锁的使用
# 一般使用try_catch_finally 来确保锁被释放

def data_operator(n):
    global balance  # 表明在这个函数中，balance 是使用的全局的
    balance += n
    balance -= n


def change_it(n):
    for item in range(0, 10 ** 6):
        with main_thread_lock:  # 这里是在操作主要数据的时候，获得锁，在执行完操作之后将锁释放，这里使用with 语句更加简洁的实现，一定要注意锁的释放
            data_operator(n)


def thread_synchronization():
    t1 = threading.Thread(target=change_it, args=(5,))
    t2 = threading.Thread(target=change_it, args=(8,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    global balance
    print(balance)


thread_synchronization()
