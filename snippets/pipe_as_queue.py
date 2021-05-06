"""This short test demonstrates that pipes can be used as queues between 2
processes. All messages are sent by the parent process (without blocking)
before any are received by the child process.
"""


import multiprocessing as mp
import time

def f(pipe: mp.Pipe):
    time.sleep(1)

    while pipe.poll(timeout=0.01):
        message = pipe.recv()
        print(message)

def main():
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=f, args=(child_conn,))
    p.start()

    for i in range(100):
        parent_conn.send(i)
    print("all messages sent")
    p.join()

if __name__ == "__main__":
    main()