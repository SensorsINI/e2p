# runs entire PDAVIS E2P demo
# author: Tobi Delbruck 2023

import multiprocessing as mp
import time
from multiprocessing import Process, Pipe, Queue

from producer import producer
from consumer import consumer
from utils.get_logger import get_logger
from utils.kbhit import KBHit

log = get_logger(__name__)

def main():
    kb=None
    try:
        kb = KBHit()  # can only use in posix terminal; cannot use from spyder ipython console for example
        kbAvailable = True
    except:
        kbAvailable = False

    def print_help():
        print('x or ESC:  exit')
        # print('space: pause/unpause')
        # print('r toggle recording')

    mp.set_start_method('spawn') # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    parent_conn, child_conn = Pipe() # half duplex, only send and recieve on other side
    log.debug('starting PDAVIS demo consumer and producer processes')
    con = Process(target=consumer, args=(child_conn,),name='consumer')
    pro = Process(target=producer, args=(parent_conn,),name='producer')
    con.start()
    pro.start()
    log.debug('waiting for consumer and producer processes to join')
    if kbAvailable:
        print_help()
    while True:
        if kbAvailable and kb.kbhit():
            ch = kb.getch()
            if ch == 'h' or ch == '?':
                print_help()
            elif ord(ch) == 27 or ch == 'x':  # ESC, 'x'
                log.info("\nterminating producer and consumer....")
                con.terminate()
                pro.terminate()
                break
        time.sleep(.3)
    con.join()
    pro.join()
    log.debug('both consumer and producer processes have joined, done')
    quit(0)

if __name__ == '__main__':
    main()

