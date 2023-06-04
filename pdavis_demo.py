# runs entire PDAVIS E2P demo
# author: Tobi Delbruck 2023

import multiprocessing as mp
import time
from multiprocessing import Process, Pipe, Queue, SimpleQueue

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
    queue=Queue()
    # parent_conn, child_conn = Pipe() # half duplex, only send and recieve on other side
    log.debug('starting PDAVIS demo consumer process')
    con = Process(target=consumer, args=(queue,),name='consumer')
    con.start()
    time.sleep(8) # give some time to load E2P
    log.debug('starting PDAVIS demo producer process')
    pro = Process(target=producer, args=(queue,),name='producer')
    pro.start()
    if kbAvailable:
        print_help()
    while True:
        # log.debug('waiting for consumer and producer processes to join')
        if kbAvailable and kb.kbhit():
            print('.',end='')
            ch = kb.getch()
            if ch == 'h' or ch == '?':
                print_help()
            elif ord(ch) == 27 or ch == 'x':  # ESC, 'x'
                log.info("\nterminating producer and consumer....")
                pro.terminate()
                con.terminate()
                break
        if not con.is_alive() or not pro.is_alive():
            log.info('either or both producer or consumer process(es) ended, terminating pdavis_demo loop')
            if pro.is_alive():
                log.debug('terminating producer')
                pro.terminate()
            if con.is_alive():
                log.debug('terminating consumer')
                con.terminate()
            break
        time.sleep(.3)
    log.debug('joining producer and consumer processes')
    pro.join()
    con.join()
    log.debug('both consumer and producer processes have joined, done')
    # quit(0)

if __name__ == '__main__':
    main()

