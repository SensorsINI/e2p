import multiprocessing as mp
from multiprocessing import Process, Pipe

from producer import producer
from consumer import consumer
from utils.get_logger import get_logger

log = get_logger(__name__)


if __name__ == '__main__':
    mp.set_start_method('spawn') # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    parent_conn, child_conn = Pipe() # half duplex, only send and recieve on other side
    log.debug('starting consumer and producer processes')
    c = Process(target=consumer, args=(child_conn,),name='consumer')
    p = Process(target=producer, args=(parent_conn,),name='producer')
    c.start()
    p.start()
    log.debug('waiting for consumer and producer processes to join')
    c.join()
    p.join()
    log.debug('both consumer and producer processes have joined, done')
