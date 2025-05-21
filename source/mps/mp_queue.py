import posix_ipc
import pickle
import mmap
from multiprocessing.reduction import ForkingPickler

class SHMQueue():
    def __init__(self, name, create=False, chunk_size=10*1024*1024, nelem=4*1024):

        if name[0] != '/':
            name = '/' + name
        if not create:
            flags = 0
            self.creator = False
        else:
            flags = posix_ipc.O_CREX
            self.creator = True
        
        self.name = name
        self.chunk_size = chunk_size
        self.nelem = nelem
        self.open_name = []
        self.open_name.append(name+"_msgq_sem")
        self.open_name.append(name+"_stats_sem")
        self.open_name.append(name+"_msg_q")
        self.open_name.append(name + "_msg_mem")
        self.open_name.append(name + "_mem_stats")

        self.q_sem = posix_ipc.Semaphore(name+"_msgq_sem", flags, initial_value=1)
        self.chunk_stats_sem = posix_ipc.Semaphore(name+"_stats_sem", flags, initial_value=1)
        
        total_size = chunk_size * nelem
        self.msg_q = posix_ipc.MessageQueue(name+"_msg_q", flags)
        
        self.msg_data = posix_ipc.SharedMemory(name + "_msg_mem", flags, size=total_size)
        self.msg_data_fd = mmap.mmap(self.msg_data.fd, self.msg_data.size)
        # print('msg data size %s' % self.msg_data.size)
        self.msg_data.close_fd()

        self.chunk_stats = posix_ipc.SharedMemory(name + "_mem_stats", flags, size=nelem+1)
        self.chunk_stats_fd = mmap.mmap(self.chunk_stats.fd, self.chunk_stats.size)
        self.chunk_stats.close_fd()

        if create:
            # init status
            self.chunk_stats_fd.seek(0)
            self.chunk_stats_fd.write(b'0'*nelem + b"\0")
    
    def _assign_chunk(self,):
        self.chunk_stats_sem.acquire()
        self.chunk_stats_fd.seek(0)
        stats_bytes = self.chunk_stats_fd.read(self.nelem)
        stats_bytes = bytearray(stats_bytes)
        found = -1
        for i in range(len(stats_bytes)):
            if stats_bytes[i] != 49: # '1'
                found = i
                stats_bytes[i] = 49
                # write out new status
                self.chunk_stats_fd.seek(0)
                self.chunk_stats_fd.write(bytes(stats_bytes))
                break
        
        if found == -1:
            raise Exception('no memory available to store msg data')
        self.chunk_stats_sem.release()

        return found
    
    def _clean_chunk(self, idx):
        self.msg_data_fd.seek(idx*self.chunk_size)
        self.msg_data_fd.write(b'\0'*self.chunk_size)
        
        # clean stats
        self.chunk_stats_sem.acquire()
        self.chunk_stats_fd.seek(0)
        stats_bytes = self.chunk_stats_fd.read(self.nelem)
        stats_bytes = bytearray(stats_bytes)
        stats_bytes[idx] = 48 # '0' ascii

        self.chunk_stats_fd.seek(0)
        self.chunk_stats_fd.write(bytes(stats_bytes))
        self.chunk_stats_sem.release()

    def put(self, obj):
        # assign a free chunk to store the data
        obj_bytes = bytes(ForkingPickler.dumps(obj))
        assert len(obj_bytes) < self.chunk_size - 1
        chunk_idx = self._assign_chunk()
        self.msg_data_fd.seek(chunk_idx * self.chunk_size)
        self.msg_data_fd.write(obj_bytes + b'\0')

        # self.q_sem.acquire()
        self.msg_q.send(pickle.dumps((chunk_idx, len(obj_bytes))))
        # self.q_sem.release()
        # print('send complete')
    
    def get(self):
        # self.q_sem.acquire()
        msg_b, _ = self.msg_q.receive()
        # self.q_sem.release()
        # print('received')

        chunk_idx, data_size = pickle.loads(msg_b)
        self.msg_data_fd.seek(chunk_idx * self.chunk_size)
        data = self.msg_data_fd.read(data_size)

        # clean data
        self._clean_chunk(chunk_idx)
        return pickle.loads(data)

    def __del__(self):
        if hasattr(self, 'msg_q'):
            self.msg_q.close()

        if hasattr(self, 'msg_data_fd'):
            self.msg_data_fd.close()

        if hasattr(self, 'chunk_stats_fd'):
            self.chunk_stats_fd.close()
        
        if hasattr(self, 'q_sem'):
            self.q_sem.close()
        
        if hasattr(self, 'chunk_stats_sem'):
            self.chunk_stats_sem.close()
        
        if self.creator:
            for name in self.open_name:
                if 'mem' in name:
                    try:
                        posix_ipc.unlink_shared_memory(name)
                        print('unlinked %s' % name)
                    except:
                        pass
                elif 'sem' in name:
                    try:
                        posix_ipc.unlink_semaphore(name)
                        print('unlinked %s' % name)
                    except:
                        pass
                else:
                    try:
                        posix_ipc.unlink_message_queue(name)
                        print('unlinked %s' % name)
                    except:
                        pass

