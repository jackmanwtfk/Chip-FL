import struct
import pickle


def sendall(conn, data):
    body = pickle.dumps(data)
    header = struct.pack('>I', len(body))
    msg = header + body
    conn.sendall(msg)
    return len(msg)


def recvall(conn):
    raw_msglen = _recvall(conn, 4)
    if not raw_msglen:
        return None        
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg = _recvall(conn, msglen)
    return pickle.loads(msg), msglen    

def _recvall(conn, n: int):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
