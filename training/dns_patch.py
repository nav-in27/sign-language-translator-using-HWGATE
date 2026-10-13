"""
DNS Resolution Patch for environments where local router/ISP DNS drops or refuses queries.
Resolves hostnames using 8.8.8.8 / 1.1.1.1 via UDP socket directly.
"""
import socket
import struct
import sys

_orig_getaddrinfo = socket.getaddrinfo
_dns_cache = {}

def resolve_dns_udp(hostname: str, dns_servers=("8.8.8.8", "1.1.1.1")) -> list:
    if hostname in _dns_cache:
        return _dns_cache[hostname]
    
    # Don't query IP addresses
    parts = hostname.split('.')
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        return [hostname]
        
    tid = 0x2711
    flags = 0x0100
    header = struct.pack('>HHHHHH', tid, flags, 1, 0, 0, 0)
    qname = b''.join(bytes([len(p)]) + p.encode('ascii') for p in hostname.split('.')) + b'\x00'
    question = qname + struct.pack('>HH', 1, 1) # A, IN

    for server in dns_servers:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(3.0)
            sock.sendto(header + question, (server, 53))
            data, _ = sock.recvfrom(2048)
            sock.close()
            
            num_answers = struct.unpack('>H', data[6:8])[0]
            offset = 12 + len(question)
            answers = []
            for _ in range(num_answers):
                if offset >= len(data):
                    break
                if (data[offset] & 0xC0) == 0xC0:
                    offset += 2
                else:
                    while offset < len(data) and data[offset] != 0:
                        offset += 1 + data[offset]
                    offset += 1
                if offset + 10 > len(data):
                    break
                rtype, rclass, ttl, rdlength = struct.unpack('>HHIH', data[offset:offset+10])
                offset += 10
                if rtype == 1 and rdlength == 4:
                    ip = socket.inet_ntoa(data[offset:offset+4])
                    answers.append(ip)
                offset += rdlength
                
            if answers:
                _dns_cache[hostname] = answers
                return answers
        except Exception:
            continue
    return []

def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        ips = resolve_dns_udp(host)
        if ips:
            for ip in ips:
                try:
                    return _orig_getaddrinfo(ip, port, family, type, proto, flags)
                except Exception:
                    continue
        raise

socket.getaddrinfo = patched_getaddrinfo
