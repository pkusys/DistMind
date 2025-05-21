#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <netinet/tcp.h>

#include <string>
#include <memory>
#include <iostream>

#include "utils/common/errno.h"

#include "tcp.h"


namespace balance {
namespace util {

using namespace std;

TcpServer::TcpServer(string address, int port, int listen_num):
_address(address), _port(port) {
    _server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_server_fd == 0) { 
        perror("util/common/TCPServer CreateSocket");
        exit(EXIT_FAILURE); 
    } 
       
    int opt = 1;
    if (setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) { 
        perror("util/common/TCPServer SetSocketOption");
        exit(EXIT_FAILURE); 
    }
    int yes = 1;
    if (setsockopt(_server_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes))) { 
        perror("util/common/TCPServer SetSocketOption");
        exit(EXIT_FAILURE); 
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET; 
    addr.sin_port = htons(_port);
    if(inet_pton(AF_INET, _address.c_str(), &addr.sin_addr) <= 0) { 
        perror("util/common/TCPServer ConvertStringAddress");
        exit(EXIT_FAILURE);
    }
    if (bind(_server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { 
        perror("util/common/TCPServer BindPort");
        exit(EXIT_FAILURE); 
    } 
    if (listen(_server_fd, listen_num) < 0) {
        perror("util/common/TCPServer Listen");
        exit(EXIT_FAILURE); 
    } 
}

TcpServer::~TcpServer() {
    close(_server_fd);
}

shared_ptr<TcpAgent> TcpServer::tcpAccept() {
    int conn_fd;
    struct sockaddr_in addr;
    size_t addrlen = sizeof(addr);
    conn_fd = accept(_server_fd, (struct sockaddr *)&addr, (socklen_t*)&addrlen);
    if (conn_fd < 0) {
        perror("util/common/TCPServer AcceptConnection");
        exit(EXIT_FAILURE); 
    }

    return shared_ptr<TcpAgent>(new TcpAgent(conn_fd));
}

TcpAgent::TcpAgent(int conn_fd): _conn_fd(conn_fd) { }

TcpAgent::~TcpAgent() {
    shutdown(_conn_fd, SHUT_RDWR);
    close(_conn_fd);
}

int TcpAgent::tcpSend(const char* data, size_t size) {
    // Define a maximum chunk size for large data transfers
    const size_t MAX_CHUNK_SIZE = 8 * 1024 * 1024; // 8MB chunks
    
    size_t total_sent = 0;
    while (total_sent < size) {
        // Determine the size of this chunk
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, size - total_sent);
          size_t chunk_sent = 0;
        while (chunk_sent < chunk_size) {
            // cout << "tcpSend: " << total_sent + chunk_sent << ", " << size << endl;
            ssize_t ret = send(_conn_fd, data + total_sent + chunk_sent, chunk_size - chunk_sent, 0);
            if (ret == -1) {
                // Check if error is retryable
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    // These errors are temporary - we can retry
                    cout << "Temporary send error (errno=" << errno << "), retrying..." << endl;
                    continue;
                }
                cout << "Send error" << ", " << ret << ", " << size << ", errno=" << errno << endl;
                return ERRNO_TCP;
            }
            chunk_sent += ret;
        }
        total_sent += chunk_sent;
    }
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpRecv(char* data, size_t size) {
    // Define a maximum chunk size for large data transfers
    const size_t MAX_CHUNK_SIZE = 8 * 1024 * 1024; // 8MB chunks
    
    size_t total_received = 0;
    while (total_received < size) {
        // Determine the size of this chunk
        size_t chunk_size = std::min(MAX_CHUNK_SIZE, size - total_received);
        
        size_t chunk_received = 0;
        while (chunk_received < chunk_size) {
            // cout << "tcpRecv: " << total_received + chunk_received << ", " << size << endl;
            ssize_t ret = recv(_conn_fd, data + total_received + chunk_received, chunk_size - chunk_received, 0);
            if (ret == 0) {
                // Connection closed by peer
                std::cout << "Connection closed by peer" << std::endl;
                return ERRNO_TCP;
            } else if (ret < 0) {
                // Check if error is retryable
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    // These errors are temporary - we can retry
                    cout << "Temporary receive error (errno=" << errno << "), retrying..." << endl;
                    continue;
                }
                std::cout << "Recv error, ret=" << ret << ", total=" << total_received 
                          << ", expected=" << size << ", errno=" << errno << std::endl;
                return ERRNO_TCP;
            }
            chunk_received += ret;
        }
        total_received += chunk_received;
    }
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpSendWithLength(const char* data, size_t size) {
    int status;
    status = tcpSend((char*)&size, sizeof(size));
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    status = tcpSend(data, size);
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpSendWithLength(const shared_ptr<char> data, size_t size) {
    int status;
    status = tcpSend((char*)&size, sizeof(size));
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    status = tcpSend(data.get(), size);
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpRecvWithLength(shared_ptr<char> &data, size_t &size) {
    // cout << "tcpRecvWithLength (1): " << sizeof(size) << endl;
    int status;

    status = tcpRecv((char*)&size, sizeof(size));
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    //if (size > 10000000)
    //    return ERRNO_TCP;
    // cout << "tcpRecvWithLength (2): " << size << endl;
    data.reset((char*)malloc(size));
    // cout << "tcpRecvWithLength (3)" << endl;
    status = tcpRecv(data.get(), size);
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    // cout << "tcpRecvWithLength (4)" << endl;
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpSendString(const string data) {
    int status;
    size_t data_size = data.size();
    status = tcpSend((char*)&data_size, sizeof(data_size));
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    status = tcpSend(data.c_str(), data_size);
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    return ERRNO_SUCCESS;
}

int TcpAgent::tcpRecvString(string &data) {
    int status;

    size_t data_size = 0;
    // cout << "tcpRecvString (1): " << sizeof(data_size) << endl;
    status = tcpRecv((char*)&data_size, sizeof(data_size));
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    // if (data_size > 1000)
    //    return ERRNO_TCP;
    // cout << "tcpRecvString (2): " << data_size << endl;
    char* buffer = (char*)malloc(data_size);
    // cout << "tcpRecvString (3)" << endl;
    memset(buffer, 0, data_size);
    // cout << "tcpRecvString (4)" << endl;
    status = tcpRecv(buffer, data_size);
    if (status != ERRNO_SUCCESS)
        return ERRNO_TCP;
    // cout << "tcpRecvString (5)" << endl;
    data = string(buffer, data_size);
    // cout << "tcpRecvString (6)" << endl;
    free(buffer);
    // cout << "tcpRecvString (7)" << endl;
    return ERRNO_SUCCESS;
}

TcpClient::TcpClient(string address, int port):
TcpAgent(0), _address(address), _port(port) {
    _conn_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_conn_fd == 0) { 
        perror("util/common/TCPClient CreateSocket");
        exit(EXIT_FAILURE);
    }

    int yes = 1;
    if (setsockopt(_conn_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes))) { 
        perror("util/common/TCPServer SetSocketOption");
        exit(EXIT_FAILURE); 
    }

    struct sockaddr_in serv_addr; 
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(_port);
    if(inet_pton(AF_INET, _address.c_str(), &serv_addr.sin_addr) <= 0) { 
        perror("util/common/TCPClient ConvertStringAddress");
        exit(EXIT_FAILURE);
    }
    if (connect(_conn_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) { 
        perror("util/common/TCPClient MakeConnection");
        exit(EXIT_FAILURE);
    } 
}

TcpClient::~TcpClient() {
    ;
}

} //namespace util
} //namespace balance