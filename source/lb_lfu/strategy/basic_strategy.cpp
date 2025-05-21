#include <unistd.h>

#include <string>
#include <memory>

#include "basic_strategy.h"

using namespace std;

const int SERVER_BUSY_THRESHOLD = 2;
const int SERVER_SATURATED_THRESHOLD = 4;

BasicStrategy _bs;

ServerSelection BasicStrategy::calculate(shared_ptr<MetadataManager> metadata, string model_name) {
    auto cache_location = metadata->get_cache_location();
    auto workload = metadata->get_workload();

    for (int server_ip : cache_location[model_name]) {
        for (auto itr_port = workload[server_ip].begin(); itr_port != workload[server_ip].end(); ++itr_port) {
            if (itr_port->second == 0) {
                cout << "Select: Cache and workload 0" << endl;
                return ServerSelection(server_ip, itr_port->first);
            }
        }
    }

    for (auto itr_ip = workload.begin(); itr_ip != workload.end(); ++itr_ip) {
        for (auto itr_port = itr_ip->second.begin(); itr_port != itr_ip->second.end(); ++itr_port) {
            if (itr_port->second == 0) {
                cout << "Select: Workload 0" << endl;
                return ServerSelection(itr_ip->first, itr_port->first);
            }
        }
    }

    for (int server_ip : cache_location[model_name]) {
        for (auto itr_port = workload[server_ip].begin(); itr_port != workload[server_ip].end(); ++itr_port) {
            if (itr_port->second < SERVER_BUSY_THRESHOLD) {
                cout << "Select: Cache and workload threshold" << endl;
                return ServerSelection(server_ip, itr_port->first);
            }
        }
    }

    int min_workload = SERVER_SATURATED_THRESHOLD;
    int ip_int = 0, port = 0;
    for (auto itr_ip = workload.begin(); itr_ip != workload.end(); ++itr_ip) {
        for (auto itr_port = itr_ip->second.begin(); itr_port != itr_ip->second.end(); ++itr_port) {
            if (itr_port->second < min_workload) {
                ip_int = itr_ip->first;
                port = itr_port->first;
                min_workload = itr_port->second;
            }
        }
    }
    cout << "Select: Minimal workload" << ", " << min_workload << endl;
    return ServerSelection(ip_int, port);
}