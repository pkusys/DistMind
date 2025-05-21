#ifndef BALANCE_SERVER_CORE_LOOP_H
#define BALANCE_SERVER_CORE_LOOP_H

#include <unistd.h>

#include <string>

bool onceHandleTask();
bool onceRecv();
bool onceCopy();

void loopHandleTask();
void loopRecv();
void loopCopy();

void createLoops(std::string cache_addr, int cache_port);
void destroyLoops();

void enqueueTask(std::string model_name);
std::pair<char*, size_t> dequeueModelInfo();
void enqueueParamGpuMemory(char* d_ptr, size_t size);
size_t dequeueParamCompletion();

#endif