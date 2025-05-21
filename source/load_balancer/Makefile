CC=g++
CPPFLAGS=-lpthread
UTILS=../../source/utils/tcp/tcp.cpp ../../source/utils/tcp/tcp_pattern.cpp
DEPS=load_balancer.cpp metadata_manager.cpp client_agent.cpp cache_agent.cpp server_agent.cpp strategy/basic_strategy.cpp

all: load_balancer.out

%.out: $(DEPS)
	$(CC) -o $@ $(DEPS) $(UTILS) $(CPPFLAGS)

clean:
	rm *.out