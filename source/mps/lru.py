class CacheNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.pre = None


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = {}
        self.size = 0
        self.cap = capacity
        # two sentinel for the begining and the end of the
        # doubly linked list
        self.leastSentinel = CacheNode(None, None)
        self.recentSentinel = CacheNode(None, None)
        self.leastSentinel.next = self.recentSentinel
        self.recentSentinel.pre = self.leastSentinel

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """

        if key in self.cache:
            node = self.cache[key]
            self.to_recent(node)
            return node.value
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        evict_key = None
        if key in self.cache:
            # change the value of node
            node = self.cache[key]
            node.value = value
            self.to_recent(node)
            self.cache[key] = node
        else:
            if self.size == self.cap:
                evict_key = self.evicts()
            node = CacheNode(key, value)
            self.cache[key] = node
            self.to_recent(node)
            self.size += 1
        return evict_key

    def to_recent(self, node):
        """"""
        if node.pre and node.next:
            # it is a existing node
            node.pre.next = node.next
            node.next.pre = node.pre
        else:
            # it is a new node
            pass
        self.recentSentinel.pre.next = node
        node.pre = self.recentSentinel.pre
        node.next = self.recentSentinel
        self.recentSentinel.pre = node

    def evicts(self):
        """
        remove the next node of leastSentinel
        """
        if self.size == 0:
            return
        node_to_evict = self.leastSentinel.next
        # update the pointer
        self.leastSentinel.next = node_to_evict.next
        node_to_evict.next.pre = self.leastSentinel
        key = node_to_evict.key
        del self.cache[node_to_evict.key]
        del node_to_evict
        self.size -= 1
        return key

    def _debug_print(self):
        node = self.leastSentinel.next
        print(node.key, node.value)
        while node.next:
            node = node.next
            print(node.key, node.value)
