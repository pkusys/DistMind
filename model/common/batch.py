def generate_batch_basic(func_list, size_list=[], max_batch_size=8 * 4096000):
    print (len(func_list))
    batch_list = [(0, 0)]
    ptr = 1
    def get_param_size(left, right):
        total_size = 0
        for _, _, param, _, _, _ in func_list[left: right]:
            for _, v in param.items():
                size = v.nelement() * v.element_size()
                padded_size = (size + 511) // 512 * 512
                total_size += padded_size
        return total_size
    while ptr <= len(func_list):
        if len(batch_list) - 1 < len(size_list):
            threshold = size_list[len(batch_list) - 1]
        else:
            # default threshold 8 * 4096000
            threshold = max_batch_size
        
        while True:
            if ptr >= len(func_list):
                break
            if get_param_size(batch_list[-1][0], ptr) > 0 and \
                get_param_size(ptr, ptr + 1) > 0 and \
                get_param_size(batch_list[-1][0], ptr + 1) > threshold:
                break
            ptr += 1

        total_size = get_param_size(batch_list[-1][0], ptr)
        batch_list.append((ptr, total_size))
        print (ptr, func_list[ptr - 1][0], total_size)
        ptr += 1

    return batch_list