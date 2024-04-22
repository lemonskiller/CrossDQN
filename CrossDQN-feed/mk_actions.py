

def generate_lists(limit_num, max_len):
    if limit_num > max_len:
        return []

    def backtrack(curr_list, num_ones):
        if len(curr_list) == max_len:
            if num_ones == limit_num:
                results.append(curr_list.copy())
            return

        curr_list.append(0)
        backtrack(curr_list, num_ones)
        curr_list.pop()

        if num_ones < limit_num:
            curr_list.append(1)
            backtrack(curr_list, num_ones + 1)
            curr_list.pop()

    results = []
    curr_list = []
    backtrack(curr_list, 0)
    return results


def generate_list(max_len, max_num, float_num=2):
    my_list = [round(max_num * i / max_len, float_num) for i in range(max_len)]
    return my_list


if __name__ == '__main__':
    # --- make actions ---
    lists = generate_lists(5, 14)
    for i in range(len(lists)):
        lists[i] = [0]+lists[i]
    print(len(lists))
    print(lists)

    # --- make values ---
    # print(generate_list(2002, 1))
