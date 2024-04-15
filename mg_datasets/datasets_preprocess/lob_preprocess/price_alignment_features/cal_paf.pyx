# -*- coding: utf-8 -*-
# @author : RenKai
# @time   : 2023/11/16 16:38

""" Price-Alignment-Features (PAF) 算法

对外提供 Bid&Ask 两个方向的特征计算接口, 以便直接调用
    - `cal_bid_paf_mat()`: bid 方向的 paf matrix
    - `cal_ask_paf_mat()`: ask 方向的 paf matrix

PAF 算法的核心是如下三大函数:
    - `merge_price_range()`: 得到价格区间
    - `gen_paf_mat()`: 构建 paf 矩阵算法
    - `fill_nan_in_paf_mat()`: 填充 paf 矩阵中的空值算法

"""

import numpy as np
cimport numpy as cnp

DEF ASK_ZERO_PRICE = (2 ** 31 - 1)  # 用极大值来表示 Ask 方的 0
DEF NAN_FEATURE_VALUE = -10  # 用 -10 表示 Feature 中的 Nan

def merge_price_range(cnp.ndarray[cnp.int32_t, ndim=1] last_price,
                      cnp.ndarray[cnp.int32_t, ndim=1] now_price,
                      bint is_bid,
                      int period_num,
                      int level_num):
    """ 对两个价格序列进行归并 (求两个集合且保持有序)

    :param last_price: 上一期价格, 注意其 shape 是不断变化的
    :param now_price: 当期价格, shape=(l)
    :param is_bid: bid or ask direction
    :param period_num: 期数
    :param level_num: 档数

    return:
        - price_range: 归并后的价格区间 shape = (1) to (period_num * level_num) => price_range_num
        - price_range_num: 归并后的有效价格数量

    NOTE:
        price 数组必须具有如下 3 个特点, 才能使用本:
            - 如果 is_bid == True 表示委托`买`, 则为从大到小排序, 不足以 0 填充
            - 如果 is_bid == False 表示委托`卖`, 则为从小到大排序, 不足以 0 填充
            - price 除 0 可以重复以外, 其他的价格数字不可能重复 (从交易所拿到的默认 LOB)
        last_price 和 now_price 不可以交换

    """

    # 构建一个定长的 price_range, 用于存放归并后的结果
    cdef cnp.ndarray[cnp.int32_t, ndim=1] price_range = np.zeros(period_num * level_num, dtype=np.int32)
    cdef int price_range_num  # 记录归并后的有效价格数量
    cdef bint flag_zero = False  # 是否取到 0 的标记
    cdef int i = 0, i_last = 0, i_now = 0  # 迭代下标
    # ---- bid situation : 直接归并即可 ---- #
    if is_bid:
        while i_last < last_price.shape[0]:  # 在 last price 数组中做遍历, 由于 last_price 不定长因此只能取 shape[0]
            if i_now < level_num:  # 情况 1. 两个数组均未遍历完成, now_price.shape[0] 是定值 level_num
                if last_price[i_last] > now_price[i_now]:  # 取较大的数 last price 放入 price_range 中
                    # 注意: 这种情况不可能取到 0, 因此无需额外判断
                    price_range[i] = last_price[i_last]
                    i += 1
                    i_last += 1
                elif now_price[i_now] > last_price[i_last]:  # 取较大的数 now price 放入 price_range 中
                    # 注意: 这种情况不可能取到 0, 因此无需额外判断
                    price_range[i] = now_price[i_now]
                    i += 1
                    i_now += 1
                else:  # 二者相等, 取 last price 并且指针同时后移
                    price_range[i] = last_price[i_last]  # == now_price[i_now]
                    # 注意: 这种情况可能取到 0, 取到 0 之后就说明 price_range 已经构造完成了
                    if price_range[i] == 0:
                        i += 1
                        flag_zero = True
                        break
                    # 非 0 则继续取
                    i += 1
                    i_last += 1
                    i_now += 1
            else:  # 情况 2. now 数组已经遍历完成, 将 last 中的剩余数字都取到
                price_range[i] = last_price[i_last]
                # 注意: 这种情况可能取到 0, 取到 0 之后就说明 price_range 已经构造完成了
                if price_range[i] == 0:
                    i += 1
                    flag_zero = True
                    break
                # 非 0 则继续取
                i += 1
                i_last += 1
        # 情况 3. last 数组已经遍历完成, 将 now 数组中剩余的数字都取到, now_price.shape[0] 是定值 level_num
        while not flag_zero and i_now < level_num:
            # 注意: 若上述两种情况已经取到了 0 即 flag_zero 为 True, 该情况直接忽略
            price_range[i] = now_price[i_now]
            # 若进入该情况，有可能取到 0, 取到 0 之后说明 price_range 已经构造完成了
            if price_range[i] == 0:
                i += 1
                break
            # 非 0 则继续取
            i += 1
            i_now += 1
        price_range_num = i
    # ---- ask situation : 需要先将价格 0 转化为 ASK_PRICE_ZERO, 然后再归并 ---- #
    else:
        # fill last price
        i = last_price.shape[0] - 1
        while i >= 0:
            if last_price[i] == 0:
                last_price[i] = ASK_ZERO_PRICE
                i -= 1
            else:
                break
        # fill now price, now_price.shape[0] 是定值 level_num
        i = level_num - 1
        while i >= 0:
            if now_price[i] == 0:
                now_price[i] = ASK_ZERO_PRICE
                i -= 1
            else:
                break
        # 归并
        i = 0
        while i_last < last_price.shape[0]:  # 在 last price 数组中做遍历
            if i_now < level_num:  # 情况 1. 两个数组均未遍历完成, now_price.shape[0] 是定值 level_num
                if last_price[i_last] < now_price[i_now]:  # 取较小的数 last price 放入 price_range 中
                    # 注意: 这种情况不可能取到 ASK_ZERO_PRICE, 因此无需额外判断
                    price_range[i] = last_price[i_last]
                    i += 1
                    i_last += 1
                elif now_price[i_now] < last_price[i_last]:  # 取较小的数 now price 放入 price_range 中
                    # 注意: 这种情况不可能取到 ASK_ZERO_PRICE, 因此无需额外判断
                    price_range[i] = now_price[i_now]
                    i += 1
                    i_now += 1
                else:  # 二者相等, 取 last price 并且指针同时后移
                    price_range[i] = last_price[i_last]  # = now_price[i_now]
                    # 注意: 这种情况可能取到 ASK_ZERO_PRICE, 取到就说明 price_range 已经构造好了
                    if price_range[i] == ASK_ZERO_PRICE:
                        i += 1
                        flag_zero = True
                        break
                    i += 1
                    i_last += 1
                    i_now += 1
            else:  # 情况 2. now 数组已经遍历完成, 将 last 中的剩余数字都取到
                price_range[i] = last_price[i_last]
                # 注意: 这种情况可能取到 ASK_ZERO_PRICE, 取到就说明 price_range 已经构造好了
                if price_range[i] == ASK_ZERO_PRICE:
                    i += 1
                    flag_zero = True
                    break
                # 非 ASK_ZERO_PRICE 则继续取
                i += 1
                i_last += 1
        # 情况 3. last 数组已经遍历完成, 将 now 数组中剩余的数字都取到, now_price.shape[0] 是定值 level_num
        while not flag_zero and i_now < level_num:
            # 注意: 若上述两种情况已经取到了 ASK_ZERO_PRICE 即 flag_zero 为 True, 该情况直接忽略
            price_range[i] = now_price[i_now]
            # 若进入该情况，有可能取到 ASK_ZERO_PRICE, 取到 ASK_ZERO_PRICE 之后说明 price_range 已经构造完成了
            if price_range[i] == ASK_ZERO_PRICE:
                i += 1
                break
            # 非 ASK_ZERO_PRICE 则继续取
            i += 1
            i_now += 1
        price_range_num = i

    # ---- 切分有意义部分的数据 + 返回 ---- #
    return price_range[:price_range_num], price_range_num

def gen_paf_mat(cnp.ndarray[cnp.int32_t, ndim=2] price, cnp.ndarray[double, ndim=3] feature, bint is_bid):
    """ 构建 price-alignment-features 矩阵

    :param price: t 期, l 档的委托价格 shape=(t, l)
    :param feature: t 期, l 档的 f 个对应特征 shape=(t, l, f)
    :param is_bid: bid or ask directtion

    return:
        - price_range : ndarray(dtype=cnp.int32_t, shape=price_range_num), t 期的价格区间序列
        - paf_mat: ndarray(dtype=double, shape=(t, price_range_num, l)), 与价格区间对齐的 paf 矩阵

    """

    # ---- Def the iter variables ---- #
    cdef int t_i, p_i, f_i

    # ---- Get the period_num, level_num and feature_num ---- #
    cdef int period_num = price.shape[0]  # == feature.shape[0]
    cdef int level_num = price.shape[1]  # == feature.shape[1]
    cdef int feature_num = feature.shape[2]  # the feature number

    # ---- Step 1. Get the price range of t periods from price array ---- #
    cdef cnp.ndarray[cnp.int32_t, ndim=1] price_range = price[0]  # set the first price to price_range
    cdef int price_range_num = level_num  # init the price range number
    # 遍历每一个 period, 归并出最终的 price range
    t_i = 1
    while t_i < period_num:
        price_range, price_range_num = merge_price_range(last_price=price_range,
                                                         now_price=price[t_i],
                                                         is_bid=is_bid,
                                                         period_num=period_num,
                                                         level_num=level_num)
        t_i += 1

    # ---- Step 2. Define the empty paf mat and set value ---- #
    # 全部初始化为 0, shape=(t, price_range_num, f)
    cdef cnp.ndarray[double, ndim=3] paf_mat = np.zeros((period_num, price_range_num, feature_num))
    # 在 price_range 中做大循环, 管理 p_i 定位价格区间的位置, 管理一个 all_period_i 的指针数组定位在每一期上的价格位置
    p_i = 0
    cdef cnp.ndarray[cnp.int_t, ndim= 1] all_period_i = np.zeros(period_num, dtype=int)
    while p_i < price_range_num:  # 在整个价格区间上做大循环
        t_i = 0
        while t_i < period_num:  # 在所有时期的价格上做小循环, 管理每个时期的价格前进指针
            # t_i 是时期, all_period_i[t_i] 是该时期上目前价格判断的位置
            if all_period_i[t_i] < level_num and price_range[p_i] == price[t_i, all_period_i[t_i]]:
                # for loop to set value
                f_i = 0
                while f_i < feature_num:
                    paf_mat[t_i, p_i, f_i] = feature[t_i, all_period_i[t_i], f_i]
                    f_i += 1
                all_period_i[t_i] += 1
            elif all_period_i[t_i] == level_num:
                # for loop to set nan
                f_i = 0
                while f_i < feature_num:
                    paf_mat[t_i, p_i, f_i] = NAN_FEATURE_VALUE
                    f_i += 1
            t_i += 1
        p_i += 1

    return price_range, paf_mat

cpdef cnp.ndarray[double, ndim=3] fill_nan_in_paf_mat(cnp.ndarray[double, ndim=3] paf_mat, str fill_nan_option):
    """ 填充 price alignment feature 矩阵中的空值 (NAN_FEATURE_VALUE)

    :param paf_mat: ndarray(dtype=double, shape=(t, price_range_num, f)), 与价格区间对齐的 paf 矩阵
    :param fill_nan_option: 填充空值的方式, 有三种选择:
        - `cut_off`: 直接 drop
        - `ffill_and_cut_off`: 先 ffill 然后再 drop
        - `ffill`: 只进行 ffill
        - `ffill_and_note_steps`: ffill 的同时记录 ffill 的步数 `ffill_steps` (这种方式会在 f 维度上增加一层) 

    return:
        - paf_mat_fill_nan : ndarray(dtype=double, shape=(t, `not_sure`, f)), 填充空值后的 paf mat
        如果使用 `ffill_and_note_steps` 填充法, 那么:
        - paf_mat_fill_nan : ndarray(dtype=double, shape=(t, `not_sure`, f + 1)), +1 的部分为 `ffill_steps`.

    """

    cdef int t_i, pr_i, f_i, no_nan_value_i
    cdef int period_num = paf_mat.shape[0], price_range_num = paf_mat.shape[1], feature_num = paf_mat.shape[2]
    cdef bint has_nan_value
    cdef bint first_is_nan_value, behind_has_nan_value
    # 定义带有 ffill_steps 特征的 paf_mat, 专门为 `ffill_and_note_steps` 而定义
    cdef cnp.ndarray[double, ndim=3] ffill_steps_paf_mat = np.zeros((period_num, price_range_num, feature_num + 1))

    # ---- Way 1. 直接 drop 有空值的数值行 ---- #
    if fill_nan_option == "cut_off":
        pr_i = price_range_num - 1  # 遍历价格的 iter
        # 反向遍历价格区间
        while pr_i > 0:
            has_nan_value = False
            # 遍历检测这一个价格对应的特征值中是否含有 NAN_FEATURE_VALUE (双重循环遍历, 尽可能尽早 break)
            t_i = 0
            while t_i < period_num:
                # 因为所有 feature 有无 nan value 是统一的, 所以只需要对第一个 feature 进行判断
                if paf_mat[t_i, pr_i, 0] == NAN_FEATURE_VALUE:
                    has_nan_value = True
                    break
                t_i += 1
            # 如果对应的特征值中没有 NAN_FEATURE_VALUE 就说明到底了, 把这一行取出来
            if not has_nan_value:
                no_nan_value_i = pr_i
                break
            # 如果对应的特征值中有 NAN_FEATURE_VALUE, 说明还没到底,仍需继续
            pr_i -= 1
        return paf_mat[:, :no_nan_value_i + 1, :]

    # ---- Way 2. 先 ffill 然后再 drop 有空值的数值行 ---- #
    elif fill_nan_option == "ffill_and_cut_off":
        pr_i = price_range_num - 1  # 遍历价格的 iter
        # 反向遍历价格区间, 对每一个价格区间都进行填空操作
        while pr_i > 0:
            first_is_nan_value = False
            behind_has_nan_value = False
            # 分析后很容易发现: ffill 后如果 paf_mat 中还有空值,
            # 那么某个价格下对应的第一个时期的特征一定有空值。所以先对第一个时期进行空值检测,
            # 如果为空, 不进行 ffill 操作, 而是直接跳到下一个价格下 (就算 fill 了也会 cut_off 掉)
            if paf_mat[0, pr_i, 0] == NAN_FEATURE_VALUE:
                first_is_nan_value = True
            if not first_is_nan_value:
                # 如果第一个值不为空, 并且上一个 price 对应的特征值为空, 那就说明其是第一个全部不会为 nan 的价格
                if pr_i == price_range_num - 1:
                    no_nan_value_i = pr_i
                elif paf_mat[0, pr_i + 1, 0] == NAN_FEATURE_VALUE:
                    no_nan_value_i = pr_i
                # 如果第一个时期空值检测后不为空, 进行 ffill 操作才有意义
                t_i = 1
                while t_i < period_num:
                    # 因为所有 feature 有无 nan value 是统一的, 所以只需要对第一个 feature 进行判断
                    if paf_mat[t_i, pr_i, 0] == NAN_FEATURE_VALUE:
                        # 检测后续有空值
                        behind_has_nan_value = True
                        # 如果有空值, 就进行 ffill 操作
                        f_i = 0
                        while f_i < feature_num:
                            paf_mat[t_i, pr_i, f_i] = paf_mat[t_i - 1, pr_i, f_i]
                            f_i += 1
                    t_i += 1
                # 注意: 填充完成后, 这一个价格对应的 feature 就不可能有空值了
            # 如果对应的特征值中没有 NAN_FEATURE_VALUE 了就说明后续的都不可能有空了, 也就不需要继续填了, 直接 break
            if not first_is_nan_value and not behind_has_nan_value:
                break
            # 否则如果有空值说明后续可能还继续会存在空值, 还需要继续向下走
            pr_i -= 1
        return paf_mat[:, :no_nan_value_i + 1, :]

    # ---- Way 3. 仅进行 ffill ---- #
    elif fill_nan_option == "ffill":
        pr_i = price_range_num - 1  # 遍历价格的 iter
        # 反向遍历价格区间, 对每一个价格区间都进行 ffill 填空操作
        while pr_i > 0:
            # 正向遍历时间步进行填充, 从 t = 1 开始
            t_i = 1
            behind_has_nan_value = False  # 初始化除第一期以外含有 nan 的标识 behind_has_nan_value 为 False
            while t_i < period_num:
                if paf_mat[t_i, pr_i, 0] == NAN_FEATURE_VALUE:
                    behind_has_nan_value = True  # 检测到空值, 直接赋值
                    # 遍历每一个特征进行填充, 只要有一个为空那所有的都应该是空
                    f_i = 0
                    while f_i < feature_num:
                        paf_mat[t_i, pr_i, f_i] = paf_mat[t_i - 1, pr_i, f_i]
                        f_i += 1
                t_i += 1
            # 如果对应该时间点的特征值中没有 NAN_FEATURE_VALUE 了就说明后续的都不可能有空了, 也就不需要继续填了, 直接 break
            if not behind_has_nan_value:
                break
            pr_i -= 1
        return paf_mat

    # ---- Way 4. 在进行 ffill 的同时, 记录 ffill steps ---- #
    elif fill_nan_option == "ffill_and_note_steps":
        ffill_steps_paf_mat[:, :, :feature_num] = paf_mat
        # 定义反向遍历价格的 iter, 基于此 iter 进行 paf_mat 的反向遍历
        pr_i = price_range_num - 1
        # 反向遍历价格区间, 对每一个价格区间都进行 ffill 填空操作
        while pr_i > 0:
            t_i = 1  # 正向遍历时间步进行填充, 从 t = 1 开始
            behind_has_nan_value = False  # 初始化除第一期以外含有 nan 的标识 behind_has_nan_value 为 False
            ffill_steps = 0  # 定义 ffill steps 为 0
            while t_i < period_num:
                if ffill_steps_paf_mat[t_i, pr_i, 0] == NAN_FEATURE_VALUE:  # 检测到空值
                    behind_has_nan_value = True  # 让含 nan 标记为 True
                    f_i = 0  # 遍历每一个特征进行填充, 只要有一个为空那所有的都应该是空
                    while f_i < feature_num:  # 完成具有实际意义的特征 ffill
                        ffill_steps_paf_mat[t_i, pr_i, f_i] = ffill_steps_paf_mat[t_i - 1, pr_i, f_i]
                        f_i += 1
                    ffill_steps += 1  # 完成一步填充 ffill steps + 1
                    ffill_steps_paf_mat[t_i, pr_i, feature_num] = ffill_steps  # ffill steps 赋值
                else:  # 检测到非空值
                    ffill_steps = 0  # ffill_steps 重置
                t_i += 1
            # 如果对应该时间点的特征值中没有 NAN_FEATURE_VALUE 了就说明后续的都不可能有空了, 也就不需要继续填了, 直接 break
            if not behind_has_nan_value:
                break
            pr_i -= 1
        return ffill_steps_paf_mat
def cal_bid_paf_mat(cnp.ndarray[cnp.int32_t, ndim=2] bid_price,
                    cnp.ndarray[double, ndim=3] bid_feature,
                    str fill_nan_option=None):
    """ 计算委托买的 paf 矩阵, 且完成空值填充

    :param bid_price: 当前委托买的订单价格, shape = (t, l)
    :param bid_feature: 当前委托买的订单数量, shape = (t, l, f)
    :param fill_nan_option: 填充空值的方式, 可能有以下 4 种选择
        - way 1 : `cut_off`
        - way 2 : `ffill_and_cut_off`
        - way 3 : `ffill`
        - way 4: `ffill_and_note_steps`
        - None: don't do the fill none operation

    :return:
        bid_price_range: the range of bid price (num_price_range)
        bid_paf_mat_fill_nan: (t, num_price_range, f) 委托买的 paf 矩阵

    """

    # ---- Step 1. 计算 volume mat ---- #
    bid_price_range, bid_paf_mat = gen_paf_mat(price=bid_price, feature=bid_feature, is_bid=True)
    cdef cnp.ndarray[double, ndim = 3]  bid_paf_mat_fill_nan

    # ---- Step 2. fill nan ---- #
    if fill_nan_option is not None:
        bid_paf_mat_fill_nan = fill_nan_in_paf_mat(paf_mat=bid_paf_mat, fill_nan_option=fill_nan_option)
        return bid_price_range, bid_paf_mat_fill_nan
    else:
        return bid_price_range, bid_paf_mat

def cal_ask_paf_mat(cnp.ndarray[cnp.int32_t, ndim=2] ask_price,
                    cnp.ndarray[double, ndim=3] ask_feature,
                    str fill_nan_option=None):
    """ 计算委托卖的 paf 矩阵, 且完成空值填充

    :param ask_price: 当前委托卖的订单价格 shape = (t, l)
    :param ask_feature: 当前委托卖的订单数量 shape = (t, l, f) 且 f = 0 为 volume; f = 1 为 amount
    :param fill_nan_option: 填充空值的方式
        - way 1 : `cut_off`
        - way 2 : `ffill_and_cut_off`
        - way 3 : `ffill`
        - way 4: `ffill_and_note_steps`
        - None: don't do the fill none operation

    :return:
        ask_price_range: the range of bid price (num_price_range)
        ask_paf_mat_fill_nan: (t, num_price_range, f) 委托卖的 paf 矩阵

    """

    # ---- Step 1. 计算 volume mat ---- #
    ask_price_range, ask_paf_mat = gen_paf_mat(price=ask_price, feature=ask_feature, is_bid=False)
    cdef cnp.ndarray[double, ndim = 3]  ask_paf_mat_fill_nan
    # ---- Step 2. fill nan ---- #
    if fill_nan_option is not None:
        ask_paf_mat_fill_nan = fill_nan_in_paf_mat(paf_mat=ask_paf_mat, fill_nan_option=fill_nan_option)
        return ask_price_range, ask_paf_mat_fill_nan
    else:
        return ask_price_range, ask_paf_mat
