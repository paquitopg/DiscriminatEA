import gc
import time
import multiprocessing
import numpy as np


g = 1000000000

def div_list(ls, n):
    """
    Divide a list into n sub-lists
    Args:
        ls: list to be divided
        n: number of sub-lists
    Returns:
        list of sub-lists
    """
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_num = [j] * n
        for i in range(k):
            ls_num[i] += 1
        ls_return = []
        ind = 0
        for i in range(n):
            ls_return.append(ls[ind : ind+ls_num[i]])
            ind += ls_num[i]
        return ls_return


def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    """
    Calculate the rank of the reference entities
    Args:
        task: list of reference entities
        sim: similarity matrix
        top_k: list of top k values
        accurate: whether to use accurate method
    Returns:
        mean: mean rank
        mrr: mean reciprocal rank
        num: number of hits at each top k
        prec_set: set of reference entities and their ranks
    """
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set


def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False,output = True):
    """
    Evaluate the alignment by similarity matrix
    Args:
        embed1: first embedding matrix
        embed2: second embedding matrix
        top_k: list of top k values
        nums_threads: number of threads
        csls: whether to use csls
        accurate: whether to use accurate method
        output: whether to output the results
    Returns:
        t_prec_set: set of reference entities and their ranks
        acc: accuracy at each top k
        t_mrr: mean reciprocal rank
    """
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, acc, t_mrr


def cal_csls_sim(sim_mat, k):
    # Cap k_eff so that k_eff + 1 never exceeds sim_mat.shape[1] - 1 (0-based)
    max_k = sim_mat.shape[1] - 2  # -2 so that k_eff+1 <= shape[1]-1
    k_eff = min(k, max_k)
    if k_eff < 1:
        k_eff = 1
    sorted_mat = -np.partition(-sim_mat, k_eff + 1, axis=1)  # Get top k+1 largest values
    nearest_k = sorted_mat[:, 0:k_eff]  # Take the k largest (nearest neighbors)
    sim_values = np.mean(nearest_k, axis=1)  # Average similarity to k nearest neighbors
    return sim_values


def CSLS_sim(sim_mat1, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values


def sim_handler(embed1, embed2, k, nums_threads):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat
