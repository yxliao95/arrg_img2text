import pstats

if __name__ == "__main__":
    cprof_file = "/home/yuxiang/liao/workspace/arrg_img2text/outputs/results/imgcls_exp1/time_statistic.cprofile"
    out_file_path = "/home/yuxiang/liao/workspace/arrg_img2text/outputs/results/imgcls_exp1/pstats.txt"

    with open(out_file_path, "w") as f:
        ps = pstats.Stats(cprof_file, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
