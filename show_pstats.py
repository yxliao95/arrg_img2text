import pstats

if __name__ == "__main__":
    cprof_file = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/2_fast_regression/time_statistic.cprofile"
    out_file_path = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/2_fast_regression/pstats.txt"

    with open(out_file_path, "w") as f:
        ps = pstats.Stats(cprof_file, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
