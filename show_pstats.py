import pstats

if __name__ == "__main__":
    cprof_file = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/1_fast_000test_preprocess_data_gpu_fp16/time_statistic.cprofile"
    out_file_path = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/1_fast_000test_preprocess_data_gpu_fp16/pstats.txt"

    with open(out_file_path, "w") as f:
        ps = pstats.Stats(cprof_file, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
