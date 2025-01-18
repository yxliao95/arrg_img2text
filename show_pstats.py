import pstats

if __name__ == "__main__":
    cprof_file = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/1_imgcls_notallimg_fast_000test/time_eval.txt"
    out_file_path = "/scratch/c.c21051562/workspace/arrg_img2text/outputs/results/1_imgcls_notallimg_fast_000test/pstats.txt"

    with open(out_file_path, "w") as f:
        ps = pstats.Stats(cprof_file, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
        