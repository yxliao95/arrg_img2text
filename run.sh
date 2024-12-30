# Server side:
# tensorboard --logdir=/home/yuxiang/liao/workspace/arrg_img2text/outputs/logs --port=6006

# Check process and kill
# ps aux | grep <进程名>
# kill <PID>

# Client side:
# ssh -L 6007:localhost:6006 yuxiang@10.97.37.97
# tensorboard --logdir=/home/yuxiang/liao/workspace/arrg_img2text/outputs/logs --port=6006