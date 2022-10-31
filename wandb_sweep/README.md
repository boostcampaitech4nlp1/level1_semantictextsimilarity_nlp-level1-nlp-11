### KoELECTRA_wandb_sweep_train.py 실행 명령어

```bash
# terminal 종료, 컴퓨터 절전 모드여도 백엔드에서 실행되는 명령어
# nohup mycommand > mycommand.out 2> mycommand.err &
nohup python3 KoELECTRA_wandb_sweep_train.py > wandb_train.out 2> wandb_train.err &

# 실행 중인지 확인하는 명령어
# ps -ef | grep mycommand
# ps -ef | grep PROCESS_ID
ps -ef | grep "python3 KoELECTRA_wandb_sweep_train.py"
ps -ef | grep 31517
```