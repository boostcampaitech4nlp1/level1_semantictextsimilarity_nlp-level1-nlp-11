### 순차적으로 여러 개의 python file을 실행하도록 만든 shell script 실행 명령어

- terminal 종료, 컴퓨터 절전 모드여도 백그라운드에서 실행되는 `nohup` 명령어와 `&`
- 아래 명령어를 실행하면 terminal에 출력되는 log는 `*.out`에 저장하고, error message는 `*.err`에 저장
- (추가) 네트워크 연결이 끊겨도 서버에서 프로그램이 계속 실행되고 wandb 연결도 끊기지 않음

```bash
# 실행 명령어
# nohup mycommand > mycommand.out 2> mycommand.err &
nohup bash train_repeat.sh > wandb_train_shell_repeat.out 2> wandb_train_shell_repeat.err &

nohup python3 'KcELECTRA_train&dev.py' > wandb_train_shell_repeat.out 2> wandb_train_shell_repeat.err &


# 실행 중인지 확인하는 명령어
# ps -ef | grep mycommand
# ps -ef | grep PROCESS_ID
ps -ef | grep "bash train_repeat.sh"
ps -ef | grep 2431
```