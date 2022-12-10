# ☃️ 엘사(E1I4)지만 자신있조 ☃️
>## Semantic Text Similarity (STS) Task


## Contributors
- **기윤호 [(Github)](https://github.com/yhkee0404) : 데이터 팀, 모델링 팀**
    - Label Imbalance 대처를 위한 RankSim 적용
    - 2-STAGE RETRAINING 등 모델 성능 개선 전략 수립

- **김남규 [(Github)](https://github.com/manstar1201) : 데이터 팀, 모델링 팀**
    - EDA
    - 데이터 증강 및 전처리
    - 하이퍼 파라미터 튜닝 등 모델 개선

- **이동찬 [(Github)](https://github.com/DongChan-Lee) : PM, 모델링 팀, 코드 및 이슈 관리팀**
    - 전체적인 프로젝트 실험 계획 수립 및 성능 개선을 위한 아이디어 제안
    - 모델 탐색 및 선정
    - 모델링 코드 구현 및 공유
    - 코드 리뷰
    - 모델 성능 개선

- **이정현 [(Github)](https://github.com/Jlnus) : 데이터 팀, 모델링 팀**
    - EDA
    - 모델 학습 전략 수립
    - 앙상블 수행
    - 모델 성능 개선

- **조문기 [(Github)](https://github.com/siryuon) : 데이터 팀, 모델링 팀**
    - 데이터 증강 및 전처리
    - 모델 탐색 및 선정
    - K-fold 베이스라인 코드 작성


## 프로젝트 구조
        ├─ baseline
        ├─ model_experiments
        ├─ wandb_sweep
        ├─ wandb_many_epochs_sh
        └─ requirements.txt
        └─ .gitignore

- model_experiments
    - 1차적으로 다양한 모델 실험

- wandb_sweep
    - 실험한 모델들 중 성능이 좋았던 모델들에 WandB sweep을 이용하여 하이퍼 파라미터 튜닝

- wandb_many_epochs_sh
    - shell script를 이용하여 wandb 반복 실행
