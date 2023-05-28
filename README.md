# Hopper_Offline
Final Project of AI3601 Reinforcement Learning, SJTU 2023, Spring.



## Environment

​	Fork from [Jidiai/Competition_Gym](https://github.com/jidiai/Competition_Gym)



## Dependency

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## Train

```shell
python scripts/train_ALGO.py
```

​	The `ALGO` could be one of AWR, BAIL, BC, BCQ, CQL, BABCQ. 

​	For detailed training parameters, please view `scripts/args.py` for more information.

## Evaluate

```bash
python run_log.py --env_name 'gym_Hopper-v2' --my_ai "AGENT"
```





