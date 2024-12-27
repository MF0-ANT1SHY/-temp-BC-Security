# State

Current state consists of 11 elements following the below format:

```
(a, b, status, advantage, specialblock, uncleblock1, ..., uncleblock6)
// 4 + 1 + 6
```

PS: 其实我没去看后面 7 个元素的具体含义和逻辑, 只是简单沿用了 SquirRL 的逻辑, 它们是负责记录叔块奖励的.

| Value | Action   | Status  | Advantage                |
| ----- | -------- | ------- | ------------------------ |
| 0     | adopt    | normal  | not acceptable           |
| 1     | match    | forking | equal                    |
| 2     | mine     |         | the attacker's is larger |
| 3     | override |         |                          |

# Action Argument

Agent 输入到 Env 的动作是一个二维的数据, 例如, (0,1); 第一位表示对应的 Action, 第二位表示 Action 的 argument.

不同值对应的 Argument 的含义如下:

| Action | Argument value | Description      |
| ------ | -------------- | ---------------- |
| adopt  | 0              |                  |
|        | 1              |                  |
| mine   | 0              | honest timestamp |
|        | 1              | maximum          |
|        | 2              | Minimum          |
