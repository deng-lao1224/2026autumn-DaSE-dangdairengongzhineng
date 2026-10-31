# 实验一：新闻文本十分类

用 scikit-learn 完成 10 类新闻文本分类。模型选择只使用带标签数据划出的验证集，测试集只在方案锁定后预测一次。

本目录对应实验报告中的结果。报告里的验证 macro-F1 0.933 是 `0.93291` 保留三位小数。

## 环境与依赖

在 Windows 上用 Anaconda 跑通，版本如下：

- 操作系统：Windows 10 / 11，64 位
- Python：3.12.7（Anaconda）
- numpy：1.26.4
- pandas：2.3.3
- scipy：1.13.1
- scikit-learn：1.5.1
- matplotlib：3.9.2

在存放本说明的目录中安装依赖：

```bash
pip install -r requirements.txt
```

请使用上面的 scikit-learn 1.5.1。换一个小版本时，SGD 的最优 `alpha` 和轮数会变，终端里的 `BEST` 就不再是报告中的模型。

## 数据与目录

从课程压缩包中取出两个 csv，按下面的相对路径放好。脚本用 `train.py` 所在目录定位文件，不要改文件夹名字。

```text
lab01/
  README.md
  requirements.txt
  train.py
  prediction.csv                          运行后生成
  data-Project1/
    data-Project1/
      train_data.csv                      带标签数据，7368 行
      test_data_unlabeled.csv             无标签测试集，2457 行
  results/                                运行后生成
    loss_curve.png
    model_compare.png
    loss_steps.csv
    loss_epochs.csv
    metrics.json
```

`train_data.csv` 有两列：

- `text`：整封邮件或新闻，含信头和正文
- `target`：类别编号，整数 0 到 9

`test_data_unlabeled.csv` 只有 `text`，没有标签。

若本目录不叫 `lab01`，只要 `train.py`、`requirements.txt` 和 `data-Project1/data-Project1/` 仍然在同一层，就可以运行。

## 运行顺序

在本目录执行一条命令即可完成检查数据、训练、评估和生成预测：

```bash
python train.py
```

脚本依次做这几件事：

1. 读取两个 csv，并打印 `labeled=7368 train_split=5894 val_split=1474 test=2457`。数字不符就先停止，检查是不是放错了文件。
2. 用随机种子 42，按类别分层划出 20% 验证集。
3. 固定 `LinearSVC(C=1)`，比较八种文本表示和清洗方式。
4. 锁定得分最高的表示，再比较朴素贝叶斯、SGD 逻辑回归、SGD 线性 SVM、LinearSVC 和 MLP。
5. 按验证集 macro-F1 选择模型，并记录逐步损失。
6. 在全部 7368 条带标签样本上，按验证阶段选出的轮数重新训练，对测试集只做 `transform` 和预测。
7. 写出 `prediction.csv` 和 `results/` 下的图表与 `metrics.json`。

## 实验配置

- 随机种子：42
- 划分：带标签数据的 80% 训练、20% 验证，`stratify` 为类别
- 词表：`max_features=12000`，`min_df=2`，`max_df=0.9`，只保留长度不少于 2 的字母词
- 最终表示：原文不做删信头，unigram sublinear TF-IDF，L2 归一化
- 小批量：256
- SGD 训练轮数上限：24
- SGD 学习率日程：`optimal`；正则为 L2
- `alpha` 候选：`0.0001`、`1/5894`、`1/(2×5894)`
- 最终模型：`SGDClassifier(loss="hinge", alpha=8.48e-5)`，验证阶段第 24 轮 macro-F1 最高，全量重训也用 24 轮
- 主指标：验证集 macro-F1；同时记录 accuracy

## 结果与提交

运行结束后，终端应出现：

```text
best feature 原文+TFIDF-sublinear
BEST SGD-SVM(alpha=8.48e-05) 0.932909839954668 0.9328358208955224
wrote ... prediction.csv n= 2457
```

与报告对应的验证 macro-F1（三位小数）为：

| 比较项 | 验证 macro-F1 |
| --- | ---: |
| 原文 + 词袋 | 0.893 |
| 原文 + TF-IDF L2 | 0.932 |
| 原文 + TF-IDF L1 | 0.895 |
| 原文 + sublinear TF-IDF | 0.932 |
| 原文 + 一元/二元 TF-IDF | 0.930 |
| 删除全部信头 | 0.860 |
| 只保留 Subject | 0.914 |
| 保留 Subject 并去停用词 | 0.916 |
| SGD 线性 SVM，alpha=8.48e-5，第 24 轮 | 0.933 |
| LinearSVC，C=1 | 0.932 |
| SGD 逻辑回归，alpha=8.48e-5 | 0.927 |
| MLP，隐层 64 | 0.922 |
| 多项式朴素贝叶斯，alpha=0.1 | 0.911 |

各文件的含义：

- `results/loss_curve.png`：选中模型的逐步 hinge 损失，以及每个 epoch 的验证损失和验证 macro-F1
- `results/model_compare.png`：每个模型家族的最佳验证 macro-F1
- `results/loss_steps.csv`：576 行逐步损失
- `results/loss_epochs.csv`：24 个 epoch 的训练损失、验证损失、accuracy 和 macro-F1
- `results/metrics.json`：全部表示、全部模型和验证集分类报告
- `prediction.csv`：测试集预测。2457 行，无表头，无索引，每行一个 0 到 9 的类别编号，顺序与 `test_data_unlabeled.csv` 相同

课程要求的压缩包只放实验报告 PDF 和 `prediction.csv`，不要把本目录打进那个压缩包。代码、本说明和依赖文件放在仓库中。

## 复现与限制

核对预测文件：

```bash
python -c "from pathlib import Path; lines=Path('prediction.csv').read_text(encoding='utf-8').splitlines(); print(len(lines), set(lines)<=set('0123456789'))"
```

正常输出是 `2457 True`。再打开 `results/metrics.json`，其中 `selected_model` 应为 `SGD-SVM(alpha=8.48e-05)`，`selected_val_macro_f1` 约为 `0.93291`。

运行时可能出现两类警告，都不改变上述选择：

- 词袋计数上的 `LinearSVC` 在 2000 次迭代内没有收敛，liblinear 会给出 `ConvergenceWarning`。该组验证 macro-F1 为 0.893，低于 TF-IDF，不作为最终模型。
- MLP 每个 epoch 的最后一批样本不足 256 条，scikit-learn 会提示 `batch_size` 被截断。这是最后一批的长度问题，训练会继续。

限制：

- 测试集标签未公布，仓库里的分数都是验证集分数。0.933 是重训之前的验证成绩，不是测试成绩。
- 全量重训把原来的验证子集也用于拟合，之后本地没有留出的带标签数据可以再给 `prediction.csv` 打分。
- 未在 scikit-learn 1.5.1 以外的版本上保证选出同一个 `alpha`。在 1.9.x 上，同脚本可能改选 `alpha=0.0001`、第 11 轮。
