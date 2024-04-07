import jieba
import matplotlib.pyplot as plt

# 读取文本文件
# txt = open("jyxstxtqj_downcc.com\\白马啸西风.txt", "r", encoding="ANSI").read()
txt = open("jyxstxtqj_downcc.com\\天龙八部.txt", "r", encoding="ANSI").read()
# 使用结巴分词进行分词
words = jieba.lcut(txt)

# 定义需要排除的特殊字符
extra_characters = {"，", "。", "\n", "：", "；", "？", "（", "）", "！", "…", "「", "」", " "}

# 统计词频
counts = {}
for word in words:
    if word not in extra_characters and word.strip() != "" and word.strip() != " ":
        counts[word] = counts.get(word, 0) + 1

# 将词频排序并获取频率列表
items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
frequencies = [count for _, count in items]

# 输出字典文件
with open("dictionary.txt", "w", encoding="utf-8") as file:
    for word, count in items:
        file.write(f"{word}: {count}\n")

# 绘制Zipf定律图表
plt.title('Zipf-Law', fontsize=18)  # 标题
plt.xlabel('Rank', fontsize=18)  # 排名
plt.ylabel('Frequency', fontsize=18)  # 频度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(range(1, len(frequencies) + 1), frequencies, 'r')  # 绘制曲线
plt.xscale('log')  # 设置x轴为对数刻度
plt.yscale('log')  # 设置y轴为对数刻度
plt.savefig('Zipf_Law.png')  # 保存图表
plt.show()
# 输出指定排名的频率值和排名与频率之积
specified_ranks = [10, 20, 30, 40, 50, 100]
for rank in specified_ranks:
    if rank <= len(items):
        word, frequency = items[rank - 1]
        product = rank * frequency
        print(f"Rank {rank}: Frequency = {frequency}, Rank * Frequency = {product}")
    else:
        print(f"Rank {rank} is beyond the total number of words.")