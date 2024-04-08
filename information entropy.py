import os
import jieba
import math

'''
# 读取取样文本文件
txt = open("jyxstxtqj_downcc.com\\白马啸西风.txt", "r", encoding="ANSI").read()
# txt = open("jyxstxtqj_downcc.com\\天龙八部.txt", "r", encoding="ANSI").read()

# 使用结巴分词进行分词
words = jieba.lcut(txt)
'''
def merge_text_files(input_files):
    merged_text = ""
    for file_name in input_files:
        with open(file_name, "r", encoding="ANSI") as file:
            merged_text += file.read()
    return merged_text
# 读取全部文件
# 指定要读取的文件夹路径
folder_path = "E:/desktop/NLP/jyxstxtqj_downcc.com"
# 获取文件夹中所有以.txt为扩展名的文件名
input_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".txt")]
# 合并多个文本文件的内容
merged_text = merge_text_files(input_files)
# 使用结巴分词进行分词
words = jieba.lcut(merged_text)

# 定义需要排除的特殊字符
extra_characters = {"，", "。", "\n", "：", "；", "？", "（", "）", "！", "…", "「", "」", " "}

# 统计字频和词频
char_counts = {}
word_counts = {}
for char in merged_text:
    if char not in extra_characters:
        char_counts[char] = char_counts.get(char, 0) + 1

for word in words:
    if word not in extra_characters and word.strip() != "" and word.strip() != " ":
        word_counts[word] = word_counts.get(word, 0) + 1

# 计算字的信息熵
char_total = sum(char_counts.values())
char_entropy = 0
for count in char_counts.values():
    probability = count / char_total
    char_entropy += probability * math.log2(probability)

char_entropy = -char_entropy

# 计算词语的信息熵
word_total = sum(word_counts.values())
word_entropy = 0
for count in word_counts.values():
    probability = count / word_total
    word_entropy += probability * math.log2(probability)

word_entropy = -word_entropy

# 输出字和词语的信息熵
print(f"Character Entropy: {char_entropy}")
print(f"Word Entropy: {word_entropy}")