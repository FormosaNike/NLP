import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False

# ========读取原始数据========
with open(r'E:\desktop\NLP\作业4\jyxstxtqj_downcc.com\天龙八部.txt', encoding='ANSI', errors='ignore') as f:
    data = f.readlines()
# 生成一个正则，负责找'()'包含的内容
pattern = re.compile(r'\(.*\)')
# 将其替换为空
data = [pattern.sub('', lines) for lines in data]
# 将.....替换为句号
data = [line.replace('……', '。') for line in data if len(line) > 1]
# 将每行的list合成一个长字符串
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)

# ========词汇表========
vocab = list(set(data))
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for i, c in enumerate(vocab)}
numdata = [char2id[char] for char in data]

# ========数据生成========
def data_generator(data,BATCH_SIZE, TIME_STEPS):
    data = data[:BATCH_NUMS * BATCH_SIZE * TIME_STEPS]
    data = np.array(data).reshape((BATCH_SIZE, -1))
    while True:
        for i in range(0, data.shape[1], TIME_STEPS):
            x = data[:, i:i + TIME_STEPS]
            y = np.roll(x, -1, axis=1)
            yield x, y

# ====================================搭建模型===================================
class RNNModel(tf.keras.Model):
    """docstring for RNNModel"""
    def __init__(self, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE):
        super(RNNModel, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.HIDDEN_LAYERS = HIDDEN_LAYERS
        self.VOCAB_SIZE = VOCAB_SIZE

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, HIDDEN_LAYERS)
        self.lstm_layers = [tf.keras.layers.LSTM(HIDDEN_LAYERS, return_sequences=True, return_state=True) for _ in
                            range(HIDDEN_LAYERS)]
        self.dense = tf.keras.layers.Dense(VOCAB_SIZE)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs)
        new_states = []
        for i in range(self.HIDDEN_LAYERS):
            x, state_h, state_c = self.lstm_layers[i](x, initial_state=states[i] if states else None,
                                                      training=training)
            new_states.append([state_h, state_c])
        x = self.dense(x)
        if return_state:
            return x, new_states
        else:
            return x

# ========定义回调函数========
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# =======预定义模型参数========
VOCAB_SIZE = len(vocab)
EPOCHS = 100
BATCH_SIZE = 32
TIME_STEPS = 60
BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 3
learning_rate = 0.01

# ===========模型训练===========
model = RNNModel(HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

train_data = data_generator(numdata, BATCH_SIZE, TIME_STEPS)

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=BATCH_NUMS, callbacks=[history])

# ===========生成文本函数===========
def generate_text(model, start_string, num_generate=100):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    states = None

    for i in range(num_generate):
        predictions, states = model(input_eval, states=states, return_state=True)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)

# ===========绘制loss曲线===========
plt.plot(history.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('seq2seq Training Loss')
plt.show()

# ===========下段文本结果===========
print(generate_text(model, start_string="那长须老者满脸得色，微微一笑，说道：“东宗已胜了三阵，看来这‘剑湖宫’又要让东宗再住五年了。辛师妹，咱们还须比下去么？”"))