# coding: utf-8
'''
참고: Step 3 가 없음. 원문이 그래서 그렇게 놔두었음.
'''

from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import zipfile

# Step 1: Download the data.
# 데이터를 다운로드함. 파일이 이미 있다면 제대로 받아졌는지 (파일 크기가 같은지) 확인.
# 다운로드 받은 후 filename을 리턴함.
print("Step 1: Download the data.")
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.
# file (zipfile) 을 읽어옴
# text8.zip 의 내용은 파일 하나임. 코드를 봐서는 ' '로 구분된 단어들인 듯.
def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return f.read(name).split()
    f.close()

words = read_data(filename)
print('Data size', len(words))
print('Sample words: ', words[:10])

# Step 2: Build the dictionary and replace rare words with UNK token.
print("\nStep 2: Build the dictionary and replace rare words with UNK token.")
vocabulary_size = 50000

def build_dataset(words):
    """
    vocabulary_size 는 사용할 빈발 단어의 수를 뜻함.
    등장 빈도가 상위 50000개 (vocabulary_size) 안에 들지 못하는 단어들은 전부 UNK로 처리한다.

    :param words: 말 그대로 단어들의 list
    :return data: indices of words including UNK. 즉 words index list.
    :return count: 각 단어들의 등장 빈도를 카운팅한 collections.Counter
    :return dictionary: {"word": "index"}
    :return reverse_dictionary: {"index": "word"}. e.g.) {0: 'UNK', 1: 'the', ...}
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) # insert index to dictionary (len이 계속 증가하므로 결과적으로 index의 효과)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data: ', data[:10])
print('Sample count: ', count[:10])
print('Sample dict: ', dictionary.items()[:10])
print('Sample reverse dict: ', reverse_dictionary.items()[:10])

data_index = 0


# Step 4: Function to generate a training batch for the skip-gram model.
print("\nStep 4: Function to generate a training batch for the skip-gram model.")
def generate_batch(batch_size, num_skips, skip_window):
    """
    minibatch를 생성하는 함수.
    data_index는 global로 선언되어 여기서는 static의 역할을 함. 즉, 이 함수가 계속 재호출되어도 data_index의 값은 유지된다.

    :param batch_size   : batch_size.
    :param num_skips    : context window 내에서 (target, context) pair를 얼마나 생성할 지.
    :param skip_window  : context window size. skip-gram 모델은 타겟 단어로부터 주변 단어를 예측하는데, skip_window가 그 주변 단어의 범위를 한정한다.
    :return batch       : mini-batch of data.
    :return labels      : labels of mini-batch. [batch_size][1] 의 2d array.
    """
    global data_index
    assert batch_size % num_skips == 0  # num_skips의 배수로 batch가 생성되므로.
    assert num_skips <= 2 * skip_window # num_skips == 2*skip_window 이면 모든 context window의 context에 대해 pair가 생성된다.
    # 즉, 그 이상 커지면 안 됨.

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    # Deques are a generalization of stacks and queues.
    # The name is pronounced "deck" and is short for "double-ended queue".
    # 양쪽에 모두 push(append) & pop 을 할 수 있음.

    # buffer = data[data_index:data_index+span] with circling
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # // 는 나머지 혹은 소수점 아래를 버리는 연산자
    # skip-gram은 타겟 단어로부터 주변의 컨텍스트 단어를 예측하는 모델이다.
    # skip-gram model을 학습하기 전에, words를 (target, context) 형태로 변환해 주어야 한다.
    # 아래 코드는 그 작업을 batch_size 크기로 수행한다.
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                # context window에서 context를 뽑아내는 작업은 랜덤하게 이루어진다.
                # 단, skip_window*2 == num_skips 인 경우, 어차피 모든 context를 다 뽑아내므로 랜덤은 별 의미가 없음. 순서가 랜덤하게 될 뿐.
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

# batch가 어떻게 구성되는지를 보기 위해 한번 뽑아서 출력:
print("Generating batch ... ")
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
print("Sample batches: ", batch[:10])
print("Sample labels: ", labels[:10])
for i in range(8):
    print(batch[i], '->', labels[i, 0])
    print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])


# Step 5: Build and train a skip-gram model.
print("\nStep 5: Build and train a skip-gram model.")
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
# [0 ~ valid_window] 의 numpy array를 만들고 거기서 valid_size 만큼 샘플링함.
# 즉, 여기서는 0~99 사이의 수 중 랜덤하게 16개를 고른 것이 valid_examples 임.
num_sampled = 64    # Number of negative examples to sample.

print("valid_examples: ", valid_examples)

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    # embedding_lookup이 GPU implementation이 구현이 안되어 있어서 CPU로 해야함.
    # default가 GPU라서 명시적으로 CPU라고 지정해줌.
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        # embedding matrix (vectors)
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 전체 embedding matrix에서 train_inputs (mini-batch; indices) 이 가리키는 임베딩 벡터만을 추출
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        # NCE loss 는 logistic regression model 을 사용해서 정의된다.
        # 즉, logistic regression 을 위해, vocabulary의 각 단어들에 대해 weight와 bias가 필요함.
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # minibatch (valid_embeddings) 와 all embeddings 사이의 cosine similarity를 계산한다.
    # 이 과정은 학습이 진행되면서 각 valid_example 들에게 가장 가까운 단어가 어떤 것인지를 보여주기 위함이다 (즉 학습 과정을 보여주기 위함).
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 6: Begin training
print("\nStep 6: Begin training")
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    tf.initialize_all_variables().run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # feed_dict를 사용해서 placeholder에 데이터를 집어넣고 학습시킴.
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Step 7: Visualize the embeddings.
print("\nStep 7: Visualize the embeddings.")
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    # 혹시 여기서 에러가 난다면, scikit-learn 과 matplotlib 을 최신버전으로 업데이트하자.
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500

    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
