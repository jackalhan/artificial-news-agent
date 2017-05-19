from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import re
import random
import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



class MyVocabulary(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        vstr = ''
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                vstr+=line
        return vstr

vocabulary_str = MyVocabulary('../../datasets/raw_data/bbc/test/').__iter__()
vocabulary_list = []
vocabulary_list = re.sub("[^\w]", " ",  vocabulary_str).lower().split() #remove quotes and split
print("Vocabulary_list ---------",vocabulary_list)

# Read the data into a list of strings.

print('Data size', len(vocabulary_list))

# Step 2: Build the dictionary.
# UNK is used for unnecessary words.

#print('Before bulding the dataset .......')

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  # Only take vocabulary_size words into consideration, other words
  # will be replaced by 'UNK' because these words doesn't happen a lot.
  print("Raw Words ==> ", words)
  count = [['UNK', -1]]
  #most_common function of Counter object is sorting the list based on
  count.extend(collections.Counter(words).most_common(n_words - 1))

  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  #print(count[0][1], unk_count)

  # print("Dictionary Values")
  # print(dictionary.values())
  # print("Dictionary Keys")
  # print(dictionary.keys())
  # print("Dictionary")
  # print(dictionary)
  # print("Zipped - Reversed Dictionary")
  # print(dict(zip(dictionary.values(), dictionary.keys())))

  # in order to sort dictionary based on values that are indexes  of each key (word)
  # It is doing it only once which consumes less memory.
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  #print(reverse_dictionary)
  print("\n\n ****** Data (Index of the Words in vocabulary list ) || Numerical representaion of the document ****** ")
  print("\n",data)
  print("\n\n ****** Count (Sorted Frequency of the words) ****** ")
  print("\n",count)
  print("\n\n ****** Dictionary (value is Index key is word) || Dictionary to map which numerical representation is mapping to what word ******")
  print("\n",dictionary)
  print("\n\n ****** Reversed Dictionary (sorted by numerical representation of the words) ****** ")
  print("\n",reversed_dictionary)

  return data, count, dictionary, reversed_dictionary

vocabulary_size = 5000 # to set a size for distinct words in vocabulary
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary_list,vocabulary_size)
# print('After bulding the dataset .......')
# print(reverse_dictionary)
del vocabulary_str # Hint to reduce memory.
del vocabulary_list # Hint to reduce memory.


#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector. 128
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.2

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 10000

with tf.Session(graph=graph) as session:
  # Initialize all variables before using them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
  # with opende_file.write(fi('D:\\1.txt','w') as outmode_file:
  #     outmonal_embeddings)
  # summary_writer = tf.summary.FileWriter('output_graph')
  # summary_writer = tf.summary.FileWriter('output_graph', session.graph)

# Step 6: Output word2vec result for future usage.

try:
  # Put words into a tmp list for zipping operation
  labels = [reverse_dictionary[i] for i in range(len(reverse_dictionary))]

  # Zip the vector representation for each word together with the word itself.
  word_embedding_list = list(zip(final_embeddings,labels))

  # word_embedding_dict stores words[key] and their vector representations[value].
  word_embedding_dic ={}
  for tuple_ in word_embedding_list:
    word_embedding_dic[tuple_[1]] = list(tuple_[0])

  with open('obj.txt','w') as fp:
    for key in word_embedding_dic:
      fp.write(key + '  ')
      for tmp in word_embedding_dic[key]:
        fp.write(str(tmp) + ' ')
      fp.write('\n')
  fp.close()

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

