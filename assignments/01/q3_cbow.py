"""Instead of implementing a CBOW for general text, I implemented for training
embedding vectors for amino acids in particular.

This code still needs to be refactored to make it more generalizable to any
text. Will do after improving my tensorflow skills
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

import utils
import word2vec_utils

# Model hyperparameters
VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# add starting and ending token
# VOCAB = ['<s>'] + VOCAB + ['</s>']
DICTIONARY = dict(zip(VOCAB, range(len(VOCAB))))
VOCAB_SIZE = len(DICTIONARY)
BATCH_SIZE = 128
# EMBED_SIZE = 128            # dimension of the word embedding vectors
EMBED_SIZE = 5                  # dimension of the word embedding vectors
WINDOW_SIZE = 3             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
# LEARNING_RATE = 1.0         # gradient descent
LEARNING_RATE = 0.001       # adam
NUM_TRAIN_STEPS = int(1e7)
visual_outdir = 'visualization'
LOG_INTERVAL_STEPS = 500       # log per this number of step

NUM_VISUALIZE = 3000        # number of tokens to visualize


class CBOWModel:
    """ Build the graph for word2vec model """
    def __init__(self, dataset, vocab_size, embed_size, batch_size,
                 num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.get_variable(
            'global_step',
            initializer=tf.constant(0),
            trainable=False
        )
        self.skip_step = LOG_INTERVAL_STEPS
        self.dataset = dataset

    def _import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.src_idx, self.tgt_idx = self.iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable(
                'embed_matrix',
                shape=[self.vocab_size, self.embed_size],
                initializer=tf.random_uniform_initializer()
            )
            embed = tf.nn.embedding_lookup(
                self.embed_matrix,
                self.src_idx,
                name='embedding'
            )
            self.embed = tf.reduce_sum(embed, axis=1)

    def _create_loss(self):
        with tf.name_scope('loss'):
            # construct variables for cross entropy loss
            w_out_init = tf.truncated_normal_initializer(
                stddev=1.0 / (EMBED_SIZE ** 0.5)
            )
            w_out = tf.get_variable(
                'w_out',
                shape=[EMBED_SIZE, VOCAB_SIZE],
                initializer=w_out_init
            )
            b_out = tf.get_variable(
                'b_out',
                initializer=tf.zeros(shape=[VOCAB_SIZE])
            )

            logits = tf.matmul(self.embed, w_out) + b_out
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.tgt_idx,
                name="entropy"
            )
            self.loss = tf.reduce_mean(entropy, name="loss")

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        # self.optimizer = tf.train.GradientDescentOptimizer(
        #     self.lr).minimize(self.loss, global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(
            self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        # defaults to saving all variables - in this case embed_matrix,
        # nce_weight, nce_bias
        saver = tf.train.Saver()

        initial_step = 0
        utils.safe_mkdir('checkpoints')
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # we use this to calculate late average loss in the last LOG_INTERVAL_STEPS
            # steps
            total_loss = 0.0
            writer = tf.summary.FileWriter(
                'graphs/cbow/lr' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()

            target_step = initial_step + num_train_steps
            for index in range(initial_step, target_step):
                try:
                    loss_batch, _, summary = sess.run(
                        [self.loss, self.optimizer, self.summary_op]
                    )
                    writer.add_summary(summary, global_step=index)

                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {0}/{1}: {2:5.6f}'.format(
                            index + 1, target_step, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/cbow', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    def visualize(self, visual_fld, num_visualize):
        """ run "'tensorboard --logdir='visualization'" to see the embeddings """

        # create the list of num_variable most common words to visualize
        # utils.safe_mkdir(visual_fld)
        word2vec_utils.most_common_words(visual_fld, num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(
                final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file, in this case the first
            # NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during
            # startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)


def read_data_and_generate_sample(file_path, window_size):
    with open(file_path) as inf:
        for line in inf:
            # 'A C G\n' => ['<s>', 'A', 'C', 'G', '</s>']
            # words = ['<s>'] + line.split() + ['</s>']
            words = line.split()
            # ignore beginning and ending to make input size constant
            num_words = len(words)
            for index, center in enumerate(words):
                if index < window_size or index > num_words - 1 - window_size:
                    continue
                context = (
                    words[index - window_size: index] +
                    words[index + 1: index + window_size + 1]
                )
                yield [DICTIONARY[_] for _ in context], DICTIONARY[center]
                # yield context, center, [DICTIONARY[_] for _ in context], DICTIONARY[center]


def batch_gen(file_path, vocab_size, batch_size, window_size):
    """Generate input/output batch in indices"""
    single_gen = read_data_and_generate_sample(file_path, window_size)

    while True:
        # None specified the unknown size of the number of context words that
        # would be used depending on the location of the center words
        src_batch = np.zeros([batch_size, window_size * 2], dtype=np.int32)
        tgt_batch = np.zeros([batch_size])
        for index in range(batch_size):
            src_batch[index], tgt_batch[index] = next(single_gen)
        yield src_batch, tgt_batch


def gen():
    yield from batch_gen(
        DATA_FILE_PATH,
        VOCAB_SIZE,
        BATCH_SIZE,
        WINDOW_SIZE
    )


def main():
    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.int32, tf.int32),
        (tf.TensorShape([BATCH_SIZE, WINDOW_SIZE * 2]),
         tf.TensorShape([BATCH_SIZE]))
    )

    # iterator = dataset.make_initializable_iterator()
    # src_idx, tgt_idx = iterator.get_next()

    # with tf.name_scope('embed'):
    #     embed_matrix = tf.get_variable(
    #         'embed_matrix',
    #         shape=[VOCAB_SIZE, EMBED_SIZE],
    #         initializer=tf.random_uniform_initializer()
    #     )
    #     embed = tf.nn.embedding_lookup(
    #         embed_matrix, src_idx, name='embedding')
    #     # see CBOW paper about this sum, axis=0 is the batch_size
    #     embed = tf.reduce_sum(embed, axis=1)

    # with tf.name_scope('loss'):
    #     # construct variables for NCE loss
    #     w_out_init = tf.truncated_normal_initializer(
    #         stddev=1.0 / (EMBED_SIZE ** 0.5)
    #     )
    #     w_out = tf.get_variable(
    #         'w_out',
    #         shape=[EMBED_SIZE, VOCAB_SIZE],
    #         initializer=w_out_init
    #     )
    #     b_out = tf.get_variable(
    #         'b_out',
    #         initializer=tf.zeros(shape=[VOCAB_SIZE])
    #     )

    #     logits = tf.matmul(embed, w_out) + b_out
    #     entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         logits=logits,
    #         # self.tgt_idx is of shape [BATCH_SIZE, 1]
    #         labels=tf.reshape(tgt_idx, [-1]),
    #         name="entropy"
    #     )
    #     loss = tf.reduce_mean(entropy, name="loss")

    # with tf.Session() as sess:
    #     sess.run(iterator.initializer)
    #     sess.run(tf.global_variables_initializer())
    #     # print(sess.run(embed))
    #     # print(sess.run([src_idx, tgt_idx]))
    #     # print(sess.run(entropy))
    #     print(sess.run(loss))

    model = CBOWModel(
        dataset,
        VOCAB_SIZE,
        EMBED_SIZE,
        BATCH_SIZE,
        NUM_SAMPLED,
        LEARNING_RATE
    )
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(visual_outdir, NUM_VISUALIZE)


if __name__ == '__main__':
    # DATA_FILE_PATH = '/projects/btl/zxue/amp/amino_acid_word2vec/data/transformed/one_peptide_per_line/w2v_nr_1e+04_seqs.txt'

    # DATA_FILE_PATH = '/projects/btl/zxue/amp/amino_acid_word2vec/lele.txt'
    DATA_FILE_PATH = sys.argv[1]

    # for k, i in enumerate(read_data_and_generate_sample(DATA_FILE_PATH, WINDOW_SIZE)):
    # for k, i in enumerate(gen()):
    #     print(i)
    #     if k == 20:
    #         break
    main()
