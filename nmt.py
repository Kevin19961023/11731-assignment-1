# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
import random
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, input_transpose
from vocab import Vocab, VocabEntry

import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # initialize neural network layers...
        # embedding
        self.encoder_embed = nn.Embedding(len(vocab.src.word2id), embed_size, padding_idx=vocab.src.word2id['<pad>'])
        self.decoder_embed = nn.Embedding(len(vocab.tgt.word2id), embed_size, padding_idx=vocab.tgt.word2id['<pad>'])

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=dropout_rate)
        self.decoder_lstm_cell = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        self.decoder_init_cell = nn.Linear(hidden_size * 2, hidden_size)

        #self.hidden_ctx_linear = nn.Linear(hidden_size * 2, hidden_size)
        # Layer for attention dropout
        self.att_drp_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Linear layer for both key and value in attention
        self.key_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.value_linear = nn.Linear(hidden_size * 2, hidden_size)

        # Linear layer for output
        self.output_linear = nn.Linear(hidden_size, len(vocab.tgt))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]], is_teacher_forcing = True):# -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_encodings, decoder_init_state, list_len = self.encode(src_sents)

        mask = torch.tensor(self.get_mask(src_sents))

        scores, padded_tgt = self.decode(src_encodings, decoder_init_state, tgt_sents, mask, is_teacher_forcing)

        return scores, padded_tgt

    '''
    All helper functions
    '''
    def sents_to_list_of_tensor(self, sents: List[List[str]], is_src: bool):
        tensor_indices = []
        for sent in sents:
            if is_src:
                tensor_indices.append(torch.tensor(self.vocab.src.words2indices(sent)))
            else:
                tensor_indices.append(torch.tensor(self.vocab.tgt.words2indices(sent)))

        return tensor_indices

    def get_mask(self, sents: List[List[str]]):
        max_s_len = max(len(s) for s in sents)
        size = len(sents)
        mask = []
        for i in range(size):
            mask.append([1.0 if j < len(sents[i]) else 0.0 for j in range(max_s_len)])
        return mask

    def dot_prod_attn(self, q, k, v, mask):
        # (batch_size, len_k, d_k)
        att_score_hidden = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).to(DEVICE)
        # (batch_size, len_k)
        mask = mask.to(DEVICE)
        #pdb.set_trace()
        att_score_weights = torch.softmax(att_score_hidden * mask.unsqueeze(1), dim=-1)
        ctx = torch.bmm(att_score_weights, v).squeeze(1)

        return ctx, att_score_weights

    def encode(self, src_sents: List[List[str]]):# -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        # transfer the sentences to the tensor
        sent_tensors = self.sents_to_list_of_tensor(src_sents, True)
        padded_sent_tensors = rnn.pad_sequence(sent_tensors).to(DEVICE)

        # Embedding the tensor
        embed_tensors = self.encoder_embed(padded_sent_tensors)
        embed_tensors = rnn.pack_padded_sequence(embed_tensors, torch.tensor([len(e) for e in src_sents]))

        encode_out, (encode_last_state, encode_last_cell) = self.encoder_lstm(embed_tensors)

        # Initial variables for decoder
        decode_cell = self.decoder_init_cell(torch.cat([encode_last_cell[0], encode_last_cell[1]], 1))
        decode_state = F.tanh(decode_cell)

        out, list_len = rnn.pad_packed_sequence(encode_out)

        return out, (decode_state, decode_cell), list_len

    def decode(self, src_encodings: Tensor, decoder_init_state, tgt_sents: List[List[str]], mask, is_teacher_forcing):# -> Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """

        hidden = decoder_init_state

        L, B, H = src_encodings.size(0), src_encodings.size(1), src_encodings.size(2)

        # Change dimension
        src_encodings = src_encodings.permute(1, 0, 2)

        # Process target sentences
        sent_tensors = self.sents_to_list_of_tensor(tgt_sents, False)
        padded_sent_tensors = rnn.pad_sequence(sent_tensors).to(DEVICE)

        embed_tensor = self.decoder_embed(padded_sent_tensors).to(DEVICE)

        # Attention
        # Prepare key value
        key, value = self.key_linear(src_encodings), self.value_linear(src_encodings)
        # Initialize attention vector
        attention = torch.zeros(B, self.hidden_size).to(DEVICE)

        scores = []
        idx = 0

        for tgt_embed in embed_tensor.split(split_size=1):
            if is_teacher_forcing:
                if idx == 0 or random.uniform(0, 1) <= 0.9:
                    # Not teacher forcing
                    inp = torch.cat([tgt_embed.squeeze(0), attention], 1)
                else:
                    inp = self.decoder_embed(torch.argmax(out, 1))
                    inp = torch.cat([inp, attention], 1)
            else:
                inp = torch.cat([tgt_embed.squeeze(0), attention], 1)

            h, cell = self.decoder_lstm_cell(inp, hidden)
            h = self.dropout(h)

            # Dot Prod Attetion
            ctx_vector, _ = self.dot_prod_attn(h, key, value, mask)

            #pdb.set_trace()

            # dropout for attention vector
            att = F.tanh(self.att_drp_linear(torch.cat([h, ctx_vector], 1)))
            att = self.dropout(att)

            # Readout layer
            out = self.output_linear(att)

            scores.append(out)

            attention = ctx_vector
            hidden = h, cell
            idx += 1

        # Remove the last one
        scores.pop()

        return torch.stack(scores), padded_sent_tensors

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70):# -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        # hidden_size = 256

        # Just implement the greedy search rather beam search
        # [62, 1, 512] ([1, 256], [1, 256])
        src_encodings, decoder_init_state, list_len = self.encode([src_sent])

        # Change dimension
        src_encodings = src_encodings.permute(1, 0, 2)
        # [62, 16]
        mask = torch.tensor(self.get_mask([src_sent]))

        hypotheses = [torch.tensor(1).numpy()]

        # Attention
        # Prepare key value
        # [62, 1, 256]
        key, value = self.key_linear(src_encodings), self.value_linear(src_encodings)
        # Initialize attention vector
        # [1, 256]
        attention = torch.zeros(1, self.hidden_size).to(DEVICE)

        pred = torch.tensor([1]).to(DEVICE)

        hidden = decoder_init_state

        for time_step in range(max_decoding_time_step):
            if hypotheses[-1] == 2:
                break
            #[1, 256]
            pred_embed = self.decoder_embed(pred)
            #pdb.set_trace()
            inp = torch.cat([pred_embed, attention], 1)
            h, cell = self.decoder_lstm_cell(inp, hidden)
            #pdb.set_trace()
            # h: [1, 256]
            ctx_vector, _ = self.dot_prod_attn(h, key, value, mask)

            #out = self.output_linear(torch.cat([h, ctx_vector], 1))
            out = self.output_linear(self.att_drp_linear(torch.cat([h, ctx_vector], 1)))

            #pdb.set_trace()
            pred = torch.argmax(out, dim=1)

            attention = ctx_vector
            hidden = h, cell

            hypotheses.append(pred.cpu().numpy()[0])

        #pdb.set_trace()
        # Remove the first and end of the hypotheses
        hypotheses = [self.vocab.tgt.id2word[idx] for idx in hypotheses[1:-1]]
        return hypotheses

    def evaluate_ppl(self, dev_data, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                #loss = -model(src_sents, tgt_sents).sum()
                scores, padded_tgt = self.forward(src_sents, tgt_sents, False)
                loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='sum').to('cuda')
                loss = loss_func(scores.permute(0, 2, 1), padded_tgt[1:, :])

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        return torch.load(model_path, map_location=lambda storage, loc: storage)

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self.state_dict(), path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]):# -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float(args['--lr'])
    cuda = bool(args['--cuda'])
    vocab = pickle.load(open(args['--vocab'], 'rb'), encoding="utf-8")
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab).to(DEVICE)

    # Initial operations for training
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    # lr scheduler -> minus lr
    vocab_mask = torch.ones(len(vocab.tgt)).to(DEVICE)
    vocab_mask[vocab.tgt['<pad>']] = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='sum').to('cuda')

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()


    print('begin Maximum Likelihood training')


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            model.train()
            model.zero_grad()
            train_iter += 1

            batch_size = len(src_sents)

            # (batch_size)
            #loss = -model(src_sents, tgt_sents)

            scores, tgt_sent_tensors = model(src_sents, tgt_sents)
            #print(scores)

            loss = loss_func(scores.permute(0, 2, 1), tgt_sent_tensors[1:, :])
            #print(loss)

            report_loss += loss.item()
            cum_loss += loss.item()
            loss.backward()
            opt.step()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size



            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    #model.save(model_save_path)
                    torch.save(model, model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        #model.load(model_save_path)
                        model = torch.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int):# -> List[List[Hypothesis]]:
    was_training = model.training
    with torch.no_grad():
        hypotheses = []
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH']).to(DEVICE)


    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            #top_hyp = hyps[0]
            #hyp_sent = ' '.join(top_hyp.value)
            hyp_sent = ' '.join(hyps)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
