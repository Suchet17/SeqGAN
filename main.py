from __future__ import print_function
from math import ceil
import sys
import os
import numpy as np

import pandas as pd
import torch
from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import generator
import discriminator
import helpers

# Stays the same
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

# Data format
VOCAB_SIZE = 4 + 1 # 'A', 'C', 'G', 'T' +  '*'
MAX_SEQ_LEN = 100 + 1
START_LETTER = 0
DATASET = '1_positive_trim100_noimplant'
input_sequences_path = f"Data/{MAX_SEQ_LEN-1}bp/{DATASET}"

# Oracle Training Hyperparams
ORACLE_TRAIN_EPOCHS = 150
ORACLE_BATCH_SIZE = 100
ORACLE_HIDDEN_DIM = 32
ORACLE_EMBEDDING_DIM = 16

# GAN Training Hyperparams
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 50
ADV_TRAIN_EPOCHS = 150
POS_NEG_SAMPLES = 1000

GEN_EMBEDDING_DIM = 16
GEN_HIDDEN_DIM = 64
GEN_LR = 1e-2

DIS_EMBEDDING_DIM = 16
DIS_HIDDEN_DIM = 64
DIS_LR = 1e-2

# =============================================================================
VERSION = '5_realCTCF'
# =============================================================================
oracle_samples_path = f"SeqGAN/{MAX_SEQ_LEN-1}bp_{DATASET}_data.pth"
oracle_state_dict_path = f'SeqGAN/{MAX_SEQ_LEN-1}bp_{DATASET}_stateDict.pth'
pretrained_gen_path = f'SeqGAN/try{VERSION}/{MAX_SEQ_LEN-1}bp_{DATASET}_pretrainedGen.pth'
pretrained_dis_path = f'SeqGAN/try{VERSION}/{MAX_SEQ_LEN-1}bp_{DATASET}_pretrainedDisc.pth'
fullytrained_gen_path = f'SeqGAN/try{VERSION}/{MAX_SEQ_LEN-1}bp_{DATASET}_fullytrainedGen.pth'
fullytrained_dis_path = f'SeqGAN/try{VERSION}/{MAX_SEQ_LEN-1}bp_{DATASET}_fullytrainedDisc.pth'

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs, f):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='', file=f, flush=True)
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          device=device)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='', file=f, flush=True)
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, device=device)

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss), file=f, flush=True)


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches, f):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, device=device)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, device=device)

    print(' oracle_sample_NLL = %.4f' % oracle_loss, file=f, flush=True)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs, f):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, device=device)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, device=device)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='', file=f, flush=True)
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='', file=f, flush=True)
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/(200.)),
                file=f, flush=True)


def prepare_dna_input(filepath):
    """
    Tokenizes input sequences and saves in torch readable format
    """
    # Load your raw text
    with open(filepath, "r") as fi:
        lines = [i.strip() for i in fi.readlines()][1::2]

    # Tokenize: split by characters or words (here: characters)
    tokenized = [list(line.strip()) for line in lines]  # e.g., ['h', 'e', 'l', 'l', 'o']

    # Build vocabulary
    flattened = [token for seq in tokenized for token in seq]
    vocab = sorted(set(flattened))
    vocab = ['*', ] + vocab  # add special tokens
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {s: i for i, s in stoi.items()}
    # vocab_size = len(vocab)
    # print(vocab_size, stoi)

    # Convert to token IDs and pad sequences
    data = []
    for seq in tokenized:
        token_ids = [stoi['*']] + [stoi[ch] for ch in seq]
        data.append(token_ids)

    data_tensor = torch.tensor(data, dtype=torch.long)
    # print("Data shape:", data_tensor.shape)  # (num_samples, seq_len + 1)
    torch.save(data_tensor, oracle_samples_path)

    with open("SeqGAN/vocab_stoi.txt", "w", encoding="utf-8") as f:
        for idx, token in stoi.items():
            f.write(f"{idx}\t{token}\n")
    with open("SeqGAN/vocab_itos.txt", "w", encoding="utf-8") as f:
        for idx, token in itos.items():
            f.write(f"{idx}\t{token}\n")


def produce_output(gen, num_samples):
    itos = {}
    with open("SeqGAN/vocab_itos.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            itos[int(line[0])] = line[1][:-1]

    samples = list(gen.sample(num_samples).to(torch.device('cpu')))
    for i in range(len(samples)):
        samples[i] = [itos[j.item()] for j in samples[i]]
        samples[i] = "".join(samples[i])
    with open(f'SeqGAN/try{VERSION}/Output.fasta', 'w') as f:
        for i in range(len(samples)):
            print(f">{i+1}\n{samples[i][1:]}", file=f, flush=True, end='\n')
    return samples


def train_oracle_MLE(oracle, data):
    inp = data[:, :-1]
    target = data[:, 1:]

    opt = torch.optim.Adam(oracle.parameters(), lr=(1e-3))
    loss_fn = nn.NLLLoss()
    for epoch in range(ORACLE_TRAIN_EPOCHS):
        total_loss = 0
        for i in range(0, len(data), ORACLE_BATCH_SIZE):
            x = inp[i:i+ORACLE_BATCH_SIZE]
            y = target[i:i+ORACLE_BATCH_SIZE]
            hidden = torch.zeros(1, x.size(0), ORACLE_HIDDEN_DIM).to(device)

            loss = 0
            for t in range(x.size(1)):
                out, hidden = oracle(x[:, t], hidden)
                loss += loss_fn(out, y[:, t])

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    return oracle


def train_GAN(oracle, oracle_samples):
    f = open(f"SeqGAN/try{VERSION}/logs.txt", 'w')
    # prepare_dna_input(input_sequences_path+'.fa")

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=device, oracle_init=False)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=device)
    oracle = oracle.to(device)
    gen = gen.to(device)
    dis = dis.to(device)
    oracle_samples = oracle_samples.to(device)

    with open(f'SeqGAN/try{VERSION}/hyperparams.txt', 'w', encoding='utf-8') as h:
        print(f"BATCH_SIZE = {BATCH_SIZE}", f"MLE_TRAIN_EPOCHS = {MLE_TRAIN_EPOCHS }",
        f"ADV_TRAIN_EPOCHS = {ADV_TRAIN_EPOCHS}", f"POS_NEG_SAMPLES = {POS_NEG_SAMPLES}",
        f"GEN_EMBEDDING_DIM = {GEN_EMBEDDING_DIM}", f"GEN_HIDDEN_DIM = {GEN_HIDDEN_DIM}",
        f"DIS_EMBEDDING_DIM = {DIS_EMBEDDING_DIM}", f"DIS_HIDDEN_DIM = {DIS_HIDDEN_DIM}",
        f"DIS_LR = {DIS_LR}", f"GEN_LR = {GEN_LR}", f"GEN_params = {helpers.count_parameters(gen)}",
        f"DIS_params = {helpers.count_parameters(dis)}", f"DATASET = {DATASET}",
        file = h, flush=True, sep = '\n')

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...', file=f, flush=True)
    gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
    train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS, f)

    torch.save(gen.state_dict(), pretrained_gen_path)
    gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...', file=f, flush=True)
    dis_optimizer = optim.Adagrad(dis.parameters(), lr=DIS_LR)
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 25, 3, f)

    torch.save(dis.state_dict(), pretrained_dis_path)
    dis.load_state_dict(torch.load(pretrained_dis_path))


    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...', file=f, flush=True)
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, device=device)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss, file=f, flush=True)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('EPOCH %d' % (epoch+1), file=f, flush=True)
        # TRAIN GENERATOR
        print('Adversarial Training Generator : ', end='', file=f, flush=True)
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 5, f)

        # TRAIN DISCRIMINATOR
        print('Adversarial Training Discriminator : ', file=f, flush=True)
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 3, 3, f)

    torch.save(dis.state_dict(), fullytrained_dis_path)
    torch.save(gen.state_dict(), fullytrained_gen_path)

    f.close()
    return gen

if __name__ == '__main__':
    if not os.path.exists(f'SeqGAN/try{VERSION}'):
        os.makedirs(f'SeqGAN/try{VERSION}')

    oracle = generator.Generator(ORACLE_EMBEDDING_DIM, ORACLE_HIDDEN_DIM, VOCAB_SIZE,
                                 MAX_SEQ_LEN, device=device, oracle_init=False)

    if not os.path.exists(oracle_samples_path):
        print("Preparing DNA Input")
        prepare_dna_input(f'Data/{MAX_SEQ_LEN-1}bp/{DATASET}.fa')
    oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)

    oracle = oracle.to(device)
    oracle_samples = oracle_samples.to(device)

    if not os.path.exists(f'SeqGAN/{DATASET}_fullytrainedOracle.pth'):
        print("Training Oracle")
        oracle = train_oracle_MLE(oracle, oracle_samples)
        torch.save(oracle.state_dict(), f'SeqGAN/{DATASET}_fullytrainedOracle.pth')

    if not os.path.exists(fullytrained_gen_path):
        print("Training GAN")
        gen = train_GAN(oracle, oracle_samples)
    else:
        gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM , VOCAB_SIZE , MAX_SEQ_LEN)
        gen.load_state_dict(torch.load(fullytrained_gen_path, weights_only=True))

    print("Generating Output Samples")
    output = produce_output(gen, 100000)

    if os.path.exists(input_sequences_path+'.tr'):
        motif = pd.read_csv(input_sequences_path+'.tr', sep='\t',
                            names=["index", 'a', 't', 'c', 'g'], skiprows=2,
                            skipfooter=1, usecols=[1,2,3,4], engine='python').to_numpy()
        np.save(f'SeqGAN/try{VERSION}/motif.npy', motif)
