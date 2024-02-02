'''
​​© 2020-2022 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

This material may be only be used, modified, or reproduced by or for the U.S. Government pursuant to the license rights granted under the clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact the Office of Technology Transfer at JHU/APL.
'''

import os
import re
import ast
import argparse
import pickle
import itertools
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import gffpandas.gffpandas as gffpd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import H3_pymol_4gms20191204 as h3

from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Concatenate, Dense, \
    Dropout, Embedding, Input, LSTM, add, LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import diff_match_patch as dmp_module

'''
Utility functions for processing and cleaning data.
'''


def split_n(line: str, n: int = 3, step: int = 1):
    '''
    function that splits the string into segments of size n with overlap step.
    inputs:
        line: string of text to split
        n: integer lengths of the substrings
        step: integer step_size of the string splitting
    output:
        list of strings of size n with an overlap of step
    '''
    if len(line) % n > 0:
        line = line + '_' * ((n - len(line) % n))
    return [line[i:i + n] for i in range(0, len(line) - (n - 1), step)]


def clean_data(df_sequences: pd.Series, n: int, step: int):
    '''
    Function for cleaning the data and splitting them into n-mers
    inputs:
        df_sequences: a pandas Series of sequences (strings of base pairs or amino acids). Sequences should not be separated by spaces.
        n: integer lengths of the substrings
        step: integer step size of the string splitting
    output:
        seq_temp: a list of the sequences from the input split using split_n and re-concatenated with spaces (as it's easier for the tokenizer to use)
        allowed_characters_short_train: a list of all the unique tokens in the data
    '''
    seq_temp = df_sequences.apply(lambda x: split_n(x, n, step))
    allowed_characters_short_train = list(set([char for character in seq_temp for char in character]))
    seq_temp = list(seq_temp)
    seq_temp = [' '.join(doc) for doc in seq_temp]
    return seq_temp, allowed_characters_short_train


def gen_bad_sequences(infenc, infdec, inputs: list, n_steps: int, batch_size: int = 150):
    '''
    Generates sequences from a model, specifically a bad one
    inputs:
        infenc: the encoder model
        infdec: the decoder model
        inputs: a list of the input sequence, the cell noise and the state noise
        n_steps: the number of steps in the prediction
        batch_size: batch size for predictions
    '''
    # encode
    cell_noise = inputs[2]
    state_noise = inputs[1]
    input = inputs[0]
    state = infenc.predict([input], batch_size=batch_size)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(len(input))]).reshape(len(input), 1, )
    # collect predictions
    noise = [state_noise, cell_noise]
    output = np.zeros(input.shape)
    for t in tqdm(range(n_steps)):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state + noise)
        # store prediction
        output[..., t:t + 1] = np.argmax(yhat, axis=2)
        # update state
        state = [h, c]
        noise = [np.zeros(cell_noise.shape), np.zeros(cell_noise.shape)]
        # update target sequence
        target_seq = np.argmax(yhat, axis=2)
    return output


def process_x_data(X_train: pd.DataFrame, tokenizer, n: int, step: int, maxlen=None):
    '''
    This function processes the given X_train data into the input and the goal prediction output (for teacher forcing)
    inputs:
        X_train: a pandas dataframe that contains the columns ParentSequence and ChildSequence
        tokenizer: a keras tokenizer for splitting the text
        n: integer lengths of the substrings
        step: integer step_size of the string splitting
        maxlen: either integer or None, it's the maximum length of a sequence. If None, uses the longest sequence. 
    returns:
        tokenized_xtrain: a numpy array of the tokenized sequences
    '''
    # preprocesses the parent and child sequences
    x_train, allowed_characters_x_train = clean_data(X_train.ParentSequence, n, step)
    # tokenizes the words
    tokenized_x_train = tokenizer.texts_to_sequences(x_train)
    if maxlen is None:
        maxlen = max([len(x) for x in tokenized_x_train])
        return_max_len = True
    else:
        return_max_len = False
    # create x and y values
    tokenized_x_train = pad_sequences(tokenized_x_train, maxlen=maxlen, padding='post')
    #
    zeros = np.zeros((tokenized_x_train.shape[0], 1))
    goal_sequences_full_X = np.concatenate([zeros, tokenized_x_train], axis=1)
    goal_sequences_full_X = goal_sequences_full_X[:, 0:maxlen]
    goal_sequences_full_X = goal_sequences_full_X.astype('int')
    if return_max_len:
        return tokenized_x_train, goal_sequences_full_X, maxlen
    else:
        return tokenized_x_train, goal_sequences_full_X


def process_y_data(Y_train: pd.DataFrame, tokenizer, n: int, step: int, maxlen: int):
    '''
    This function processes the given Y_train data into the input and the goal prediction output (for teacher forcing).
    inputs:
        Y_train: a pandas dataframe that contains the columns ParentSequence and ChildSequence
        tokenizer: a keras tokenizer for splitting the text
        n: integer lengths of the substrings
        step: integer step_size of the string splitting
        maxlen: either integer or None, it's the maximum length of a sequence. If None, uses the longest sequence. 
    returns:
        tokenized_xtrain: a numpy array of the tokenized sequences
        goalsequence: a numpy array of the right shifted version of the tokenized_ytrain data with the last column taken off and the first column 0s.
    '''
    # preprocesses the parent and child sequences
    y_train, allowed_characters_y_train = clean_data(Y_train.ChildSequence, n, step)
    # tokenizes the words
    tokenized_y_train = tokenizer.texts_to_sequences(y_train)
    # create x and y values
    tokenized_y_train = pad_sequences(tokenized_y_train, maxlen=maxlen, padding='post')
    #
    zeros = np.zeros((tokenized_y_train.shape[0], 1))
    goal_sequences_full_Y = np.concatenate([zeros, tokenized_y_train], axis=1)
    goal_sequences_full_Y = goal_sequences_full_Y[:, 0:maxlen]
    goal_sequences_full_Y = goal_sequences_full_Y.astype('int')
    #
    return tokenized_y_train, goal_sequences_full_Y


def preprocess_data(train_data_file: str, train_data_dup_file: str, test_data_file: str, test_data_dup_file: str,
                    num_words: int,
                    tokenizer_file=None, raw_training_data_file=None, raw_testing_data_file=None):
    '''
    Function for checking for the existence and loading in or generating data for training and building the tokenizer.
    inputs:
        train_data_file: the csv containing the training data with columns "Parent Sequence" and "Child Sequence" where the parent and child are not the same
        train_data_dup_file: the csv containing the training data with columns "Parent Sequence" and "Child Sequence" where the parent and child are the same
        test_data_file: the csv containing the test data with columns "Parent Sequence" and "Child Sequence" where the parent and child are not the same
        test_data_dup_file: the csv containing the test data with columns "Parent Sequence" and "Child Sequence" where the parent and child are the same
        num_words: total number of words in the tokenizer
        tokenizer_file: if it exists already, the name of the tokenizer file. If not, it will generate a new one
        raw_training_data_file and raw_testing_data_file: if test_data_file does not exist, it will load in the data in raw_training_data_file and raw_testing_data_file, split them into test and training randomly and save them according to the names in train_data_file, train_data_dup_file, test_data_file and test_data_dup_file
    returns:
        tokenizer
    '''
    # Checks if the tokenizer exists. If it not, it will load the data for generating it.
    if tokenizer_file is None:
        # Checks if the csv test_data_file exists that contains the parent child pairs for the test data. If not, then it will generate new ones from the raw_training_data_file csv containing all the parent-child pairs.
        if os.path.isfile(test_data_file):
            X_test2 = pd.read_csv(test_data_dup_file)
            X_train2 = pd.read_csv(train_data_dup_file)
            X_test = pd.read_csv(test_data_file)
            X_train = pd.read_csv(train_data_file)
        else:
            # load all the data
            X_train = pd.read_csv(raw_training_data_file)
            X_test = pd.read_csv(raw_testing_data_file)
            X_all = pd.concat([X_train, X_test])
            # Separate data into test and training based on unique parent sequences after removing pairs with a levenshtein distance greater than 10
            X_all['Diff'] = X_all.apply(lambda x: calculate_levenshtein(x), axis=1)
            X_removed = X_all.loc[X_all['Diff'] > 10]
            X_removed.to_csv("../data/removedPairs.csv", index=False)
            # separates out instances where the parent sequence is equal to the child sequence
            not_duplicates = X_all['ParentSequence'] != X_all['ChildSequence']
            X_all = X_all.loc[not_duplicates]
            # splitting on unique parents so the same parent isn't in the test and training set
            unique_parents = X_all['ParentSequence'].drop_duplicates()
            msk = np.random.uniform(0, 1, len(unique_parents))
            in_test = msk > .9
            in_train = msk <= .9
            #
            unique_parents_test = unique_parents.loc[in_test]
            unique_parents_train = unique_parents.loc[in_train]
            unique_parents_train_list = unique_parents_train.tolist()
            unique_parents_train_list = "|".join(unique_parents_train_list)
            unique_parents_test_list = unique_parents_test.tolist()
            unique_parents_test_list = "|".join(unique_parents_test_list)
            #
            X_test = X_all.loc[X_all['ParentSequence'].str.contains(unique_parents_test_list)]
            X_train = X_all.loc[X_all['ParentSequence'].str.contains(unique_parents_train_list)]
            duplicates = X_test['ParentSequence'] == X_test['ChildSequence']
            X_test_dup = X_test.loc[duplicates]
            duplicates = X_train['ParentSequence'] == X_train['ChildSequence']
            X_train_dup = X_train.loc[duplicates]
            #
            not_duplicates = X_test['ParentSequence'] != X_test['ChildSequence']
            X_test = X_test.loc[not_duplicates]
            not_duplicates = X_train['ParentSequence'] != X_train['ChildSequence']
            X_train = X_train.loc[not_duplicates]
            #
            new_data_directory = os.path.split(test_data_file)[0]
            train_data_file = os.path.join(new_data_directory,'trainUniqueParentsDiffParentChild.csv')
            test_data_file = os.path.join(new_data_directory,'testUniqueParentsDiffParentChild.csv')
            train_data_dup_file = os.path.join(new_data_directory,'trainUniqueParentsSameParentChild.csv')
            test_data_dup_file = os.path.join(new_data_directory,'testUniqueParentsSameParentChild.csv')

            X_test.to_csv(test_data_file, index=False)
            X_train.to_csv(train_data_file, index=False)
            X_test_dup.to_csv(test_data_dup_file, index=False)
            X_train_dup.to_csv(train_data_dup_file, index=False)
        # Calculates the levenshtein distance and removes ones where the distances is greater than 10
        X_train['Diff'] = X_train.apply(lambda x: calculate_levenshtein(x), axis=1)
        X_test['Diff'] = X_test.apply(lambda x: calculate_levenshtein(x), axis=1)
        X_train = X_train.loc[X_train['Diff'] <= 10]
        X_test = X_test.loc[X_test['Diff'] <= 10]
        # preprocesses the parent and child sequences
        x_train, allowed_characters_x_train = clean_data(X_train.ParentSequence)
        y_train, allowed_characters_y_train = clean_data(X_train.ChildSequence)
        x_test, allowed_characters_x_test = clean_data(X_test.ParentSequence)
        y_test, allowed_characters_y_test = clean_data(X_test.ChildSequence)
        #
        x_train2, allowed_characters_x_train2 = clean_data(X_train2.ParentSequence)
        y_train2, allowed_characters_y_train2 = clean_data(X_train2.ChildSequence)
        x_test2, allowed_characters_x_test2 = clean_data(X_test2.ParentSequence)
        y_test2, allowed_characters_y_test2 = clean_data(X_test2.ChildSequence)
        allwords = y_test + y_train + x_train + x_test + y_test2 + y_train2 + x_test2 + x_train2
        # tokenizes the words
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(allwords)
        with open("tokenizer_" + str(num_words) + ".pkl", 'wb') as f:
            pickle.dump(tokenizer, f)
        # tokenizer.texts_to_matrix(allwords,mode = 'tfidf')
    else:
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
    return tokenizer


def load_or_create_unmatching_files(unmatching_data_file: str, unmatching_parent_file: str, unmatching_child_file: str,
                                    tokenized_x_train=None, tokenized_y_train=None, bad_models=None,
                                    hidden_space_dim=128, num_unmatching: int = 10000,  num_bad_generated: int = 10000,
                                    reverse_word_map: dict = None, step_size: int = 1, window_size: int = 3):
    '''
    Function for either loading in or creating unmatching pairs of sequences and bad sequences
    inputs:
        unmatching_data_file: name of file containing the levenshtein difference between the fake/bad generated sequences. It should be an npy file.
        unmatching_parent_file: fake or bad parent sequences file. It should be an npy file
        unmatching_child_file: fake or bad child sequences file. It should be an npy file.
        tokenized_x_train: training data to be randomly assigned to the fake parent-child pairs for the parent sequences
        tokenized_y_train: training data to be randomly assigned to the fake parent-child pairs for the child sequences
        bad_models: a list of two lists for the encoder and decoder model json and h5 files for generating bad data. [[encoder.json, encoder.h5],[decoder.json,decoder.h5]]
        num_unmatching: number of examples not matching
        num_bad_generated: number of examples that are poorly generated
    returns:
        unmatching_sequences_child: the tokenized child sequences that are not part of valid parent-child pairs,
        unmatching_sequences_parent:  the tokenized parent sequences that are not part of valid parent-child pairs,
        unmatching_sequences_diff: the levenshtein distances of the invalid parent-child pairs
    '''
    if os.path.isfile(unmatching_data_file):
        # these are a mix of unmatching sequences created from randomly pairing sequences together, and using sequences created
        # from a bad GAN to help
        unmatching_sequences_child = np.load(unmatching_child_file)
        unmatching_sequences_parent = np.load(unmatching_parent_file)
        unmatching_sequences_diff = np.load(unmatching_data_file)
    else:
        unmatching_sequences_child = []
        unmatching_sequences_parent = []
        unmatching_sequences_child_original = []
        unmatching_sequences_parent_original = []
        unmatching_levenshtein_distance = []
        while len(unmatching_sequences_parent) < num_unmatching:
            select_rand_row_temp = np.random.randint(0, tokenized_x_train.shape[0], size=1)
            select_rand_parent_temp = np.random.randint(0, tokenized_x_train.shape[0], size=1)
            y_random_child = tokenized_y_train[select_rand_row_temp]
            y_random_childs_parent = tokenized_x_train[select_rand_row_temp]
            y_not_parent_temp = tokenized_x_train[select_rand_parent_temp]
            # testing to see if the randomly selected child is not a child of the randomly selected parent
            if (y_not_parent_temp == y_random_childs_parent).all():
                child = tokenized_y_train[select_rand_row_temp]
                not_parent = tokenized_x_train[select_rand_parent_temp]
                lev_dist = np.sum(compare_sequences_levenshtein("".join(decode_sequence(not_parent[0], reverse_word_map, step_size, window_size)),
                                                                "".join(decode_sequence(child[0], reverse_word_map, step_size, window_size))) == np.Inf)
                if lev_dist > 15:
                    unmatching_sequences_child.append(y_random_child)
                    unmatching_sequences_parent.append(y_not_parent_temp)
                    unmatching_sequences_child_original.append(child)
                    unmatching_sequences_parent_original.append(not_parent)
                    unmatching_levenshtein_distance.append(lev_dist)
        #
        if bad_models is not None:
            bad_encoder_files = bad_models[0]
            bad_decoder_files = bad_models[1]
            infenc = load_model_json(bad_encoder_files[0], bad_encoder_files[1])
            infdec = load_model_json(bad_decoder_files[0], bad_decoder_files[1])
            infenc, infdec = check_bad_model(infenc, infdec, bad_encoder_files[1], bad_decoder_files[1])
            while len(unmatching_sequences_parent) < (num_unmatching + num_bad_generated):
                select_rand_parent_temp = np.random.randint(0, tokenized_x_train.shape[0], size=1000)
                X_batch = tokenized_x_train[select_rand_parent_temp]
                hidden_states = np.random.normal(0, 1, size=(1000, hidden_space_dim * 2))
                cell_states = np.random.normal(0, 1, size=(1000, hidden_space_dim * 2))
                # testing to see if the randomly selected child is not a child of the randomly selected parent
                y_hat_batch = gen_bad_sequences(infenc, infdec, [X_batch, hidden_states, cell_states],
                                                tokenized_x_train.shape[1], 1)
                for i in range(y_hat_batch.shape[0]):
                    parentseq = "".join(decode_sequence(X_batch[i], reverse_word_map, step_size, window_size))
                    childseq = ''.join(decode_sequence(y_hat_batch[i], reverse_word_map, step_size, window_size))
                    lev_dist = np.sum(compare_sequences_levenshtein(parentseq, childseq) == np.Inf)
                    # looks for sequences that are only very bad because we want it to know these are very off.
                    if lev_dist > 65:
                        unmatching_sequences_child.append(X_batch[i])
                        unmatching_sequences_parent.append(y_hat_batch[i])
                        unmatching_levenshtein_distance.append(lev_dist)
        #
        for i in range(len(unmatching_sequences_child)):
            if np.ndim(unmatching_sequences_child[i]) == 1:
                unmatching_sequences_child[i] = np.expand_dims(unmatching_sequences_child[i], 0)
                unmatching_sequences_parent[i] = np.expand_dims(unmatching_sequences_parent[i], 0)
            if np.ndim(unmatching_sequences_child[i]) == 3:
                unmatching_sequences_child[i] = unmatching_sequences_child[i][0]
                unmatching_sequences_parent[i] = unmatching_sequences_parent[i][0]
        #
        unmatching_sequences_child = np.concatenate(unmatching_sequences_child)
        unmatching_sequences_parent = np.concatenate(unmatching_sequences_parent)
        unmatching_sequences_diff = np.array(unmatching_levenshtein_distance)
        np.save(unmatching_parent_file, unmatching_sequences_parent)
        np.save(unmatching_child_file, unmatching_sequences_child)
        np.save(unmatching_data_file, unmatching_sequences_diff)
    return unmatching_sequences_child, unmatching_sequences_parent, unmatching_sequences_diff


def check_bad_model(infenc, infdec, bad_encoder_weights_file: str, bad_decoder_weights_file: str):
    '''
    This code checks that the models loaded from file are good and don't have a tensorflow version issue because the models can be old.
    inputs:
        infenc: encoder model
        infdec: decoder model
        bad_encoder_weights_file: the .h5 file location with the weights for the bad encoder model
        bad_decoder_weights_file: the .h5 file location with the weights for the bad decoder model
    returns
        infenc2: a working version of the bad encoder model
        infdec2: a working version of the bad decoder model
    '''
    num_words = infenc.layers[1].input_dim
    embedding_dim = infenc.layers[1].output_dim
    hidden_space_dim = infenc.layers[2].backward_layer.units
    autoencoder, infenc2, infdec2 = define_generator(hidden_space_dim, num_words, embedding_dim)
    infenc2.load_weights(bad_encoder_weights_file)
    infdec2.load_weights(bad_decoder_weights_file)
    return infenc2, infdec2


def predict_sequence(infenc, infdec, inputs: list, n_steps: int):
    '''
    Generates sequences from a model
    inputs
        infenc: the encoder model
        infdec: the decoder model
        input: a list of the input sequence, the cell noise and the state noise
        n_steps: the number of steps in the prediction
    returns
        numpy arrays of predicted sequences
    
    '''
    # encode
    cell_noise = inputs[2]
    state_noise = inputs[1]
    x_data = inputs[0]
    state = infenc.predict_on_batch([inputs[0]])
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(len(x_data))]).reshape(len(x_data), 1, )
    # collect predictions
    noise = [state_noise, cell_noise]
    output = np.zeros(x_data.shape[0:2])
    for t in tqdm(range(n_steps)):
        # predict next char
        yhat, h, c = infdec.predict_on_batch([target_seq] + state + noise)
        output[..., t:t + 1] = np.argmax(yhat, axis=2)
        # update state
        state = [h, c]
        noise = [np.zeros(cell_noise.shape), np.zeros(cell_noise.shape)]
        # update target sequence
        target_seq = output[..., t:t + 1]
    return output


def train_paired(full_train: list, full_test: list, unmatching_files: list, model_file_names: list,
                 hidden_space_dim: int = 128, num_words: int = 4500, embedding_dim: int = 250, losses: list = [[], []],
                 epochs: int = 1, batch_size: int = 256, gen_loops: int = 5,
                 disc_loops: int = 5, first_pass=False, remove_parent_to_parent_training=False, batch_count=None):
    '''
    Function for training the model
    Inputs:
        full_train: A list of the training data containing [parent sequences, child sequences, goal sequences (child but shifted one)]
        full_test: A list of the test data containing [parent sequences, child sequences, goal sequences (child but shifted one)]
        unmatching_files: a list of the unmatching/bad sequences [child sequences, parent sequences, levenshtein distances]
        model_file_names: a list of the model file names for the pretrained model [encoder file name, decoder file name, autoencoder file name]
        hidden_space_dim: The hidden space dimension of the encoder/decoder
        num_words: size of the number of words in the embedding space
        embedding_dim: the shape of the embedding matrix
        losses: any previous losses that the new ones can be appended to
        epochs: number of training epochs
        batch_size: batch size for model training
        gen_loops: the number of epochs for training the generator at each step
        disc_loops: the number of epochs for training the generator at each step
        first_pass: whether this is the first time training a model or if there is further training on an existing model
        remove_parent_to_parent_training: allow instances where the child is equal to the parent.
    Returns:
        d_losses: the history of losses for the discriminator
        Glosses: the history of losses for the generator 
        filenames: names of the saved models
    '''
    # puts the data into it's proper variables
    pt_encoder_file_name, pt_decoder_file_name, pt_autoencoder_file_name = model_file_names
    encoder_file_name, decoder_file_name, autoencoder_file_name = [re.sub('pretrained', '', f) for f in
                                                                   model_file_names]
    discriminator_file_name = re.sub("encoder", "discriminator", encoder_file_name)
    loss_file_name = re.sub("encoder", "Loss", encoder_file_name)
    d_losses = losses[0]
    g_losses = losses[1]

    unmatching_sequences_child = np.load(unmatching_files[0])
    unmatching_sequences_parent = np.load(unmatching_files[1])
    x_train_data = full_train[0]
    y_train_data = full_train[1]
    x_test_data = full_test[0]
    y_test_data = full_test[1]
    goal_sequences_full = full_train[2]
    goal_sequences_full_test = full_test[2]
    train_duplicates = []
    parent_parent = []

    #
    # for loop that removes duplicated (parent->parent) sequences
    test_duplicates = []
    for i in range(len(x_test_data)):
        if (x_test_data[i] != y_test_data[i]).any():
            test_duplicates.append(i)
    #
    x_test_data = x_test_data[test_duplicates]
    y_test_data = y_test_data[test_duplicates]
    goal_sequences_full_test = goal_sequences_full_test[test_duplicates]
    #
    if remove_parent_to_parent_training:
        # for loop that removes duplicated (parent->parent) sequences
        for i in range(len(x_train_data)):
            if (x_train_data[i] != y_train_data[i]).any():
                train_duplicates.append(i)
            else:
                parent_parent.append(i)
        #
        x_train_par_to_par = x_train_data[parent_parent]
        y_train_par_to_par = y_train_data[parent_parent]
        x_train_data = x_train_data[train_duplicates]
        y_train_data = y_train_data[train_duplicates]
        goal_sequences_full = goal_sequences_full[train_duplicates]
        unmatching_sequences_child = np.concatenate([unmatching_sequences_child, y_train_par_to_par])
        unmatching_sequences_parent = np.concatenate([unmatching_sequences_parent, x_train_par_to_par])
    #
    # creates the number of batches
    if batch_count is None:
        batch_count = x_train_data.shape[0] / batch_size

    file_names = [encoder_file_name, decoder_file_name, autoencoder_file_name, discriminator_file_name]
    #
    for e in range(1, epochs + 1):
        if (e == 1) & (first_pass):
            print("loading autoencoder")
            for double in range(2):
                K.clear_session()
                autoencoder, infenc, infdec = define_generator(hidden_space_dim, num_words, embedding_dim)
                autoencoder.load_weights(pt_autoencoder_file_name)
                discriminator = define_discriminator(embedding_dim, num_words, hidden_space_dim,
                                                    load_file=pt_encoder_file_name)
                gan = define_GAN(discriminator, autoencoder, hidden_space_dim)
        else:
            # loads the models with the saved weights
            for double in range(2):
                K.clear_session()
                autoencoder, infenc, infdec = define_generator(hidden_space_dim, num_words, embedding_dim)
                autoencoder.load_weights(pt_autoencoder_file_name)
                discriminator = define_discriminator(embedding_dim, num_words, hidden_space_dim)
                gan = define_GAN(discriminator, autoencoder, hidden_space_dim)
            infdec.load_weights(decoder_file_name)
            infenc.load_weights(encoder_file_name)
            autoencoder.load_weights(autoencoder_file_name)
            discriminator.load_weights(discriminator_file_name)  # 
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        maxlen = x_train_data.shape[-1]
        #
        for _ in tqdm(range(batch_count)):
            # get a random set of input noise and images
            draw_sequences = np.random.randint(0, x_train_data.shape[0], size=batch_size)
            X_batch = x_train_data[draw_sequences]
            Y_batch_real = y_train_data[draw_sequences]
            #
            # Generate fake sequences
            hidden_states = np.random.normal(0, 1, size=(batch_size, hidden_space_dim * 2))
            cell_states = np.random.normal(0, 1, size=(batch_size, hidden_space_dim * 2))
            Yhat_batch = predict_sequence(infenc, infdec, [X_batch, hidden_states, cell_states], X_batch.shape[1])
            #
            # Generates pairs of parents and non-children sequences to pair them with
            draw_sequences_unmatching = np.random.randint(0, unmatching_sequences_parent.shape[0], size=batch_size)
            unmatching_child_batch = unmatching_sequences_child[draw_sequences_unmatching]
            unmatching_parent_batch = unmatching_sequences_parent[draw_sequences_unmatching]
            #
            # combines all the training data for the discriminator
            X_prime = np.concatenate([X_batch, X_batch, unmatching_parent_batch])
            X = np.zeros((batch_size * 3, maxlen, num_words))
            X[0:batch_size, ...] = to_categorical(Y_batch_real, num_words)
            X[batch_size:batch_size * 2, ...] = to_categorical(Yhat_batch, num_words)
            X[2 * batch_size:batch_size * 3, ...] = to_categorical(unmatching_child_batch, num_words)
            #
            # label assignment: Real Parent - Real Child: 1
            #                   Real Parent - Generated Child: 0 when using binary crossentropy or -1 when using Wasserstein
            #                   Fake Parent - Real Child: 0 when using binary crossentropy or -1 when using Wasserstein            
            y_wasser = np.ones((3 * batch_size, 1))
            y_wasser[:batch_size, 0] = -1
            #
            # train discriminator
            discriminator.trainable = True
            dL = []
            for disc_iter in range(disc_loops):
                print('Fit Discriminator Epoch ' + str(disc_iter + 1) + '/' + str(disc_loops))
                dloss = discriminator.train_on_batch([X_prime, X], y_wasser)
                print('Discriminator Loss: ' + str(dloss))
                dL.append(dloss)
            # ##
            # ##train generator
            # creates noise vectors for the model
            hidden_states = np.random.normal(0, 1, size=(len(draw_sequences), hidden_space_dim * 2))
            cell_states = np.random.normal(0, 1, size=(len(draw_sequences), hidden_space_dim * 2))
            #
            # randomly draws sequences to be used for training the encoder-decoder
            x_train_data_rand = x_train_data[draw_sequences]
            goal_sequences_full_rand = goal_sequences_full[draw_sequences]
            y_train_data_rand = y_train_data[draw_sequences]
            y_wasser_output = -1 * np.ones((len(draw_sequences), 1))
            discriminator.trainable = False
            gL = []
            for gen_iters in range(gen_loops):
                print('Fit Generator Epoch ' + str(gen_iters + 1) + '/' + str(gen_loops))
                gloss = gan.train_on_batch(
                    [x_train_data_rand, hidden_states, cell_states, goal_sequences_full_rand.astype("int")],
                    [y_wasser_output, np.expand_dims(y_train_data_rand, -1)])
                print('Generator Loss: ' + str(gloss))
                gL.append(gloss)
        discriminator.save_weights(discriminator_file_name)
        infdec.save_weights(decoder_file_name)
        infenc.save_weights(encoder_file_name)
        autoencoder.save_weights(autoencoder_file_name)
        d_losses.append([dL])
        g_losses.append([gL])
        with open(loss_file_name, 'wb') as f:
            pickle.dump([d_losses, g_losses], f)

    return d_losses, g_losses, file_names


def proper_format_changes(changes: list, proper_formatting_convention: pd.DataFrame):
    '''
    Formatting the mutations in a biology specific notation
    inputs
        changes: list of changes (original amino acid, location, expected amino acid, generated amino acid)
        proper_formatting_convention: the dataframe containing the start of a segment of the protein, the end, and the region name
                returns the list changes, but reformatted to reflect notation meaningful for biologists and bioinformaticists, specifically (original amino acid)(location)(new amino acid)
    '''
    reformatted_changes = []
    for i in range(len(changes)):
        locations = proper_formatting_convention.loc[
            (proper_formatting_convention.start <= changes[i][0]) & (proper_formatting_convention.end >= changes[i][0])]
        regions = [str(x) + ": " for x in list(locations.index)]
        numbers = [str(x) for x in list(changes[i][0] + 1 - locations.start)]
        reformatted = [r + changes[i][1][1] + n + changes[i][2][1] for r, n in zip(regions, numbers)]
        reformatted_changes.append(reformatted)
    return reformatted_changes


def proper_format_changes_partially(changes: list, proper_formatting_convention: pd.DataFrame):
    '''
    Formatting the mutations in a biology specific notation
    inputs
        changes: list of changes (original amino acid, location, expected amino acid, generated amino acid)
        proper_formatting_convention: the dataframe containing the start of a segment of the protein, the end, and the region name
    returns
        the list changes, but reformatted to reflect notation meaningful for biologists and bioinformaticists, specifically (original amino acid)(location)(new amino acid)
    '''
    reformatted_changes = []
    for i in range(len(changes)):
        locations = proper_formatting_convention.loc[
            (proper_formatting_convention.start <= changes[i][0]) & (proper_formatting_convention.end >= changes[i][0])]
        regions = [str(x) + ": " for x in list(locations.index)]
        numbers = [str(x) for x in list(changes[i][0] + 1 - locations.start)]
        reformatted = [r + changes[i][1][1] + n + changes[i][3][1] for r, n in zip(regions, numbers)]
        reformatted_changes.append(reformatted)
    return reformatted_changes


'''
Functions used for evaluating the results
'''


def split_multiple_changes(idx: int, initial: str, final: str):
    '''
    Function that splits a change that lists multiple differences in one set
    inputs:
        idx: the initial index of the change location
        initial: the original string
        final: the new string that initial was changed to
    returns:
        a reformatted list version of the changes but broken down into individual changes. 
    '''
    maxlen = max(len(initial[1]), len(final[1]))
    if len(initial[1]) == maxlen:
        final = (final[0], list(final[1] + '_' * (maxlen - len(final[1]))))
    if len(final[1]) == maxlen:
        initial = (initial[0], list(initial[1] + '_' * (maxlen - len(initial[1]))))
    split_change = []
    for i in range(maxlen):
        split_change.append((idx + i, (initial[0], initial[1][i]), (final[0], final[1][i])))
    return split_change


def find_where_changes_made(reference_sequences: list, gen_sequence: list):
    '''
    Finds all the changes that are made explicitly and reports where they are made in the format
    inputs:
        reference_sequences: a list of strings that is the sequence that is being compared against ['a','b','c']
        gen_sequence: a string that is the sequence that is being compared ['a','b','d']
    returns:
        a list of all the changes and the locations where they were made
    '''
    dmp = dmp_module.diff_match_patch()
    reference_sequences = "".join(reference_sequences)
    gen_sequence = "".join(gen_sequence)
    diff = dmp.diff_main(reference_sequences, gen_sequence)
    dmp.diff_cleanupSemantic(diff)
    diff_skip_first = diff[1:] + ['end']
    match_count = []
    changes_made = []
    skip = False
    index = 0
    for d, e in zip(diff, diff_skip_first):
        if skip:
            skip = False
            continue
        #
        if d[0] == 0:
            match_count = match_count + [0] * len(d[1])
            index = index + len(d[1])
        #
        if (d[0] == -1) & (e[0] == 1):
            match_count = match_count + [np.inf] * max(len(d[1]), len(e[1]))
            index = index + len(d[1])
            changes_made = changes_made + [(index + 1 - len(d[1]), d, e)]
            skip = True
        #
        if (d[0] == -1) & (e[0] == 0):
            match_count = match_count + [np.inf] * len(d[1])
            index = index + len(d[1])
            changes_made = changes_made + [(index, d)]
        #
        if (d[0] == 1) & (e[0] == 0):
            match_count = match_count + [np.inf] * len(d[1])
            index = index + len(d[1])
            changes_made = changes_made + [(index, d)]
        #
        if (d[0] == -1) & (e == 'end'):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [(index + 1, d)]
        #
        if (d[0] == 1) & (e == 'end'):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [(index + 1, d)]
        #
    # Cleans up the changes that were found
    match_count = np.array(match_count)
    split_change = []
    for change in changes_made:
        idx = change[0]
        if len(change) == 3:
            split_change = split_change + split_multiple_changes(idx, change[1], change[2])
        if len(change) == 2:
            if change[1][0] == 1:
                initial = (-1, '_')
                final = change[1]
            if change[1][0] == -1:
                initial = change[1]
                final = (1, '_')
            split_change = split_change + split_multiple_changes(idx, initial, final)
    return split_change


def compare_sequences_levenshtein(reference_sequences: str, gen_sequence: str):
    '''
    Finds the number of changes made
    inputs:
        reference_sequences: a list of strings that is the sequence that is being compared against ['a','b','c']
        gen_sequence: a string that is the sequence that is being compared ['a','b','d']
    returns: 
        A comparison of where changes are made on a given string to as to calculate levelshtein distance
    '''
    dmp = dmp_module.diff_match_patch()
    reference_sequences = "".join(reference_sequences)
    gen_sequence = "".join(gen_sequence)
    diff = dmp.diff_main(reference_sequences, gen_sequence)
    dmp.diff_cleanupSemantic(diff)
    diff_skip_first = diff[1:] + ['end']
    match_count = []
    changes_made = []
    skip = False
    for d, e in zip(diff, diff_skip_first):
        if skip:
            skip = False
            continue
        #
        if d[0] == 0:
            match_count = match_count + [0] * len(d[1])
        #
        if (d[0] == -1) & (e[0] == 1):
            match_count = match_count + [np.inf] * max(len(d[1]), len(e[1]))
            changes_made = changes_made + [(d, e)]
            skip = True
        #
        if (d[0] == -1) & (e[0] == 0):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [d]
        #
        if (d[0] == -1) & (e == 'end'):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [d]
        if (d[0] == 1) & (e[0] == 0):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [d]
        if (d[0] == 1) & (e == 'end'):
            match_count = match_count + [np.inf] * len(d[1])
            changes_made = changes_made + [d]
            #
    match_count = np.array(match_count)
    return match_count


def calculate_levenshtein(X: pd.DataFrame, col1: str = 'ParentSequence', col2: str = 'ChildSequence'):
    '''
    Calculates the levnshtein distance
    inputs 
        X: a pandas dataframe with the columsn col1 and col2 containing strings for comparison
        col1: a string, the column name of the reference string (original string) in X that is being compared to
        col2: a string, the column name of the comparison string (new string) in X that is being compared to
    returns:
        an integer that is the edit distance
    '''
    return np.sum(compare_sequences_levenshtein(X[col1], X[col2]) == np.Inf)


def decode_sequence(seq: np.ndarray, reverse_word_map: dict, step_size: int = 1, window_size: int = 3):
    '''
    Maps the numeric encoding of the amino acids
    inputs:
        seq: the tokenized sequence
        reverse_word_map: a dictionary of the numbers -> characters
        step_size: the step_size of the moving window for decoding/tokenizing the sequence
        window_size: the integer window_size for the length of a tokenized
    returns
        a string version of the sequence
    '''
    seq_amino_split = []
    for num in seq:
        if num > 0:
            seq_amino_split.append(reverse_word_map[num])
    seq_amino = seq_amino_split[0]
    for i in range(1, len(seq_amino_split)):
        seq_amino = seq_amino + seq_amino_split[i][(window_size - step_size):]
    return seq_amino


def compare_sequences(rel_sequences: pd.DataFrame):
    '''
    Function for comparing parents and children/generated sequences
    inputs:
        rel_sequences: a dataframe containing the ParentSequence, generatedSequence, ChildSequence, ParentChildChange (changes between parent and child), ParentGenChange (changes between parent and child), ParentGenDiff (levenshtein distance between parent and generated), Diff (levenshtein distance between parent and child, ParentID (indicating the parent's id number
    
    returns
        A dictionary evaluating the generated sequences and comparisons to them.
    '''
    rel_sequences['ParentChildChange'] = rel_sequences['ParentChildChange']
    # calculating the levenshtein distance between the parents and generated sequences and finding where the differences are
    rel_sequences['ParentGenDiff'] = calculate_levenshtein(rel_sequences, 'ParentSequence', 'generatedSequence')
    rel_sequences['ParentGenChange'] = find_where_changes_made(rel_sequences['ParentSequence'],
                                                               rel_sequences['generatedSequence'])
    # creating a list of the known mtuations that were caught, mutations that were missed, and extra mutations (as well as partially caught mutations
    mutations_caught = list(set(rel_sequences['ParentGenChange']).intersection(set(rel_sequences['ParentChildChange'])))
    mutations_missed = list(set(rel_sequences['ParentChildChange']).difference(set(rel_sequences['ParentGenChange'])))
    extra_mutations_temp = list(
        set(rel_sequences['ParentGenChange']).difference(set(rel_sequences['ParentChildChange'])))
    partial_mutations = []
    extra_mutations = extra_mutations_temp.copy()
    for j in range(len(extra_mutations_temp)):
        placeholder = [x for x in mutations_missed if x[0] == extra_mutations_temp[j][0]]
        if len(placeholder) > 0:
            extra_mutations = list(set(extra_mutations).difference(set([extra_mutations_temp[j]])))
        mutations_missed = set(mutations_missed).difference(placeholder)
        placeholder = [(x[0], x[1], x[2], extra_mutations_temp[j][2]) for x in placeholder]
        partial_mutations = partial_mutations + placeholder
    temp = {'ParentSequence': rel_sequences['ParentSequence'], 'ChildSequence': rel_sequences['ChildSequence'],
            'PredictedSequence': rel_sequences['generatedSequence'],
            'diffFromParent': rel_sequences['ParentGenDiff'], 'ParentChildDiff': rel_sequences['Diff'],
            'whereParentGeneratedDifferences': [rel_sequences['ParentGenChange']],
            'whereParentChildDifferences': [rel_sequences['ParentChildChange']],
            'index': [rel_sequences['ParentID']],
            'CompletelyCaughtMutations': mutations_caught,
            'CompletelyExtraMutations': extra_mutations,
            'CompletelyMissedMutations': mutations_missed,
            'PartiallyCaughtMutations': partial_mutations}
    return temp


def pair_generated_and_parent_clean_mutations(data):
    '''
    Better formats the caught, missed, partially caught, and extra mutations
    inputs:
        data: a tuple of the parent sequence and the corresponding rows in the data frame with that parent sequence
    returns:
        a pandas Series of cleaned up mutation locations that aggregates the changes made across all generated sequences for that parent. The data is formatted as (location in the string, (-1,changed amino acid), (1,new amino acid))
    '''
    partially_mutations = [t for T in data[1].PartiallyCaughtMutations.tolist() for t in T]
    missed_mutations = [t for T in data[1].CompletelyMissedMutations.tolist() for t in T]
    caught_mutations = [t for T in data[1].CompletelyCaughtMutations.tolist() for t in T]
    extra_mutations = [t for T in data[1].CompletelyExtraMutations.tolist() for t in T]
    partially_mutations_part = [p[0:3] for p in partially_mutations]
    missed_mutations = list(set(missed_mutations).difference(set(caught_mutations)))
    missed_mutations = list(set(missed_mutations).difference(partially_mutations_part))
    caught_mutations = list(set(caught_mutations))
    data_sub = data[1].iloc[0]
    data_sub.CompletelyExtraMutations = list(set(extra_mutations))
    data_sub.CompletelyMissedMutations = missed_mutations
    data_sub.CompletelyCaughtMutations = caught_mutations
    data_sub.PartiallyCaughtMutations = partially_mutations
    data_sub.PredictedSequence = data[1].PredictedSequence.tolist()
    return data_sub


def gff_name_format(filename):
    '''
    For loading in a gff file containing the information about proper formatting for describing mutations in biological notation
    inputs:
        filename: the location of the gff file
    returns:
        a formatted pandas dataframe that contains the proper formatting structure.
    '''
    #
    annotation = gffpd.read_gff3(filename)
    df = annotation.df
    df = df.loc[~((df.type == "CDS") | (df.type == "region"))]
    #
    df['attributes'] = df.attributes.apply(lambda x: {y.split(" = ")[0]: y.split(" = ")[1] for y in x.split(';')})
    df['name'] = ''
    df.loc[df.type == 'gene', 'name'] = df.attributes.loc[df.type == 'gene'].apply(lambda x: x['gene'])
    df.loc[df.type.str.contains('mature_protein_region'), 'name'] = df.attributes.loc[
        df.type.str.contains('mature_protein_region')].apply(lambda x: x['product'])
    df.loc[df.type.str.contains('prime_UTR'), 'name'] = df.type.loc[df.type.str.contains('prime_UTR')]
    df.loc[df.type == 'stem_loop', 'name'] = df.type.loc[df.type == 'stem_loop'] + [str(x + 1) for x in list(
        range(len(df.type.loc[df.type == 'stem_loop'])))]
    df = df[['name', 'start', 'end']].drop_duplicates()
    df = df.drop(['name'], axis=1)
    return df


def find_sneath_index_file(sneath_file_name):
    '''
    Function for finding the Sneath Index file
    inputs: 
        sneath_file_name: location of the Sneath matrix
    returns: 
        the Sneath similarity matrix 
    '''
    try:
        sneath = pd.read_csv(sneath_file_name, index_col=[0])
        sneath = sneath.loc[~(sneath.isnull().sum(axis=1) > 0)]
    except:
        found_sneaths_index = False
        sneath_directory = None
        for root, dir, files in os.walk("..//..", topdown=False):
            for name in files:
                if re.search(sneath_file_name, os.path.join(root, name)):
                    found_sneaths_index = True
                    sneath_directory = root
                    break
            if found_sneaths_index:
                break
        try:
            sneath = pd.read_csv(os.path.join(sneath_directory, sneath_file_name), index_col=[0])
            sneath = sneath.loc[~(sneath.isnull().sum(axis=1) > 0)]
        except TypeError:
            print("No Sneath Index File found in this directory.")
    return sneath


def evaluate_model_quick(infenc, infdec, unique_test_parents: pd.DataFrame, X_test: pd.DataFrame,
                         reverse_word_map: dict,
                         num_words: int, step_size: int, tokenizer, maxlen: int, proper_formatting: pd.DataFrame,
                         tries: int = 1, window_size: int = 3, hidden_space_dim: int = 128,
                         max_allowed_differences=500, batch_size=10, drop_unknowns=False):
    '''
    A function for quickly evaluating a model.
    inputs:
        infenc: the encoder from the generator of the GAN
        infdec: the decoder from the generator of the GAN
        unique_test_parents: A dataframe of just the unique parent sequences (X_test deduplicated on the parent sequences)
        X_test: All the parent_child pairs in the data
        reverse_word_map: the reverse of the tokenizer (it's a dictionary where the keys are integers and the values are n-mers
        num_words: the number of words allowed in the tokenizer, at most, may be more than the number of words in the tokenizer
        step_size: the step_size of the sliding window over the n-mers
        tokenizer: the object for converting text sequences to integers
        maxlen: the longest the sequences can get
        proper_formatting: a dataframe containing the information on how protein is separated into different markers
        tries: The number of generated sequences per unique parent
        window_size: the size of the window on the n-mers. If using 3mers, window_size is 3
        hidden_space_dim: the size of the hidden space in the GAN
        max_allowed_differences: Max allowed differences between a parent and generated after which it is discarded
        batch_size: how many sequences are generated at a given time. This is to help make it easier with longer sequences and less memoryview
        drop_unknowns: boolean, drops the sequences that contain unknown tokens (tokens that are out of vocabulary).
    returns the analysis of the sequences at a parent level and looking at each sequence
    '''
    tqdm.pandas()
    # drop sequences with unknown tokens (tokens that are out of vocabulary
    if drop_unknowns:
        contains_unknown_token = X_test.ParentSequence.apply(
            lambda x: np.array(tokenizer.texts_to_sequences(split_n(x))))
        X_test = X_test.loc[~contains_unknown_token.apply(lambda x: any([t == [] for t in x]))]
        parent_child_pairs = find_changes_parent_to_child(X_test)
        unique_test_parents_missing = unique_test_parents.ParentSequence.apply(
            lambda x: np.array(tokenizer.texts_to_sequences(split_n(x))))
        unique_test_parents = unique_test_parents.loc[
            ~unique_test_parents_missing.apply(lambda x: any([t == [] for t in x]))]
        parent_tokenized = unique_test_parents.ParentSequence.apply(
            lambda x: np.array([t[0] for t in tokenizer.texts_to_sequences(split_n(x))]))
    else:
        # keep them and replace them with XXX, or UNKNOWN
        unk = len(reverse_word_map) + 1
        reverse_word_map[unk] = 'XXX'
        parent_child_pairs = find_changes_parent_to_child(X_test)
        parent_tokenized = unique_test_parents.ParentSequence.apply(
            lambda x: np.array([t[0] if len(t) > 0 else unk for t in tokenizer.texts_to_sequences(split_n(x))]))
    #
    # process the seqences and pad them so predictions can be made on them
    parent_tokenized = pad_sequences(parent_tokenized.tolist(), maxlen=maxlen, padding='post')
    parent_tokenized = np.repeat(parent_tokenized, batch_size, axis=0)
    tqdm.pandas()
    print('generating sequences')
    generated_sequences = []
    unique_test_parents_repeated = []
    # loops through the data generating tries sequences, batch_size at a time. If there are 24 unique parents, and tries is 50, and batchsize is 10, it will generate 240 sequences and repeat 5 times.
    for i in tqdm(range(0, tries, batch_size)):
        unique_test_parents_repeated.append(unique_test_parents.ParentSequence.repeat(batch_size))
        hidden_states = np.random.normal(0, 1, size=(batch_size * len(unique_test_parents), hidden_space_dim * 2))
        cell_states = np.random.normal(0, 1, size=(batch_size * len(unique_test_parents), hidden_space_dim * 2))
        temp = np.squeeze(
            predict_sequence(infenc, infdec, [parent_tokenized, hidden_states, cell_states], parent_tokenized.shape[1]))
        generated_sequences.append(temp)
    unique_test_parents_repeated = pd.concat(unique_test_parents_repeated)
    generated_sequences = np.concatenate(generated_sequences)
    generated_sequences = [np.array(x) for x in generated_sequences.tolist()]
    generated_sequences = pd.Series(generated_sequences, name="generatedSequence",
                                    index=unique_test_parents_repeated.index)
    # removes any generated sequences that contain tokens outside the allowable tokens
    generated_sequences = generated_sequences.loc[~(generated_sequences.apply(max) > max(reverse_word_map.keys()))]
    # decodes the sequences back into strings of amino acids
    generated_sequences = generated_sequences.progress_apply(
        lambda x: ''.join(decode_sequence(x, reverse_word_map, step_size, window_size)).upper())
    parents_and_generated = pd.DataFrame(generated_sequences).merge(pd.DataFrame(unique_test_parents), left_index=True,
                                                                    right_index=True)
    parents_and_generated = parent_child_pairs.merge(parents_and_generated, left_on="ParentSequence",
                                                     right_on="ParentSequence")
    results = []
    # compares the parents to the generated sequences to see which mutations were caught, missed, extra, and partially caught
    for i in tqdm(parents_and_generated.iterrows()):
        results.append(compare_sequences(i[1]))
    results = pd.DataFrame(results)
    # previews the results
    print(results[['ParentSequence', 'PredictedSequence', 'diffFromParent']])
    # removes sequences that are too different from the parent. This could also be done without ground trught.
    results = results.loc[results.diffFromParent <= max_allowed_differences]
    parent_child_pairs = []
    # reformats the results to make it cleaner
    for parent_seq_group in tqdm(results.groupby("ParentSequence")):
        all_caught_mutations = dict(
            Counter([t for T in parent_seq_group[1].CompletelyCaughtMutations.tolist() for t in T]))
        all_partial_mutations = dict(
            Counter([t for T in parent_seq_group[1].PartiallyCaughtMutations.tolist() for t in T]))
        all_missed_mutations = dict(
            Counter([t for T in parent_seq_group[1].CompletelyMissedMutations.tolist() for t in T]))
        all_extra_mutations = dict(
            Counter([t for T in parent_seq_group[1].CompletelyExtraMutations.tolist() for t in T]))
        temp = pair_generated_and_parent_clean_mutations(parent_seq_group).to_frame().transpose()
        temp['missedCounts'] = [all_missed_mutations]
        temp['caughtCounts'] = [all_caught_mutations]
        temp['partialCounts'] = [all_partial_mutations]
        temp['extraCounts'] = [all_extra_mutations]
        temp['averageDiffFromParent'] = [np.mean(parent_seq_group[1].diffFromParent)]
        temp['stdDiffFromParent'] = [np.std(parent_seq_group[1].diffFromParent)]
        parent_child_pairs.append(temp)
    if len(parent_child_pairs) == 0:
        print("All generated sequences were bad")
        return None, None
    parent_child_pairs = pd.concat(parent_child_pairs)
    print(parent_child_pairs)
    # formats the results by looking at all the sequences individually, and by looking at them grouped by parent sequence
    parent_child_pairs['numMutationsCaught'] = parent_child_pairs.CompletelyCaughtMutations.apply(lambda x: len(x))
    parent_child_pairs['numPartialMutationsCaught'] = parent_child_pairs.PartiallyCaughtMutations.apply(
        lambda x: len(x))
    parent_child_pairs['numMissedMutations'] = parent_child_pairs.CompletelyMissedMutations.apply(lambda x: len(x))
    #
    tqdm.pandas()
    results['properFormatCompletelyCaughtMutations'] = results['CompletelyCaughtMutations'].progress_apply(
        lambda x: proper_format_changes(x, proper_formatting))
    results['properFormatCompletelyExtraMutations'] = results['CompletelyExtraMutations'].progress_apply(
        lambda x: proper_format_changes(x, proper_formatting))
    results['properFormatPartiallyCaughtMutations'] = results['PartiallyCaughtMutations'].progress_apply(
        lambda x: proper_format_changes_partially(x, proper_formatting))
    #
    parent_child_pairs['properFormatCompletelyCaughtMutations'] = parent_child_pairs[
        'CompletelyCaughtMutations'].progress_apply(
        lambda x: proper_format_changes(x, proper_formatting))
    parent_child_pairs['properFormatCompletelyExtraMutations'] = parent_child_pairs[
        'CompletelyExtraMutations'].progress_apply(
        lambda x: proper_format_changes(x, proper_formatting))
    parent_child_pairs['properFormatPartiallyCaughtMutations'] = parent_child_pairs[
        'PartiallyCaughtMutations'].progress_apply(
        lambda x: proper_format_changes_partially(x, proper_formatting))
    results['ChildGenDiff'] = results.progress_apply(lambda x: np.min(
        [np.sum(compare_sequences_levenshtein(y, x.PredictedSequence) == np.Inf) for y in x.ChildSequence]), axis=1)
    print(results[['diffFromParent', 'ParentChildDiff']])
    return results, parent_child_pairs


def find_changes_parent_to_child(parent_child_pairs: pd.DataFrame):
    '''
    Finds the ground truth mutations in the parents and children
    inputs:
        parent_child_pairs: a dataframe of the parent child pairs
    returns:
        a dataframe of the changes made to the parent child pairs
    '''
    parent_child_pairs['ParentChildChange'] = parent_child_pairs.apply(
        lambda x: find_where_changes_made(x['ParentSequence'], x['ChildSequence']), axis=1)
    parent_child_pairs = parent_child_pairs.groupby('ParentSequence').agg(lambda x: tuple(x)).applymap(
        list).reset_index()
    parent_child_pairs['ParentChildChange'].apply(lambda x: list(itertools.chain(x))[0])
    parent_child_pairs['ParentChildChange'] = parent_child_pairs['ParentChildChange'].apply(
        lambda x: [t for T in x for t in T])
    return parent_child_pairs


def true_positive_metric(results: pd.DataFrame, sneath: pd.DataFrame):
    '''
    Calculates the true postive rate and false positive rate and weights them using the Sneath Index matrix. 
    This is done by treating Extra mutations as false positives, correct mutations and true positives, missed mutations as false negatives. 
    True negatives have no meaning here.
    Weighting is done by checking if the similarity between two amino acids is >0.85. If so, then the mutations is deemed ok if it was an extra mutation.
    inputs:
        results: pd.DataFrame of the results with the columns CompletelyCaughtMutations, PartiallyCaughtMutations, CompletelyMissedMutations, CompletelyExtraMutations
        sneath: a pd.DataFrame of the Sneath Similarity matrix
    returns:
        true positive score
        false positive score
        false negative score
        weighted true positive score using the Sneath similarity matrix
        weighted false positive score using the Sneath similarity matrix
        weighted false negative score using the Sneath similarity matrix
        
    '''
    sneath_sub = sneath * (sneath >= .85).astype(int)
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_pos_weight = 0
    false_pos_weight = 0
    false_neg_weight = 0
    gen_mut_counts, child_mut_counts, correct_mut_counts = {}, {}, {}
    results['Missed'] = results.apply(lambda x: list(x.PartiallyCaughtMutations) + list(x.CompletelyMissedMutations),
                                      axis=1)
    results['Extra'] = results.apply(lambda x: list(x.PartiallyCaughtMutations) + (x.CompletelyExtraMutations), axis=1)
    for i in tqdm(results.index):
        true_pos = true_pos + len(results.at[i, 'CompletelyCaughtMutations'])
        false_pos = false_pos + len(results.at[i, 'Extra'])
        false_neg = false_neg + len(results.at[i, 'Missed'])
        #
        extra = results.at[i, 'Extra']
        extra = [x for x in extra if x[2][1] != 'B']
        extra = [x for x in extra if x[2][1] != 'J']
        extra = [x for x in extra if x[1][1] != 'B']
        extra = [x for x in extra if x[1][1] != 'J']
        weights_extra = [sneath_sub.loc[x[1][1], x[2][1]] for x in extra if
                         ((x[2][1] != 'X') & (x[1][1] != '*') & (x[2][1] != '*'))]
        missed = results.at[i, 'Missed']
        missed = [x for x in missed if x[2][1] != 'B']
        missed = [x for x in missed if x[2][1] != 'J']
        missed = [x for x in missed if x[1][1] != 'B']
        missed = [x for x in missed if x[1][1] != 'J']
        weights_missed = [sneath_sub.loc[x[1][1], x[2][1]] for x in missed if
                          ((x[2][1] != 'X') & (x[1][1] != '*') & (x[2][1] != '*'))]
        true_pos_weight = true_pos_weight + len(results.at[i, 'CompletelyCaughtMutations'])
        true_pos_weight = true_pos_weight + np.sum(weights_extra)
        false_pos_weight = false_pos_weight + np.sum([1 - x for x in weights_extra])
        true_pos_weight = true_pos_weight + np.sum(weights_missed)
        false_neg_weight = false_neg_weight + np.sum([1 - x for x in weights_missed])
        for value in results.at[i, 'CompletelyCaughtMutations']:
            if value not in correct_mut_counts:
                correct_mut_counts[value] = 1
            else:
                correct_mut_counts[value] = correct_mut_counts[value] + 1
            if (value not in gen_mut_counts) & (value not in child_mut_counts):
                gen_mut_counts[value] = 1
                child_mut_counts[value] = 1
            elif (value not in gen_mut_counts) & (value in child_mut_counts):
                gen_mut_counts[value] = 1
                child_mut_counts[value] = child_mut_counts[value] + 1
            elif (value in gen_mut_counts) & (value not in child_mut_counts):
                gen_mut_counts[value] = gen_mut_counts[value] + 1
                child_mut_counts[value] = 1
            else:
                gen_mut_counts[value] = gen_mut_counts[value] + 1
                child_mut_counts[value] = child_mut_counts[value] + 1
        for value in results.at[i, 'CompletelyExtraMutations']:
            if value not in gen_mut_counts:
                gen_mut_counts[value] = 1
            else:
                gen_mut_counts[value] = gen_mut_counts[value] + 1
        for value in results.at[i, 'CompletelyMissedMutations']:
            if value not in child_mut_counts:
                child_mut_counts[value] = 1
            else:
                child_mut_counts[value] = child_mut_counts[value] + 1
    #
    print("\t True Positive (%): " + str(round(true_pos / (true_pos + false_neg), 3)))
    print("\t False Positive (%): " + str(round(false_pos / (true_pos + false_pos), 3)))
    print("\t Weighted True Positive (%): " + str(round(true_pos_weight / (true_pos_weight + false_neg_weight), 3)))
    print("\t Weighted False Positive (%): " + str(round(false_pos_weight / (true_pos_weight + false_pos_weight), 3)))
    return true_pos, false_pos, false_neg, true_pos_weight, false_pos_weight, false_neg_weight

def read_in_predictions(parent_level_file_name: str, sequence_level_file_name: str):
    '''
    Code for reading in predictions that have already been made and saved to files
    input:
        parent_level_file_name: parent level sequence file name
        sequence_level_file_name: sequence level file name
    returns
        parent_level_results_unk
        sequence_level_results_unk
    '''
    generic = lambda x: ast.literal_eval(x)
    conv = {'ChildSequence': generic,
            'whereParentGeneratedDifferences': generic,
            'PredictedSequence': generic,
            'ParentChildDiff': generic,
            'whereParentGeneratedDifferences': generic,
            'whereParentChildDifferences': generic,
            'CompletelyCaughtMutations': generic,
            'CompletelyExtraMutations': generic,
            'CompletelyMissedMutations': generic,
            'PartiallyCaughtMutations': generic,
            'missedCounts': generic,
            'caughtCounts': generic,
            'partialCounts': generic,
            'extraCounts': generic,
            'properFormatCompletelyCaughtMutations': generic,
            'properFormatCompletelyExtraMutations': generic,
            'properFormatPartiallyCaughtMutations': generic
            }
    parent_level_results_unk = pd.read_csv(parent_level_file_name, converters=conv)
    conv.pop('PredictedSequence')
    conv.pop('CompletelyMissedMutations')
    sequence_level_results_unk = pd.read_csv(sequence_level_file_name, converters=conv)
    sequence_level_results_unk['CompletelyMissedMutations'] = sequence_level_results_unk['CompletelyMissedMutations'].str.replace('set\(\)', '{}')
    sequence_level_results_unk['CompletelyMissedMutations'] = sequence_level_results_unk['CompletelyMissedMutations'].apply(generic)
    return parent_level_results_unk, sequence_level_results_unk


def confusion_matrix_analysis(tresults: pd.DataFrame, analysis_type: str, results_file_name: str, sneath: pd.DataFrame,
                              directory: str, file: str):
    '''
    Generates a confusion matrix of the predicted and ground truth changes.
    inputs:
        tresults: either the sequence level results or the parent level results
        analysis_type: a string "parent_level_results" or "sequence_level_results"
        results_file_name: file to save the resulting confusion matrix to.
        sneath: Sneath similiarity matrix
        directory: the directory where the file is saved to
        file: the name of the file containing the basic statistical analysis results.
    '''
    type_text = ''
    if analysis_type == 'parent_level_results':
        type_text = 'Parent Sequence level'
        by_parent_sequence = True
    elif analysis_type == "sequence_level_results":
        type_text = 'Sequence level'
        by_parent_sequence = False
    else:
        raise ValueError("Invalid analysis type. Must be either sequence_level_results or parent_level_results")
    df_confusion, predictions_vs_actual = create_confusion_matrices_for_predictions(tresults, results_file_name,
                                                                                    directory,
                                                                                    by_parent_sequence=by_parent_sequence)
    missing = list(set(sneath.columns).difference(set(list(df_confusion.index))))
    for c in missing:
        df_confusion.loc[c] = 0
    missing = list(set(sneath.columns).difference(set(list(df_confusion.columns))))
    for c in missing:
        df_confusion[c] = 0
    df_confusion_sub = df_confusion.loc[sneath.columns, sneath.columns]
    df_confusion_sub = df_confusion_sub.fillna(0)
    accuracy = np.sum(np.trace(df_confusion_sub)) / (np.sum(np.sum(df_confusion_sub)))
    print("***********************************************************************************************")
    print("Accuracy for " + type_text + ": " + str(accuracy))
    sneath_sub = sneath * (sneath >= .85).astype(int)
    df_confusion = pd.crosstab(predictions_vs_actual.actual, predictions_vs_actual.predicted, rownames=['Actual'],
                               colnames=['Predicted'], margins=True)
    missing = list(set(sneath.columns).difference(set(list(df_confusion.index))))
    for c in missing:
        df_confusion.loc[c] = 0
    missing = list(set(sneath.columns).difference(set(list(df_confusion.columns))))
    for c in missing:
        df_confusion[c] = 0
    df_confusion_sub = df_confusion.loc[sneath.columns, sneath.columns]
    df_confusion_sub = df_confusion_sub.fillna(0)
    weighted_accuracy = np.sum(np.sum(sneath_sub * df_confusion_sub)) / (np.sum(np.sum(df_confusion_sub)))
    print("Weighted accuracy for " + type_text + ": " + str(weighted_accuracy))
    file.write("*********************************************************************************************** \n")
    file.write("Accuracy for " + type_text + ": " + str(accuracy) + ' \n')
    file.write("Weighted accuracy for " + type_text + ": " + str(weighted_accuracy) + ' \n')
    return


def basic_statistical_analysis(parent_level_results: pd.DataFrame, sequence_level_results: pd.DataFrame,
                               sneath: pd.DataFrame, directory: str, file: str):
    '''
    Basic statistical analysis of the data including graph generation
    inputs
        parent_level_results: the parent level analysis of the generated sequences
        sequence_level_results: the sequence level analysis of the generated sequences
        sneath: The sneath similarity matrix
        directory: the directory the output is written to
        file: the name of the file the output is written to.
    '''
    mean_mutations_caught = np.mean((sequence_level_results.CompletelyCaughtMutations.apply(lambda x: len(x) > 0)))
    print("Mean Number of Mutations Caught For Generated Sequences: " + str(mean_mutations_caught))
    file.write("Mean Number of Mutations Caught For Generated Sequences: " + str(mean_mutations_caught) + '\n')
    #
    # percent of mutations caught
    mutations_caught_percent = np.sum(parent_level_results['numMutationsCaught']) / (np.sum(
        parent_level_results['numMutationsCaught'] + parent_level_results['numMissedMutations'] + parent_level_results[
            'numPartialMutationsCaught']))
    print("Percent of Mutations Caught For Each Parent Sequences: " + str(mutations_caught_percent))
    file.write("Percent of Mutations Caught For Each Parent Sequences: " + str(mutations_caught_percent) + '\n')
    # percent of mutations caught including partial mutations
    mutations_caught_and_partial_percent = np.sum(
        parent_level_results['numMutationsCaught'] + parent_level_results['numPartialMutationsCaught']) / (np.sum(
        parent_level_results['numMutationsCaught'] + parent_level_results['numMissedMutations'] + parent_level_results[
            'numPartialMutationsCaught']))
    print("Percent of Mutations Caught and partially caught For Each Parent Sequences: " + str(
        mutations_caught_and_partial_percent))
    file.write("Percent of Mutations Caught and partially caught For Each Parent Sequences: " + str(
        mutations_caught_and_partial_percent) + "\n")
    # percent of mutations missed
    mutations_missed_percent = np.sum(parent_level_results['numMissedMutations']) / (np.sum(
        parent_level_results['numMutationsCaught'] + parent_level_results['numMissedMutations'] + parent_level_results[
            'numPartialMutationsCaught']))
    print("Percent of Mutations Missed For Each Parent Sequences: " + str(mutations_missed_percent))
    file.write("Percent of Mutations Missed For Each Parent Sequences: " + str(mutations_missed_percent) + "\n")
    # percent of parents with more than one mutation caught or partially caught
    multiple_mutations_caught = np.sum(
        (parent_level_results['numMutationsCaught'] + parent_level_results['numPartialMutationsCaught']) > 0) / len(
        parent_level_results)
    multiple_mutations_caught_correct = np.sum((parent_level_results['numMutationsCaught']) > 0) / len(
        parent_level_results)
    print("Percent of parents with more than one mutation caught or partially caught For Each Parent Sequences: " + str(
        multiple_mutations_caught))
    file.write(
        "Percent of parents with more than one mutation caught or partially caught For Each Parent Sequences: " + str(
            multiple_mutations_caught) + "\n")
    print("Percent of parents with more than one mutation caught for Each Parent Sequences: " + str(
        multiple_mutations_caught_correct))
    file.write("Percent of parents with more than one mutation caught for Each Parent Sequences: " + str(
        multiple_mutations_caught_correct) + "\n")
    print("********************************************************************************************************")

    # fraction of mutations caught for each parent
    # average number of  differences from parent to generated
    print("********************************************************************************************************")
    mean_differences = np.mean(sequence_level_results['diffFromParent'])
    median_differences = np.median(sequence_level_results['diffFromParent'])
    std_differences = np.std(sequence_level_results['diffFromParent'])
    print("Mean Levenshtein Distance between Parent and Generated: " + str(mean_differences))
    print("Median Levenshtein Distance between Parent and Generated: " + str(median_differences))
    print("Standard Deviation of the Levenshtein Distance between Parent and Generated: " + str(std_differences))
    file.write("Mean Levenshtein Distance between Parent and Generated: " + str(mean_differences) + "\n")
    file.write("Median Levenshtein Distance between Parent and Generated: " + str(median_differences) + "\n")
    file.write(
        "Standard Deviation of the Levenshtein Distance between Parent and Generated: " + str(std_differences) + "\n")
    #
    mean_differences_child = np.mean(sequence_level_results['ChildGenDiff'])
    median_differences_child = np.median(sequence_level_results['ChildGenDiff'])
    std_differences_child = np.std(sequence_level_results['ChildGenDiff'])
    print("Mean Levenshtein Distance between Child and Generated: " + str(mean_differences_child))
    print("Median Levenshtein Distance between Child and Generated: " + str(median_differences_child))
    print("Standard Deviation of the Levenshtein Distance between Child and Generated: " + str(std_differences_child))
    file.write("Mean Levenshtein Distance between Child and Generated: " + str(mean_differences_child) + "\n")
    file.write("Median Levenshtein Distance between Child and Generated: " + str(median_differences_child) + "\n")
    file.write("Standard Deviation of the Levenshtein Distance between Child and Generated: " + str(
        std_differences_child) + "\n")
    #
    print("********************************************************************************************************")
    print("Sequence Level True Positive and False Positive rates:")
    true_pos, false_pos, false_neg, true_pos_weight, false_pos_weight, false_neg_weight = true_positive_metric(
        sequence_level_results, sneath)
    file.write("Sequence Level True Positive and False Positive rates: \n")
    file.write("\t True Positive (%): " + str(round(true_pos / (true_pos + false_neg), 3)) + '\n')
    file.write("\t False Positive (%): " + str(round(false_pos / (true_pos + false_pos), 3)) + '\n')
    file.write("\t Weighted True Positive (%): " + str(
        round(true_pos_weight / (true_pos_weight + false_neg_weight), 3)) + '\n')
    file.write("\t Weighted False Positive (%): " + str(
        round(false_pos_weight / (true_pos_weight + false_pos_weight), 3)) + '\n')
    print('Parent Level True Positive and False Positive rates: ')
    true_pos, false_pos, false_neg, true_pos_weight, false_pos_weight, false_neg_weight = true_positive_metric(
        parent_level_results, sneath)
    file.write("Parent Level True Positive and False Positive rates: \n")
    file.write("\t True Positive (%): " + str(round(true_pos / (true_pos + false_neg), 3)) + '\n')
    file.write("\t False Positive (%): " + str(round(false_pos / (true_pos + false_pos), 3)) + '\n')
    file.write("\t Weighted True Positive (%): " + str(
        round(true_pos_weight / (true_pos_weight + false_neg_weight), 3)) + '\n')
    file.write("\t Weighted False Positive (%): " + str(
        round(false_pos_weight / (true_pos_weight + false_pos_weight), 3)) + '\n')
    #
    percentage = list(
        (parent_level_results['numMutationsCaught'] + parent_level_results['numPartialMutationsCaught']) / (
                parent_level_results['numMutationsCaught'] + parent_level_results['numPartialMutationsCaught'] +
                parent_level_results['numMissedMutations']))
    counts = list((parent_level_results['numMutationsCaught'] + parent_level_results['numPartialMutationsCaught']))
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.set_size_inches(11, 8)
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=20, labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('figure', titlesize=20)
    axs[0].hist(counts, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axs[1].hist(percentage, bins=11)
    axs[0].set_title('Number of Mutations Caught')
    axs[1].set_title('Percent of Mutations Caught')
    axs[0].set_xlabel('Number of Mutations Caught')
    axs[1].set_xlabel('Percent of Mutations Caught')
    axs[0].set_ylabel('Counts')
    plt.savefig(os.path.join(directory, "HistogramOfCaughtAndPartiallyCaughtMutations.png"))
    #
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    N, bins, patches = axs.hist(sequence_level_results.diffFromParent, bins=41, range=(0, 41))
    fig.set_size_inches(11, 8)
    plt.title("Histogram of Levenshtein Distances Between Parent and Generated")
    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Counts")
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('figure', titlesize=20)
    plt.savefig(os.path.join(directory, "HistogramofLevenshteinParentAndGeneratedAllSequences.png"))
    #
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    N, bins, patches = axs.hist(parent_level_results.averageDiffFromParent, bins=41, range=(0, 41))
    fig.set_size_inches(11, 8)
    plt.title("Histogram of Levenshtein Distances Between Parent and Generated")
    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Counts")
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('figure', titlesize=20)
    plt.savefig(os.path.join(directory, "HistogramofLevenshteinParentAndGeneratedByParent.png"))
    #
    num_replicates = sequence_level_results.groupby(['PredictedSequence']).size()
    num_replicates = num_replicates.reset_index().rename(columns={0: 'num_replicates'}).sort_values('num_replicates')
    num_replicates['cumulative_sum'] = num_replicates.cumsum()['num_replicates']
    plt.figure(figsize=(12, 8.5))
    plt.plot(num_replicates['num_replicates'], num_replicates['cumulative_sum'])
    plt.xlabel("Replicates per Generated Sequence")
    plt.ylabel("Cumulative Number of Generated Sequences")
    plt.savefig(os.path.join(directory, "cumulative_replicates_per_sequence.png"))


def make_protein_visuals(X_train: pd.DataFrame, X_valid: pd.DataFrame, results: pd.DataFrame, directory: str,
                         include_histogram=True):
    '''
    Function for generating the protein plots with the counts and locations of mutations made on the proteins. This can include the histogram to the left if include_histogram=True
    inputs:
        X_train: the training data parent sequences
        X_valid: the validation data parent sequences
        results: The sequence level results 
        directory: the directory the output gets written to
        include_histogram: boolean indicating if you want a histogram printed next to the data with locations of mutations 
    '''
    file_names = ['train.txt', 'test.txt', 'gen.txt']
    file_names = [os.path.join(directory, f) for f in file_names]
    columns = ['ParentChildChange', 'ParentChildChange', 'whereParentGeneratedDifferences']
    datas = [X_train, X_valid, results]
    for df, fl, c in zip(datas, file_names, columns):
        results_counts = df[c].apply(lambda x: mutation_locations(x)).explode().value_counts().sort_index()
        results_counts = results_counts.reindex(list(range(0, 567)), fill_value=0).drop(0)
        results_counts.to_csv(os.path.join(directory, fl), index=False, header=False)
    if include_histogram:
        generate_protein_plots(file_names, directory, X_train=X_train, X_valid=X_valid, results=results)
    else:
        generate_protein_plots(file_names, directory, include_histogram=False)


def generate_protein_plots(file_names: list, directory: str, title_font_size: int = 24, figsize=(30, 20),
                           text_font_size: int = 20, textx: int = -25,
                           texty: int = 400, include_histogram=True, X_train=None, X_valid=None, results=None,
                           proper_formatting_convention=None):
    '''
    Function for generating the 3D protein plot. If pymol isn't a valid package, it skips this code
        filenames: list of file names where the data for generating the plot is stored
        directory: the location of the directory for saving data to
        title_font_size: an integer giving the font size of the title
        figsize: a tuple of the plot size
        text_font_size: integer text font size
        textx: Integer text x location,
        texty: Integer text y location, 
        include_histogram: boolean if you want the histogram present in the image or not. If yes, True
        X_train: pd.DataFrame of training data. Must include if include_histogram is True
        X_valid: pd.DataFrame of validation data. Must include if include_histogram is True 
        results: pd.DataFrame of sequence level results
        proper_formatting_convention: pd.DataFrame of proper formatting location
    '''
    try_pymol = False
    try:
        import pymol
        try_pymol = True
    except:
        print(
            "Could not find the package pymol, so code is skipping the generation of the protein visualization, but the inputs required for it will be generated.")
        if include_histogram:
            with open(os.path.join(directory, "outputresults.pkl"), 'wb') as f:
                pickle.dump([results, X_train, X_valid], f)
        print("Data saved to %s for easy implementation on another system." % os.path.join(directory,
                                                                                           "outputresults.pkl"))
    if try_pymol:
        new_file_names = []
        plot_type = ['cartoon']
        for file in file_names:
            for j in plot_type:
                parser = argparse.ArgumentParser(description='Generate images from the output of mutation_rate.sh.')
                args = parser.parse_args()
                args.depth = 0
                args.input = os.path.join(directory, file)
                args.freq = 0.3
                args.out = os.path.join(directory, '4gms-' + file.split(".")[0] + j)
                args.show = j
                new_file_names.append(h3.main(args))
        cartoon_files = [f for f in new_file_names if re.search("cartoon", f)]
        if include_histogram:
            fig, axs = plt.subplots(3, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 1]})
            ax0 = [a[0] for a in axs]
            ax = [a[1] for a in axs]
        else:
            fig, ax = plt.subplots(3, 1, figsize=figsize)
        titles = ['Train', 'Validation', 'Generated']
        ylab = ['Observed Mutations', 'Observed Mutations', 'Predicted Mutations']
        fig.tight_layout()
        if include_histogram:
            parent_child_pairs_train = find_changes_parent_to_child(X_train)
            parent_child_pairs_valid = find_changes_parent_to_child(X_valid)
            plot_gen(parent_child_pairs_train, 'ParentChildChange', ax0[0], 'Training Child',
                     proper_formatting_convention)
            plot_gen(parent_child_pairs_valid, 'ParentChildChange', ax0[1], 'Validation Child',
                     proper_formatting_convention)
            plot_gen(results, 'whereParentGeneratedDifferences', ax0[2], 'Validation Generated',
                     proper_formatting_convention)
        for i, cf in enumerate(cartoon_files):
            ax[i].axis('off')
            ax[i].set_title(titles[i], fontsize=title_font_size)
            ax[i].text(textx, texty, ylab[i], rotation=90, fontsize=text_font_size)
            img = mpimg.imread(cf)
            ax[i].imshow(img)
            fig.tight_layout()
        fig.savefig(os.path.join(directory, "4gms-train_test_gen_cartoon_counts.png"))


def plot_gen(df: pd.DataFrame, column: str, ax, fig_title: str, color_blocks: dict = None,
             proper_formatting_convention=None):
    '''
    Function for generating a histogram of the counts and locations of mutations
    inputs:
        df: pd.DataFrame containing the information for plotting into a histogram
        column: the column that should be counted as a histogram
        ax: the plot the histogram will be plotted on 
        fig_title: the title of the histogram
        color_blocks: dictionary of the locations of bands of color where the key is the color and the values are lists of lists of x values spanning the color blocks.
        proper_formatting_convention: pd.DataFrame of proper formatting convention
    '''
    plt.rcParams.update({'font.size': 16})
    if color_blocks is None:
        color_blocks = {'red': [[137, 177], [202, 245]],
                        'yellow': [[52, 76], [276, 279], [300, 305]],
                        'blue': [[28, 41], [346, 362]]}
    if proper_formatting_convention is None:
        proper_formatting_convention = pd.DataFrame.from_dict({'start': {'Single Peptide': 1, 'HA1': 17, 'HA2': 346},
                                                               'end': {'Single Peptide': 16, 'HA1': 345, 'HA2': 566}})
    texts = {}
    labels = [0]
    lastend = 0
    for n, row in proper_formatting_convention.iterrows():
        texts[n] = (row['end'] - row['start']) / 2 + row['start']
        labels += list(range(1, row['end'] + 1 - lastend))
        lastend = row['end']
    #
    results_counts = df[column].apply(lambda x: mutation_locations(x)).explode().value_counts().sort_index()
    results_counts = results_counts.reindex(list(range(0, proper_formatting_convention['end'].max() + 2)),
                                            fill_value=0).drop(0)
    fig = results_counts.plot.bar(ax=ax, color='black')
    fig.set_xticklabels(labels, rotation=0, fontsize=16)
    textyloc = int(results_counts.max() * -.15)
    for t in texts:
        ax.text(texts[t], textyloc, r'' + t, fontsize=20)
        ax.text(texts[t], textyloc, r'' + t, fontsize=20)
        ax.text(texts[t], textyloc, r'' + t, fontsize=20)
    fig.grid(visible=True, which='major', linestyle='--', linewidth=.5)
    fig.locator_params(axis="x", nbins=40)
    fig.set_title(fig_title, fontsize=24)
    for i in color_blocks:
        for window in color_blocks[i]:
            rect1 = matplotlib.patches.Rectangle((window[0], 0),
                                                 window[1] - window[0], int(results_counts.max() * 1.1),
                                                 color=i, alpha=.25)
            ax.add_patch(rect1)


def plot_mutation_locations(X_train: pd.DataFrame, X_valid: pd.DataFrame, results: pd.DataFrame, directory: str):
    '''
    Plots the histograms of mutations locations for training, testing, and generated
    inputs:
        X_train: pd.DataFrame training data
        X_valid: pd.DataFrame, validation data
        results: pd.DataFrame, sequence level results
        directory: str, directory where the results get saved to.
    '''
    fig, axs = plt.subplots(3, figsize=(20, 10))
    fig.tight_layout()
    parent_child_pairs_train = find_changes_parent_to_child(X_train)
    parent_child_pairs_valid = find_changes_parent_to_child(X_valid)
    plot_gen(parent_child_pairs_train, 'ParentChildChange', axs[0], 'Training Child')
    plot_gen(parent_child_pairs_valid, 'ParentChildChange', axs[1], 'Validation Child')
    plot_gen(results, 'whereParentGeneratedDifferences', axs[2], 'Validation Generated')
    plt.savefig(os.path.join(directory, "PlotMutationLocationsHistogram.png"))
    plt.close()


def plot_gen_subs(df: pd.DataFrame, column: str, ax, text: str):
    '''
    Plots the heatmap of mutation frequency
    inputs:
        df: pd.DataFrame containing the parent-child/generated pairs and the changes that were made to them
        column: column name containing the differences between the two sequences
        ax: the plot
        text: str, title of the plot
    returns:
        new_df: pd.DataFrame, the cleaned up change matrix that was plotted.
    '''
    p2gdf = pd.DataFrame([s for p in df[column].apply(mutation_transitions).tolist() for s in p],
                         columns=['From', 'To'])
    count_series = p2gdf.groupby(['From', 'To']).size()
    new_df = count_series.to_frame(name='size').reset_index()
    new_df = new_df.pivot("From", "To", "size")
    new_df = new_df.fillna(0)
    new_df = new_df.drop(columns=[c for c in ['B', 'J', 'X'] if c in new_df.columns])
    sns.heatmap(new_df, cmap="Greens", ax=ax)
    ax.set_title(text, fontdict={'fontsize': 25})
    plt.rcParams.update({'font.size': 20})
    plt.yticks(rotation=0, fontsize=20)
    return new_df


def plot_diff_subs(df1: pd.DataFrame, df2: pd.DataFrame, ax, text: str):
    '''
    Plots the heatmap of mutation rate changes form parent-child and parent-generated
    inputs:
        df1: pd.DataFrame containing the parent-child pairs and the changes that were made to them
        df2: pd.DataFrame containing the parent-generated pairs and the changes that were made to them
        ax: the plot
        text: str, title of the plot
    returns:
        new_df: pd.DataFrame, the cleaned up change matrix that was plotted.
    '''

    df1 = df1 / np.sum(np.sum(df1))
    df2 = df2 / np.sum(np.sum(df2))
    new_df = df1 - df2
    new_df = new_df.drop(columns=[c for c in ['B', 'J', 'X'] if c in new_df.columns])
    sns.heatmap(new_df, cmap="RdBu", ax=ax)
    ax.set_title(text, fontdict={'fontsize': 25})
    plt.rcParams.update({'font.size': 20})
    plt.yticks(rotation=0, fontsize=20)
    return new_df


def mutation_transitions(x):
    '''
    Formats the changes that are made to a form From and To
    '''
    if len(x) == 0:
        return []
    if isinstance(x[0], list):
        if len(x[0]) == 0:
            return []
        else:
            return [(loc[1][1], loc[2][1]) for loc in x[0]]
    else:
        if len(x) == 0:
            return []
        else:
            return [(loc[1][1], loc[2][1]) for loc in x]


def mutation_locations(all_mutations):
    '''
    Returns the positions of mutations
    '''
    if len(all_mutations) == 0:
        return []
    if isinstance(all_mutations[0], list):
        if len(all_mutations[0]) == 0:
            return []
        else:
            return [loc[0] for loc in all_mutations[0]]
    else:
        if len(all_mutations) == 0:
            return []
        else:
            return [loc[0] for loc in all_mutations]


def plot_mutation_subs(X_train: pd.DataFrame, X_valid: pd.DataFrame, results: pd.DataFrame, directory: str):
    '''
    Plots the confusion matrices
    inputs:        
        X_train: the training parent sequences
        X_valid: the test/validation parent sequences
        results: the sequence level results
        directory: The location where the outputput is saved
    returns:
        three pd.DataFrames containing the difference between the training and validation mutation rates, the training and generated mutation rates, and the validation and generated mutation rates.
    '''
    fig, axs12 = plt.subplots(2, 3, figsize=(40, 20))
    plt.rcParams.update({'font.size': 20})
    axs = axs12[0]
    ax2 = axs12[1]
    fig.suptitle('Mutation Counts Per Amino Acid', fontsize=30)
    fig.tight_layout()
    parent_child_pairs_train = find_changes_parent_to_child(X_train)
    parent_child_pairs_valid = find_changes_parent_to_child(X_valid)
    contains_training = plot_gen_subs(parent_child_pairs_train, 'ParentChildChange', axs[0], 'Training Set')
    contains_validation = plot_gen_subs(parent_child_pairs_valid, 'ParentChildChange', axs[1], 'Validation Child')
    contains_generated = plot_gen_subs(results, 'whereParentGeneratedDifferences', axs[2], 'Validation Generated')
    #
    plt.figtext(0.5, 0.5, r'$\Delta(Amino Acid Mutation Frequency)$', ha="center", va="top", fontsize=30)
    train_validation_freq_diff = plot_diff_subs(contains_validation, contains_training, ax2[0],
                                                "Validation vs. Training")
    train_gen_freq_diff = plot_diff_subs(contains_generated, contains_training, ax2[1], "Generated vs. Training")
    validation_gen_freq_diff = plot_diff_subs(contains_generated, contains_validation, ax2[2],
                                              "Generated vs. Validation")
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(directory, "HeatmapPlots.png"))
    plt.close()
    return train_validation_freq_diff, train_gen_freq_diff, validation_gen_freq_diff


def create_confusion_matrices_for_predictions(test_results, results_file_name, directory, by_parent_sequence=True):
    '''
    Generates a confusion matrix of the predicted and ground truth changes.
    inputs:
        test_results: either the sequence level results or the parent level results
        analysis_type: a string "parent_level_results" or "sequence_level_results"
        results_file_name: file to save the resulting confusion matrix to.
        sneath: Sneath similiarity matrix
        directory: the directory where the file is saved to
        file: the name of the file containing the basic statistical analysis results.
    returns:
        df_confusion pd.DataFrame of the confusion matrix
        predictions_vs_actual pd.Dataframe of predictions vs the actual results :
    '''
    test_results.CompletelyCaughtMutations = test_results.CompletelyCaughtMutations.apply(
        lambda x: {y[0]: y[1:3] for y in x})
    list_extra_mutations = [list(x) for x in test_results.CompletelyExtraMutations.tolist()]
    extra_mutations_flatten = [{'actual': x[1][1], 'predicted': x[2][1]} for extra_muts in list_extra_mutations for x in
                               extra_muts if x[1][1] != x[2][1]]
    list_caught_mutations = [list(x.values()) for x in test_results.CompletelyCaughtMutations.tolist()]
    caught_mutations_flatten = [{'actual': x[1][1], 'predicted': x[1][1]} for caught_muts in list_caught_mutations for x
                                in caught_muts]
    list_missed_mutations = [list(x) for x in test_results.CompletelyMissedMutations.tolist()]
    missed_mutations_flatten = [{'actual': x[2][1], 'predicted': x[1][1]} for missed_muts in list_missed_mutations for x
                                in missed_muts]
    list_partially_caught_mutations = [list(x) for x in test_results.PartiallyCaughtMutations.tolist()]
    partially_caught_mutations_flatten = [{'actual': x[1][1], 'predicted': x[2][1]} for partially_caught_muts in
                                          list_partially_caught_mutations for x in partially_caught_muts]
    #
    # "confusion matrix" for partial mutations
    predictions_vs_actual = pd.concat([pd.DataFrame(partially_caught_mutations_flatten)])
    df_confusion = pd.crosstab(predictions_vs_actual.actual, predictions_vs_actual.predicted, rownames=['Actual'],
                               colnames=['Predicted'], margins=True)
    #
    # "confusion matrix" for partial, missed, caught, mutations
    if by_parent_sequence:
        predictions_vs_actual = pd.concat(
            [pd.DataFrame(partially_caught_mutations_flatten), pd.DataFrame(missed_mutations_flatten),
             pd.DataFrame(caught_mutations_flatten)])
        df_confusion = pd.crosstab(predictions_vs_actual.actual, predictions_vs_actual.predicted, rownames=['Actual'],
                                   colnames=['Predicted'], margins=True)
        df_confusion.to_csv(os.path.join(directory,
                                         "PartialCaughtMissed_" + results_file_name + "_confusionmatrix_parent_level_results.csv"))
    else:
        predictions_vs_actual = pd.concat(
            [pd.DataFrame(partially_caught_mutations_flatten), pd.DataFrame(caught_mutations_flatten),
             pd.DataFrame(extra_mutations_flatten)])
        df_confusion = pd.crosstab(predictions_vs_actual.actual, predictions_vs_actual.predicted, rownames=['Actual'],
                                   colnames=['Predicted'], margins=True)
        df_confusion.to_csv(os.path.join(directory,
                                         "PartialCaughtExtra_" + results_file_name + "_confusionmatrix_sequence_level_results.csv"))
    return df_confusion, predictions_vs_actual


'''
Functions related to Model training
'''


def Wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def define_generator(hidden_space_dim: int, num_tokens: int, latent_dim: int):
    '''
    Defines and generates the models:
    inputs:
        hidden_space_dim: the dimensions of the LSTM encoder/decoder
        num_tokens: number of words in the embedding layers
        latent_dim: the dimensions of the embedding        
    returns
        encoder
        decoder
        autoencoder
    '''
    # define training encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(num_tokens, latent_dim)
    encoder_inputs_emb = encoder_embedding(encoder_inputs)
    encoder_BiLSTM = Bidirectional(LSTM(hidden_space_dim, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_BiLSTM(encoder_inputs_emb)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    encoder_model = Model([encoder_inputs], encoder_states)
    # define training decoder
    decoder_shape = hidden_space_dim * 2
    decoder_inputs = Input(shape=(None,))
    state_noise_inputs = Input(shape=(decoder_shape,), name='state_noise')
    cell_noise_inputs = Input(shape=(decoder_shape,), name='cell_noise')
    decoder_inputs_emb = Embedding(num_tokens, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(decoder_shape, return_sequences=True, return_state=True)
    state_enc = add([encoder_states[0], state_noise_inputs])
    noise_enc = add([encoder_states[1], cell_noise_inputs])
    encoder_states = [state_enc, noise_enc]
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_emb, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, state_noise_inputs, cell_noise_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    # define inference decoder
    decoder_state_input_h = Input(shape=(decoder_shape,))
    decoder_state_input_c = Input(shape=(decoder_shape,))
    decoder_hidden_input = add([decoder_state_input_h, state_noise_inputs])
    decoder_cell_input = add([decoder_state_input_c, cell_noise_inputs])
    decoder_states_inputs = [decoder_hidden_input, decoder_cell_input]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + [decoder_state_input_h, decoder_state_input_c, state_noise_inputs, cell_noise_inputs],
        [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


def define_discriminator(latent_dim: int, num_words: int, hidden_space_dim: int,
                        load_file: str = '../utilities/Influenza_biLSTM_encoder_model_128_4500_weightsV3.h5'):
    '''
    Defines and Discriminator model:
    inputs:
        latent_dim: the dimensions of the embedding
        num_words: number of words in the embedding layers
        hidden_space_dim: the dimensions of the LSTM encoder/decoder
        load_file: a pretrained encoder to preseed the weights (which helps with linking.
    returns:
        the discriminator
    '''
    # Encoder for tokenized input
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(num_words, latent_dim)
    encoder_inputs_emb = encoder_embedding(encoder_inputs)
    encoder_BiLSTM = Bidirectional(LSTM(hidden_space_dim, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_BiLSTM(encoder_inputs_emb)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_state = [state_h, state_c]
    encoder_model = Model([encoder_inputs], encoder_state)
    #
    # encoder for vectorized input where the tokenized version cannot be used because it would create a discontinuity.
    one_hot_inputs = Input(shape=(None, num_words))
    encoder_inputs_one_hot = Dense(latent_dim, activation='linear', use_bias=False)(one_hot_inputs)
    encoder_outputs_one_hot, forward_h_one_hot, forward_c_one_hot, backward_h_one_hot, backward_c_one_hot = encoder_BiLSTM(
        encoder_inputs_one_hot)
    state_h_one_hot = Concatenate()([forward_h_one_hot, backward_h_one_hot])
    state_c_one_hot = Concatenate()([forward_c_one_hot, backward_c_one_hot])
    encoder_state_one_hot = [state_h_one_hot, state_c_one_hot]
    encoder_model_one_hot = Model([one_hot_inputs], encoder_state_one_hot)
    encoder_model.load_weights(load_file)
    encoder_model_one_hot.layers[1].set_weights(encoder_embedding.get_weights())
    #
    # merging the two encoders
    x2 = encoder_model([encoder_inputs])
    x3 = encoder_model_one_hot([one_hot_inputs])
    concat_input = Concatenate()(x2 + x3)
    #
    x4 = Dropout(0.2)(concat_input)
    x4 = BatchNormalization()(x4)  # was 0.35
    x4 = Dense(128)(x4)
    x4 = LeakyReLU(0.1)(x4)
    x4 = Dropout(0.2)(x4)
    x4 = BatchNormalization()(x4)  # was 0.35
    x4 = Dense(64)(x4)
    x4 = LeakyReLU(0.1)(x4)
    x4 = Dropout(0.2)(x4)
    x4 = BatchNormalization()(x4)  # was 0.35
    output_class = Dense(1, activation='linear')(x4)  # was 0.35
    discriminator = Model([encoder_inputs, one_hot_inputs], [output_class])
    discriminator.compile(loss=Wasserstein_loss, optimizer='Adam', metrics=['acc'])
    return discriminator


def define_GAN(discriminator, autoencoder, hidden_space_dim: int):
    '''
    Defines the GAN.
    inputs: 
        discriminator
        autoencoder
        encoder
        decoder
        hidden_space_dim 
    returns:
        gan model
    '''
    state_noise_inputs = Input(shape=(hidden_space_dim * 2,))
    cell_noise_inputs = Input(shape=(hidden_space_dim * 2,))
    parentSequence = Input(shape=(None,))
    goal_inputs = Input(shape=(None,))
    generated_sequence = autoencoder([parentSequence, state_noise_inputs, cell_noise_inputs, goal_inputs])
    output_class = discriminator([parentSequence, generated_sequence])
    gan = Model([parentSequence, state_noise_inputs, cell_noise_inputs, goal_inputs], [output_class, generated_sequence])
    gan.compile(loss=[Wasserstein_loss, 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.001))
    return (gan)


def load_model_json(model_filename, model_weights_filename):
    # loads a model with a json and corresponding weights files.
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model
