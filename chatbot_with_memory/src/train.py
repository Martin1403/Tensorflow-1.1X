import os
import pickle
import shutil

from src.utils.coloured import Print
from src.utils.model import Seq2SeqModel, tf
from src.utils.functions import *
from src.utils.measure import timer


@timer
def start_training(args):
    Print(f'b([INFO]) w(Start training...)')
    word2id = create_vocabulary(args.lin)
    id2word = {i: symbol for symbol, i in word2id.items()}

    lines = get_lines(args.lin)
    input_sequence, output_sequence, memory_n = prepare_dialog(args.con)

    # Model
    tf.compat.v1.reset_default_graph()
    model = Seq2SeqModel(8, word2id, id2word, max_decode=args.maxlen)
    model.declare_placeholders()
    model.create_embeddings(args.emb)
    model.build_encoder(args.hidden)
    model.memory_network(args.hidden)
    model.build_decoder(args.hidden, len(word2id.keys()), word2id['^'], word2id['$'])
    model.compute_loss()
    model.perform_optimization()

    # session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) if args.tf == 'gpu' else tf.Session()
    session = tf.compat.v1.Session(
        config=tf.ConfigProto(allow_soft_placement=True)) if args.tf == 'gpu' else tf.compat.v1.Session()

    session.run(tf.compat.v1.global_variables_initializer())

    n_epochs = args.epoch
    batch_size = args.batch
    max_len = args.maxlen
    learning_rate = args.rate
    dropout_keep_probability = args.dropout

    number_epoch = 1
    for epoch in range(n_epochs):
        if number_epoch % int(args.decstep) == 0:
            learning_rate = float(args.decrate) * learning_rate
        Print(f'b([INFO]) w(Train epoch: {epoch + 1})')
        n_step = int(len(input_sequence[0]) / batch_size)
        for n_iter, (X_batch, Y_batch) in enumerate(
                generate_batches_no_memory(input_sequence[0], output_sequence[0], lines, batch_size=batch_size)):
            input_seq, x_len1 = batch_to_ids(X_batch, word2id, max_len=max_len)
            output_seq, y_len1 = batch_to_ids(Y_batch, word2id, max_len=max_len)
            loss = model.train_on_batch(session, input_seq, x_len1, output_seq, y_len1, [[0, 0]], [1], 0, learning_rate,
                                        dropout_keep_probability, 2)
            if n_iter % 200 == 0:
                Print(f"b([INFO]) w(Dialogue training, Memory size: 0, Epoch: [%d/%d], step: [%d/%d], loss: %f)"
                      % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

        for memory in memory_n.keys():
            n_step = int(len(input_sequence[memory]) / batch_size)
            for n_iter, (X_batch, Y_batch, Z_batch) in enumerate(
                    generate_batches_memory(input_sequence[memory], output_sequence[memory], memory_n[memory], lines,
                                            batch_size=batch_size)):
                input_seq, x_len1 = batch_to_ids(X_batch, word2id, max_len=max_len)
                output_seq, y_len1 = batch_to_ids(Y_batch, word2id, max_len=max_len)
                memory_seq, mem_len = batch_to_ids(Z_batch, word2id, max_len=max_len)

                loss = model.train_on_batch(session, input_seq, x_len1, output_seq, y_len1, memory_seq, mem_len, memory,
                                            learning_rate, dropout_keep_probability, 1)
                if n_iter % 200 == 0:
                    Print(f"b([INFO]) w(Dialogue training, Memory size: %d, Epoch: [%d/%d], step: [%d/%d], loss: %f)" %
                          (memory, epoch + 1, n_epochs, n_iter + 1, n_step, loss))

        number_epoch += 1

    # Save model
    shutil.rmtree(args.model, ignore_errors=True)
    os.makedirs(args.model, exist_ok=True)
    saver = tf.compat.v1.train.Saver()
    saver.save(session, f'{args.model}/model.ckpt')
    pickle.dump(word2id, open(f"{args.model}/word2id.pkl", "wb"))
    pickle.dump(id2word, open(f"{args.model}/id2word.pkl", "wb"))
    pickle.dump(args, open(f"{args.model}/args.pkl", "wb"))
