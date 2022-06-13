import pickle
from src.utils.model import Seq2SeqModel, tf
from src.utils.coloured import Print, Execute


def start_chat(args):
    """
    Interacting with model.
    :param args: argparse
    :return: None
    """
    tf.reset_default_graph()

    word2id = pickle.load(open(f"{args.model}/word2id.pkl", "rb"))
    id2word = pickle.load(open(f"{args.model}/id2word.pkl", "rb"))
    args = pickle.load(open(f"{args.model}/args.pkl", "rb"))

    model = Seq2SeqModel(14, word2id, id2word, args.maxlen)
    model.declare_placeholders()
    model.create_embeddings(args.emb)
    model.build_encoder(args.hidden)
    model.memory_network(args.hidden)
    model.build_decoder(args.hidden, len(word2id.keys()), word2id['^'], word2id['$'])

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) if args.tf == 'gpu' else tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, f"{args.model}/model.ckpt")

    print(Execute('Enter text... Quit[ctrl-c] Reset[reset-r]'))
    Print('b([EXAMPLE INPUT]) w(Hello how are you?)')
    Print('b([EXAMPLE INPUT]) w(reset-r)')
    Print('b([EXAMPLE INPUT]) w(ctrl-c)')

    text = input(f'TEXT: ').lower()
    Print(f'b([INPUT]) w({text})')

    while 'ctrl-c' not in text:
        if 'reset-r' in text:
            Print('r([RESET]) w(Cleaning memory...)')
            model.empty_memory(10)

        output = model.predict_sentence(text, session)
        Print(f'm([OUTPUT]) w({output[:-1]})')
        text = input(f'TEXT: ').lower()
        Print(f'b([INPUT]) w({text})')
