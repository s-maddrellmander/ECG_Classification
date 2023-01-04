import logging
import time
from jsonargparse import ArgumentParser


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info(f"\U00002B50 {self.name}:")
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time.time() - self.t
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Timedelta makes for human readable format - not safe for maths operations
        logging.info(
            f"\U0001F551 Elapsed time for {self.name}: {elapsed_time} (HH:MM:SS)"
        )


def parse_cmd_args():
    parser = ArgumentParser()
    parser.add_argument("--model.name", default="Transformer")
    # Add Transformer arguments
    parser.add_argument("--model.n_layers",
                        default=6,
                        type=int,
                        help="Number of Transformer encoder layers")
    parser.add_argument("--model.n_hid",
                        default=32,
                        type=int,
                        help="Transformer hidden size")
    parser.add_argument("--model.emb_dim",
                        default=32,
                        type=int,
                        help="Embedding hidden dimmension")
    parser.add_argument("--model.n_token",
                        default=256,
                        type=int,
                        help="Number of embedding tokens")
    parser.add_argument("--model.n_head",
                        default=4,
                        type=int,
                        help="Number of heads for multi-headed self attention")
    parser.add_argument("--model.dropout",
                        default=0.1,
                        type=float,
                        help="Dropout rate for Transformer layers")
    # Add model arguments to work with new transformer architecture
    parser.add_argument("--model.hidden_size", default=128, type=int, help="Transformer Encoder hidden dimmension")
    parser.add_argument("--model.intermediate_size", default=128, type=int, help="Feed Forward intermediate dimmension")
    parser.add_argument("--model.embed_dim", default=128, type=int, help="Transformer Encoder embedding dimmension")
    parser.add_argument("--model.num_heads", default=4, type=int, help="Number of heads for Multi-Headed self attention")
    parser.add_argument("--model.num_hidden_layers", default=4, type=int, help="Number of Multi-Headed self attention layers in the encoder")
    parser.add_argument("--model.hidden_dropout_prob", default=0.1, type=float, help="Dropout Probability")
    parser.add_argument("--model.vocab_size", default=10000, type=int, help="Vocab size for embedding")  # NOTE: This will go later
    parser.add_argument("--model.max_position_embeddings", default=1000, type=int, help="Max sequence length")
    parser.add_argument("--model.num_labels", type=int, default=5, help="Number of class labels to predict")
    
    
    
    # Training dynamics arguments
    parser.add_argument("--model.learning_rate",
                        default=0.001,
                        type=float,
                        help="Learning rate for the model")
    parser.add_argument("--model.epochs",
                        default=10,
                        type=int,
                        help="Number of epochs to train for")

    args = parser.parse_args()

    return args
