from src.transformer import transformer

src_vocab_size = 101
tgt_vocab_size = 101

model = transformer(src_vocab_size, tgt_vocab_size)