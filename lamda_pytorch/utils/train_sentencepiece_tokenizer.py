import io
import sentencepiece as spm
from datasets import load_dataset

dataset = load_dataset('conceptofmind/pile_wikipedia_en', split='train', streaming=True, data_files="data/train-0000*-of-00212.parquet")

def batch_iterator(dataset):
    for i in dataset:
        yield i["text"]
        
model = io.BytesIO()

print("\nBeginning sentencepiece tokenizer training...")

spm.SentencePieceTrainer.train(
    sentence_iterator = batch_iterator(dataset), 
    model_writer=model,
    vocab_size=32000,
    model_type='unigram', #formerly BPE but unigram is apparently used in the HF tokenizers for models like T5
)

print("\nWriting to file...")
with open('out.model', 'wb') as f:
    f.write(model.getvalue())
   
#Test the tokenizer :)
KYS = "Keep Your Smile. Keep on smiling"
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.encode_as_pieces(KYS)) #wikipedia_32k_tokenizer: ['▁Keep', '▁Your', '▁Smile', '.', '▁Keep', '▁on', '▁', 's', 'mil', 'ing']
print(sp.encode_as_ids(KYS)) #wikipedia_32k_tokenizer: [15135, 2427, 16767, 5, 15135, 24, 17, 8, 10884, 42]
