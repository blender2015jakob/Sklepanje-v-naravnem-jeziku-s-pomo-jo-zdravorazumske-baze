#add embedding
from sentence_transformers import SentenceTransformer
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

print("get all available devices:", torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

#import atomic dataset

import pandas as pd

location = "../sloatomic2020/"


#read in the data
df = pd.read_csv(location + "sloatomic_train_fixed_embedding.tsv", sep="\t", header=0)

#print length of the dataset
print(len(df))

#go trough until it doesent have nothing in embedding, and is not equal to previous head_event
for i in range(1000000, len(df)):
	if df["embedding"][i] != df["embedding"][i] and df["head_event"][i] != df["head_event"][i-1]:
		try:
			#calculate embedding on gpu
			emb = model.encode(df["head_event"][i], convert_to_tensor=True).to(device)
			#add the embedding to the dataframe
			df['embedding'][i] = emb
			
		except:
			print("Error in row: ", i)

	print(i)

print("shranjujem")
df.to_csv(location + "sloatomic_train_fixed_embedding_final.tsv", sep="\t", index=False)
print("shranjeno")