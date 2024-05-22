#import data
import pandas as pd
from sklearn.model_selection import train_test_split

path_in = 'SI-NLI/'
path_out = 'Fixed_dataset/'

dev = pd.read_csv(path_in + 'dev.tsv', sep='\t')
train = pd.read_csv(path_in + 'train.tsv', sep='\t')

#dev -> test
dev.to_csv(path_out + 'test.tsv', sep='\t', index=False)

#train -> split train and dev
train, dev = train_test_split(train, test_size=0.1, random_state=42)

train.to_csv(path_out + 'train.tsv', sep='\t', index=False)
dev.to_csv(path_out + 'dev.tsv', sep='\t', index=False)
