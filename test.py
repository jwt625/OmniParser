

#%%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/OmniParser")
print(tokenizer)

# %%
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/OmniParser-v2.0", trust_remote_code=True)
# %%
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
# %%
classifier('i guess they could also "choose" to not interact, and the incoming fluxon just get reflected. How is the interaction "guaranteed" is my main question, there has to be a rate of some sort?')
# %%
classifier('The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.')

# %%
