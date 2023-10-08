# multi-task learning based on BART, no relationship-aware attentions
# Loading parameters from pre-trained BART

# some modification from with_led
# 1. use pre-trained Tokenizer and Model from BART
# 2. remove global attention
# 3. add <doc-sep> into the tokenizer

# Using the original LED code from huggingface transformers:
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/bart
# The commit on Dec 9, 2022