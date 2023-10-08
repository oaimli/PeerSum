from tokenization_led_fast import LEDTokenizerFast
from modeling_led import LEDForConditionalGeneration
pretrained_model = "allenai/PRIMERA"
tokenizer = LEDTokenizerFast.from_pretrained(pretrained_model)
model = LEDForConditionalGeneration.from_pretrained(pretrained_model)