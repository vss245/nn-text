import utils
#LSTM to generate text
output = utils.get_text()
chars = list(set(output)) #unique characters
size = len(chars)
#map between indexes and characters
idx_char = {idx:char for idx, char in enumerate(chars)} #a number for every char
char_idx= {char:idx for idx, char in enumerate(chars)} #a char for every number
print(idx_char)
