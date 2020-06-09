# seq2seq, teacher0

# attention
Attention
	units = self.hid_dim*2
	self.w1 = nn.Linear(self.hid_dim*2, units, bias=False) # for encoder_outputs
	self.w2 = nn.Linear(self.hid_dim*2, units, bias=False)  # for decoder_hidden
	self.w3 = nn.Linear(units, 1, bias=False)
