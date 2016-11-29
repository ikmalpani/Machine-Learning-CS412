function build_nn(n_inputs, opt,n_output)

	local network= nn.Sequential()
	local p=opt.dropout_prob
	-- Activation fn could be Tanh as well, will change according to the requirements
	network.add:(nn.Linear(n_inputs, opt.n_hidden))
	network.add:(nn.Dropout(p))
	network.add:(nn.Sigmoid())

	-- Since it is a simple set of features, multi-layer is not required, will change as per the performance of the NN
	[[

	for i=1,opt.n_layers-1 do
		network.add:(nn.Linear(n_hidden, n_hidden))
		network.add:(nn.Dropout(p))
		network.add:(nn.Sigmoid())
	end

	]]

	network.add:(nn.Linear(opt.n_hidden, n_output))
	network.add:(nn.Sigmoid()) -- Ideal for binary classification	
	--network.add:(nn.LogSoftMax()) -- Choose this if classes are more than 2

	local criterion=nn.MSECriterion() -- If using this, output must be binary or a 1-hot encoded Tensor, use with Sigmoid
	
	--local criterion=nn.ClassNLLCriterion() --Use this if using SoftMax at output


	return network, criterion

end
