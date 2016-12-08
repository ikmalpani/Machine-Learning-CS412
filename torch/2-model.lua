function build_nn(n_inputs,opt,n_output)

	--require "nn"
	local n_hidden=opt.n_hidden
	local network= nn.Sequential()
	
	-- Activation fn could be Tanh as well, will change according to the requirements

	network:add(nn.Linear(n_inputs, n_hidden+10))
	network:add(nn.Sigmoid())

	network:add(nn.Linear(n_hidden+10, n_hidden))
	network:add(nn.Sigmoid())


	network:add(nn.Linear(n_hidden, n_output))
	--network.add:(nn.Sigmoid()) -- Ideal for binary classification	
	network:add(nn.LogSoftMax()) -- Choose this if classes are more than 2

	--local criterion=nn.MSECriterion() -- If using this, output must be binary or a 1-hot encoded Tensor, use with Sigmoid
	
	criterion=nn.ClassNLLCriterion() --Use this if using SoftMax at output

	print(network)
	return network, criterion

end
