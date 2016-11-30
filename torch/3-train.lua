function train_nn(input, target, model, criterion, opt, num_folds)
	local error = 0

	-- Get weights and gradient
	local w, dl_dw = model:getParameters()

	-- Optimization function
	opfunc = function(w_new)
	  -- Copy the weights if they were updated in the last iteration
	  -- They are vectors (or Tensors in the world of Torch), so we need to use
	  -- copy() funciton
	  if w ~= w_new then
	    w:copy(w_new)
	  end

	  -- select a new training sample
	  _nidx_ = (_nidx_ or 0) + 1
	  if _nidx_ > #input then _nidx_ = 1 end

	  --for _nidx_  = 1, #target do
		  local x = torch.Tensor(#input[1])
			local y = torch.Tensor(1)

			for j = 1, #input[1] do
				x[j] = input[_nidx_][j]
			end
			y[1] = target[_nidx_]

			--dataset[_nidx_] = {x, y}
		--end

	  -- Reset the gradients (by default, they are always accumulated)
	  dl_dw:zero()

	  -- Evaluate the loss function and its derivative with respect to w
	  -- Step 1: Compute the prediction
	  -- Step 2: Compute the loss (error)
	  -- Step 3: Compute the gradient of the loss
	  -- Step 4: Adjust the weights of the net
	  local prediction = model:forward(x)
	  local loss_w = criterion:forward(prediction, y)
	  local df_dw = criterion:backward(prediction, y)
	  model:backward(x, df_dw)

	  -- return loss and its derivative
	  return loss_w, dl_dw
	end

	for i = 1, #target do
		w_new, fs = optim.sgd(opfunc, w, opt)
		error = error + fs[1]
	end

	return model, error/#target
end

function test_nn(input, label, model)
	--[[
	-- Step over every test example and perform feedforward evaluation
	for i=1, dataset_test:size(1) do
		local x = torch.Tensor(#input[1])
		local y = torch.Tensor(1)

		for j = 1, #input[1] do
			x[j] = input[_nidx_][j]
		end
		y[1] = target[_nidx_]
		
		local pred=model:forward(x)
		
		-- If MSE and Sigmoid at output, we shall use this way to evaluate our output
		if pred[1]>=0.5 then
			pred[1]=1
		else
			pred[1]=0
		end

		if pred[1]==(y[1]) then
			pos=pos+1
		end

		tot=tot+1
		
		-- Use this method of evaluation for NLLCriterion and SoftMax

		if math.floor(pred[1])==math.floor(y[1]) then
			pos=pos+1
		end

		tot=tot+1

	end
	
	-- Return Accuracy
 	--return ((pos/tot)*100)
 	--]]
end
