function train_nn(dataset_train, model, criterion, opt)
		

	-- Perform BP training using the model constructed
	trainer=nn.StochasticGradient(model, criterion)

	-- Set hyperparameters
	trainer.learningRate=opt.learningRate
	trainer.maxIteration=opt.maxIter
	trainer.shuffleIndices=opt.shuffle

	-- Train
	trainer:train(dataset_train)

	return model
end

function argmax(v)
	local maxvalue=torch.max(v)
	for i=1,v:size(1) do
		if v[i]==maxvalue then
			return i
		end
	end
end

function test_nn(dataset_test, model)
tot=0
pos=0
	-- Step over every test example and perform feedforward evaluation
	for i=1, dataset_test:size(1) do
		local x=dataset_test[i][1]
		local y=torch.Tensor(1)		
		y[1]=dataset_test[i][2]
		
		local pred=argmax(model:forward(x))
		--[[
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
		--]]
		-- Use this method of evaluation for NLLCriterion and SoftMax

		
		if math.floor(pred)==math.floor(y[1]) then
			pos=pos+1
		end

		tot=tot+1
		

	end
	
	-- Return Accuracy
 	return ((pos/tot)*100)
end
