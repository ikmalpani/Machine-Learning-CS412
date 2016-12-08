--[[
-- Creator: Jillian Aurisano, Sharath Kalkur, Aditi Mallavarapu, Keshav Malpani, Suganya Sivakumar, Hai Tran
-- Date: Nov 15, 2016
-- File: 0-doall.lua
-- 
-- We divide our training process into steps: 
--   + Final data formatting to match the model that we are going to train 
--     ("1-data.lua")
--   + Construct model ("2-model.lua")
--   + Train & test model ("3-train.lua")
-- This file takes control of all files above and executes the entire process.
-- User interactions (graph plot, model saving/loading, etc.) are also handled
-- here.
--]]

-------------------------------------------------------------------------------
-- Libraries
--
-- If you have not installed any of these library, simply open command line
-- and type: luarocks install <library name>
--
require "optim"
require "nn"
require "gnuplot"


-------------------------------------------------------------------------------
-- Constants.
--
DATASET_SIZE = 1
--n_inputs = 124

-- Classes: Gesture/non-gesture
n_output = 4
classes = {'1', '2','3','4'}

-- k-fold cross-validation
NUM_FOLDS = 10

-- Size of each cross-validation subset
SUB_SIZE = math.floor(DATASET_SIZE / NUM_FOLDS)

-------------------------------------------------------------------------------
-- Command line inputs
-- These are the flags that we pass in when we run in command line mode.
-- Reference: https://github.com/torch/tutorials/blob/master/2_supervised/doall.lua
-- Flags that are unnecessary/undecided at the moment are commented out. 
--
print("==> processing options")

cmd = torch.CmdLine()
cmd:text()
cmd:text("Kinect Gesture Classification")
cmd:text()
cmd:text("Options:")
-- Global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- Data:
--cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- Model:
cmd:option("-architecture", "nn", "type of architecture to construct: svm | hmm | nn")
cmd:option("-n_layers", 1, "number of hidden layers (NN only)")
cmd:option("-n_hidden", 30, "number of hidden nodes per layer (NN only)")
-- Loss:
--cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- Training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-shuffle',true,'Shuffle Examples')
--cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-dropout_prob', 0, 'Dropout Probability')
--cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 500, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- Number of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Seed for random number generator
--torch.manualSeed(torch.Generator(), 0)

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


-------------------------------------------------------------------------------
-- Run
--
-- Load functions from other files
--dofile "1-data.lua"
dofile "data.lua"
dofile "2-model.lua"
dofile "3-train.lua"
print("==> Run")


local dataset_train=loadData("train75.csv")
local dataset_test=loadData("test25.csv")

n_inputs=dataset_train[1][1]:size()
n_inputs=tonumber(n_inputs[1])


  local model,criterion = build_nn(n_inputs,opt,4)

 -- Train model. We may output more than one variables here.
train_result = train_nn(dataset_train, model, criterion, opt)
torch.save('test_network.th', train_result)
-- Test model. Again, there may be more than one variables here.
test_result= test_nn(dataset_test, train_result)

print("Test Accuracy is ".. test_result.."%")
print("Goodbye!")
