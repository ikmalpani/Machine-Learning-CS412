--[[
-- Creator: Hai Tran
-- Date: Nov 26, 2016
-- File: hmm.lua
--
-- HMMs (Hidden Markov Models) are trained in a different way compared to other 
-- models: we mainly work with states instead of features, and the probability 
-- of each state is pre-estimate before training. Therefore, I create this 
-- separated file to deal with HMMs.
-- 
--]]


-------------------------------------------------------------------------------
-- Libraries
--
-- If you have not installed any of these library, simply open command line
-- and type: luarocks install <library name>
--
require "gnuplot"


-------------------------------------------------------------------------------
-- Constants.
--
DATASET_SIZE = 1
n_inputs = 124

-- Classes: Gesture/non-gesture
n_class = 5
classes = {'1', '0'}

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
-- Training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
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
-- Load data
--
function load_data(data_path)

end


-------------------------------------------------------------------------------
-- Main
--
-- Data location
local data_path = "../data/"

-- Create a collection of CSV files


-- Load data.
local data, target = load_data(data_collection)

-- 
