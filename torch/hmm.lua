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

-- Classes: Gesture/non-gesture
--n_class = 4
classes = {"free_gestures", "init_gesture", "move_gesture", "stop_gesture"}  -- {1, 2, 3, 4}

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
print("==> Processing command line inputs")

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


-------------------------------------------------------------------------------
-- Global variables
--

-- Data & label
local data, label = {}, {}
-- A collection of CSV files
local data_files = {}


-------------------------------------------------------------------------------
-- String splitter
-- Reference: http://stackoverflow.com/questions/1426954/split-string-in-lua
--
function mysplit(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=0
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    i = i + 1
    t[i] = str
  end
  return t
end


-------------------------------------------------------------------------------
-- Load data
--
function load_file(file)
  local curr_size = #data

  local f = assert(io.open(file, "r"))
  local line

  -- The first two lines contain no data.
  f:read()
  f:read()

  -- Read all the way to the last line.
  while true do
    line = f:read()

    if line ~= nil then
      curr_size = curr_size + 1
      data[curr_size] = {}
      local values = mysplit(line, ",")

      -- Data
      for i = 1, #values do
        data[curr_size][1] = tonumber(values[1])
        if i > 3 then
          data[curr_size][i-2] = tonumber(values[i])
        end
      end

      -- Label
      if values[3] == "free_gestures" then label[curr_size] = 1
      elseif values[3] == "init_gesture" then label[curr_size] = 2
      elseif values[3] == "gesture" then label[curr_size] = 3
      elseif values[3] == "stop_gesture" then label[curr_size] = 4 
      end

    else break end
  
  end
  f:close()
end


-------------------------------------------------------------------------------
-- Pre-estimate state probability and transition probability between states
-- based on a small part of the dataset.
--
function calc_probability()
  print("\n==> Initiating probabilities...")
  local transition_prob = {}
  local emission_prob = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, 
                          {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}

  -- Load file(s)
  table.insert(data_files, "../KinectData1/data time structure/aditi1_time/Time_struct_subject0_trial0_aditi1.csv")
  table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial0_suganya1.csv")
  for key, file in pairs(data_files) do
    load_file(file)
  end

  -- Count labels & transitions
  -- free_gestures = 1, init_gesture = 2, gesture = 3, stop_gesture = 4
  local count_label = {0.0, 0.0, 0.0, 0.0}
  local count_transitions = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, 
                            {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}
  -- Iterate through the sample(s)
  for i = 1, #label do
    if label[i] == 1 then
      count_label[1] = count_label[1] + 1    
    elseif label[i] == 2 then
      count_label[2] = count_label[2] + 1
    elseif label[i] == 3 then
      count_label[3] = count_label[3] + 1
    elseif label[i] == 4 then
      count_label[4] = count_label[4] + 1
    end
    -- Transition
    if i < #label and label[i] ~= nil and label[i+1] ~= nil then 
      count_transitions[ label[i] ][ label[i+1] ]
        = count_transitions[ label[i] ][ label[i+1] ] + 1
    end
  end

  -- Normalize the vectors
  count_label = normalize(count_label)
  transition_prob[1] = normalize(count_transitions[1])
  transition_prob[2] = normalize(count_transitions[2])
  transition_prob[3] = normalize(count_transitions[3])
  transition_prob[4] = normalize(count_transitions[4])

  -- Emission probability is a diagonal matrix
  for i = 1, #classes do
    emission_prob[i][i] = count_label[i]
  end

  return torch.Tensor(emission_prob), torch.Tensor(transition_prob)
end


-------------------------------------------------------------------------------
-- Forward-backward algorithm
-- Reference: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
function fwd_bwk(label, start_prob, transition_prob, emission_prob, end_state)
  -- Forward
  local fwd = {}
  local f_prev = torch.Tensor(#classes)
  local p_fwd = 0

  for i = 1, #label do
    fwd[i] = {}
    local x_i = label[i]
    local prev_f_sum = 0
    local f_curr = torch.Tensor(#classes)

    for st = 1, #classes do
      if i == 1 then
        prev_f_sum = start_prob[st]
      else
        for k = 1, #classes do
          prev_f_sum = prev_f_sum + f_prev[k] * transition_prob[k][st]
        end
      end
      f_curr[st] = emission_prob[st][x_i] * prev_f_sum
      fwd[i][st] = f_curr[st]
    end
    f_prev = f_curr
  end

  for k = 1, #classes do
    p_fwd = p_fwd + f_prev[k] * transition_prob[k][end_state]
  end

  -- Backward
  local bkw = {}
  local b_prev = torch.Tensor(#classes)
  local p_bkw = 0

  for i = #label, 1, -1 do
    local x_i_plus = label[i]
    local b_curr = torch.Tensor(#classes)
    bkw[i] = {}

    for st = 1, #classes do
      if i == 1 then
        b_curr[st] = transition_prob[st][end_state]
      else
        for l = 1, #classes do
          b_curr[st] = b_curr[st] + 
              transition_prob[st][l] * emission_prob[l][x_i_plus] * b_prev[l]
        end
      end
      bkw[i][st] = b_curr[st]
    end
    b_prev = b_curr
  end

  for l = 1, #classes do
    p_bkw = p_bkw + start_prob[l] * emission_prob[l][label[1]] * b_prev[l]
  end

  -- Posterior
  local posterior = {}
  for i = 1, #label do
    posterior[i] = {}
    for st = 1, #classes do
      posterior[i][st] = fwd[i][st] * bkw[i][st] / p_fwd
    end
  end

  return p_fwd, p_bkw, posterior
end


-------------------------------------------------------------------------------
-- Normalize a numeric vector
--
function normalize(table)
  local sum = 0.0
  for i = 1, #table do
    sum = sum + table[i]
  end
  for i = 1, #table do
    table[i] = table[i] / sum
  end
  return table
end


-------------------------------------------------------------------------------
-- Main
--

-- Pre-estimate the probabilities
local emission_prob, transition_prob = calc_probability()
local start_prob = torch.Tensor({0.25, 0.25, 0.25, 0.25})

-- Print starting probabilities
print("\nEmission Probability:")
print(emission_prob)
print("\nTransition Probability:")
print(transition_prob)

-- Reset the data collection
data_files, data, label = {}, {}, {}

-- Gather all data files
table.insert(data_files, "../KinectData1/data time structure/aditi1_time/Time_struct_subject0_trial0_aditi1.csv")
table.insert(data_files, "../KinectData1/data time structure/aditi1_time/Time_struct_subject0_trial1_aditi1.csv")
table.insert(data_files, "../KinectData1/data time structure/aditi1_time/Time_struct_subject0_trial2_aditi1.csv")
table.insert(data_files, "../KinectData1/data time structure/aditi1_time/Time_struct_subject0_trial3_aditi1.csv")
table.insert(data_files, "../KinectData1/data time structure/hai1_time/Time_struct_subject0_trial0_hai1.csv")
table.insert(data_files, "../KinectData1/data time structure/hai1_time/Time_struct_subject0_trial1_hai1.csv")
table.insert(data_files, "../KinectData1/data time structure/hai1_time/Time_struct_subject0_trial2_hai1.csv")
table.insert(data_files, "../KinectData1/data time structure/hai1_time/Time_struct_subject0_trial3_hai1.csv")
table.insert(data_files, "../KinectData1/data time structure/hai1_time/Time_struct_subject0_trial4_hai1.csv")
table.insert(data_files, "../KinectData1/data time structure/keshav1_time/Time_struct_subject0_trial0_keshav1.csv")
table.insert(data_files, "../KinectData1/data time structure/keshav1_time/Time_struct_subject0_trial1_keshav1.csv")
table.insert(data_files, "../KinectData1/data time structure/keshav1_time/Time_struct_subject0_trial2_keshav1.csv")
table.insert(data_files, "../KinectData1/data time structure/keshav1_time/Time_struct_subject0_trial3_keshav1.csv")
table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial0_suganya1.csv")
table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial1_suganya1.csv")
table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial2_suganya1.csv")
table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial3_suganya1.csv")
table.insert(data_files, "../KinectData1/data time structure/suganya1_time/Time_struct_subject0_trial4_suganya1.csv")

-- Load data.
print("\n==> Loading data...")
for key, file in pairs(data_files) do
  load_file(file)
end

local p_fwd, p_bkw, posterior = 
    fwd_bwk(label, start_prob, transition_prob, emission_prob, #classes)

--print(p_fwd)
--print(p_bkw)
print(posterior[#posterior])

