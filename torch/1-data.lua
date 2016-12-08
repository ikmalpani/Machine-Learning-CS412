-- References: http://stackoverflow.com/questions/22935906/string-format-and-gsub-in-lua 
-- http://mdtux89.github.io/2015/12/11/torch-tutorial.html

-- Function to split the data at commas
function string:splitAtCommas()
  local sep, values = ",", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) values[#values+1] = c end)
  return values
end

-- Load the features and labels into Tensor
function loadData(dataFile)
  local dataset = {}
local i=0
  for line in io.lines(dataFile) do
    local values = line:splitAtCommas()
    local y = torch.Tensor(1)
    y[1] = values[#values]
    values[#values] = nil
    local x = torch.Tensor(values)
    dataset[i] = {x, y}
    i = i + 1
  end
  function dataset:size() return (i - 1) end 
  return dataset
end
