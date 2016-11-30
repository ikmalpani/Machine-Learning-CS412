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
function load_data(file)
  local data, label = {}, {}
  local curr_size = 0

  local f = assert(io.open(file, "r"))
  local line

  -- Read all the way to the last line.
  while true do
    line = f:read()

    if line ~= nil then
      curr_size = curr_size + 1
      data[curr_size] = {}
      local values = mysplit(line, ",")

      -- Data
      for i = 1, #values-1 do
          data[curr_size][i] = tonumber(values[i])
      end

      -- Label
      label[curr_size] = values[#values]

    else break end
  
  end
  f:close()

	return data,label
end
