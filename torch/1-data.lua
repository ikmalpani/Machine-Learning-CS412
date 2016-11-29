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
        --data[curr_size][1] = tonumber(values[1])
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

	return data,label
end
