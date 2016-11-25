print("==> Loading data...")

DATASET_SIZE = 1

--------------------------------------------------------------------------------
-- Load data from csv file

function load_data(path)
	local file = assert(io.open(path, "r"))
	local data, label = {}, {}
	local line

	for i = 0, 144 do
		line = file:read()
		for j = 1, 143 do
			data = line[i][j]
		label = line[i][144]
	end

	return data, label
end


--------------------------------------------------------------------------------
-- Test
--
--data, label = load_data("Time_struct_subject0_trial0_aditi2.csv")
--print(data)
