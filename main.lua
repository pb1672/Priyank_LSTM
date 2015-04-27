require 'cunn';
require('nngraph')
stringx = require('pl.stringx')
require 'io'
ptb = require('data')

-- parameters used for the pretrained model
params = {batch_size=40,
          seq_length=35,
          layers=2,
          decay=1.3,
          rnn_size=1000,
          dropout=0.65,
          init_weight=0.05,
          mom=0.7,
          lr=0.3,
          vocab_size=50,
          max_epoch=4,
          max_max_epoch=50,
          max_grad_norm=5}

function compute_probability()
        -- get the index in the vocab map of the character
    idx = ptb.vocab_map[predictor]
      -- fill p with the same character
    for i=1,params.batch_size do p[i] = idx end
    perp_tmp, model.s[1], prediction_temp = unpack(model.rnns[1]:forward({p, q, model.s[0]}))
    prediction_final = prediction_temp[1]:clone():float()
      
      --  probability in pred 


    for i=1,prediction_final:size(1) do
      io.write(prediction_final[i]..' ')
      io.flush()
    end
      -- replace initial state  with previous state
    g_replace_table(model.s[0], model.s[1])
    io.write('\n')
    io.flush()
end

-- function to move data to GPU
function transfer_data(x)
  return x:cuda()
end

function reset_state()
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end



function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function check_character(line)
  if line.code == "vocab" then
    print("Character is not there in vocabulary: ", line.word)
    flag=1
  else
   print(line)
   print("Please try again")
  end
end


function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  for i = 2,#line do
    -- check to see if the word is in the vocabulary
    if not ptb.vocab_map[line[i]] then error({code="vocab", word = line[i]}) end
  end
  return line
end

state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}

model = torch.load('lmodel_best_3.net')

p = transfer_data(torch.zeros(params.batch_size))
q = transfer_data(torch.ones(params.batch_size))
io.write("OK GO\n")
io.flush()
reset_state()
g_disable_dropout(model.rnns)
g_replace_table(model.s[0], model.start_s)
flag=0

while true do
  local check, line = pcall(readline)
  if not check then
    f=check_character(line)
  else
      -- if a space or blank is entered, an underscore is returned as predicting character
    if next(line) == nil then
      predictor = ' '
    else
      predictor = line[1]
    end

    compute_probability()
  end
end


