--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
ptb = require('data')


-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
params = {batch_size=32,
          seq_length=50,
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

function transfer_data(x)
  return x:cuda()
end

stringx = require('pl.stringx')
require 'io'

model = {}

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  ---including the pred in module
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s),pred})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  ---including pred here
  model.pred = transfer_data(torch.zeros(params.seq_length,params.batch_size,params.vocab_size))
end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i], model.pred[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    ---backward pass for pred
    local dpred = transfer_data(torch.zeros(params.batch_size,params.vocab_size))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, dpred})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end 
  paramdx:mul(params.mom):add(1,paramdx)

   -- update gradients
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  ----exp(5.6 * perp / len)---avg length of word is 5.6
  print("Validation set perplexity : " .. g_f3(torch.exp(5.6 * perp / len)))
  g_enable_dropout(model.rnns)
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    -- check to see if the word is in the vocabulary
    if not ptb.vocab_map[line[i]] then error({code="vocab", word = line[i]}) end
  end
  return line
end
function  generate_next(i,len,p,q,prev_char)
  --  index in the vocab map of the word
  local idx = ptb.vocab_map[prev_char]
  for i=1,params.batch_size do p[i] = idx end
  -- local s = model.s[i - 1]
  perp_tmp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({p, q, model.s[0]}))
  xx = pred_tmp[1]:clone():float()
  xx = torch.multinomial(torch.exp(xx),1)
  io.write(ptb.inverse_vocab_map[xx[1]]..' ')
  g_replace_table(model.s[0], model.s[1])
  prev_char = ptb.inverse_vocab_map[xx[1]]
end

function query_sentences()
  while true do
    print("Query: len word1 word2 etc")
    local check, line = pcall(readline)
    if not check then
      if line.code == "EOF" then
        break 
      elseif line.code == "vocab" then
        print("Character not in vocabulary: ", line.word)
      elseif line.code == "init" then
        print("Start with a number")
      else
        print(line)
        print("Failed, try again")
      end
    else
      for i = 2, #line do 
        io.write(line[i]..' ') 
      end
      -- generate next characater in sequence
      len = line[1]
      g_disable_dropout(model.rnns)
      g_replace_table(model.s[0], model.start_s)
      -- character that will be used to predict the next
      prev_char = line[#line]
      -- tensors to hold words
      local p = transfer_data(torch.zeros(params.batch_size))
      local q = transfer_data(torch.ones(params.batch_size))
      for i = 1,len do
        generate_next(i,len,p,q,prev_char)
      end
      io.write('\n')
    end
  end
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

--function main()
g_init_gpu(arg)
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
-- state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
print("Network parameters:")
print(params)
states = {state_train, state_valid}
for _, state in pairs(states) do
 reset_state(state)
end
setup()
step = 0
epoch = 0
total_cases = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
-- perps
while epoch < params.max_max_epoch do
 perp = fp(state_train)
 if perps == nil then
   perps = torch.zeros(epoch_size):add(perp)
 end
 perps[step % epoch_size + 1] = perp
 step = step + 1
 bp(state_train)
 total_cases = total_cases + params.seq_length * params.batch_size
 epoch = step / epoch_size
 if step % torch.round(epoch_size / 10) == 10 then
   wps = torch.floor(total_cases / torch.toc(start_time))
   since_beginning = g_d(torch.toc(beginning_time) / 60)
   ---perps=exp(5.6*perps:mean())----avg length of word is 5.6
   print('epoch = ' .. g_f3(epoch) ..
         ', train perp. = ' .. g_f3(torch.exp(5.6 * perps:mean())) ..
         ', wps = ' .. wps ..
         ', dw:norm() = ' .. g_f3(model.norm_dw) ..
         ', lr = ' ..  g_f3(params.lr) ..
         ', since beginning = ' .. since_beginning .. ' mins.')
 end
 if step % epoch_size == 0 then
   run_valid()
   print("Saving model")
   torch.save('lmodel_best_3.net',model)
   if epoch > params.max_epoch then
       params.lr = params.lr / params.decay
   end
 end
 if step % 33 == 0 then
   cutorch.synchronize()
   collectgarbage()
 end
end
-- run_test()
print("Training is over.")
query_sentences()
--end