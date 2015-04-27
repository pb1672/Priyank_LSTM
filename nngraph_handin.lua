require 'nngraph';
require 'nn';

torch.manualSeed(123)

t1 = torch.Tensor{2,4,6}
t2 = torch.Tensor{3,6,9}
t3 = torch.Tensor{4,8,12}

--  output through  feed forward 

test = nn.Linear(3,3)
ff_out = torch.add(t1,torch.cmul(t2,test:forward(t3)))
print('Output via feed forward is:',ff_out)

torch.manualSeed(123)

--  nngraph that computes z = x1 + x2 * linear(x3) 
x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Linear(3,3)()

--  elementwise multiplication between x2 and x3 and results into y
y = nn.CMulTable()({x2,x3})
--  table  having addition of x1 and y
z = nn.CAddTable()({x1,y})
-- create a graph node using x1,x2,x3 as inputs
mod = nn.gModule({x1,x2,x3},{z})

--output through gmodule(nngraph)
gmod_out = mod:forward({t1,t2,t3})


print('Output via gModul is: ', gmod_out)
