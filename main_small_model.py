import numpy as np

from models.discrete_tandem_queue import Model as M2
from models.discrete_tandem_queue import param_list as P2
from models.garnet import Model as M1
from models.garnet import param_list as P1
from models.rooms_sparse import Model as M3
from models.rooms_sparse import param_list as P3
from models.weakly_coupled import Model as M4
from models.weakly_coupled import param_list as P4
from exam_my.exam_experinment import Experience
exp = Experience()

model_dict = {'0': [M1, P1],
              '1': [M2, P2],
              '2': [M3, P3],
              '3': [M4, P4]}
a = [0]
spar = [0.3, 0.5, 1.0]
state_dim = [100, 100, 100]
for i in a:
    for j in range(3):
        params = model_dict['{}'.format(i)][1][0]
        model_ = model_dict['{}'.format(i)][0]
        params['state_dim'] = state_dim[j]
        params['sparsity_transition'] = spar[j]
        model = model_(params)
        model.create_model()

        exp.run(model, 20000)

t = 0
time_list = [20000, 1000, 20000]
for i in [1, 2, 3]:
    params = model_dict['{}'.format(i)][1][0]
    model_ = model_dict['{}'.format(i)][0]
    model = model_(params)
    model.create_model()
    exp.run(model, time_list[i - 1])
    t += 1
