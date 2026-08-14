import os,sys
sys.path.insert(0,'test/bw1100')
from phase2_dsv3_op_catalog import compile_dsv3
from phase3_dsv3_common import collect_layer_times
from phase3_dsv3_common import sum_pred
l,a,s,e=compile_dsv3('models/deepseek-v3-671b.json','systems/BW1100.json',matrix_dtype='float8',vector_dtype='bfloat16',seq_size=1024,microbatch_size=1,expert_par=1)
print(s.matrix_dtype, s.grouped_moe_times.keys())
for x in l._moe_layers:
 if 'MlpBlock_MoE_' in x.name and hasattr(x,'batch_seq'):
  print(x.name,x.batch_seq,x.c_in,x.c_out,x.weight_multiplier,x.flop_multiplier,x.bytes_per_element,s.get_grouped_moe_time(x.name,x.batch_seq,x.c_in,x.c_out,x.weight_multiplier,x.flop_multiplier,x.bytes_per_element),x.compute_flops_time('fw'))
for r in collect_layer_times(l,'moe',stages=('fw',)):
 if 'MlpBlock_MoE_' in r.name: print('ROW',r.name,r.pred_f_s)
print('SUM',sum_pred(collect_layer_times(l,'moe',stages=('fw',)),'fw'))
