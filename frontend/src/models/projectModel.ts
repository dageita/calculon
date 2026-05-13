const ProjectModel = ({ set, get }: any = {}) => ({
  curMode: 'guide', // 当前模式， 默认Guide
  curGpu: null as any, // 当前选择的GPU
  curNetwork: null as any, // 当前选择的GPU
  curModel: null as any, // 当前选择的Model
  modelMetrics: null as any, // Model Metrics，根据所选择的model和minibatch size计算而来
  otherConfig: {
    // optimization_strategy: 'No recomputation'
  } as any, // 其他配置
  totalConfig: {
    // data_parallel_degree: 0,
    // number_of_input_tokens: 0,
    // epochs: 0
  } as any,
  recommendConfig: {},
  result: null as any, // 计算结果
  latest_result: null as any, //上一次计算结果
  bm_result: null as any, // benchmark 解析结果
  curIteration: 0, // 当前指针
  autoRecalc: false,
  loading: false,
  showError: false, //  是否显示错误提示
  errorMsg: '', // 错误信息
  changeLog: {
    field: '',
    new_value: '',
    pre_value: ''
  },
  /** 设备上 minibatch 需能被 microbatch 整除（与 UI 提示一致）；无 minibatch_size 时不拦。 */
  checkSize: () => {
    const { curModel, otherConfig } = get();
    if (!curModel || otherConfig?.microbatch_size == null) {
      return false
    }
    const mb = otherConfig.microbatch_size
    if (mb < 1) return false
    if (curModel.minibatch_size != null) {
      if (curModel.minibatch_size % mb !== 0) return false
    }
    return true
  },
  /** batch_size % (data_par * microbatch_size) === 0，与后端 run_calculate 一致。 */
  checkBatchDpMicro: () => {
    const { otherConfig } = get()
    const bs = otherConfig?.batch_size
    const dp = otherConfig?.data_par
    const micro = otherConfig?.microbatch_size
    if (bs == null || dp == null || micro == null) return false
    if (bs < 1 || dp < 1 || micro < 1) return false
    const denom = dp * micro
    return bs % denom === 0
  },
  /** tensor_par * pipeline_par * data_par === num_procs */
  checkParallelProd: () => {
    const { curGpu, otherConfig } = get()
    const tp = otherConfig?.tensor_par
    const pp = otherConfig?.pipeline_par
    const dp = otherConfig?.data_par
    const n = curGpu?.num_procs
    if (tp == null || pp == null || dp == null || n == null) return false
    if (tp < 1 || pp < 1 || dp < 1 || n < 1) return false
    return tp * pp * dp === n
  },
  checkPipeline: () => {
    const { curModel, otherConfig } = get();
    if (!curModel) {
      return false
    }
    const pp = otherConfig?.pipeline_par
    if (pp == null || pp < 1) return false
    const layers = curModel.num_layers ?? curModel.num_blocks
    if (layers == null) return true
    return layers % pp === 0
  },
  checkTotalConfig: () => {
    const { totalConfig } = get();
    const { data_parallel_degree, number_of_input_tokens, epochs } = totalConfig || {}
    if (data_parallel_degree && number_of_input_tokens && epochs) {
      return true
    }
    return false
  },
  setProject: (pro: any) => {
    set((state: any) => {
      return Object.assign(state, pro);
    });
  },
  setOtherConfig: (params: any) => {
    set((state: any) => {
      state.otherConfig = {
        ...state.otherConfig,
        ...params
      }
      return state
    });
  },
  setChangeLog: (field: string, new_value?: any, pre_value?: any) => {
    set((state: any) => {
      if (new_value != pre_value) {
        state.changeLog = {
          field,
          new_value,
          pre_value
        }
      }

      return state
    });
  },
  setRecommendConfig: (key: string, val: any) => {
    set((state: any) => {
      state.recommendConfig[key] = val
      return state
    });
  },
});

export default ProjectModel;
