import { FC, Fragment } from 'react';
import { Result, Divider, Button, Alert, Descriptions } from 'antd'
import { useImmer } from 'use-immer';
import useModel from 'flooks';
import styles from './index.less';
import ProjectModel from '@/models/projectModel';
import BenchPanel from './benchmark'
import { LoadingOutlined, CaretDownOutlined, CaretRightOutlined, ExportOutlined } from '@ant-design/icons';
import { sum } from 'lodash';
import Steps from '../guide-steps'
import OptimalSteps from '../optimal-steps'
import FileSaver from 'file-saver'
import { exportResult } from '@/services';
import LogModel from '@/models/logModel';
import { useTranslation } from 'react-i18next';
import BaseTL from '../timelines/base-timeline';
import TLEventChart from '../timelines/timeline-events';

const COLOR_MAPPING: any = {
  warmup: {
    label: 'Warmup time',
    color: '#3793FF',
    key: 'warmup_time'
  },
  forward: {
    label: 'Forward time',
    color: '#92CC76',
    key: 'forward_time'
  },
  backward: {
    label: 'Backward time',
    color: '#AAE7FF',
    key: 'backward_time'
  },
  cooldown: {
    label: 'Cooldown time',
    color: '#FAC858',
    key: 'cooldown_time'
  },
  allReduce: {
    label: 'All Reduce time',
    color: '#EF6666',
    key: 'allreduce_time'
  }
}

export interface IPanelRightProps { }
const PanelRight: FC<IPanelRightProps> = (props) => {
  const { t } = useTranslation();
  const { result, latest_result, bm_result, loading, curGpu, curMode, curModel, otherConfig, totalConfig,
    setProject, autoRecalc } = useModel(ProjectModel);
  const { changeLog, autoCalculated } = useModel(LogModel);
  const [state, setState] = useImmer({
    memoryCollapse: false,
    computationCollapse: true,
    communicationCollapse: true,
    timelineCollapse: false,
    excutionsCollapse: false,
    optionalCollapse: false,
    summaryCollapse: false
  });
  // const readExcelFile = async () => {
  //   setProject({
  //     result: null
  //   })
  //   const readRes = await readFile()
  //   setProject({
  //     result: {
  //       timeline: readRes
  //     }
  //   });
  // }
  const dataParse = (d: number, toGB?: boolean) => {
    if (!d) return d
    if (toGB) {
      d = d / (1024 * 1024 * 1024)
    }
    // 整数
    if (d.toString() === d.toFixed(0)) {
      return d
    }
    // 大于1的浮点数，保留2位
    if (d > 1) {
      return d.toFixed(2)
    }
    // 小于1的浮点数，保留6位
    return d.toFixed(6)
  }
  const { warmup_time, forward_time, backward_time, cooldown_time, allreduce_time, num_microbatches } = result?.timeline || {}
  const totalTime = sum([warmup_time, forward_time * num_microbatches, backward_time * num_microbatches, cooldown_time, allreduce_time])
  const loopTotalTime = (forward_time + backward_time) * num_microbatches
  const calcLength = (time: number, isMulti?: boolean) => {
    if (isMulti) {
      return `${(time / loopTotalTime) * (100 - Math.ceil(num_microbatches / 10))}%`
    }
    return `${(time / totalTime) * 98}%`
  }
  const checkChanged = (val: any, preVal: any) => {
    if (curMode !== 'guide') {
      return ''
    }
    if (preVal && val !== preVal) {
      return styles.changed
    }
    return ''
  }
  const renderLoopTime = (index: number) => {
    return <Fragment key={index}>
      <div key={index} className={styles.timeline_inner_block} style={{
        width: calcLength(forward_time, true),
        backgroundColor: COLOR_MAPPING['forward'].color
      }}>
      </div>
      <div key={`${index}_1`} className={styles.timeline_inner_block} style={{
        width: calcLength(backward_time, true),
        backgroundColor: COLOR_MAPPING['backward'].color
      }}>
      </div></Fragment>
  }
  const renderMultiLoopTime = () => {
    const numsArray = []
    for (let i = 0; i < num_microbatches; i++) {
      numsArray.push(i)
    }
    return numsArray.map((_, index) =>
      renderLoopTime(index)
    )
  }
  const checkMemoryOverall = () => {
    if (result.memory_usage && curGpu) {
      return result.memory_usage.overall_usage >= curGpu.memory * 1024 * 1024 * 1024
    }
    return false
  }
  const exportResultFile = () => {
    exportResult({
      ...result, cluster: curGpu, model: curModel, other_config: otherConfig, input_config: totalConfig
    }).then((res: any) => {
      FileSaver.saveAs(res, "llm-training-calculator.xlsx");
    })
  }
  // const renderTip = (time: number, title: string) => {
  //   return <div className={styles.pop_tip}>
  //     <div>{title}(GPU usage)</div>
  //     {/* <div>{dataParse(time)} ({((time / totalTime) * 100).toFixed(2)}%)</div> */}
  //     <div>{dataParse(time)} (0%)</div>
  //   </div>
  // }
  // const renderDetail = () => {
  //   return <PopPanel />
  // }
  if (loading) {
    return <div className={styles.loading}><LoadingOutlined /></div>
  }
  if (result && result.error) {
    return <Result
      status="error"
      title={t('Trainning failed')}
      subTitle={t("Please check , modify the input and try again.")}
      extra={[
      ]}
    >
    </Result>
  }
  if (!result && curMode === 'guide' ) {
    return <div className={styles.content}>
      <div className={styles.empty_steps} >
        <div><Steps />
        </div>
      </div>
    </div>
  }
  if (!result &&  curMode === 'optimal') {
    return <div className={styles.content}>
      <div className={styles.empty_steps} >
        <div><OptimalSteps />
        </div>
      </div>
    </div>
  }
  
  if ((!result && curMode === 'custom') || (!bm_result && curMode === 'benchmark')) {
    return <div className={styles.content}>
      <div className={styles.empty} >
        <div className={styles.empty_icon}></div>
        <div className={styles.empty_tip}>
          {t('wait calc')}
        </div>
      </div >
    </div>
  }
  if (curMode === 'benchmark') {
    return <div className={styles.content}>
      <BenchPanel />
    </div>
  }
  return (
    <div className={styles.content}>
      {autoRecalc && autoCalculated && changeLog.field &&
        <Alert
          message={`${changeLog.field} changed !`}
          type="success"
          closable
        />}
      <div className={styles.result}>
        <div className={styles.result_group}>
          {/* Memory */}
          {result.memory_usage && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>
                Memory
              </div>
              <div className={styles.result_group_collapse}>{!state.memoryCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, memoryCollapse: !state.memoryCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, memoryCollapse: !state.memoryCollapse })
                }} />}
              </div>
            </div>
            {!state.memoryCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                <Descriptions.Item label="Optimizer States">
                  {result.memory_usage.optimizer}
                </Descriptions.Item>
                <Descriptions.Item label="Weights">{result.memory_usage.weights}</Descriptions.Item>
                <Descriptions.Item label="Activation">{result.memory_usage.activation}</Descriptions.Item>
                <Descriptions.Item label="Activation Gradients">{result.memory_usage.activation_gradients}</Descriptions.Item>
                <Descriptions.Item span={1} label="Overall Usage">{result.memory_usage.overall_usage}</Descriptions.Item>
              </Descriptions>
            </div>}
            <Divider />
          </>}
          {/* Computation Time */}
          {result.computation && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>Computation Time</div>
              <div className={styles.result_group_collapse}>{!state.computationCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, computationCollapse: !state.computationCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, computationCollapse: !state.computationCollapse })
                }} />}
              </div>
            </div>
            {!state.computationCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                {/* <Descriptions.Item label="Per-device LLM blocks">
                  {result.computation.per_device_blocks}
                </Descriptions.Item> */}
                <Descriptions.Item label="Number of microbatches">{result.computation.num_microbatches}</Descriptions.Item>
                <Descriptions.Item label="Per-batch forward computation time(s)">{result.computation.batch_forward_computation_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-microbatch forward computation time(s)">{result.computation.microbatch_forward_computation_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-batch backward computation time(s)">{result.computation.batch_backward_computation_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item span={1} label="Per-microbatch backward computation time(s)">{result.computation.microbatch_backward_computation_time.toFixed(6)}</Descriptions.Item>
              </Descriptions>
            </div>}
            <Divider />
          </>}
          {/* Communication Time */}
          {result.communication && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>Communication Time</div>
              <div className={styles.result_group_collapse}>{!state.communicationCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, communicationCollapse: !state.communicationCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, communicationCollapse: !state.communicationCollapse })
                }} />}
              </div>
            </div>
            {!state.communicationCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                <Descriptions.Item label="DP communication size">
                  {result.communication.dp_comm_size}
                </Descriptions.Item>
                <Descriptions.Item label="TP forward communication size">{result.communication.tp_comm_fw_size}</Descriptions.Item>
                <Descriptions.Item label="TP backward communication size">{result.communication.tp_comm_bw_size}</Descriptions.Item>
                <Descriptions.Item label="PP forward communication size">{result.communication.pp_comm_fw_size}</Descriptions.Item>
                <Descriptions.Item label="PP backward communication size">{result.communication.pp_comm_bw_size}</Descriptions.Item>
                <Descriptions.Item label="Per-batch DP communication time(s)">{result.communication.batch_dp_comm_time.toFixed(6)}</Descriptions.Item>

                <Descriptions.Item label="Per-batch TP communication time(s)">{result.communication.batch_tp_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-batch TP forward communication time(s)">{result.communication.batch_tp_fw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-microbatch TP forward communication time(s)">{result.communication.microbatch_tp_fw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-batch TP backward communication time(s)">{result.communication.batch_tp_bw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-microbatch TP backward communication time(s)">{result.communication.microbatch_tp_bw_comm_time.toFixed(6)}</Descriptions.Item>

                <Descriptions.Item label="Per-batch PP communication time(s)">{result.communication.batch_pp_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-batch PP forward communication time(s)">{result.communication.batch_pp_fw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-microbatch PP forward communication time(s)">{result.communication.microbatch_pp_fw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item label="Per-batch PP backward communication time(s)">{result.communication.batch_pp_bw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item  label="Per-microbatch PP backward communication time(s)">{result.communication.microbatch_pp_bw_comm_time.toFixed(6)}</Descriptions.Item>
                <Descriptions.Item span={1} label="Total Comm Time">{result.communication.total_comm_time.toFixed(6)}</Descriptions.Item>

              </Descriptions>
            </div>}
            <Divider />
          </>}
          {/*  Timeline */}
          {curMode !=='optimal'&&<div className={styles.result_group_header}>
            <div className={styles.result_group_title}>
              Timeline
              {curMode === 'custom' ? <div className={styles.result_group_more}>
                <div style={{ paddingRight: 10 }}>
                  Totoal number of gpus:
                </div>
                <div>
                  {result.total_time.totoal_number_of_gpus}</div>
              </div> :
                <div className={styles.result_group_collapse}>{!state.timelineCollapse ?
                  <CaretDownOutlined onClick={() => {
                    setState({ ...state, timelineCollapse: !state.timelineCollapse })
                  }} /> :
                  <CaretRightOutlined onClick={() => {
                    setState({ ...state, timelineCollapse: !state.timelineCollapse })
                  }} />}
                </div>}
            </div>
          </div>}
          {/* {!state.timelineCollapse && <div className={styles.result_group_content}>
            <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
              <Descriptions.Item label="Per-device LLM blocks">
                {result.timeline.per_device_blocks}
              </Descriptions.Item>
              <Descriptions.Item label="Number of microbatches">{result.timeline.num_microbatches}</Descriptions.Item>
              <Descriptions.Item label="Warmup time">{result.timeline.warmup_time.toFixed(6)}</Descriptions.Item>
              <Descriptions.Item label="Cooldown time">{result.timeline.cooldown_time.toFixed(6)}</Descriptions.Item>
              <Descriptions.Item span={1} label="Batch total time">{result.timeline.batch_total_time.toFixed(6)}</Descriptions.Item>
            </Descriptions>
          </div>} */}
        </div>
        {/* <BaseTL result={{ ...result, other_config: curMode === 'guide' ? otherConfig : result.other_config }} latest_result={latest_result} curMode={curMode}></BaseTL> */}
        
        
        {!state.timelineCollapse && curMode !=='optimal' &&<TLEventChart  result={result.timeline_events}></TLEventChart>}

        {/* {curMode === 'guide' && <div className={styles.export_btn}>
          <Button type="primary" icon={<ExportOutlined />} onClick={exportResultFile}>
            {t('export')}
          </Button>
        </div>} */}
          {result.executions && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>
                Executions
              </div>
              <div className={styles.result_group_collapse}>{!state.excutionsCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, excutionsCollapse: !state.excutionsCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, excutionsCollapse: !state.excutionsCollapse })
                }} />}
              </div>
            </div>
            {!state.excutionsCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                <Descriptions.Item label="Bad Executions">
                  {result.executions.bad_executions}
                </Descriptions.Item>
                <Descriptions.Item label="Calculation Rate">{result.executions.calculation_rate}</Descriptions.Item>
                <Descriptions.Item label="Good Executions">{result.executions.good_executions}</Descriptions.Item>
                <Descriptions.Item span={1} label="Total Executions">{result.executions.total_executions}</Descriptions.Item>
              </Descriptions>
            </div>}
            <Divider />
          </>}

          {result.optimal_result && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>
                Optimal Result
              </div>
              <div className={styles.result_group_collapse}>{!state.optionalCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, optionalCollapse: !state.optionalCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, optionalCollapse: !state.optionalCollapse })
                }} />}
              </div>
            </div>
            {!state.optionalCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                <Descriptions.Item label="Gpu Numbers">{result.optimal_result.gpu_numbers}</Descriptions.Item>
                <Descriptions.Item label="Tensor Parallel">{result.optimal_result.tensor_parallel}</Descriptions.Item>
                <Descriptions.Item label="Pipeline Parallel">{result.optimal_result.pipeline_parallel}</Descriptions.Item>
                <Descriptions.Item label="Data Parallel">{result.optimal_result.data_parallel}</Descriptions.Item>
                <Descriptions.Item  label="Batch Size">{result.optimal_result.batch_size}</Descriptions.Item>
                <Descriptions.Item label="Microbatch Size">{result.optimal_result.microbatch_size}</Descriptions.Item>
                <Descriptions.Item label="Datatype">{result.optimal_result.datatype}</Descriptions.Item>
                <Descriptions.Item label="Fused Activation">{result.optimal_result.fused_activation}</Descriptions.Item>
                <Descriptions.Item label="Attention Type">{result.optimal_result.attention_type}</Descriptions.Item>
                <Descriptions.Item label="Activation Recompute">
                  {result.optimal_result.activation_recompute}
                </Descriptions.Item>
                <Descriptions.Item label="Pipeline Interleaving">{result.optimal_result.pipeline_interleaving}</Descriptions.Item>
                <Descriptions.Item label="Optimizer Sharding">{result.optimal_result.optimizer_sharding.toString()}</Descriptions.Item>
                <Descriptions.Item label="Tensor Parallel Common Type">{result.optimal_result.tensor_parallel_comm_type}</Descriptions.Item>
                <Descriptions.Item label="Tensor Parallel Overlap">{result.optimal_result.tensor_parallel_overlap}</Descriptions.Item>
                <Descriptions.Item label="Sequence Parallel Allgather Redo">{result.optimal_result.sequence_parallel_allgather_redo.toString()}</Descriptions.Item>
                <Descriptions.Item label="Data Parallel Overlap">{result.optimal_result.data_parallel_overlap.toString()}</Descriptions.Item>
                <Descriptions.Item  label="Weight Offload">{result.optimal_result.weight_offload.toString()}</Descriptions.Item>               
                <Descriptions.Item label="Activations Offload">{result.optimal_result.activations_offload.toString()}</Descriptions.Item>              
                <Descriptions.Item label="Optimizer Offload">{result.optimal_result.optimizer_offload.toString()}</Descriptions.Item>
                <Descriptions.Item span={1} label="Training">{result.optimal_result.training.toString()}</Descriptions.Item>
              </Descriptions>
            </div>}
            <Divider />
          </>}

          {result.summary && <>
            <div className={styles.result_group_header}>
              <div className={styles.result_group_title}>
                Summary
              </div>
              <div className={styles.result_group_collapse}>{!state.summaryCollapse ?
                <CaretDownOutlined onClick={() => {
                  setState({ ...state, summaryCollapse: !state.summaryCollapse })
                }} /> :
                <CaretRightOutlined onClick={() => {
                  setState({ ...state, summaryCollapse: !state.summaryCollapse })
                }} />}
              </div>
            </div>
            {!state.summaryCollapse && <div className={styles.result_group_content}>
              <Descriptions colon={false} className='customize-des' column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }} title="">
                <Descriptions.Item label="Batch Total Time">
                  {result.summary.batch_total_time.toFixed(6)}
                </Descriptions.Item>
                <Descriptions.Item label="Global Batch Size">{result.summary.global_batch_size}</Descriptions.Item>
                <Descriptions.Item label="Local Batch Size">{result.summary.local_batch_size}</Descriptions.Item>
                <Descriptions.Item  label="Total Efficiency">{result.summary.total_efficiency}</Descriptions.Item>
                <Descriptions.Item span={1} label="Totoal Number Of Gpus">{result.summary.totoal_number_of_gpus}</Descriptions.Item>
              </Descriptions>
            </div>}
            <Divider />
          </>}

      </div>
    </div >
  );
};

export default PanelRight;
