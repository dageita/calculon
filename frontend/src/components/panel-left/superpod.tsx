import { InputNumber, Select } from 'antd';
import { useEffect, useRef, useState } from 'react';
import useModel from 'flooks';
import ProjectModel from '@/models/projectModel';
import LogModel from '@/models/logModel';
import { getDataTypes } from '@/services';
import styles from './index.less';

const DEFAULT_TRAINING_SAMPLES = 1_000_000;

const SuperpodPanel = () => {
  const { curGpu, curNetwork, otherConfig, setOtherConfig, setProject } = useModel(ProjectModel);
  const { setChangeLog } = useModel(LogModel);
  const [matrixTypes, setMatrixTypes] = useState<any[]>([]);
  const [vectorTypes, setVectorTypes] = useState<any[]>([]);
  const previousGpuRef = useRef<string | undefined>(undefined);

  const setValue = (key: string, value: any, label: string) => {
    setChangeLog(label, value, otherConfig?.[key]);
    setOtherConfig({ [key]: value });
  };

  useEffect(() => {
    if (!curGpu?.value) return;
    getDataTypes(curGpu.value).then((result: any) => {
      const options = (items: string[] = []) => items.map((value) => ({ value, label: value }));
      setMatrixTypes(options(result.matrix_datatypes || result.datatypes || []));
      setVectorTypes(options(result.vector_datatypes || result.datatypes || []));
    });
  }, [curGpu?.value]);

  useEffect(() => {
    if (!curGpu) return;
    const procs = Number(curGpu.num_procs || 1);
    const intraLatency = Number(curGpu.intra_latency ?? 0);
    const interLatency = Number(curGpu.inter_latency ?? 0);
    const resetHardwareDefaults = previousGpuRef.current !== curGpu.value;
    const currentScaleUp = Number(otherConfig.max_scale_up_size);
    const rangeDefault = (key: string, fallback: number) =>
      resetHardwareDefaults ? fallback : otherConfig[key] ?? fallback;
    setOtherConfig({
      objectives: otherConfig.objectives?.length
        ? otherConfig.objectives : [otherConfig.objective || 'throughput'],
      training_samples: otherConfig.training_samples || DEFAULT_TRAINING_SAMPLES,
      max_global_batch_size: otherConfig.max_global_batch_size || otherConfig.global_batch_size || procs,
      max_scale_up_size: resetHardwareDefaults
        || !Number.isFinite(currentScaleUp) || currentScaleUp < 1
        || currentScaleUp > procs
        ? procs : currentScaleUp,
      intra_bandwidth_start: rangeDefault('intra_bandwidth_start', 100),
      intra_bandwidth_stop: rangeDefault('intra_bandwidth_stop', 500),
      intra_bandwidth_step: rangeDefault('intra_bandwidth_step', 50),
      inter_bandwidth_start: rangeDefault('inter_bandwidth_start', 12.5),
      inter_bandwidth_stop: rangeDefault('inter_bandwidth_stop', 62.5),
      inter_bandwidth_step: rangeDefault('inter_bandwidth_step', 5),
      intra_latency_start: rangeDefault('intra_latency_start', intraLatency * 0.5),
      intra_latency_stop: rangeDefault('intra_latency_stop', intraLatency * 1.5),
      intra_latency_step: rangeDefault('intra_latency_step', Math.max(intraLatency * 0.25, 0.000000001)),
      inter_latency_start: rangeDefault('inter_latency_start', interLatency * 0.5),
      inter_latency_stop: rangeDefault('inter_latency_stop', interLatency * 1.5),
      inter_latency_step: rangeDefault('inter_latency_step', Math.max(interLatency * 0.25, 0.000000001)),
      placement_policies: otherConfig.placement_policies?.length
        ? otherConfig.placement_policies : ['tp-ep-cp-pp-dp'],
      max_candidates: otherConfig.max_candidates || 512,
      hardware_top_n: otherConfig.hardware_top_n || 10,
    });
    previousGpuRef.current = curGpu.value;
  }, [curGpu?.value, curGpu?.num_procs]);

  const numberInput = (
    key: string,
    label: string,
    options: { min?: number; max?: number; precision?: number; step?: number } = {},
  ) => (
    <InputNumber
      className={styles.number_item}
      style={{ width: '100%' }}
      value={otherConfig[key]}
      onChange={(value) => setValue(key, value, label)}
      {...options}
    />
  );

  const scientificFormatter = (
    value: any,
    info: { userTyping: boolean; input: string },
  ) => {
    if (info?.userTyping) return info.input;
    if (value === undefined || value === null || value === '') return '';
    const numberValue = Number(value);
    return Number.isFinite(numberValue)
      ? numberValue.toExponential(1)
      : String(value);
  };

  const rangeInput = (
    prefix: string,
    label: string,
    options: any,
  ) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {(['start', 'stop', 'step'] as const).map((part) => (
        <InputNumber
          key={part}
          addonBefore={part === 'start' ? 'Start' : part === 'stop' ? 'Stop' : 'Step'}
          className={styles.number_item}
          style={{ width: '100%' }}
          value={otherConfig[`${prefix}_${part}`]}
          onChange={(value) => setValue(
            `${prefix}_${part}`,
            value,
            `${label} ${part}`,
          )}
          {...options}
        />
      ))}
    </div>
  );

  const gpuNumbers = Math.max(1, Number(curGpu?.num_procs || 1));
  const clampedScaleUp = Math.min(
    gpuNumbers,
    Math.max(1, Number(otherConfig.max_scale_up_size || gpuNumbers)),
  );

  return (
    <div className={styles.nest}>
      <p className={styles.section_title}>GPU Numbers</p>
      <InputNumber className={styles['input-num-content']} min={1} precision={0}
        value={curGpu?.num_procs} onChange={(value) => setProject({
          curGpu: {
            ...curGpu,
            num_procs: value,
          },
        })} />

      <p className={styles.section_title}>Max Scale-up Size</p>
      <InputNumber
        className={styles.number_item}
        style={{ width: '100%' }}
        min={1}
        max={gpuNumbers}
        precision={0}
        value={clampedScaleUp}
        formatter={(value, info) => {
          const typed = Number(info?.input);
          if (info?.userTyping && Number.isFinite(typed)) {
            return String(Math.min(gpuNumbers, Math.max(1, typed)));
          }
          const numeric = Number(value);
          return Number.isFinite(numeric)
            ? String(Math.min(gpuNumbers, Math.max(1, numeric))) : '';
        }}
        parser={(value) => {
          const numeric = Number(value);
          return Number.isFinite(numeric)
            ? Math.min(gpuNumbers, Math.max(1, numeric)) : gpuNumbers;
        }}
        onChange={(value) => setValue(
          'max_scale_up_size',
          Math.min(gpuNumbers, Math.max(1, Number(value || gpuNumbers))),
          'Max Scale-up Size',
        )}
        onBlur={() => setOtherConfig({
          max_scale_up_size: clampedScaleUp,
        })}
      />
      <div className={styles.to_tips}>Maximum GPUs in one scale-up pod. GPU Numbers is fixed; all divisors up to this limit are searched and Scale-up Size × Scale-out Size always equals GPU Numbers.</div>

      <p className={styles.section_title}>Search Objective</p>
      <Select
        mode="multiple"
        allowClear={false}
        style={{ width: '100%' }}
        value={otherConfig.objectives?.length ? otherConfig.objectives : ['throughput']}
        options={[
          { value: 'throughput', label: 'Maximize throughput' },
          { value: 'batch_time', label: 'Minimize batch time' },
          { value: 'mfu', label: 'Maximize MFU' },
          { value: 'time_to_train', label: 'Minimize time-to-train' },
        ]}
        onChange={(value) => setOtherConfig({
          objectives: value.length ? value : ['throughput'],
          training_samples: otherConfig.training_samples || DEFAULT_TRAINING_SAMPLES,
        })}
      />
      <div className={styles.to_tips}>Multiple objectives use Pareto filtering and a normalized balanced compromise.</div>

      <p className={styles.section_title}>Max Global Batch Size</p>
      <InputNumber className={styles.number_item} min={1} precision={0} value={otherConfig.max_global_batch_size}
        onChange={(value) => setValue('max_global_batch_size', value, 'Max Global Batch Size')} />
      {otherConfig.objectives?.includes('time_to_train') && <>
        <p className={styles.section_title}>Training Samples</p>
        <InputNumber className={styles.number_item} min={1} precision={0} value={otherConfig.training_samples || DEFAULT_TRAINING_SAMPLES}
          onChange={(value) => setValue('training_samples', value, 'Training Samples')} />
      </>}

      <p className={styles.section_title}>Intra-node Bandwidth Range (GB/s)</p>
      {rangeInput('intra_bandwidth', 'Intra Bandwidth', { min: 0.1, precision: 1, step: 0.1 })}
      <p className={styles.section_title}>Inter-node Aggregate Bandwidth Range (GB/s per Scale-up Pod)</p>
      {rangeInput('inter_bandwidth', 'Inter Bandwidth', { min: 0.1, precision: 1, step: 0.1 })}
      <div className={styles.to_tips}>Converted to per-GPU bandwidth for each candidate by dividing by its Scale-up Size.</div>
      <p className={styles.section_title}>Intra-node Latency Range (s)</p>
      {rangeInput('intra_latency', 'Intra Latency', {
        min: 0, step: 0.0001, formatter: scientificFormatter, parser: Number,
      })}
      <p className={styles.section_title}>Inter-node Latency Range (s)</p>
      {rangeInput('inter_latency', 'Inter Latency', {
        min: 0, step: 0.0001, formatter: scientificFormatter, parser: Number,
      })}
      <div className={styles.to_tips}>Hardware ranges use axis sweeps plus low-discrepancy joint sampling for a global balanced search.</div>
      <p className={styles.section_title}>Placement Policies (inner → outer)</p>
      <Select mode="tags" style={{ width: '100%' }} value={otherConfig.placement_policies}
        options={[
          { value: 'tp-ep-cp-pp-dp', label: 'TP → EP → CP → PP → DP' },
          { value: 'tp-cp-ep-pp-dp', label: 'TP → CP → EP → PP → DP' },
          { value: 'ep-tp-cp-pp-dp', label: 'EP → TP → CP → PP → DP' },
        ]}
        onChange={(value) => setValue('placement_policies', value, 'Placement Policies')} />

      <p className={styles.section_title}>Maximum Total Search Executions</p>
      <InputNumber className={styles.number_item} min={1} precision={0} value={otherConfig.max_candidates}
        onChange={(value) => setValue('max_candidates', value, 'Maximum Candidates')} />

      <p className={styles.section_title}>Matrix datatype (GEMM)</p>
      <Select style={{ width: '100%' }} options={matrixTypes} value={otherConfig.matrix_dtype}
        onChange={(value) => setValue('matrix_dtype', value, 'Matrix Data Type')} />
      <p className={styles.section_title}>Vector datatype (Norm/Act)</p>
      <Select style={{ width: '100%' }} options={vectorTypes} value={otherConfig.vector_dtype}
        onChange={(value) => setValue('vector_dtype', value, 'Vector Data Type')} />
    </div>
  );
};

export default SuperpodPanel;

