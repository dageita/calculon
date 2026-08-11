import { Select, InputNumber, Slider, Popover } from 'antd';
import useModel from 'flooks';
import styles from './index.less';
import ProjectModel from '@/models/projectModel';
import { InfoCircleOutlined } from '@ant-design/icons';
import LogModel from '@/models/logModel';
import { getDataTypes } from '@/services';
import { useImmer } from 'use-immer';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const RECOMPUTE_OPTIONS = [
  { key: 'full', label: 'full', value: 'full' },
  { key: 'attn_only', label: 'attn_only', value: 'attn_only' },
  { key: 'none', label: 'none', value: 'none' },
];

const OPTIMIZER_SHARDING_OPTIONS = [
  { key: 'true', label: 'True', value: 'true' },
  { key: 'false', label: 'False', value: 'false' },
];

// Keep recommend APIs that still expect legacy optimization_strategy labels in sync.
const recomputeToStrategy = (recompute: string) => {
  if (recompute === 'full') return 'Full recomputation';
  if (recompute === 'attn_only') return 'Attention-only recomputation';
  return 'None recomputation';
};

// 参数配置列表
const PARAMS_LIST = [
  {
    title: 'Tensor parallel degree',
    key: 'tensor_par',
    min: 0,
    max: 8,
    precision: 0,
    step: 1,
  },
  {
    title: 'Pipeline parallel degree',
    key: 'pipeline_par',
    min: 0,
    max: 10000,
    precision: 0,
    step: 1,
  },
  {
    title: 'Data parallel degree',
    key: 'data_par',
    min: 0,
    max: 10000,
    precision: 0,
    step: 1,
  },
  {
    title: 'Expert parallel degree',
    key: 'expert_par',
    min: 1,
    max: 10000,
    precision: 0,
    step: 1,
  },
  {
    title: 'Context parallel degree',
    key: 'context_par',
    min: 1,
    max: 10000,
    precision: 0,
    step: 1,
  },
];

const OtherPanel = (props) => {
  const {
    setProject,
    setOtherConfig,
    otherConfig,
    recommendConfig,
    curModel,
    curGpu,
    checkSize,
    checkPipeline,
  } = useModel(ProjectModel);
  const { t } = useTranslation();
  const { setChangeLog } = useModel(LogModel);

  const [state, setState] = useImmer({
    matrixDataTypeList: [],
    vectorDataTypeList: [],
    lastGpuValue: null,
  });

  const dpDegree = otherConfig?.data_par || 0;
  const optimizerShardingEnabled = dpDegree > 1;
  const expertParallelEnabled = Boolean(curModel?.num_experts);

  // 设置参数值并记录变更日志
  const setParamValue = (key, val, title) => {
    setChangeLog(title, val, otherConfig?.[key]);
    setOtherConfig({ [key]: val });
  };

  const setActivationRecompute = (val: string) => {
    setChangeLog(
      'Activation recompute',
      val,
      otherConfig?.activation_recompute,
    );
    setOtherConfig({
      activation_recompute: val,
      optimization_strategy: recomputeToStrategy(val),
    });
  };

  const setOptimizerSharding = (val: boolean) => {
    if (!optimizerShardingEnabled && val) {
      return;
    }
    setChangeLog(
      'Optimizer sharding',
      String(val),
      String(otherConfig?.optimizer_sharding),
    );
    setOtherConfig({ optimizer_sharding: val });
  };

  // 计算最小值
  const calcMin = (cf) => cf.min;

  // 计算最大值
  const calcMax = () => {
    return curGpu.num_procs;
  };

  const loadDataTypes = async (gpuName?: string) => {
    const name = gpuName || curGpu?.value || curGpu?.name;
    if (!name) return;
    try {
      const result = (await getDataTypes(name)) as any;
      const toOpts = (arr: string[] = []) =>
        arr.map((item: string) => ({
          key: item,
          label: item,
          value: item,
        }));
      const matrixList = toOpts(
        result.matrix_datatypes || result.datatypes || [],
      );
      const vectorList = toOpts(
        result.vector_datatypes || result.datatypes || [],
      );
      setState((prev) => ({
        ...prev,
        matrixDataTypeList: matrixList,
        vectorDataTypeList: vectorList,
      }));
    } catch (e) {
      console.error('loadDataTypes failed', e);
      setState((prev) => ({
        ...prev,
        matrixDataTypeList: [],
        vectorDataTypeList: [],
      }));
    }
  };

  // GPU 变化时重新拉取 matrix/vector dtype
  useEffect(() => {
    const gpuName = curGpu?.value || curGpu?.name;
    if (!gpuName) return;
    if (state.lastGpuValue && state.lastGpuValue !== gpuName) {
      setOtherConfig({
        matrix_dtype: undefined,
        vector_dtype: undefined,
      });
    }
    setState((prev) => ({ ...prev, lastGpuValue: gpuName }));
    loadDataTypes(gpuName);
  }, [curGpu?.value, curGpu?.name]);

  // DP<=1 时强制关闭 optimizer sharding
  useEffect(() => {
    if (!optimizerShardingEnabled && otherConfig?.optimizer_sharding) {
      setOtherConfig({ optimizer_sharding: false });
    }
  }, [optimizerShardingEnabled, otherConfig?.optimizer_sharding]);

  // 渲染微批次大小设置组件
  const renderMicrobatchSize = () => (
    <div>
      {recommendConfig.recomended_microbatch && (
        <div className={styles.slider_tip}>
          {t('microbatch recommend', {
            value: recommendConfig.recomended_microbatch,
          })}
        </div>
      )}
      <InputNumber
        className={styles.number_item}
        precision={0}
        min={1}
        max={curModel?.minibatch_size}
        value={otherConfig?.microbatch_size}
        onChange={(val) =>
          setParamValue('microbatch_size', val, 'Microbatch size')
        }
        addonAfter={
          <Popover
            content={<div>Need to be able to divide minibatch size.</div>}
          >
            <InfoCircleOutlined style={{ cursor: 'pointer' }} />
          </Popover>
        }
      />
      {!checkSize() && curModel?.minibatch_size && (
        <div className={styles.error_tip}>
          Need to be able to divide minibatch size({curModel?.minibatch_size}).
        </div>
      )}
    </div>
  );
  // 渲染微批次大小设置组件
  const renderBatchSize = () => (
    <div>
      {recommendConfig.recomended_microbatch && (
        <div className={styles.slider_tip}>
          {t('microbatch recommend', {
            value: recommendConfig.recomended_microbatch,
          })}
        </div>
      )}
      <InputNumber
        className={styles.number_item}
        precision={0}
        min={1}
        max={curModel?.batch_size}
        value={otherConfig?.batch_size}
        onChange={(val) => setParamValue('batch_size', val, 'Batch size')}
        addonAfter={
          <Popover
            content={
              <div>
                Need to be able to divide (data parallel degree * microbatch
                size).
              </div>
            }
          >
            <InfoCircleOutlined style={{ cursor: 'pointer' }} />
          </Popover>
        }
      />
      {!checkSize() && curModel?.batch_size && (
        <div className={styles.error_tip}>
          Need to be able to divide data parallel degree ({curModel?.batch_size}
          ).
        </div>
      )}
    </div>
  );

  return (
    <div className={styles.nest}>
      <p className={styles.section_title}>{t('optimization strategy')}</p>

      <p className={styles.section_title} style={{ marginTop: 8 }}>
        {t('activation_recompute')}
      </p>
      <div className={styles['group-content']}>
        <Select
          options={RECOMPUTE_OPTIONS}
          placeholder={t('Please select')}
          value={otherConfig['activation_recompute']}
          onChange={(val) => setActivationRecompute(val)}
        />
      </div>

      <p className={styles.section_title} style={{ marginTop: 8 }}>
        {t('optimizer_sharding')}
      </p>
      <div className={styles.slider_tip}>
        <span
          style={{ color: optimizerShardingEnabled ? undefined : '#ff4d4f' }}
        >
          {optimizerShardingEnabled
            ? t('optimizer_sharding_tip')
            : t('optimizer_sharding_dp_tip')}
        </span>
      </div>
      <div className={styles['group-content']}>
        <Select
          options={OPTIMIZER_SHARDING_OPTIONS}
          placeholder={t('Please select')}
          value={otherConfig['optimizer_sharding'] ? 'true' : 'false'}
          disabled={!optimizerShardingEnabled}
          onChange={(val) => setOptimizerSharding(val === 'true')}
        />
      </div>

      <div className={styles.slider_tip}>
        <span
          style={{
            color:
              otherConfig['tensor_par'] *
                otherConfig['pipeline_par'] *
                otherConfig['data_par'] *
                (otherConfig['expert_par'] || 1) *
                (otherConfig['context_par'] || 1) ==
              curGpu.num_procs
                ? ''
                : '#ff4d4f',
          }}
        >
          {t('pp_dp_tp_recommend', { value: curGpu.num_procs })}
        </span>
      </div>

      <div className={styles['group_slider']}>
        {PARAMS_LIST.map((cf) => (
          <div className={styles['group-list-item']} key={cf.key}>
            <div className={styles['item-wrapper']}>
              <span>{cf.title}</span>
              <InputNumber
                precision={cf.precision || 0}
                width={100}
                min={calcMin(cf)}
                max={calcMax()}
                value={otherConfig[cf.key]}
                disabled={
                  ['expert_par', 'context_par'].includes(cf.key) &&
                  !expertParallelEnabled
                }
                onChange={(val) => setParamValue(cf.key, val, cf.title)}
              />
            </div>

            <Slider
              min={cf.min}
              max={calcMax()}
              onChange={(val) => setParamValue(cf.key, val, cf.title)}
              value={otherConfig[cf.key]}
              disabled={
                ['expert_par', 'context_par'].includes(cf.key) &&
                !expertParallelEnabled
              }
              step={cf.step}
            />

            {cf.key === 'pipeline_par' &&
              !checkPipeline() &&
              curModel?.minibatch_size && (
                <div className={styles.error_tip}>
                  {t('pipeline divide tips')}({curModel?.num_layers}).
                </div>
              )}
          </div>
        ))}
      </div>

      <p className={styles.section_title}>{t('batch size')}</p>
      <div className={styles.batch_size}>
        <span
          style={{
            color:
              otherConfig['microbatch_size'] * otherConfig['data_par'] <=
              otherConfig['batch_size']
                ? ''
                : '#ff4d4f',
          }}
        >
          {t('batch_recommend')}
        </span>
      </div>
      <div className={styles.section_content}>{renderBatchSize()}</div>

      <p className={styles.section_title}>{t('microbatch')}</p>
      <div className={styles.section_content}>{renderMicrobatchSize()}</div>

      <p className={styles.section_title}>{t('compute_precision')}</p>
      <div className={styles.slider_tip}>
        <span>{t('compute_precision_tip')}</span>
      </div>
      <p className={styles.section_title} style={{ marginTop: 8 }}>
        {t('matrix_dtype')}
      </p>
      <div className={styles['group-content']}>
        <Select
          options={state.matrixDataTypeList}
          placeholder={t('Select matrix datatype')}
          value={otherConfig['matrix_dtype']}
          onChange={(val) =>
            setParamValue('matrix_dtype', val, 'Matrix Data Type')
          }
        />
      </div>

      <p className={styles.section_title}>{t('vector_dtype')}</p>
      <div className={styles['group-content']}>
        <Select
          options={state.vectorDataTypeList}
          placeholder={t('Select vector datatype')}
          value={otherConfig['vector_dtype']}
          onChange={(val) =>
            setParamValue('vector_dtype', val, 'Vector Data Type')
          }
        />
      </div>
    </div>
  );
};

export default OtherPanel;
