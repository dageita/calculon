import { FC, useEffect } from 'react';
import { useImmer } from 'use-immer';
import {
  Select,
  Divider,
  InputNumber,
  Input,
  Drawer,
  Button,
  message,
} from 'antd';
import useModel from 'flooks';
import styles from './index.less';
import ProjectModel from '@/models/projectModel';
import { getModelList, getParameterMetrics } from '@/services';
import Empty from '../empty';
import LogModel from '@/models/logModel';
import { PlusOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const PARAMS_LIST = [
  {
    title: 'Model Type',
    key: 'name',
  },
  {
    title: 'Token length',
    key: 'seq_size',
  },
  {
    title: 'Number of attention heads',
    key: 'attn_size',
  },
  {
    title: 'Attention head size',
    key: 'attn_heads',
  },
  {
    title: 'Number of KV heads (GQA)',
    key: 'kv_heads',
  },
  {
    title: 'RoPE theta',
    key: 'rope_theta',
  },
  { title: 'RMSNorm', key: 'rms_norm' },
  { title: 'QK RMSNorm', key: 'qk_norm' },
  { title: 'FFN type', key: 'ffn_type' },
  { title: 'Untied token embedding / LM head', key: 'untied_embeddings' },
  {
    title: 'Hidden layer size',
    key: 'hidden',
  },
  {
    title: 'Feedforward dimension size',
    key: 'feedforward',
  },
  {
    title: 'Number of layers',
    key: 'num_blocks',
  },
  {
    title: 'Vocabulary size',
    key: 'vocab_size',
  },
  {
    title: 'Number of routed experts (MoE)',
    key: 'num_experts',
  },
  {
    title: 'Experts per token (MoE top-k)',
    key: 'moe_topk',
  },
  {
    title: 'Normalize top-k probabilities (MoE)',
    key: 'norm_topk_prob',
  },
  {
    title: 'Router auxiliary loss coefficient',
    key: 'router_aux_loss_coef',
  },
  {
    title: 'Number of shared experts',
    key: 'num_shared_experts',
  },
  {
    title: 'MoE feedforward dimension size',
    key: 'moe_feedforward',
  },
  {
    title: 'First-k dense layers',
    key: 'first_k_dense',
  },
  {
    title: 'MoE layer frequency',
    key: 'moe_layer_freq',
  },
  {
    title: 'Per-token KV size (CP)',
    key: 'kv_size',
  },
  {
    title: 'Q LoRA rank (MLA)',
    key: 'q_lora_rank',
  },
  {
    title: 'KV LoRA rank (MLA)',
    key: 'kv_lora_rank',
  },
  {
    title: 'QK nope head dim (MLA)',
    key: 'qk_nope_head_dim',
  },
  {
    title: 'QK rope head dim (MLA)',
    key: 'qk_rope_head_dim',
  },
  {
    title: 'V head dim (MLA)',
    key: 'v_head_dim',
  },
  // minibatch_size
];

// MoE 字段对 dense 模型可选，不参与必填校验
const MOE_KEYS = [
  'num_experts',
  'moe_topk',
  'norm_topk_prob',
  'router_aux_loss_coef',
  'num_shared_experts',
  'moe_feedforward',
  'first_k_dense',
  'moe_layer_freq',
  'kv_size',
  'kv_heads',
  'rope_theta',
  'rms_norm',
  'qk_norm',
  'ffn_type',
  'untied_embeddings',
  'q_lora_rank',
  'kv_lora_rank',
  'qk_nope_head_dim',
  'qk_rope_head_dim',
  'v_head_dim',
];

const NUM_PARAMS_LIST = [
  {
    title: 'Total parameters',
    key: 'total_parameters',
  },
  {
    title: 'Word embedding',
    key: 'word_embedding',
  },
  {
    title: 'Self attention',
    key: 'self_attention',
  },
  {
    title: 'Feed forward',
    key: 'feed_forward',
  },
  {
    title: 'Position embedding',
    key: 'position_embedding',
  },
];

export interface IModelSelectionProps {}
const ModelSelection: FC<IModelSelectionProps> = (props) => {
  const { setProject, setOtherConfig, curModel, modelMetrics } =
    useModel(ProjectModel);
  const { setChangeLog } = useModel(LogModel);
  const { t } = useTranslation();

  const handleItemClick = (key: string, item: any) => {
    setChangeLog('Model', item?.name, curModel?.name);
    setProject({
      curModel: {
        ...item,
        ...item?.obj,
      },
    });
    // EP has no work to distribute for dense models. Reset any stale MoE
    // setting so the parallelism product cannot claim inactive GPUs.
    if (!item?.obj?.num_experts) {
      setOtherConfig({ expert_par: 1, context_par: 1 });
    }
  };

  const [state, setState] = useImmer({
    MODEL_LIST: [] as any[],
    showAddModal: false,
    newModel: {} as any,
  });

  const loadModelList = async () => {
    const localItems =
      JSON.parse(localStorage.getItem('local_models') || '[]') || [];
    const modelRes: any = await getModelList();
    if (modelRes.error) return;
    setState((prev) => ({
      ...prev,
      MODEL_LIST: [
        ...modelRes.map((item: any) => {
          return {
            key: item.name,
            label: item.name,
            value: item.name,
            obj: item,
          };
        }),
        ...localItems,
      ],
    }));
  };
  const showAddModal = () => {
    setState({
      ...state,
      showAddModal: true,
    });
  };
  const closeAddModal = () => {
    setState({
      ...state,
      showAddModal: false,
    });
  };
  const setNewModel = (newItem: any) => {
    setState({
      ...state,
      newModel: newItem,
    });
  };
  const addItemToList = () => {
    const isNotComplete = PARAMS_LIST.find(
      (p) => !MOE_KEYS.includes(p.key) && !state.newModel[p.key],
    );
    if (isNotComplete) {
      message.warn('Please fill it out completely!');
      return;
    }
    const newItem = {
      ...state.newModel,
      key: state.newModel.name,
      label: state.newModel.name,
      value: state.newModel.name,
    };
    const newModelList = [...state.MODEL_LIST, newItem];
    setState({
      ...state,
      MODEL_LIST: newModelList,
      showAddModal: false,
    });
    setProject({
      curModel: {
        ...newItem,
      },
    });
    const localItems =
      JSON.parse(localStorage.getItem('local_models') || '[]') || [];
    localStorage.setItem(
      'local_models',
      JSON.stringify([...localItems, newItem]),
    );
  };

  useEffect(() => {
    loadModelList();
  }, []);

  return (
    <div className={styles.model_wrapper}>
      <p className={styles.section_title}>{t('select title')} Model</p>
      <div className={styles.section_content}>
        <Select
          options={state.MODEL_LIST}
          value={curModel?.value}
          placeholder={t('Please select')}
          onChange={handleItemClick}
          dropdownRender={(menu) => (
            <>
              {menu}
              <Divider />
              <Button
                type="link"
                icon={<PlusOutlined />}
                style={{ padding: '0 10px' }}
                onClick={showAddModal}
              >
                {t('add item')}
              </Button>
            </>
          )}
        ></Select>
      </div>

      <p className={styles.section_title}>
        {/* Parameters */}
        {t('parameters')}
      </p>
      <div>
        {curModel?.value ? (
          <div className={styles.gpu_params}>
            {PARAMS_LIST.filter(
              (pItem) =>
                !MOE_KEYS.includes(pItem.key) || curModel[pItem.key] != null,
            ).map((pItem, _idx, arr) => (
              <div key={_idx}>
                <div className={styles.gpu_params_item}>
                  <div className={styles.gpu_params_label}>{pItem.title}</div>
                  <div className={styles.gpu_params_value}>
                    {curModel[pItem.key]}
                  </div>
                </div>
                {_idx < arr.length - 1 && <Divider />}
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.to_tips}>
            <Empty />
          </div>
        )}
      </div>

      <Drawer
        title={t('add item')}
        placement="right"
        width={600}
        // getPopupContainer={(node: any) => {
        //   if (node) {
        //     return node.parentNode;
        //   }
        //   return document.body;
        // }}
        onClose={closeAddModal}
        open={state.showAddModal}
      >
        <div className="gpu_params">
          {PARAMS_LIST.map((pItem, _idx) => (
            <div key={_idx}>
              <div className="gpu_params_item">
                <div className="gpu_params_label">{pItem.title}</div>
                <div className="gpu_params_value">
                  {pItem.key === 'name' ? (
                    <Input
                      required
                      className="number_controls"
                      value={state.newModel[pItem.key]}
                      onChange={(e: any) => {
                        setNewModel({
                          ...state.newModel,
                          [pItem.key]: e.target.value,
                        });
                      }}
                    />
                  ) : (
                    <InputNumber
                      controls={false}
                      required
                      className="number_controls"
                      value={state.newModel[pItem.key]}
                      onChange={(val: any) => {
                        setNewModel({
                          ...state.newModel,
                          [pItem.key]: val,
                        });
                      }}
                    />
                  )}
                </div>
              </div>
              {_idx < PARAMS_LIST.length - 1 && <Divider />}
            </div>
          ))}
        </div>
        <div className="add-item-footer">
          <Button onClick={closeAddModal}>CANCEL</Button>
          <Button type="primary" onClick={addItemToList}>
            ADD
          </Button>
        </div>
      </Drawer>
    </div>
  );
};

export default ModelSelection;
